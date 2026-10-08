r"""
cnn_sweep.py
============

**数層の CNN の学習解 θ\* における RLCT を、層数・活性化・テイラー次数・
タスク別にまとめる。**

------------------------------------------------------------------
何を測っているのか
------------------------------------------------------------------
`DeepCNN` を

    [Conv2d -> act -> (MaxPool2d)] x depth -> Flatten -> Linear -> act -> Linear

の形で作り、分類 (交差エントロピー) または回帰 (二乗誤差) で**学習させてから**、
その θ\* における局所 RLCT を計算する。軸は 4 本:

  depth        : 畳み込み層の数 (1, 2, 3, …)
  act          : 'relu' か 'tanh'
  taylor_order : tanh を θ\* の前活性化まわりで何次まで展開するか
                 (**ReLU には無関係**。区分線形なのでセルを固定すれば厳密)
  loss         : 'classification' か 'regression'

ReLU と tanh は**別のモデル**なので λ を直接比べる意味はない。比べられるのは

  - 同じ活性化で層数を変えたときの λ の動き
  - tanh で次数を上げたときに λ が止まるか (止まれば打ち切りは局所構造を
    変えていない)

------------------------------------------------------------------
λ の意味 — 必ず下界
------------------------------------------------------------------
- `params='all'` なら全パラメータを動かすので、その θ\* における
  **部分空間への制限なしの** λ。ただし ReLU のセル境界に θ\* が乗っていると
  錐への制限をしていないぶん下界になる
- `params='head'` などで一部だけ動かすと、凍結した方向を除いた部分空間の
  値なので**全体の λ の下界**になる (`K = x²+y²` は λ=1 だが `y=0` に
  制限すると 1/2)

生成元の本数はデータ点数で決まる (分類は `n_data x (C-1)`、回帰は
`n_data x n_out`)。**パラメータ数より生成元が少ないと残りは全部「自由方向」**
になり、λ はデータの少なさで決まってしまって層数の情報が出ない。
`--n-data` はパラメータ数と同程度まで取ること。レポートに `gens` と `free` を
並べてあるのはこの確認のため。

------------------------------------------------------------------
使い方
------------------------------------------------------------------
    python cnn_sweep.py --depths 1,2,3 --orders 1,2,3 --losses both
    python cnn_sweep.py --acts relu --depths 1,2,3,4 --n-data 12
    python cnn_sweep.py --params head --max-vars 10 --timeout 300
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from typing import Dict, List, Optional, Sequence

import sympy as sp

__all__ = ["run_one", "sweep", "summarize", "Spec"]


# ----------------------------------------------------------------------
# 1 ケース
# ----------------------------------------------------------------------
class Spec(dict):
    """1 ケースの指定。dict なので pickle してワーカに渡せる。"""

    @property
    def label(self) -> str:
        a = self["act"]
        o = f" n={self['taylor_order']}" if a != "relu" else ""
        return (f"{self['loss'][:3]} {a}{o} depth={self['depth']}")


def _build_and_train(spec: Dict):
    r"""モデルを作って学習させ、(model, ds, train_report, info) を返す。

    **θ\* が良い解でなければ λ はその点の値でしかない**ので、`fit`
    (分類は精度、回帰は R²) が `fit_min` に届くまで**種を振り直す**。
    深い ReLU では初期値によって全ユニットが死んで学習が始まらないことが
    あり、その点の λ を表に載せても層数の比較にならない。どの種を使ったか、
    何回試したかは記録して報告する。
    """
    import torch

    from train_rlct import (DeepCNN, synthetic_classification,
                           synthetic_regression, train)

    shape = (spec.get("channels_in", 1), spec["size"], spec["size"])
    seeds = spec.get("seeds") or [spec["seed"]]
    fit_min = spec.get("fit_min", 0.95)
    best = None
    for i, sd in enumerate(seeds):
        if spec["loss"] == "regression":
            ds = synthetic_regression(n=spec["n"], shape=shape,
                                      n_out=spec["n_out"], seed=sd)
            n_out = spec["n_out"]
        else:
            ds = synthetic_classification(n=spec["n"], shape=shape,
                                          n_classes=spec["n_classes"], seed=sd)
            n_out = spec["n_classes"]
        torch.manual_seed(sd)
        model = DeepCNN(ds.shape, n_out, depth=spec["depth"],
                        channels=spec["channels"], hidden=spec["hidden"],
                        kernel=spec["kernel"], act=spec["act"])
        rep = train(model, ds, epochs=spec["epochs"], lr=spec["lr"], seed=sd)
        if best is None or rep.final_acc > best[2].final_acc:
            best = (model, ds, rep, dict(seed_used=sd, tries=i + 1))
        if rep.final_acc >= fit_min:
            return model, ds, rep, dict(seed_used=sd, tries=i + 1,
                                        fit_ok=True)
    m, d, r, info = best
    # fit_min に届かなかったので全部の種を試している。`tries` は「最良の種の
    # 番号」ではなく**実際に試した本数**を報告する (でないと 4 種試して駄目
    # だったのが「1 種試行」と出て、調べ足りないように見える)
    info["fit_ok"] = False
    info["tries"] = len(seeds)
    return m, d, r, info


def _params_for(model, mode: str, max_vars: int) -> Optional[List[str]]:
    """動かすパラメータを選ぶ。

    'all'  : 全部 (部分空間への制限がないので一番素直)
    'head' : 出力側から max_vars 個ぶん (`train_rlct._param_subset` と同じ)
    'conv' : 畳み込み層のパラメータだけ (層数を変えると変数の数が変わる)
    """
    named = list(model.named_parameters())
    if mode == "all":
        return None

    if mode == "conv":
        # 畳み込み層を**入力側から**、予算に収まるぶんだけ層ごと丸ごと取る。
        # 層数を変えても最初の conv は同じ形なので、「同じパラメータを
        # 動かして上に層を積む」比較になる。
        by_layer: Dict[str, List] = {}
        for n, p in named:
            if p.dim() == 4 or (p.dim() == 1 and
                                n.rsplit(".", 1)[0] in by_layer):
                by_layer.setdefault(n.rsplit(".", 1)[0], []).append((n, p))
        out: List[str] = []
        tot = 0
        for pref in by_layer:                       # dict は挿入順
            grp = by_layer[pref]
            if not any(p.dim() == 4 for _, p in grp):
                continue
            k = sum(p.numel() for _, p in grp)
            if tot + k > max_vars:
                break
            out += [n for n, _ in grp]
            tot += k
        return out or None

    # mode == 'head': 出力側から**連続して**取る。予算に入らないものが出たら
    # そこで止める (飛ばして先の層を拾うと、層をまたいだ中途半端な部分空間に
    # なって層数の比較にならない)
    out, tot = [], 0
    for n, p in reversed(named):
        k = p.numel()
        if tot + k > max_vars:
            break
        out.append(n)
        tot += k
    return list(reversed(out))


def run_one(spec: Dict) -> Dict:
    """1 ケースを解く。例外も timeout も記録として返す。"""
    sys.path[:0] = ["/home/claude"]
    import timing
    from torch_rlct import torch_local_rlct
    from train_rlct import _stratified

    rec: Dict = dict(spec)
    t0 = time.time()
    try:
        model, ds, trep, tinfo = _build_and_train(spec)
        rec.update(tinfo)
        rec["n_params"] = sum(p.numel() for p in model.parameters())
        rec["fit"] = round(float(trep.final_acc), 4)
        rec["train_loss"] = round(float(trep.final_loss), 6)
        nd = spec["n_data"]
        X = ds.X[_stratified(ds, nd)]
        rec["t_train"] = round(time.time() - t0, 2)
        names = _params_for(model, spec["params"], spec["max_vars"])
        rec["n_moved"] = ("all" if names is None
                          else sum(dict((n, p.numel()) for n, p
                                        in model.named_parameters())[n]
                                   for n in names))
        with timing.record("cnn_sweep", depth=spec["depth"]) as _:
            pass
        t1 = time.time()
        rep = torch_local_rlct(
            model, X.reshape(X.shape[0], -1), loss=spec["loss"],
            params=names, input_shape=ds.shape,
            max_vars=spec["max_vars_hard"],
            max_denominator=spec["max_denominator"],
            taylor_order=spec["taylor_order"],
            zero_tol=spec["zero_tol"],
            max_depth=spec["max_depth"])
        L = rep.local
        # 生成元の本数は決まっている: 分類は n_data x (C-1)、回帰は n_data x n_out
        per = (spec["n_classes"] - 1 if spec["loss"] == "classification"
               else spec["n_out"])
        rec.update(status="ok",
                   rlct=str(L.rlct), mult=L.multiplicity,
                   n_gens=nd * per,
                   n_vars=L.n_vars_initial,
                   free=len(L.free_vars), gauge=len(L.gauge_fixed),
                   regular=len(L.regular_vars), core=len(L.core_vars),
                   lam_regular=str(L.lam_regular),
                   lam_core=str(L.lam_core),
                   core_route=getattr(L, "core_route", ""),
                   cells=getattr(rep, "cells", None),
                   dead=len(getattr(rep, "dead_units", []) or []),
                   t_rlct=round(time.time() - t1, 2))
        r = timing.last_record()
        if r:
            rec["phases"] = {k: round(v, 2) for k, v in r.exclusive.items()}
    except Exception as e:                                # noqa: BLE001
        rec.update(status="error", note=f"{type(e).__name__}: {str(e)[:100]}")
    rec["wall"] = round(time.time() - t0, 2)
    return rec


# ----------------------------------------------------------------------
# 掃引
# ----------------------------------------------------------------------
def _worker(spec: Dict, q) -> None:
    sys.path[:0] = ["/home/claude"]
    q.put(run_one(spec))


def sweep(specs: Sequence[Dict], *, timeout: int = 300,
          verbose: bool = True) -> List[Dict]:
    """各ケースを**別プロセス**で解く。1 つが止まっても掃引は進む。"""
    mp.set_start_method("fork", force=True)
    out: List[Dict] = []
    for sp_ in specs:
        q = mp.Queue()
        p = mp.Process(target=_worker, args=(dict(sp_), q))
        t = time.time()
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate(); p.join()
            rec = dict(sp_); rec.update(status="timeout", wall=float(timeout))
        else:
            try:
                rec = q.get_nowait()
            except Exception:
                rec = dict(sp_)
                rec.update(status="crash", wall=round(time.time() - t, 2))
        out.append(rec)
        if verbose:
            print(_line(rec), flush=True)
    return out


def _line(r: Dict) -> str:
    head = (f"  {r['loss'][:3]:<4} {r['act']:<4} "
            f"n={r['taylor_order'] if r['act'] != 'relu' else '-':<2} "
            f"depth={r['depth']:<2}")
    if r.get("status") != "ok":
        return (f"{head} {r.get('status'):<8} {r.get('note', '')} "
                f"({r.get('wall')}s)")
    return (f"{head} lambda={r['rlct']:<6} m={r['mult']:<2} "
            f"par={r['n_params']:<3} var={r['n_vars']:<3} "
            f"gens={r.get('n_gens', '?'):<3} "
            f"free={r['free']:<3} gauge={r['gauge']:<2} "
            f"reg={r['regular']:<3} core={r['core']:<2} "
            f"fit={r['fit']:.2f}"
            + (f"(seed {r['seed_used']})" if r.get("tries", 1) > 1 else "")
            + f" {r['wall']}s")


def summarize(recs: List[Dict]) -> str:
    """層数ごと・次数ごとの表と、判定の文章を作る。"""
    L: List[str] = []
    ok = [r for r in recs if r.get("status") == "ok"]

    # --- 表 1: 層数 x (活性化, 次数) ---
    losses = sorted({r["loss"] for r in recs})
    for loss in losses:
        rows = [r for r in recs if r["loss"] == loss]
        cols = sorted({(r["act"], r["taylor_order"] if r["act"] != "relu"
                        else -1) for r in rows},
                      key=lambda c: (c[0] != "relu", c[1]))
        depths = sorted({r["depth"] for r in rows})
        L.append(f"\n--- {loss} : lambda (行=層数, 列=活性化/次数) ---")
        hdr = "  depth |" + "".join(
            f" {a if o < 0 else a + ' n=' + str(o):>10} |" for a, o in cols)
        L.append(hdr)
        L.append("  " + "-" * (len(hdr) - 2))
        for d in depths:
            cells = []
            for a, o in cols:
                m = [r for r in rows if r["depth"] == d and r["act"] == a
                     and (o < 0 or r["taylor_order"] == o)]
                if not m:
                    cells.append(f" {'':>10} |")
                elif m[0].get("status") == "ok":
                    cells.append(f" {m[0]['rlct']:>10} |")
                else:
                    cells.append(f" {m[0].get('status', '?')[:10]:>10} |")
            L.append(f"  {d:<5} |" + "".join(cells))

    # --- 判定 1: tanh の次数で lambda は止まるか ---
    L.append("\n--- tanh の打ち切り次数について ---")
    groups: Dict = {}
    for r in ok:
        if r["act"] == "relu":
            continue
        groups.setdefault((r["loss"], r["depth"]), []).append(r)
    if not groups:
        L.append("  tanh で解けたケースがありません。")
    for (loss, d), g in sorted(groups.items()):
        g = sorted(g, key=lambda r: r["taylor_order"])
        vals = [(r["taylor_order"], sp.nsimplify(r["rlct"])) for r in g]
        s = ", ".join(f"n={o}: {v}" for o, v in vals)
        if len(vals) < 2:
            L.append(f"  {loss} depth={d}: {s}  -> 1 次数しか解けていないので"
                     "判定できません")
        elif len({v for _, v in vals}) == 1:
            L.append(f"  {loss} depth={d}: {s}  -> 全次数一致。"
                     "打ち切りは局所構造を変えていないと見られます")
        else:
            last = vals[-1][1]
            stable = [o for o, v in vals if v == last]
            if len(stable) >= 2 and stable[-1] == vals[-1][0]:
                L.append(f"  {loss} depth={d}: {s}  -> n >= {min(stable)} で"
                         f"{last} に落ち着いています")
            else:
                L.append(f"  {loss} depth={d}: {s}  -> **まだ動いています。"
                         "確定値として扱ってはいけません**")

    # --- 判定 2: 層数を上げると lambda はどう動くか ---
    L.append("\n--- 層数について ---")
    byact: Dict = {}
    for r in ok:
        key = (r["loss"], r["act"],
               r["taylor_order"] if r["act"] != "relu" else -1)
        byact.setdefault(key, []).append(r)
    for (loss, a, o), g in sorted(byact.items(),
                                  key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        g = sorted(g, key=lambda r: r["depth"])
        if len(g) < 2:
            continue
        s = ", ".join(f"d={r['depth']}: {r['rlct']}" for r in g)
        name = a if o < 0 else f"{a} n={o}"
        vals = [sp.nsimplify(r["rlct"]) for r in g]
        if len(set(vals)) == 1:
            verdict = "層数を変えても lambda は同じ"
        elif all(b >= a2 for a2, b in zip(vals, vals[1:])):
            verdict = "層数とともに単調に増加"
        elif all(b <= a2 for a2, b in zip(vals, vals[1:])):
            verdict = "層数とともに単調に減少"
        else:
            verdict = "単調ではない"
        L.append(f"  {loss} {name}: {s}  -> {verdict}")

    # --- 注意書き ---
    bad = [r for r in ok if r.get("fit_ok") is False]
    if bad:
        L.append("\n[注意] 種を振り直しても学習しきれなかったケースがあります。"
                 "θ* が良い解でないので lambda は「その点の」値でしかなく、"
                 "**層数の比較には使えません**:")
        for r in bad:
            L.append(f"    {_spec_name(r)}  fit={r['fit']:.2f} "
                     f"({r.get('tries')} 種試行)")
    retried = [r for r in ok if r.get("tries", 1) > 1 and r.get("fit_ok")]
    if retried:
        L.append("\n[注意] 既定の種では学習できず、別の種を使ったケース:")
        for r in retried:
            L.append(f"    {_spec_name(r)}  seed={r.get('seed_used')} "
                     f"({r.get('tries')} 種目)")
    thin = [r for r in ok
            if r.get("n_gens", 10 ** 9) < r.get("n_vars", 0) - r.get("free", 0)]
    if thin:
        L.append("\n[注意] 生成元の本数が拘束された方向の数より少ないケースが"
                 "あります。データ点を増やすと lambda が変わりえます "
                 "(--n-data):")
        for r in thin:
            L.append(f"    {_spec_name(r)}  gens={r['n_gens']} < "
                     f"{r['n_vars'] - r['free']}")
    free = [r for r in ok if r.get("free", 0) > 0.6 * max(r.get("n_vars", 1), 1)]
    if free:
        L.append("\n[注意] 自由方向が変数の 6 割を超えているケースがあります。"
                 "lambda に効いているのは残りの方向だけです。原因は "
                 "(a) データ点が足りない か "
                 "(b) 隠れ層のボトルネックや死んだユニットで、動かしている"
                 "パラメータが出力に効いていない のどちらか。"
                 "gens と dead を見て切り分けてください:")
        for r in free:
            L.append(f"    {_spec_name(r)}  free={r['free']}/{r['n_vars']} "
                     f"gens={r.get('n_gens')} dead={r.get('dead')}")
    L.append("\n[注意] ReLU と tanh は別のモデルなので lambda を直接"
             "比べる意味はありません。層数と次数の**動き**だけが比較対象です。")
    L.append("[注意] θ* がセル境界に乗っている場合、錐への制限をしていないので"
             "lambda は下界です。部分空間に制限した場合も下界です。")
    return "\n".join(L)


def _spec_name(r: Dict) -> str:
    o = "" if r["act"] == "relu" else f" n={r['taylor_order']}"
    return f"{r['loss']} {r['act']}{o} depth={r['depth']}"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--acts", default="relu,tanh")
    ap.add_argument("--orders", default="1,2,3",
                    help="tanh のテイラー次数 (relu では無視)")
    ap.add_argument("--losses", default="both",
                    choices=("classification", "regression", "both"))
    # データとモデルの形
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--channels", type=int, default=1, help="畳み込みの出力 ch")
    ap.add_argument("--kernel", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--out-dim", type=int, default=1, help="回帰の出力次元")
    ap.add_argument("--n", type=int, default=48, help="学習データ点数")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="",
                    help="fit が --fit-min に届くまで順に試す種 (例 0,1,2,3)")
    ap.add_argument("--fit-min", type=float, default=0.95,
                    help="分類は精度、回帰は R^2 のしきい値")
    # RLCT
    ap.add_argument("--n-data", type=int, default=10,
                    help="RLCT を見るデータ点数 (生成元の本数を決める)")
    ap.add_argument("--params", default="all",
                    choices=("all", "head", "conv"))
    ap.add_argument("--max-vars", type=int, default=12,
                    help="params=head/conv のときに選ぶ変数の上限")
    ap.add_argument("--max-vars-hard", type=int, default=80,
                    help="これを超えたら例外 (暴走防止)")
    ap.add_argument("--max-denominator", type=int, default=16)
    ap.add_argument("--zero-tol", type=float, default=1e-2)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import rlct_cache
    if args.cache:
        rlct_cache.enable(args.cache)
    else:
        rlct_cache.disable()

    depths = [int(v) for v in args.depths.split(",") if v]
    acts = [v for v in args.acts.split(",") if v]
    orders = [int(v) for v in args.orders.split(",") if v]
    losses = (["classification", "regression"] if args.losses == "both"
              else [args.losses])

    base = dict(size=args.size, channels=args.channels, kernel=args.kernel,
                hidden=args.hidden, n_classes=args.classes,
                n_out=args.out_dim, n=args.n, epochs=args.epochs, lr=args.lr,
                seed=args.seed,
                seeds=[int(v) for v in args.seeds.split(",") if v]
                or [args.seed],
                fit_min=args.fit_min,
                n_data=args.n_data, params=args.params,
                max_vars=args.max_vars, max_vars_hard=args.max_vars_hard,
                max_denominator=args.max_denominator, zero_tol=args.zero_tol,
                max_depth=args.max_depth)
    specs: List[Dict] = []
    for loss in losses:
        for act in acts:
            for d in depths:
                if act == "relu":
                    specs.append(dict(base, loss=loss, act=act, depth=d,
                                      taylor_order=0))
                else:
                    for o in orders:
                        specs.append(dict(base, loss=loss, act=act, depth=d,
                                          taylor_order=o))

    print("===== 数層 CNN の学習解における RLCT =====")
    print(f"  画像 {args.channels}x{args.size}x{args.size}, conv ch="
          f"{args.channels}, kernel={args.kernel}, hidden={args.hidden}, "
          f"動かす変数={args.params}")
    print(f"  RLCT のデータ点 {args.n_data} 本, zero_tol={args.zero_tol}, "
          f"max_denominator={args.max_denominator}")
    print(f"  {len(specs)} ケース, 1 ケース {args.timeout}s まで\n")
    recs = sweep(specs, timeout=args.timeout)
    print("\n===== まとめ =====")
    print(summarize(recs))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(recs, fh, ensure_ascii=False, indent=1, default=str)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
