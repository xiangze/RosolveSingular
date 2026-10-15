r"""
taylor_sweep.py
===============

Transformer 1 層の**解 θ\* のまわりで softmax を 1 次・2 次・3 次… と
テイラー展開したときの RLCT** を並べて出す。

------------------------------------------------------------------
何を測っているのか
------------------------------------------------------------------
softmax は多項式でないので、このコードは

    E_ij = e*_ij · Σ_{k ≤ n} δ_ij^k / k!,   δ_ij = S_ij(θ) − S*_ij

と **n 次で打ち切った**モデルを解析する。n を変えると**モデルが変わる**ので
λ も変わりうる。したがって意味のある問いは

    「n を上げていくと λ は止まるか、止まるならどこか」

である。softmax は解析的なので、局所の特異性を捉えきる次数を超えれば λ は
動かなくなるはずで、どこで止まるかがそのまま「この θ\* の特異性の深さ」の
目安になる。止まらない/途中で跳ねる場合は、打ち切りが局所構造を変えている
ことを意味するので、その旨を報告する。

n = 0 は特別で、注意パターンを θ\* で凍結したモデル (A = A\*) になる。
W_Q, W_K, b_Q, b_K が生成元に現れなくなるので、value/output 回路と FFN
だけの別のモデルであり、比較の基準として並べてある。

------------------------------------------------------------------
問題の立て方 (回帰 / 分類)
------------------------------------------------------------------
fiber ideal は θ\* 自身を真のパラメータとして定義する (realizable)。すなわち
データのラベル・目標値は θ\* の出力そのもので、**θ\* は損失 0 の大域解**。
これは特異学習理論の標準的な設定。

  loss='regression'
      入力列 X^(s) ごとに、層の出力 Z^(s)_{i,c} をすべて合わせる条件。
          g = Z_{i,c}(θ) − Z_{i,c}(θ*)
  loss='classification'
      cls トークン (位置 0) に線形ヘッドを載せて C クラス分類にする。
      p = softmax(logit) が一致するのは**ロジットの差が一致するとき**なので
          g_{s,c} = (logit_c − logit_C)(θ) − (logit_c − logit_C)(θ*),  c ≠ C
      を生成元にする。ロジット全体の平行移動 (ヘッドのバイアスの (1,…,1)
      方向) は p を変えないので対称性として渡す。

どちらも分母 D_i (= softmax の分母、θ\* で単元) を払って多項式にしている。

------------------------------------------------------------------
使い方
------------------------------------------------------------------
    python taylor_sweep.py --loss regression --orders 0,1,2,3 --timeout 900
    python taylor_sweep.py --loss classification --classes 3 --seqs 3
    python taylor_sweep.py --degeneracy zero_ffn --orders 0,1,2
    python taylor_sweep.py --source trained --loss classification

``--timeout`` は既定 120s だが、実測では最小の形 (d_k=d_ff=1, 列 1 本) の 3 次が
414s かかる。3 次まで見るなら ``--timeout 900`` 程度を渡すこと。
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

__all__ = ["build_problem", "run_one", "sweep", "random_sequences"]


# ----------------------------------------------------------------------
# データ (入力列)
# ----------------------------------------------------------------------
def random_sequences(n_seq: int, T: int, d_model: int, *, seed: int = 0,
                     lo: int = -2, hi: int = 2, den: int = 2):
    """有理数の入力列を n_seq 本つくる。"""
    rng = random.Random(seed)
    return [[[sp.Rational(rng.randint(lo, hi), den) for _ in range(d_model)]
             for _ in range(T)] for _ in range(n_seq)]


# ----------------------------------------------------------------------
# 問題の組み立て
# ----------------------------------------------------------------------
@dataclass
class Problem:
    generators: List[sp.Expr]
    variables: List[sp.Symbol]
    symmetry_vectors: List[Dict[sp.Symbol, sp.Rational]]
    notes: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


def build_problem(Xs, star, *, taylor_order: int, loss: str = "regression",
                  head=None, n_classes: int = 3, pool: str = "cls",
                  max_denominator: int = 64,
                  prefix: str = "t") -> Problem:
    """入力列の集合に対する fiber ideal を組む。

    Xs   : 入力列のリスト。各要素は (T x d_model)。
    star : TransformerStar (θ\*)
    head : 分類ヘッド (W: d_model x C, b: C)。loss='classification' で必要。

    `transformer_layer_ideal` を列ごとに呼び、**同じ prefix なので同じ記号**が
    使われることを利用して生成元を連結する。対称性は θ\* だけで決まるので
    1 本目のものを使う。
    """
    from transformer_rlct import transformer_layer_ideal

    if not Xs:
        raise ValueError("入力列が空です")
    if pool != "cls":
        raise NotImplementedError(
            "pool は 'cls' (位置 0) のみ。平均プーリングだと行ごとの分母の積に"
            "なって次数が倍々になるため。")
    per: List = []
    for X in Xs:
        per.append(transformer_layer_ideal(
            X, star, taylor_order=taylor_order,
            max_denominator=max_denominator, prefix=prefix))
    I0 = per[0]
    variables = list(I0.variables)
    symv = [dict(v) for v in I0.symmetry_vectors]
    notes = list(I0.notes)

    if loss == "regression":
        gens = [g for I in per for g in I.generators]
        meta = dict(n_seq=len(Xs), loss=loss, **I0.meta)
        meta["n_gens"] = len(gens)
        return Problem(gens, variables, symv, notes, meta)

    if loss != "classification":
        raise ValueError(f"未知の loss: {loss}")

    # --- 分類ヘッド -----------------------------------------------------
    d = star.d_model
    if head is None:
        rng = random.Random(12345)
        head = ([[sp.Rational(rng.randint(-3, 3), 2) for _ in range(n_classes)]
                 for _ in range(d)],
                [sp.Rational(rng.randint(-2, 2), 2) for _ in range(n_classes)])
    hW_star, hb_star = head
    C = len(hb_star)
    uW = [[sp.Symbol(f"{prefix}hW_{a}_{c}", real=True) for c in range(C)]
          for a in range(d)]
    ub = [sp.Symbol(f"{prefix}hb_{c}", real=True) for c in range(C)]
    for row in uW:
        variables.extend(row)
    variables.extend(ub)
    pW = [[hW_star[a][c] + uW[a][c] for c in range(C)] for a in range(d)]
    pb = [hb_star[c] + ub[c] for c in range(C)]

    ref = C - 1
    gens: List[sp.Expr] = []
    for I in per:
        Z0s, D0s = I.Z_sym[0], I.D_sym[0]          # 分母を払った cls トークン
        Z0n, D0n = I.Z_star[0], I.D_star[0]
        for c in range(C):
            if c == ref:
                continue
            # (logit_c − logit_ref) * D  を多項式で組む
            num = sp.expand(sum(Z0s[a] * (pW[a][c] - pW[a][ref])
                                for a in range(d))
                            + (pb[c] - pb[ref]) * D0s)
            vstar = sp.cancel(
                sum(Z0n[a] * (hW_star[a][c] - hW_star[a][ref])
                    for a in range(d)) / D0n
                + (hb_star[c] - hb_star[ref]))
            g = sp.expand(num - vstar * D0s)
            if g != 0:
                gens.append(g)
    # ロジット全体の平行移動 b_h -> b_h + t·1 は softmax を変えない
    symv.append({u: sp.Integer(1) for u in ub})
    notes.append("分類: cls トークン (位置 0) に線形ヘッドを載せ、"
                 "ロジットの差を生成元にしました。ロジットの平行移動は"
                 "対称性としてゲージ固定しています。")
    meta = dict(n_seq=len(Xs), loss=loss, n_classes=C, **I0.meta)
    meta["n_params"] = len(variables)
    meta["n_gens"] = len(gens)
    meta["n_symmetries"] = len(symv)
    return Problem(gens, variables, symv, notes, meta)


# ----------------------------------------------------------------------
# 1 つの次数を解く (子プロセスで呼ばれる)
# ----------------------------------------------------------------------
def run_one(order: int, cfg: Dict) -> Dict:
    """order 次で打ち切った問題の λ を求め、記録を返す。"""
    import rlct_cache
    import timing
    from nn_rlct import local_rlct_from_ideal
    from transformer_rlct import random_layer

    if cfg.get("cache"):
        rlct_cache.enable(cfg["cache"])
    else:
        rlct_cache.disable()

    rec: Dict = {"order": order}
    t0 = time.time()
    try:
        star, head = _make_star(cfg)
        Xs = cfg["Xs"] if "Xs" in cfg else random_sequences(
            cfg["n_seq"], cfg["T"], cfg["d_model"], seed=cfg["seed"])
        with timing.record(f"order{order}"):
            prob = build_problem(Xs, star, taylor_order=order,
                                 loss=cfg["loss"], head=head,
                                 n_classes=cfg["n_classes"],
                                 max_denominator=cfg["max_denominator"])
            rec["n_params"] = prob.meta["n_params"]
            rec["n_gens"] = prob.meta["n_gens"]
            rec["n_sym"] = prob.meta["n_symmetries"]
            rec["max_terms"] = max(
                (len(sp.Poly(g, *prob.variables).monoms())
                 for g in prob.generators), default=0)
            rec["build_s"] = round(time.time() - t0, 2)
            loc = local_rlct_from_ideal(
                prob.generators, prob.variables,
                symmetry_vectors=prob.symmetry_vectors,
                max_depth=cfg["max_depth"])
        r = timing.last_record()
        rec.update(status="ok", rlct=str(loc.rlct), mult=loc.multiplicity,
                   free=len(loc.free_vars), gauge=len(loc.gauge_fixed),
                   regular=len(loc.regular_vars), core=len(loc.core_vars),
                   core_route=loc.core_route,
                   lam_regular=str(loc.lam_regular),
                   lam_core=str(loc.lam_core))
        if r:
            rec["phases"] = {k: round(v, 2) for k, v in r.exclusive.items()}
    except Exception as e:                                # noqa: BLE001
        rec.update(status="error", note=f"{type(e).__name__}: {str(e)[:80]}")
    rec["wall"] = round(time.time() - t0, 2)
    return rec


def _make_star(cfg: Dict):
    """θ\* (と分類ヘッド) を用意する。random か学習済み。"""
    from transformer_rlct import random_layer

    if cfg.get("source") == "trained":
        return _trained_star(cfg)
    star = random_layer(d_model=cfg["d_model"], d_k=cfg["d_k"],
                        d_ff=cfg["d_ff"], n_heads=cfg["n_heads"],
                        seed=cfg["seed"], degeneracy=cfg["degeneracy"])
    head = None
    return star, head


def _trained_star(cfg: Dict):
    """TinyViT を学習させ、エンコーダ層 0 とヘッドを θ\* として取り出す。"""
    import torch

    from torch_rlct import transformer_star_from_torch
    from train_rlct import TinyViT, load_dataset, train

    torch.manual_seed(cfg["seed"])
    ds = load_dataset("synthetic", n=cfg.get("n_data", 48),
                      shape=(1, cfg.get("img", 4), cfg.get("img", 4)),
                      n_classes=cfg["n_classes"], seed=cfg["seed"])
    vit = TinyViT(ds.shape, ds.n_classes, patch=cfg.get("patch", 2),
                  d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                  d_ff=cfg["d_ff"], depth=1, layernorm=False)
    train(vit, ds, epochs=cfg.get("epochs", 300), lr=0.05, seed=cfg["seed"])
    star = transformer_star_from_torch(vit.layers[0],
                                       max_denominator=cfg["max_denominator"])
    # 分類ヘッドも学習済みのものを使う
    from torch_rlct import _rat
    Wh = vit.head.weight.detach().t()          # (d_model, C)
    bh = vit.head.bias.detach()
    head = ([[_rat(float(v), 1e-12, cfg["max_denominator"]) for v in row]
             for row in Wh],
            [_rat(float(v), 1e-12, cfg["max_denominator"]) for v in bh])
    # 入力列は学習後のトークン列 (層 0 の入力)
    with torch.no_grad():
        Z = vit.forward_upto(ds.X[:cfg["n_seq"]], 0)
    cfg["Xs"] = [[[_rat(float(v), 1e-12, cfg["max_denominator"])
                   for v in tok] for tok in seq] for seq in Z.tolist()]
    return star, head


# ----------------------------------------------------------------------
# 掃引
# ----------------------------------------------------------------------
def _worker(order: int, cfg: Dict, q) -> None:
    sys.path[:0] = ["/home/claude"]
    q.put(run_one(order, cfg))


def sweep(orders: Sequence[int], cfg: Dict, *, timeout: int = 120,
          verbose: bool = True) -> List[Dict]:
    """各次数を**別プロセス**で解く。1 つが止まっても掃引は進む。"""
    mp.set_start_method("fork", force=True)
    out: List[Dict] = []
    for o in orders:
        q = mp.Queue()
        p = mp.Process(target=_worker, args=(o, cfg, q))
        t = time.time()
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate(); p.join()
            rec = {"order": o, "status": "timeout", "wall": float(timeout)}
        else:
            try:
                rec = q.get_nowait()
            except Exception:
                rec = {"order": o, "status": "crash",
                       "wall": round(time.time() - t, 2)}
        out.append(rec)
        if verbose:
            print(_line(rec), flush=True)
    return out


def _line(r: Dict) -> str:
    if r.get("status") != "ok":
        return (f"  order={r['order']:<2} {r.get('status'):<8} "
                f"{r.get('note', '')} ({r.get('wall')}s)")
    return (f"  order={r['order']:<2} lambda={r['rlct']:<8} m={r['mult']:<3} "
            f"params={r['n_params']:<4} gens={r['n_gens']:<3} "
            f"max項={r['max_terms']:<6} "
            f"free={r['free']:<3} gauge={r['gauge']:<3} "
            f"reg={r['regular']:<3} core={r['core']:<3} "
            f"({r['lam_regular']}+{r['lam_core']}) {r['wall']}s")


def summarize(recs: List[Dict]) -> str:
    """λ が次数とともに止まっているかを判定して文章にする。"""
    ok = [r for r in recs if r.get("status") == "ok" and r["order"] >= 1]
    L = []
    if len(ok) < 2:
        L.append("1 次以上で解けた次数が 1 つ以下なので、収束は判定できません。")
        return "\n".join(L)
    vals = [(r["order"], sp.nsimplify(r["rlct"])) for r in ok]
    L.append("1 次以上の lambda: " + ", ".join(f"n={o}: {v}" for o, v in vals))
    last = vals[-1][1]
    stable = [o for o, v in vals if v == last]
    if len(set(v for _, v in vals)) == 1:
        L.append(f"-> 試した全次数で lambda = {last} に一致。"
                 "打ち切りは局所構造を変えていないと見られます。")
    elif len(stable) >= 2 and stable[-1] == vals[-1][0]:
        L.append(f"-> n >= {min(stable)} で lambda = {last} に落ち着いています。"
                 "その次数で局所構造を捉えきったと見るのが自然です。")
    else:
        L.append("-> まだ動いています。次数を上げるか、形を小さくして"
                 "もう少し先まで確かめる必要があります。"
                 "**この状態の値を確定値として扱ってはいけません。**")
    z = [r for r in recs if r.get("status") == "ok" and r["order"] == 0]
    if z:
        L.append(f"(参考) n=0 (注意を θ* で凍結) は lambda = {z[0]['rlct']}。"
                 "W_Q, W_K が自由方向になる別のモデルの値です。")
    return "\n".join(L)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="0,1,2,3")
    ap.add_argument("--loss", default="regression",
                    choices=("regression", "classification"))
    ap.add_argument("--source", default="random", choices=("random", "trained"))
    ap.add_argument("--T", type=int, default=2, help="系列長")
    ap.add_argument("--d-model", type=int, default=2)
    ap.add_argument("--d-k", type=int, default=1)
    ap.add_argument("--d-ff", type=int, default=1)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--seqs", type=int, default=1, help="入力列の本数")
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--degeneracy", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-denominator", type=int, default=64)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = dict(loss=args.loss, source=args.source, T=args.T,
               d_model=args.d_model, d_k=args.d_k, d_ff=args.d_ff,
               n_heads=args.heads, n_seq=args.seqs, n_classes=args.classes,
               degeneracy=args.degeneracy, seed=args.seed,
               max_denominator=args.max_denominator,
               max_depth=args.max_depth, cache=args.cache or None)
    orders = [int(v) for v in args.orders.split(",") if v != ""]
    print(f"===== softmax の打ち切り次数ごとの RLCT =====")
    print(f"  loss={args.loss} source={args.source} T={args.T} "
          f"d_model={args.d_model} d_k={args.d_k} d_ff={args.d_ff} "
          f"heads={args.heads} seqs={args.seqs} "
          f"degeneracy={args.degeneracy}")
    print(f"  θ* 自身を真のパラメータとする realizable な設定 "
          f"(θ* は損失 0 の大域解)")
    recs = sweep(orders, cfg, timeout=args.timeout)
    print("\n===== まとめ =====")
    print(summarize(recs))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(recs, fh, ensure_ascii=False, indent=1, default=str)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
