r"""
exp/shape_sweep.py — **形状のパラメータ掃引**を外側から回すドライバ

`exp/cnn_sweep.py` と `exp/taylor_sweep.py` は、層数・活性化・テイラー次数は
リストで受け取るが、**形状 (チャネル数・カーネル・隠れ幅・d_model など) は
スカラー**である。形状も振りたい場合はこのドライバを使う。

    # 計画と見積りだけ (何も計算しない)
    python -m exp.shape_sweep --grid cnn-small --dry-run

    # 実行 (途中で止めても同じコマンドで再開する)
    python -m exp.shape_sweep --grid cnn-small --out runs/cnn.jsonl

    # 表と CSV
    python -m exp.shape_sweep --report runs/cnn.jsonl

------------------------------------------------------------------
設計
------------------------------------------------------------------
* 形状の組ごとに**子プロセス**で内側の掃引を起動する。内側もさらに次数ごとに
  プロセスを分けているので、1 つが固まっても掃引全体は進む。
* 結果は **JSONL に 1 行 1 レコードで追記**する。各行に形状の欄が入るので、
  あとから pandas でも素の jq でも集計できる。
* **再開できる**。既に JSONL にある (grid, 形状, 層数, 活性化, 次数, 損失) の
  組は飛ばす。長い掃引を分割して回せる。
* **タイムアウトと失敗を落とさない**。`status` が `ok` / `timeout` /
  `error` / `skipped` のいずれかで必ず記録される。λ が出なかったことも
  データであり、黙って消すと「解ける範囲」の表が嘘になる。
* 乱数種は固定し、`--seeds` の再試行は内側に任せる (θ\* が良い解でないと
  λ の比較に意味がないため。`cnn_sweep` の `--fit-min` を参照)。

------------------------------------------------------------------
実測にもとづく見積り (1 形状あたり)
------------------------------------------------------------------
CNN (size=8, ch=2, kernel=2, hidden=4, n_data=8, params=conv, max_vars=10):

    分類    relu  depth 1/2/3 = 121 / 203 / 225 秒
            tanh  次数 1 で 31 / 97 秒、**次数 2 以上は 600 秒で未完**
    回帰    relu  115 / 116 / 249 秒、tanh 次数 1 で 36 秒

Transformer (d_model=2, d_k=1, d_ff=1, T=2, seqs=1, 非退化):

    次数 0 / 1 / 2 / 3 = 0.1 / 1.8 / 34 / 414 秒
    退化点 (uniform_att) は次数 1 で 18 秒、2 以上は未完

**次数 2 以上と、形状を上げた場合はほぼ未完になる。**見積りは
`--dry-run` が「最悪の壁時計時間」として出すので、投じる計算量を先に決められる。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------
# 形状の格子
# ----------------------------------------------------------------------
# 「同じ変数を振って層を積む」比較を壊さないため、params=conv のときは
# max_vars を第 1 畳み込みのパラメータ数に合わせる
# (kernel^2 * channels * in_ch + channels)。cnn_sweep の
# `--params conv` の説明と `guards.cnn_depth_tests` を参照。
GRIDS: Dict[str, Dict] = {
    # --- CNN: 層数 x 活性化 x 次数、形状は小さめを 4 通り ------------
    "cnn-small": dict(
        kind="cnn",
        depths="1,2,3", acts="relu,tanh", orders="1,2", losses="both",
        shapes=[
            dict(size=8, channels=2, kernel=2, hidden=4, n_data=8,
                 params="conv", max_vars=10),
            dict(size=8, channels=2, kernel=3, hidden=4, n_data=8,
                 params="conv", max_vars=20),
            dict(size=8, channels=3, kernel=2, hidden=4, n_data=8,
                 params="conv", max_vars=15),
            dict(size=12, channels=2, kernel=2, hidden=4, n_data=8,
                 params="conv", max_vars=10),
        ],
        timeout=600,
    ),
    # --- CNN: データ点数と隠れ幅だけを振る (識別可能性の効果を見る) ---
    "cnn-data": dict(
        kind="cnn",
        depths="1,2", acts="relu", orders="1", losses="both",
        shapes=[
            dict(size=8, channels=2, kernel=2, hidden=h, n_data=n,
                 params="conv", max_vars=10)
            for h in (2, 4, 8) for n in (4, 8, 16)
        ],
        timeout=600,
    ),
    # --- Transformer: 次数を固定して形状を振る ----------------------
    "tf-fixed": dict(
        kind="transformer",
        orders="0,1", losses="regression,classification",
        shapes=[
            dict(T=2, d_model=2, d_k=1, d_ff=1, heads=1, seqs=1,
                 degeneracy="none"),
            dict(T=2, d_model=2, d_k=1, d_ff=2, heads=1, seqs=1,
                 degeneracy="none"),
            dict(T=2, d_model=2, d_k=2, d_ff=2, heads=1, seqs=1,
                 degeneracy="none"),
            dict(T=3, d_model=2, d_k=1, d_ff=1, heads=1, seqs=1,
                 degeneracy="none"),
            dict(T=2, d_model=2, d_k=1, d_ff=1, heads=1, seqs=2,
                 degeneracy="none"),
            dict(T=2, d_model=2, d_k=1, d_ff=1, heads=1, seqs=1,
                 degeneracy="uniform_att"),
            dict(T=2, d_model=2, d_k=1, d_ff=2, heads=1, seqs=1,
                 degeneracy="dead_ffn"),
        ],
        timeout=600,
    ),
    # --- Transformer: 次数そのものを伸ばす (最小形状だけ) -----------
    "tf-orders": dict(
        kind="transformer",
        orders="0,1,2,3", losses="regression",
        shapes=[dict(T=2, d_model=2, d_k=1, d_ff=1, heads=1, seqs=1,
                     degeneracy="none")],
        timeout=1800,
    ),
}

# 実測にもとづく 1 ケースあたりの秒数の目安 (見積りだけに使う)
_COST = {
    ("cnn", "relu", 1): 130, ("cnn", "relu", 2): 210, ("cnn", "relu", 3): 250,
    ("cnn", "tanh", 1): 100, ("cnn", "tanh", 2): 600, ("cnn", "tanh", 3): 600,
    ("tf", 0): 1, ("tf", 1): 20, ("tf", 2): 600, ("tf", 3): 1800,
}


# ----------------------------------------------------------------------
# 計画
# ----------------------------------------------------------------------
def _plan(name: str) -> Tuple[Dict, List[Dict]]:
    if name not in GRIDS:
        raise SystemExit(f"格子 {name!r} は未定義。--list で一覧。")
    g = GRIDS[name]
    jobs = []
    for i, shp in enumerate(g["shapes"]):
        jobs.append(dict(grid=name, kind=g["kind"], shape_id=i, shape=shp))
    return g, jobs


def _estimate(g: Dict, jobs: List[Dict]) -> float:
    """最悪の壁時計時間 (秒)。タイムアウトで打ち切られる分も満額で数える。"""
    orders = [int(v) for v in str(g["orders"]).split(",") if v != ""]
    losses = (["classification", "regression"] if g.get("losses") == "both"
              else [v for v in str(g.get("losses", "regression")).split(",") if v])
    tot = 0.0
    for _j in jobs:
        if g["kind"] == "cnn":
            depths = [int(v) for v in g["depths"].split(",")]
            acts = [v for v in g["acts"].split(",")]
            for d, a, o, _l in itertools.product(depths, acts, orders, losses):
                if a == "relu" and o != orders[0]:
                    continue            # relu は次数に無関係 (1 回だけ)
                tot += min(_COST.get(("cnn", a, d), 250), g["timeout"])
        else:
            for o, _l in itertools.product(orders, losses):
                tot += min(_COST.get(("tf", o), 600), g["timeout"])
    return tot


# ----------------------------------------------------------------------
# 実行
# ----------------------------------------------------------------------
def _order_of(rec: Dict):
    """次数の欄は掃引によって名前が違う。

    cnn_sweep  は Spec をそのまま返すので `taylor_order`
    taylor_sweep は `order`
    """
    v = rec.get("taylor_order", rec.get("order"))
    return None if v is None else int(v)


def _mult_of(rec: Dict):
    """位数の欄も名前が違う (cnn: multiplicity / taylor: mult)。"""
    return rec.get("mult", rec.get("multiplicity"))


def _key(rec: Dict) -> Tuple:
    """再開のための識別子。"""
    return (rec.get("grid"), rec.get("shape_id"), rec.get("depth"),
            rec.get("act"), _order_of(rec), rec.get("loss"))


def _load_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_key(json.loads(line)))
            except Exception:                         # noqa: BLE001
                continue
    return done


def _cmd(g: Dict, job: Dict, tmp_out: str, extra: List[str],
         loss: Optional[str] = None) -> List[str]:
    shp = job["shape"]
    if g["kind"] == "cnn":
        c = [sys.executable, "-m", "exp.cnn_sweep",
             "--depths", g["depths"], "--acts", g["acts"],
             "--orders", str(g["orders"]), "--losses", g.get("losses", "both"),
             "--size", str(shp["size"]), "--channels", str(shp["channels"]),
             "--kernel", str(shp["kernel"]), "--hidden", str(shp["hidden"]),
             "--n-data", str(shp["n_data"]), "--params", shp["params"],
             "--max-vars", str(shp["max_vars"]),
             "--timeout", str(g["timeout"]), "--out", tmp_out]
    else:
        c = [sys.executable, "-m", "exp.taylor_sweep",
             "--orders", str(g["orders"]),
             "--T", str(shp["T"]), "--d-model", str(shp["d_model"]),
             "--d-k", str(shp["d_k"]), "--d-ff", str(shp["d_ff"]),
             "--heads", str(shp["heads"]), "--seqs", str(shp["seqs"]),
             "--degeneracy", shp["degeneracy"],
             "--timeout", str(g["timeout"]), "--out", tmp_out]
        # taylor_sweep の --loss は **単数**。宣言した損失を 1 つずつ回す。
        if loss:
            c += ["--loss", loss]
    return c + extra


def run(name: str, out: str, *, extra: List[str], hard_factor: float = 3.0,
        verbose: bool = True) -> int:
    g, jobs = _plan(name)
    done = _load_done(out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    losses = (["classification", "regression"] if g.get("losses") == "both"
              else [v for v in str(g.get("losses", "regression")).split(",") if v])
    # 形状ごとに「揃っているはずの組」を数え、**全部あるときだけ**飛ばす。
    # 1 件でもあれば飛ばす作りにすると、途中で止めた形状の残りが黙って
    # 失われる (λ が出なかったことも記録したい掃引では致命的)。
    orders = [int(v) for v in str(g["orders"]).split(",") if v != ""]
    if g["kind"] == "cnn":
        depths = [int(v) for v in g["depths"].split(",")]
        acts = [v for v in g["acts"].split(",")]
        expect = set()
        for d, a, o, lo in itertools.product(depths, acts, orders, losses):
            # relu は次数に無関係なので内側が 1 回しか出さない
            expect.add((d, a, orders[0] if a == "relu" else o, lo))
    else:
        expect = {(None, None, o, lo) for o, lo in
                  itertools.product(orders, losses)}

    n_new = 0
    for job in jobs:
        have = {(k[2], k[3], k[4], k[5]) for k in done
                if k[0] == name and k[1] == job["shape_id"]}
        if have and expect <= have:
            if verbose:
                print(f"[skip] 形状 {job['shape_id']} は {len(have)} 件揃っている")
            continue
        if have and verbose:
            print(f"[redo] 形状 {job['shape_id']} は {len(have)}/{len(expect)} 件"
                  "しか無いのでやり直す (重複は --report が最後の値を使う)")
        # cnn_sweep は --losses が複数形なので 1 回で済む。
        # taylor_sweep は --loss が単数なので損失ごとに回す。
        passes = [None] if g["kind"] == "cnn" else list(losses)
        for _loss in passes:
            n_new += _one_pass(g, job, out, extra, hard_factor, losses,
                               _loss, verbose)
        continue

    if verbose:
        print(f"\n{n_new} 件を {out} に追記した")
    return 0


def _one_pass(g: Dict, job: Dict, out: str, extra: List[str],
              hard_factor: float, losses: List[str], loss: Optional[str],
              verbose: bool) -> int:
    name = job["grid"]
    if True:
        tmp = (f"{out}.part{job['shape_id']}"
               + (f".{loss}" if loss else "") + ".json")
        cmd = _cmd(g, job, tmp, extra, loss)
        # 内側のタイムアウトは次数ごと。外側はその hard_factor 倍を上限にする
        n_loss = 1 if loss else max(1, len(losses))
        wall = int(g["timeout"] * hard_factor * n_loss * 3)
        if verbose:
            print(f"\n[形状 {job['shape_id']}]"
                  + (f" loss={loss}" if loss else "") + f" {job['shape']}")
            print("  " + " ".join(cmd))
        t0 = time.time()
        status, note = "ok", ""
        try:
            pr = subprocess.run(cmd, cwd=_ROOT, timeout=wall,
                                capture_output=True, text=True)
            if pr.returncode != 0:
                status, note = "error", (pr.stderr or "")[-400:]
        except subprocess.TimeoutExpired:
            status, note = "timeout", f"外側の上限 {wall}s を超えた"
        el = round(time.time() - t0, 1)
        n_new = 0
        recs: List[Dict] = []
        if os.path.exists(tmp):
            try:
                with open(tmp, encoding="utf-8") as fh:
                    recs = json.load(fh)
            except Exception as e:                    # noqa: BLE001
                note = note or f"出力を読めない: {e}"
            finally:
                os.remove(tmp)
        if not recs:
            # 1 件も出なかったことも記録する (黙って消さない)
            recs = [dict(status=status or "error", note=note)]
        with open(out, "a", encoding="utf-8") as fh:
            for r in recs:
                r = dict(r)
                r.update(grid=name, kind=g["kind"], shape_id=job["shape_id"],
                         **{f"shape_{k}": v for k, v in job["shape"].items()})
                r.setdefault("status", "ok")
                if loss:
                    r["loss"] = loss          # taylor_sweep は記録に入れない
                if status != "ok":
                    r["driver_status"] = status
                    r["driver_note"] = note
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                n_new += 1
        if verbose:
            n_ok = sum(1 for r in recs if r.get("status") == "ok")
            print(f"  -> {el}s  レコード {len(recs)} 件 (ok {n_ok})"
                  + (f"  [{status}] {note[:80]}" if status != "ok" else ""))
        return n_new


# ----------------------------------------------------------------------
# 集計
# ----------------------------------------------------------------------
def report(path: str, csv_out: str = "") -> int:
    rows: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("レコードが無い")
        return 1
    kind = rows[0].get("kind", "cnn")
    print(f"{len(rows)} レコード  格子 {sorted({r.get('grid') for r in rows})}")
    st: Dict[str, int] = {}
    for r in rows:
        st[r.get("status", "?")] = st.get(r.get("status", "?"), 0) + 1
    print("  状態: " + ", ".join(f"{k}={v}" for k, v in sorted(st.items())))

    if kind == "cnn":
        cols = ("shape_channels", "shape_kernel", "shape_hidden",
                "shape_n_data", "shape_size")
        hdr = ("ch", "k", "hid", "n", "size", "loss", "act", "ord",
               "d1", "d2", "d3")
        print("\n  " + "  ".join(f"{h:>8}" for h in hdr))
        seen = {}
        for r in rows:
            key = tuple(r.get(c) for c in cols) + (
                r.get("loss"), r.get("act"), _order_of(r))
            seen.setdefault(key, {})[r.get("depth")] = r
        for key in sorted(seen, key=lambda k: [str(x) for x in k]):
            cell = []
            for d in (1, 2, 3):
                r = seen[key].get(d)
                if r is None:
                    cell.append("-")
                elif r.get("status") == "ok":
                    cell.append(str(r.get("rlct", "?")))
                else:
                    cell.append(r.get("status", "?")[:7])
            print("  " + "  ".join(f"{str(v):>8}" for v in list(key) + cell))
    else:
        hdr = ("T", "d_model", "d_k", "d_ff", "seqs", "degen", "loss",
               "ord", "lambda", "m", "wall")
        print("\n  " + "  ".join(f"{h:>9}" for h in hdr))
        for r in sorted(rows, key=lambda r: (str(r.get("shape_degeneracy")),
                                             r.get("shape_d_k") or 0,
                                             _order_of(r) or 0)):
            vals = (r.get("shape_T"), r.get("shape_d_model"), r.get("shape_d_k"),
                    r.get("shape_d_ff"), r.get("shape_seqs"),
                    r.get("shape_degeneracy"), r.get("loss"), _order_of(r),
                    r.get("rlct") if r.get("status") == "ok" else r.get("status"),
                    _mult_of(r), r.get("wall"))
            print("  " + "  ".join(f"{str(v):>9}" for v in vals))

    if csv_out:
        import csv
        keys: List[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n-> {csv_out}")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="形状のパラメータ掃引 (cnn_sweep / taylor_sweep を外から回す)")
    ap.add_argument("--grid", default="", help=f"格子: {', '.join(GRIDS)}")
    ap.add_argument("--list", action="store_true", help="格子の一覧と見積り")
    ap.add_argument("--dry-run", action="store_true",
                    help="計画と見積りだけ出す (計算しない)")
    ap.add_argument("--out", default="", help="追記する JSONL")
    ap.add_argument("--report", default="", help="集計する JSONL")
    ap.add_argument("--csv", default="", help="--report の CSV 出力先")
    ap.add_argument("--hard-factor", type=float, default=3.0,
                    help="外側の上限 = 内側 timeout x これ x 損失数 x 3")
    ap.add_argument("rest", nargs="*",
                    help="-- の後ろは内側の掃引にそのまま渡す")
    a = ap.parse_args(list(argv) if argv is not None else None)

    if a.report:
        return report(a.report, a.csv)
    if a.list or not a.grid:
        print("使える格子:")
        for name in GRIDS:
            g, jobs = _plan(name)
            est = _estimate(g, jobs)
            print(f"  {name:<12} {g['kind']:<12} 形状 {len(jobs):>2} 通り  "
                  f"最悪 {est / 3600:.1f} 時間")
        if not a.grid:
            print("\n--grid <名前> で実行、--dry-run で計画だけ")
        return 0

    g, jobs = _plan(a.grid)
    est = _estimate(g, jobs)
    print(f"格子 {a.grid}: {g['kind']}、形状 {len(jobs)} 通り")
    print(f"  内側 timeout {g['timeout']}s  最悪の壁時計 {est / 3600:.1f} 時間")
    for j in jobs:
        print(f"  形状 {j['shape_id']}: {j['shape']}")
    if a.dry_run:
        print("\n(--dry-run なので実行しない)")
        return 0
    if not a.out:
        print("\n--out を指定してください (JSONL に追記、再開に使う)")
        return 1
    return run(a.grid, a.out, extra=list(a.rest),
               hard_factor=a.hard_factor)


if __name__ == "__main__":
    raise SystemExit(main())
