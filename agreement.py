r"""
bench/agreement.py — **既知の閉形式との一致表**

厳密計算を名乗るには、独立に真値が分かっている族で一致を示す必要がある。
ここは「どの族でどれだけ合っているか」を 1 枚の表にするためのもの。

**status を必ず併記する。**値が合っていても証明書が 'proved' でなければ
「偶然合っている」可能性が残り、値が違っていても 'proved' でなければ
「誤りだと分かっている」ことになる。この区別が無い一致表は意味がない。

収録している族
--------------
A. **正則点: λ = d/2**
   原点で勾配が独立な d 本の生成元 (⟨x_1,…,x_d⟩ そのもの) なら
   K = Σ x_i² で λ = d/2, m = 1。最も基本的な検算。

B. **単項式イデアル: Howald / Lin Thm 1.3**
   I が単項式イデアルなら λ = 1/ℓ、ℓ は Newton 多面体の τ-距離
   (τ = (0,…,0) なら ℓ = min{t : t·(1,…,1) ∈ P(I)})。
   Lin の `rlct.lib` が「単項式イデアルでは厳密」と書いているのはこれ。
   ここでは τ-距離を**別経路の LP で独立に**計算して突き合わせる。

C. **一般の位置の中心的超平面配置** (`known_families.hyperplane_cases`)
   λ = min( min_i 1/m_i , n/Σm_i )。実数体上でニュートン多面体に関して
   退化するので高速経路が効かず、解消の本体を踏む。

D. **斉次形式** (`known_families.homogeneous_cases`)
   d 次斉次で実零点が原点だけなら λ = n/d。

E. **Vandermonde 行列型** (`known_families.vandermonde_cases`)
   Aoyagi の公式。**現状 H ≥ 3 で過大評価する**ので、ここでは
   「status が proved でないこと」が通れば合格とする扱いも併記する
   (`guards.vandermonde_never_proved_wrong_tests` と同じ基準)。

G. **深層線形網** (`dln.dln_cases`)
   Aoyagi (Neural Networks 172:106132, 2024) Theorem 1 の閉形式。
   **LLC の数値推定が校正に使っている唯一の族**なので、ここでの一致が
   一番効く。`dln.self_check()` が公式の実装自体を同論文 Theorem 2 と
   657 件で突き合わせている。

F. **行列分解 (深層線形網の最小形)**
   A (m×h) · B (h×n) の原点でのファイバー。単項式になる最小の場合
   (m=n=h=1 で K = a²b², λ = 1/2, m = 2) だけ真値を入れてある。
   一般の深層線形網は **G** に入れた (`bench/dln.py`)。

使い方
------
    cd bench && PYTHONPATH=..:. python agreement.py
    cd bench && PYTHONPATH=..:. python agreement.py --families A,B,C --timeout 60
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

__all__ = ["regular_cases", "monomial_cases", "factorization_cases",
           "run_agreement", "summarize"]


# ----------------------------------------------------------------------
# A. 正則点
# ----------------------------------------------------------------------
def regular_cases(dims: Sequence[int] = (1, 2, 3, 4, 5)) -> List[Tuple]:
    """⟨x_1,…,x_d⟩ は正則で λ = d/2, m = 1。"""
    out = []
    for d in dims:
        v = sp.symbols(f"x1:{d+1}", real=True)
        f = sp.expand(sum(x ** 2 for x in v))
        out.append((f"regular d={d}", f, tuple(v), "regular",
                    sp.Rational(d, 2), 1))
    return out


# ----------------------------------------------------------------------
# B. 単項式イデアル (Howald / Lin Thm 1.3)
# ----------------------------------------------------------------------
def _tau_distance(exps: Sequence[Sequence[int]], n: int) -> sp.Rational:
    r"""Newton 多面体の τ-距離 (τ = 0)。

    ℓ = min{ t >= 0 : t·(1,…,1) ∈ P(I) }、P(I) = conv(∪_i (a_i + R_{>=0}^n))。
    LP で解く。

    **Watanabe の λ は 1/(2ℓ)、1/ℓ ではない。** Lin の Theorem 1.2 は
    `(2λ, θ) = min_x RLCT_x(I;φ)` で、RLCT の定義が
    `ζ(z) = ∫ (Σ f_i²)^{−z/2}` と z/2 乗になっているため、
    Lin の RLCT は Watanabe の λ の **2 倍**である。

    手計算で確認した 3 例 (いずれも 1/(2ℓ) が正しい):
      I = ⟨x₁x₂⟩     ℓ=1    λ=1/2   (f = x₁²x₂², k=(2,2), h=0 → min 1/2)
      I = ⟨x₁²,x₂²⟩  ℓ=1    λ=1/2   (f = x₁⁴+x₂⁴, Newton 距離 2 → 1/2)
      I = ⟨x₁,…,x_d⟩ ℓ=1/d  λ=d/2   (正則点)
      I = ⟨x₁³,x₂²⟩  ℓ=6/5  λ=5/12  (f = x₁⁶+x₂⁴, Newton 距離 12/5)

    **解消器とは独立の経路**なので一致検査の意味がある。
    """
    from scipy.optimize import linprog

    # t·1 ∈ conv(a_i) + R_+^n  <=>  ∃ w >= 0, Σw=1, s >= 0:
    #     t·1 = Σ w_i a_i + s   を満たす最小の t
    m = len(exps)
    # 変数: [t, w_1..w_m, s_1..s_n]
    nv = 1 + m + n
    c = [0.0] * nv
    c[0] = 1.0
    A_eq = []
    b_eq = []
    for j in range(n):
        row = [0.0] * nv
        row[0] = 1.0                       # t
        for i in range(m):
            row[1 + i] = -float(exps[i][j])
        row[1 + m + j] = -1.0              # s_j
        A_eq.append(row)
        b_eq.append(0.0)
    row = [0.0] * nv                       # Σ w_i = 1
    for i in range(m):
        row[1 + i] = 1.0
    A_eq.append(row)
    b_eq.append(1.0)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * nv, method="highs")
    if not res.success:
        raise RuntimeError(f"τ-距離の LP が解けません: {res.message}")
    # LP は浮動小数なので有理数に直す。分母を小さく取らないと
    # 1000000000/1666666667 のような見かけだけの有理数になる
    return sp.Rational(float(res.x[0])).limit_denominator(10 ** 4)


def monomial_cases(seed: int = 7, count: int = 14) -> List[Tuple]:
    r"""単項式イデアル。λ = 1/ℓ (ℓ は τ-距離) を独立な LP で出す。

    f = Σ (x^{a_i})² として渡す。平方和なので符号の相殺がなく、
    イデアル ⟨x^{a_i}⟩ の RLCT と一致する。
    """
    import random

    rng = random.Random(seed)
    out = []
    specs = [(2, [[2, 0], [0, 2]]), (2, [[3, 0], [0, 2]]),
             (2, [[1, 1]]), (2, [[2, 1], [1, 3]]),
             (3, [[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
             (3, [[1, 1, 0], [0, 1, 1]]),
             (3, [[3, 1, 0], [0, 2, 2]]),
             (2, [[4, 0], [0, 3], [2, 1]])]
    while len(specs) < count:
        n = rng.choice((2, 3))
        k = rng.randint(1, 3)
        es = [[rng.randint(0, 3) for _ in range(n)] for _ in range(k)]
        if all(sum(e) == 0 for e in es):
            continue
        es = [e for e in es if sum(e) > 0]
        if es:
            specs.append((n, es))
    for n, es in specs[:count]:
        v = sp.symbols(f"x1:{n+1}", real=True)
        f = sp.expand(sum(sp.prod([vi ** e for vi, e in zip(v, a)]) ** 2
                          for a in es))
        if f == 0 or f.is_number:
            continue
        try:
            ell = _tau_distance(es, n)
        except Exception:                                 # noqa: BLE001
            continue
        if ell == 0:
            continue
        # Watanabe の λ = 1/(2ℓ) (Lin Thm 1.2 の 2λ を戻す)
        out.append((f"monomial n={n} {es}", f, tuple(v), "monomial",
                    sp.Rational(1, 2) / ell, None))
    return out


# ----------------------------------------------------------------------
# F. 行列分解 (深層線形網の最小形)
# ----------------------------------------------------------------------
def factorization_cases() -> List[Tuple]:
    r"""A·B の原点でのファイバー。**単項式になる最小の場合のみ**真値あり。

    m=n=h=1: K = (ab)² は単項式なので λ = 1/2, m = 2 (τ-距離 2)。
    それ以外 (h >= 2 や m,n >= 2) の閉形式は Aoyagi (2024) の深層線形網の
    公式が必要で、**完全な記述を取得できていないため真値を入れていない**
    (真値 None = 難所の探りとして走らせるだけ)。
    """
    out = []
    a, b = sp.symbols("a b", real=True)
    out.append(("factor 1x1x1", sp.expand((a * b) ** 2), (a, b),
                "factorization", sp.Rational(1, 2), 2))
    # 真値なし: 2x1x2 と 2x2x2 (難所の探り)
    A = sp.Matrix(2, 1, sp.symbols("a1 a2", real=True))
    B = sp.Matrix(1, 2, sp.symbols("b1 b2", real=True))
    P = A * B
    f = sp.expand(sum(e ** 2 for e in P))
    out.append(("factor 2x1x2 (真値なし)", f,
                tuple(sp.symbols("a1 a2 b1 b2", real=True)),
                "factorization", None, None))
    return out


# ----------------------------------------------------------------------
# 実行
# ----------------------------------------------------------------------
def _all_cases(families: Sequence[str]) -> List[Tuple]:
    cases: List[Tuple] = []
    if "A" in families:
        cases += regular_cases()
    if "B" in families:
        cases += monomial_cases()
    if "C" in families or "D" in families or "E" in families:
        import known_families as kf
        if "C" in families:
            cases += [(n, f, g, fam, lam, None)
                      for n, f, g, fam, lam in kf.hyperplane_cases()]
        if "D" in families:
            cases += [(n, f, g, fam, lam, None)
                      for n, f, g, fam, lam in kf.homogeneous_cases()]
        if "E" in families:
            cases += [(n, f, g, fam, lam, None)
                      for n, f, g, fam, lam in kf.vandermonde_cases()]
    if "F" in families:
        cases += factorization_cases()
    if "G" in families:
        from dln import dln_cases
        cases += dln_cases(max_vars=12)
    return cases


def run_agreement(families: Sequence[str] = ("A", "B", "C", "D", "E", "F"),
                  *, max_depth: int = 14, timeout: float = 120.0,
                  verbose: bool = True) -> List[Dict]:
    """各ケースで `rlct_certified` を回し、真値と突き合わせる。"""
    import multiprocessing as mp

    from certify import rlct_certified

    def _work(f, g, md, q):
        try:
            q.put(rlct_certified(f, g, max_depth=md))
        except Exception as e:                            # noqa: BLE001
            q.put(("error", f"{type(e).__name__}: {str(e)[:80]}", ""))

    mp.set_start_method("fork", force=True)
    out: List[Dict] = []
    for (name, f, g, fam, truth, th) in _all_cases(families):
        q = mp.Queue()
        p = mp.Process(target=_work, args=(f, g, max_depth, q))
        t = time.time()
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate(); p.join()
            # timeout は何も主張していないので honest=True。
            # ここを設定し忘れると `r.get("honest")` が None になり、
            # まとめの「不一致を proved」に数えられてしまう (実際に誤報した)
            rec = dict(name=name, family=fam, truth=str(truth),
                       status="timeout", lam=None, route="",
                       agree=False, honest=True,
                       wall=round(time.time() - t, 1))
        else:
            try:
                lam, status, route = q.get_nowait()
            except Exception:                             # noqa: BLE001
                lam, status, route = None, "crash", ""
            agree = (truth is not None and lam is not None
                     and sp.nsimplify(lam) == sp.nsimplify(truth))
            rec = dict(name=name, family=fam, truth=str(truth),
                       lam=str(lam), status=str(status), route=str(route),
                       agree=bool(agree),
                       # 値が違っても proved でなければ「誤りだと分かっている」
                       honest=bool(truth is None or agree
                                   or str(status) != "proved"),
                       wall=round(time.time() - t, 1))
        out.append(rec)
        if verbose:
            mark = ("一致" if rec.get("agree") else
                    ("真値なし" if truth is None else
                     ("不一致(proved でない)" if rec.get("honest")
                      else "**不一致を proved**")))
            print(f"  {name[:40]:<42} 真値 {str(truth):>8}  "
                  f"λ={str(rec.get('lam')):>8}  {rec['status']:<9} "
                  f"{mark}  {rec['wall']}s", flush=True)
    return out


def summarize(recs: List[Dict]) -> str:
    L: List[str] = []
    fams = sorted({r["family"] for r in recs})
    L.append("\n--- 族ごとの一致 ---")
    hdr = (f"  {'族':<16} {'件数':>4} {'一致':>5} {'proved':>7} "
           f"{'不一致を proved':>15}")
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))
    bad_total = 0
    for fam in fams:
        rs = [r for r in recs if r["family"] == fam]
        withtruth = [r for r in rs if r["truth"] != "None"]
        ag = [r for r in withtruth if r.get("agree")]
        pv = [r for r in rs if r["status"] == "proved"]
        bad = [r for r in withtruth if not r.get("honest")]
        bad_total += len(bad)
        L.append(f"  {fam:<16} {len(rs):>4} "
                 f"{len(ag)}/{len(withtruth):<3} {len(pv):>7} {len(bad):>15}")
    L.append(f"\n  **不一致を proved として返したケース: {bad_total} 件**"
             + ("  (0 件であることが健全性の条件)" if bad_total == 0
                else "  <- 要修正"))
    to = [r for r in recs if r["status"] == "timeout"]
    if to:
        L.append(f"  timeout: {len(to)} 件")
    dis = [r for r in recs if r["truth"] != "None" and not r.get("agree")
           and r["status"] != "timeout"]
    if dis:
        L.append("\n  値が真値と違うケース (status 併記):")
        for r in dis:
            L.append(f"    {r['name'][:46]:<48} λ={r['lam']:>8} "
                     f"真値 {r['truth']:>8}  status={r['status']}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="A,B,C,D,E,F,G")
    ap.add_argument("--max-depth", type=int, default=14)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import rlct_cache
    rlct_cache.disable()          # 一致表はキャッシュ無しで取る

    fams = [v.strip() for v in args.families.split(",") if v.strip()]
    print("===== 既知の閉形式との一致表 =====")
    print(f"  族 {fams}, 1 ケース {args.timeout}s まで, キャッシュ無効\n")
    recs = run_agreement(fams, max_depth=args.max_depth, timeout=args.timeout)
    print("\n===== まとめ =====")
    print(summarize(recs))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(recs, fh, ensure_ascii=False, indent=1, default=str)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
