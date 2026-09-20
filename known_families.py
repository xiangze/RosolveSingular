r"""
bench/known_families.py — 真値が独立に分かり、かつ難しい族。

真値を保つ変換で例を増やしても、同じ証明経路に落ちるだけで検出力は上がらない。
必要なのは「真値が別途分かる」と「実装の難所を踏む」を両立する族である。

  A. 一般の位置にある中心的超平面配置
       f = prod_i l_i^{m_i}   (l_i は一般の位置の 1 次形式)
       実数体上の零点集合は原点以外に滑らかな点を持つので
         lambda = min( min_i 1/m_i ,  n / sum_i m_i )
       前者は超平面の滑らかな点からの寄与、後者は原点での錐の寄与。
       座標超平面の場合 (単項式) とは違い、ニュートン多面体では捉えられない
       (実数体上で退化する) ので、既存の高速経路が効かない。

  B. Vandermonde 行列型 (Aoyagi-Watanabe)
       I = < sum_h c_h a_h^k : k = 1..K >
       三層ニューラルネットや正規混合の fiber ideal。実数体上でニュートン
       多面体に関して退化することが知られており、トーリック経路が使えない。

       真値は Aoyagi (Entropy 21(6):561, 2019) の Theorem 6 (H<=3) と
       N=1 の厳密式から取る (vandermonde.py)。位数 theta も N=1 なら分かる。
       加えて aoyagi_lemmas.py が同論文 Section 3 の補題 (イデアル不変性・
       単調性・分離・最深点) を **真値を使わない検査** として実装している。

  C. 斉次形式
       f が d 次斉次で実零点集合が原点だけ (定値) なら lambda = n/d。
       実零点に滑らかな点があれば lambda = min(1, n/d) (既約・被約の場合)。

有効な真値がある族は oracle に値を入れ、無いものは None にして
「難所の探り」として使う。
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import sympy as sp

Case = Tuple[str, sp.Expr, Tuple[sp.Symbol, ...], str, Optional[sp.Rational]]


def _V(n, prefix="x"):
    return sp.symbols(f"{prefix}0:{n}", real=True)


# ----------------------------------------------------------------------
# A. 一般の位置の超平面配置
# ----------------------------------------------------------------------
def hyperplane_cases(seed: int = 11) -> List[Case]:
    rng = random.Random(seed)
    out: List[Case] = []
    for n in (2, 3):
        v = _V(n)
        for d in (2, 3, 4, 5):
            for mult in ([1] * d, [2] + [1] * (d - 1)):
                forms = []
                for _ in range(d):
                    while True:
                        c = [rng.randint(-3, 3) for _ in range(n)]
                        if any(c):
                            break
                    forms.append(sum(ci * vi for ci, vi in zip(c, v)))
                f = sp.expand(sp.prod([l ** m for l, m in zip(forms, mult)]))
                if f == 0 or f.is_number:
                    continue
                # 真値の式は「一般の位置」が前提。任意の n 個の係数ベクトルが
                # 一次独立であることを確かめ、満たさない配置は捨てる
                # (満たさないと真値が変わる。実際これを確かめずに使って
                #  偽の不一致を出した)。
                import itertools as _it
                M = [[sp.Poly(l, *v).coeff_monomial(vi) for vi in v]
                     for l in forms]
                generic = True
                for sub in _it.combinations(range(d), min(n, d)):
                    if sp.Matrix([M[t] for t in sub]).rank() < min(n, d):
                        generic = False
                        break
                if not generic:
                    continue
                lam = min(sp.Rational(1, max(mult)),
                          sp.Rational(n, sum(mult)))
                out.append((f"arrangement n={n} d={d} m={mult}", f, v,
                            "hyperplane", lam))
    return out


# ----------------------------------------------------------------------
# B. Vandermonde 行列型 (難所の探り、真値なし)
# ----------------------------------------------------------------------
def vandermonde_cases() -> List[Case]:
    """Aoyagi の公式を真値とする Vandermonde 行列型 (vandermonde.py)。"""
    from vandermonde import vandermonde_cases as vc
    return [(name, f, gens, fam, lam) for name, f, gens, fam, lam, _th in vc()]


# ----------------------------------------------------------------------
# C. 斉次形式
# ----------------------------------------------------------------------
def homogeneous_cases() -> List[Case]:
    out: List[Case] = []
    for n in (2, 3, 4):
        v = _V(n)
        for d in (2, 4, 6):
            f = sum(vi ** d for vi in v)            # 定値 (d が偶数)
            out.append((f"definite sum x^{d} n={n}", f, v, "homogeneous",
                        sp.Rational(n, d)))
    x, y, z = sp.symbols("x y z", real=True)
    # 不定値で既約・被約 -> min(1, n/d)
    out.append(("indefinite x^3-x*y^2 (n=2,d=3)", x**3 - x * y**2, (x, y),
                "homogeneous", min(sp.Integer(1), sp.Rational(2, 3))))
    out.append(("indefinite x^3+y^3+z^3 (n=3,d=3)", x**3 + y**3 + z**3,
                (x, y, z), "homogeneous", sp.Integer(1)))
    return out


def all_cases() -> List[Case]:
    return hyperplane_cases() + homogeneous_cases() + vandermonde_cases()


def _demo():
    import time
    from certify import rlct_certified

    print("=" * 78)
    print("真値が独立に分かる難しい族での検証")
    print("=" * 78)
    n_ok = n_bad = n_probe = 0
    for name, f, gens, fam, lam0 in all_cases():
        st = time.time()
        try:
            lam, status, how = rlct_certified(f, gens, timeout_ms=5000,
                                              max_depth=14)
        except Exception as e:                        # noqa: BLE001
            lam, status, how = None, "error", str(e)[:30]
        if lam0 is None:
            n_probe += 1
            mark = "(真値なし・探り)"
        elif lam == lam0:
            n_ok += 1
            mark = "OK"
        else:
            n_bad += 1
            mark = f"!!! 真値 {lam0}"
        print(f"  {name:<34} lambda={str(lam):<8} {status:<8} "
              f"{how:<18} {time.time()-st:5.1f}s {mark}")
    print(f"\n  真値と一致 {n_ok} / 不一致 {n_bad} / 探り {n_probe}")


if __name__ == "__main__":
    _demo()
