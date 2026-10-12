r"""
bench_strata.py — 層の公式を Aoyagi の閉形式と突き合わせる

  python bench_strata.py            # L = 2,3,4 の掃引
  python bench_strata.py --quick

検査する 3 つのこと
  (1) 健全性: λ_strata ≥ λ_Aoyagi (L5 から必ず成り立つはず。破れたら実装のバグ)
  (2) 等号:   λ_strata = λ_Aoyagi  (予想)
  (3) 位数:   m_nested = θ_Aoyagi  (予想。鎖だけの規則 m_chain と比較)
  (4) 正則部分 (r > 0): Jacobian の厳密な階数 ρ が 2·(−r²/2 + r(H1+H_{L+1})/2)
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from fractions import Fraction

import sympy as sp

from dln import lambda_dln
from strata_closer import (dln_strata, strata_bound, jacobian_rank_at,
                           nested_multiplicity)
from strata_closer import top_multiplicity, flat_multiplicity


def sweep(maxdim: int, depths, max_vars: int, rmax: int):
    for L in depths:
        for H in itertools.product(range(1, maxdim + 1), repeat=L + 1):
            nv = sum(H[s] * H[s + 1] for s in range(L))
            if nv > max_vars:
                continue
            for r in range(0, min(rmax, min(H)) + 1):
                yield tuple(H), r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        cases = list(sweep(3, (2, 3), 12, 1))
    else:
        cases = list(sweep(4, (2,), 32, 2)) + list(sweep(3, (3,), 27, 1)) \
            + list(sweep(2, (4, 5), 20, 1))
    print(f"{len(cases)} ケース")
    stats = dict(n=0, sound=0, lam=0, m_nested=0, m_chain=0, m_pair=0, rho=0, rho_n=0)
    bad = []
    cache = {}
    t0 = time.time()
    for H, r in cases:
        lamA, thA = lambda_dln(H, r)
        lamA = Fraction(int(sp.numer(lamA)), int(sp.denom(lamA)))
        Hr = tuple(h - r for h in H)
        if r:
            rho = jacobian_rank_at(H, r)
            base = Fraction(-r * r, 2) + Fraction(r * (H[0] + H[-1]), 2)
            stats["rho_n"] += 1
            stats["rho"] += Fraction(rho, 2) == base
            if Fraction(rho, 2) != base:
                bad.append(f"ρ H={H} r={r}: ρ/2={Fraction(rho,2)} 対 {base}")
        else:
            rho = 0
        if min(Hr) == 0:
            lamS, mN, mC, mP = Fraction(rho, 2), 1, 1, 1
        else:
            if Hr not in cache:
                st = dln_strata(Hr)
                lam0, mN0, tied = strata_bound(st)
                mP0 = flat_multiplicity(st, lam0)
                mC0 = nested_multiplicity([rp for _, rp in tied],
                                          chains_only=True)
                cache[Hr] = (lam0, mN0, mC0, mP0)
            lam0, mN, mC, mP = cache[Hr]
            lamS = Fraction(rho, 2) + lam0
        stats["n"] += 1
        stats["sound"] += lamS >= lamA
        stats["lam"] += lamS == lamA
        stats["m_nested"] += (lamS == lamA and mN == thA)
        stats["m_pair"] += (lamS == lamA and mP == thA)
        stats["m_chain"] += (lamS == lamA and mC == thA)
        if lamS != lamA or mN != thA or mP != thA:
            bad.append(f"H={H} r={r}: 層 (λ={lamS}, m_top={mN}, 鎖 {mC}, flat {mP}) "
                       f"対 Aoyagi (λ={lamA}, θ={thA})")
    n = stats["n"]
    print(f"所要 {time.time()-t0:.1f}s")
    print(f"  健全 λ_strata ≥ λ_Aoyagi : {stats['sound']}/{n}")
    print(f"  λ 一致                   : {stats['lam']}/{n}")
    print(f"  m 一致 (top 成分の数)    : {stats['m_nested']}/{n}")
    print(f"  m 一致 (flat 束の最長鎖) : {stats['m_pair']}/{n}")
    print(f"  m 一致 (鎖だけ)          : {stats['m_chain']}/{n}")
    if stats["rho_n"]:
        print(f"  r>0 正則部分 ρ/2 = 基底項: {stats['rho']}/{stats['rho_n']}")
    for b in bad[:30]:
        print("   ", b)
    if len(bad) > 30:
        print(f"    … ほか {len(bad)-30} 件")
    ok_v = vandermonde_part()
    return 0 if (stats["sound"] == n and ok_v) else 1


def vandermonde_part() -> bool:
    """Vandermonde 族 (M=N=1): 滑らかな対称性の層だけでは上界にとどまることを確かめる。

    健全性 (上界 ≥ 真値) が破れたら False。
    """
    from vandermonde import vandermonde_ideal, lambda_N1
    from strata_closer import linear_strata_bound
    print("Vandermonde (M=N=1): 対称性の線形層による上界")
    ok = True
    for H, Q in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2)]:
        gens, X = vandermonde_ideal(1, 1, H, Q)
        a = [x for x in X if x.name.startswith("a")]
        bb = [x for x in X if x.name.startswith("b")]
        forms = (a + bb
                 + [bb[i] - bb[j] for i, j in itertools.combinations(range(H), 2)]
                 + [a[i] + a[j] for i, j in itertools.combinations(range(H), 2)])
        ub, T, c, o = linear_strata_bound(gens, X, forms)
        lam, _ = lambda_N1(1, H, Q)
        lam = Fraction(int(sp.numer(lam)), int(sp.denom(lam)))
        ok &= ub >= lam
        print(f"  H={H} Q={Q}: 上界 {ub} (codim {c}, ord {o})  真値 {lam}  "
              f"{'一致' if ub == lam else '上界のみ'}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
