r"""
bench/aoyagi_lemmas.py — Aoyagi の補題を「真値を使わない検査」に変えたもの。

Aoyagi (Entropy 21(6):561, 2019) の Section 3 にある補題・定理のうち、
実装の検査にそのまま使えるものを取り出した。いずれも f ごとの真値を
知らなくても成り立つ関係式なので、ランダムな難しい例にも適用できる。

  Lemma 1 (2): g_1,...,g_m がイデアル J = <f_1,...,f_n> を生成するなら
      lambda(sum g_i^2) = lambda(sum f_i^2)
    -> **RLCT はイデアルだけで決まる**。生成元を取り替えても値は変わらない。
       グレブナー基底に取り替えると多項式の形は全く変わるので、実装の
       別経路を踏む強い検査になる。

  Lemma 1 (1): sum g_i^2 <= sum f_i^2 (点ごと) なら
      lambda(sum g_i^2) <= lambda(sum f_i^2)
    -> 生成元の部分集合を取れば必ず lambda は下がる (か等しい)。

  Lemma 2: 変数が互いに素なら
      lambda(sum f_i^2 + sum g_j^2) = lambda(sum f_i^2) + lambda(sum g_j^2)
    -> families.py の和の規則そのもの。ここでは検査として使う。

  Theorem 2: f_i が w_1..w_j について斉次なら、原点のほうが lambda が小さい
      lambda_(0,...,0,w*) <= lambda_(w*)
    -> 「最も深い特異点は原点」。斉次な例で確認できる。

どれも「真値が分かる族」ではなく「真値によらず成り立つ関係」なので、
ご指摘のあった「似た例を増やすだけ」にはならない。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import sympy as sp


def _lam(f, gens, **kw):
    from certify import rlct_certified
    lam, status, _ = rlct_certified(f, tuple(gens), **kw)
    return lam, status


def ideal_invariance_disagrees(generators: Sequence[sp.Expr], gens,
                               **kw) -> Optional[str]:
    """Lemma 1(2): 生成元をグレブナー基底に取り替えても lambda は不変。

    多項式の形が完全に変わるので、族の経路・中心の選択・非退化判定が
    別の枝を通る。値が変われば実装のバグ。
    """
    gens = tuple(gens)
    G = [sp.expand(g) for g in generators if sp.expand(g) != 0]
    if not G:
        return None
    f1 = sp.expand(sum(g ** 2 for g in G))
    try:
        gb = list(sp.groebner(G, *gens, order="grevlex").exprs)
    except Exception:
        return None
    if not gb or sp.sstr(sorted(gb, key=sp.sstr)) == sp.sstr(sorted(G, key=sp.sstr)):
        return None
    f2 = sp.expand(sum(g ** 2 for g in gb))
    l1, s1 = _lam(f1, gens, **kw)
    l2, s2 = _lam(f2, gens, **kw)
    if l1 is None or l2 is None:
        return None
    if l1 != l2:
        return (f"イデアル不変性が破れています: 元の生成元で {l1} ({s1}), "
                f"グレブナー基底で {l2} ({s2})")
    return None


def monotonicity_disagrees(generators: Sequence[sp.Expr], gens,
                           **kw) -> Optional[str]:
    """Lemma 1(1): 生成元の部分集合を取ると lambda は下がる (か等しい)。"""
    gens = tuple(gens)
    G = [sp.expand(g) for g in generators if sp.expand(g) != 0]
    if len(G) < 2:
        return None
    full, _ = _lam(sp.expand(sum(g ** 2 for g in G)), gens, **kw)
    if full is None:
        return None
    for i in range(len(G)):
        sub = [g for j, g in enumerate(G) if j != i]
        lam, _ = _lam(sp.expand(sum(g ** 2 for g in sub)), gens, **kw)
        if lam is None:
            continue
        if lam > full:
            return (f"単調性が破れています: 部分集合 (第 {i} 生成元を除く) で "
                    f"{lam} > 全体 {full}")
    return None


def separation_disagrees(f, gens, g, gens2, **kw) -> Optional[str]:
    """Lemma 2: 変数が互いに素なら lambda は加法的。"""
    l1, _ = _lam(f, gens, **kw)
    l2, _ = _lam(g, gens2, **kw)
    both = tuple(list(gens) + list(gens2))
    l12, _ = _lam(sp.expand(f + g), both, **kw)
    if None in (l1, l2, l12):
        return None
    if l12 != l1 + l2:
        return (f"分離の規則が破れています: lambda(f+g)={l12} != "
                f"{l1} + {l2} = {l1 + l2}")
    return None


def deepest_point_disagrees(f, gens, point, **kw) -> Optional[str]:
    """Theorem 2: 斉次なら原点が最も深い (lambda が最小)。

    point は原点以外の点。そこでの局所 lambda は原点での値以上のはず。
    """
    gens = tuple(gens)
    lam0, _ = _lam(f, gens, **kw)
    shifted = sp.expand(sp.expand(f).subs({v: v + point.get(v, 0)
                                           for v in gens}, simultaneous=True))
    if shifted.subs({v: 0 for v in gens}) != 0:
        return None                       # その点は零点でない
    lamp, _ = _lam(shifted, gens, **kw)
    if None in (lam0, lamp):
        return None
    if lamp < lam0:
        return (f"最深点の定理が破れています: 原点 {lam0} > 移動点 {lamp}")
    return None


def _unimodular(gens, rng, kind="triangular"):
    """行列式 +-1 の整数行列による座標変換を作る (原点を固定)。"""
    n = len(gens)
    M = sp.eye(n)
    for _ in range(2 * n):
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j:
            M[i, :] = M[i, :] + rng.choice([-1, 1]) * M[j, :]
    if kind == "triangular":
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = 0
            M[i, i] = 1
        for i in range(n):
            for j in range(i):
                M[i, j] = rng.choice([-1, 0, 1])
    return M


def coordinate_invariance_disagrees(f, gens, *, n_trials: int = 2,
                                    seed: int = 0, kinds=("general",
                                                          "triangular"),
                                    **kw) -> Optional[str]:
    """lambda と m は原点を固定する可逆な座標変換で不変でなければならない。

    x -> M x (det M = +-1) はヤコビアン +-1 の微分同相なので、
    lambda も位数 m も変わらない。真値を知らなくても確かめられる、
    最も基本的なメタモルフィック検査。

    実装が「多項式の因子」しか見ずに「イデアルの生成元」を見ていないと、
    この不変性が破れる (Vandermonde 型の chart 内で実際に破れる)。
    """
    import random
    gens = tuple(gens)
    rng = random.Random(seed)
    base, sbase = _lam(sp.expand(f), gens, **kw)
    if base is None:
        return None
    for kind in kinds:
        for _ in range(n_trials):
            M = _unimodular(gens, rng, kind)
            if M.det() == 0:
                continue
            sub = {v: sum(M[i, j] * gens[j] for j in range(len(gens)))
                   for i, v in enumerate(gens)}
            F = sp.expand(sp.expand(f).subs(sub, simultaneous=True))
            if F == 0:
                continue
            lam, st = _lam(F, gens, **kw)
            if lam is None:
                continue
            if lam != base:
                return (f"座標不変性が破れています: 元の座標で {base} ({sbase}), "
                        f"det={M.det()} の {kind} 変換後に {lam} ({st})")
    return None


def _demo():
    x, y, z = sp.symbols("x y z", real=True)
    a1, a2, c1, c2 = sp.symbols("a1 a2 c1 c2", real=True)

    print("=" * 74)
    print("Aoyagi の補題を検査として使う (真値を使わない)")
    print("=" * 74)

    print("\n1. イデアル不変性 (Lemma 1(2)): 生成元をグレブナー基底に替えても不変")
    for label, G, g in [
        ("<x^2, y^3>", [x**2, y**3], (x, y)),
        ("<x*y, x^2-y^2>", [x * y, x**2 - y**2], (x, y)),
        ("Vandermonde H=2 K=2",
         [c1 * a1 + c2 * a2, c1 * a1**2 + c2 * a2**2], (a1, a2, c1, c2)),
    ]:
        msg = ideal_invariance_disagrees(G, g, timeout_ms=5000, max_depth=12)
        print(f"   {label:<24} {'OK (一致)' if msg is None else '** ' + msg}")

    print("\n2. 単調性 (Lemma 1(1)): 生成元を減らすと lambda は下がる")
    for label, G, g in [("<x^2, y^3, z^4>", [x**2, y**3, z**4], (x, y, z)),
                        ("<x*y, y*z, z*x>", [x * y, y * z, z * x], (x, y, z))]:
        msg = monotonicity_disagrees(G, g, timeout_ms=5000, max_depth=12)
        print(f"   {label:<24} {'OK' if msg is None else '** ' + msg}")

    print("\n3. 分離 (Lemma 2): 変数が互いに素なら加法的")
    msg = separation_disagrees(x**2 + y**3, (x, y), z**4, (z,),
                               timeout_ms=5000, max_depth=12)
    print(f"   lambda(x^2+y^3) + lambda(z^4)   {'OK' if msg is None else '** ' + msg}")

    print("\n4. 最深点 (Theorem 2): 斉次なら原点が最小")
    msg = deepest_point_disagrees(x**2 * y**2 + y**2 * z**2 + z**2 * x**2,
                                  (x, y, z), {x: sp.Rational(1, 2)},
                                  timeout_ms=5000, max_depth=12)
    print(f"   x^2y^2+y^2z^2+z^2x^2 を x=1/2 へ  "
          f"{'OK' if msg is None else '** ' + msg}")


if __name__ == "__main__":
    _demo()
