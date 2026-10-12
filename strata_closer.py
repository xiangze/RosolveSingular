r"""
strata_closer.py — 対称性の層 (軌道型) から λ を読む

主張 (L5, 無条件に健全)
------------------------
θ* を通る任意の滑らかな実部分多様体 S について

    λ(K, θ*) ≤ codim S / ord_S K

(S を中心にブローアップすると例外因子が h = codim S − 1, k = ord_S K を
持ち、被積分関数が正なので極は打ち消されない)。

予想 (層の公式, 等号は条件つき)
--------------------------------
対称群の軌道型の層 {S} すべてについて最小を取ると等号になり、

    λ = min_S codim S / ord_S K

検証結果 (bench_strata.py, 2026-10-12)
--------------------------------------
深層線形網 509 件 (L = 2..5, r ≤ 2):
  * λ は 509/509 で Aoyagi の閉形式と一致。最小比は常に W_0 の top 成分
    (ord_I = 1, 比 = codim/2) で取られる (Pepin Lehalleur–Rimányi と整合)。
  * top 成分の数は、Aoyagi の Ŝ のデータで C(ℓ, a) に**厳密に**等しい (204/204)。
  * 位数: 「m = top 成分の数」は 507/509。外れる 2 件 ((2,2,2,2,2),
    (2,2,2,2,2,2)) は 2 ≤ a ≤ ℓ−2 の場合で、Aoyagi は θ = a(ℓ−a)+1、
    成分数は C(ℓ,a)。両者は a ∈ {0,1,ℓ−1,ℓ} でだけ一致する。
    → 位数は「成分の数」ではなく、成分のなす何らかの順序の最長鎖
      (Gr(a,ℓ) の Bruhat 順序の階数 + 1 = a(ℓ−a)+1 と同じ値) らしい。未解決。
  * r > 0 の正則部分: θ* での Jacobian の厳密な階数 ρ について
    ρ/2 = −r²/2 + r(H1+H_{L+1})/2 が 268/268 で成立 ((L4) の検証)。
Vandermonde (M=N=1): 滑らかな対称性の層は H ≥ 2 で上界にしかならない
  (H=2: 1 対 3/4、H=3: 3/2 対 1)。真値には重み付きの中心が要る。

このモジュールはこれを **深層線形網 (r = 0)** で実装し、Aoyagi の
閉形式 (dln.lambda_dln, Theorem 2 と 657 件で照合済み) と突き合わせる。

深層線形網の層
--------------
重み (A_1, …, A_L) は等向 A_n 型箙 (n = L+1) の表現。GL(d_1)×…×GL(d_n)
(入出力の端点も含む) の作用で積 A_L⋯A_1 = 0 は不変であり、軌道は
**有限個** (Gabriel)。軌道は区間 [a,b] ⊂ {1..n} の重複度 n_[a,b] で分類され、
閉包順序は部分積の階数 r_ab の成分ごとの大小で与えられる
(Abeasis–Del Fra–Kraft)。これが W_0 の対称性の層そのもの。

各軌道 O について
  * codim O : 代表点での Lie 環作用の接写像の階数から**厳密に**計算する
              (区間の Hom 公式は使わない — 独立な検算になる)
  * ord_O K : 代表点 p と一般の方向 v で ord_t I(p + t v) を計算し 2 倍する
              (O は等質なので代表点で足りる。v が一般なら法方向の最低次が
              消えずに残る)

r > 0 は (L4) で r = 0 に帰着する: θ* での Jacobian の階数 ρ について
λ(H, r) = ρ/2 + λ(H − r, 0)。ρ は厳密に計算して閉形式の基底項と照合する。
"""
from __future__ import annotations

import itertools
import random
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

import sympy as sp

__all__ = ["dln_strata", "strata_bound", "jacobian_rank_at", "lambda_dln_strata"]

Interval = Tuple[int, int]          # 0-based の節点 a..b (両端含む)


# ----------------------------------------------------------------------
# 軌道の列挙 (区間の重複度)
# ----------------------------------------------------------------------
def _intervals(n: int) -> List[Interval]:
    return [(a, b) for a in range(n) for b in range(a, n)]


def _orbits(d: Sequence[int]) -> List[Dict[Interval, int]]:
    """次元ベクトル d の A_n 型箙の全軌道 (区間の重複度) を列挙する。"""
    n = len(d)
    ivs = _intervals(n)
    out = []

    def rec(i, rem, cur):
        if i == len(ivs):
            if all(x == 0 for x in rem):
                out.append(dict(cur))
            return
        a, b = ivs[i]
        cap = min(rem[a:b + 1])
        for k in range(cap + 1):
            if k:
                cur[(a, b)] = k
            elif (a, b) in cur:
                del cur[(a, b)]
            rem2 = rem[:]
            for j in range(a, b + 1):
                rem2[j] -= k
            rec(i + 1, rem2, cur)
        cur.pop((a, b), None)

    rec(0, list(d), {})
    return out


def _representative(d, orb) -> List[sp.Matrix]:
    """軌道の代表点 φ_k : V_k → V_{k+1} (k = 0..n−2)。行列は d_{k+1} × d_k。"""
    n = len(d)
    basis = [[] for _ in range(n)]          # basis[k] = [(区間, コピー)]
    for (a, b), mult in sorted(orb.items()):
        for c in range(mult):
            for k in range(a, b + 1):
                basis[k].append(((a, b), c))
    mats = []
    for k in range(n - 1):
        M = sp.zeros(d[k + 1], d[k])
        idx = {key: i for i, key in enumerate(basis[k + 1])}
        for j, key in enumerate(basis[k]):
            (a, b), c = key
            if k + 1 <= b:
                M[idx[key], j] = 1
        mats.append(M)
    return mats


def _rank_profile(orb, n) -> Dict[Interval, int]:
    """r_ab = φ_{b−1}⋯φ_a の階数 = [a,b] を含む区間の本数。"""
    return {(a, b): sum(m for (c, e), m in orb.items() if c <= a and b <= e)
            for a in range(n) for b in range(a, n)}


# ----------------------------------------------------------------------
# codim と ord
# ----------------------------------------------------------------------
def _orbit_codim(d, mats) -> int:
    """codim O = dim Rep − rank(gl(d) → T_p Rep)。厳密な有理数の階数。"""
    n = len(d)
    rep_dim = sum(d[k] * d[k + 1] for k in range(n - 1))
    cols = []
    for node in range(n):
        for i in range(d[node]):
            for j in range(d[node]):
                E = sp.zeros(d[node], d[node])
                E[i, j] = 1
                vec = []
                for k in range(n - 1):
                    # X_{k+1} φ_k − φ_k X_k
                    T = sp.zeros(d[k + 1], d[k])
                    if node == k + 1:
                        T += E * mats[k]
                    if node == k:
                        T -= mats[k] * E
                    vec.extend(list(T))
                cols.append(vec)
    if not cols:
        return rep_dim
    rank = sp.Matrix(cols).rank()
    return rep_dim - rank


def _product(mats):
    P = mats[0]
    for M in mats[1:]:
        P = M * P
    return P


def _ideal_order(d, mats, rng, star=None, tries: int = 4) -> int:
    """ord_O I: 生成元 = (φ_{n−2}⋯φ_0 − P*) の成分。一般方向 v で t の位数。

    行列多項式 Σ_k C_k t^k を整数係数のまま掛ける (厳密、sympy を通さない)。
    v を数回取り直して最小を取る (一般性が外れると位数が上がる側にしか
    ずれないので、最小は真値に収束する)。
    """
    import numpy as np
    best = None
    for _ in range(tries):
        polys = []
        for M in mats:
            C0 = np.array(M.tolist(), dtype=object).reshape(M.rows, M.cols)
            C1 = np.array([[rng.choice([-1, 1]) * rng.randint(1, 97)
                            for _ in range(M.cols)] for _ in range(M.rows)],
                          dtype=object).reshape(M.rows, M.cols)
            polys.append([C0, C1])
        P = polys[0]
        for Q in polys[1:]:
            R = [np.zeros((Q[0].shape[0], P[0].shape[1]), dtype=object)
                 for _ in range(len(P) + len(Q) - 1)]
            for i, A in enumerate(Q):
                for j, B in enumerate(P):
                    R[i + j] = R[i + j] + A.dot(B)
            P = R
        if star is not None:
            P[0] = P[0] - np.array(star.tolist(), dtype=object)
        o = next((k for k, C in enumerate(P) if any(x != 0 for x in C.flat)),
                 None)
        if o is None:
            o = 10 ** 9          # 生成元が恒等的に 0 (起こらないはず)
        best = o if best is None else min(best, o)
    return best


# ----------------------------------------------------------------------
# 層の公式
# ----------------------------------------------------------------------
def dln_strata(H: Sequence[int], *, seed: int = 0):
    """H = (H^(1) 出力, …, H^(L+1) 入力), r = 0 の W_0 の全層。

    返り値: [(区間の重複度, codim, ord_I, 階数プロファイル)]
    ord_K = 2 ord_I。W_0 に含まれる (ord_I ≥ 1) ものだけ。
    """
    d = list(reversed([int(h) for h in H]))
    n = len(d)
    rng = random.Random(seed)
    out = []
    for orb in _orbits(d):
        if orb.get((0, n - 1), 0):          # 全区間があると積 ≠ 0
            continue
        mats = _representative(d, orb)
        cod = _orbit_codim(d, mats)
        if cod == 0:
            continue                         # W_0 が開 (起こらない: 積=0 は真部分集合)
        oI = _ideal_order(d, mats, rng)
        out.append((orb, cod, oI, _rank_profile(orb, n)))
    return out


def _leq(rp1, rp2) -> bool:
    return all(rp1[k] <= rp2[k] for k in rp1)


def strata_bound(strata) -> Tuple[Fraction, int, List]:
    """λ_strata = min codim/(2 ord_I)、m = top 成分の数 (下の注意を参照)。"""
    ratios = [(Fraction(c, 2 * o), orb, rp) for orb, c, o, rp in strata]
    lam = min(r for r, _, _ in ratios)
    tied = [(orb, rp) for r, orb, rp in ratios if r == lam]
    return lam, top_multiplicity(strata, lam), tied


def top_multiplicity(strata, lam) -> int:
    r"""最小比を取る層のうち、余次元が最小のもの (= W_0 の top 次元成分) の数。

    他セッションの縮約ランク回帰の観察「m = 点を通る top 次元成分の数」を
    一般の深さに延長した規則。**深層線形網で 2 ≤ a ≤ ℓ−2 のとき破れる**
    (成分数 C(ℓ,a) に対し Aoyagi の θ = a(ℓ−a)+1)。L = 2 では常に ℓ ≤ 2 なので
    破れが現れない — 縮約ランク回帰で 14/14 成り立った理由。
    """
    tied = [(c, rp) for orb, c, o, rp in strata if Fraction(c, 2 * o) == lam]
    cmin = min(c for c, _ in tied)
    return sum(1 for c, _ in tied if c == cmin)


def flat_multiplicity(strata, lam) -> int:
    r"""W_0 の top 成分の交わりがなす束 (flat の束) の最長鎖の長さ。

    超平面配置で「交わりの束の階数」に当たる量。各成分の閉包を
    「その成分以下の軌道の集合」(閉包順序 = 階数の大小) として持ち、
    部分集合の交わりを全部作って、包含で真に減る最長の鎖を数える。
    ⟨ab⟩: 直線 ⊃ 原点 で 2。⟨abc⟩: 平面 ⊃ 直線 ⊃ 原点 で 3。
    """
    tied = [(c, rp) for orb, c, o, rp in strata if Fraction(c, 2 * o) == lam]
    cmin = min(c for c, _ in tied)
    tops = [rp for c, rp in tied if c == cmin]
    allrp = [rp for _, _, _, rp in strata]
    down = [frozenset(i for i, q in enumerate(allrp) if _leq(q, t))
            for t in tops]
    flats = set()
    for k in range(1, len(down) + 1):
        for T in itertools.combinations(range(len(down)), k):
            flats.add(frozenset.intersection(*(down[i] for i in T)))
    flats = sorted(flats, key=len, reverse=True)
    best = {}
    for F in flats:                       # 大きい順: 上から鎖を伸ばす
        best[F] = 1 + max((best[G] for G in best if F < G), default=0)
    return max(best.values())


def nested_multiplicity(tied_profiles, *, chains_only: bool = False) -> int:
    r"""最小比を取る層の「入れ子集合」の最大サイズ。

    De Concini–Procesi の wonderful model と同じ考え方: 最小比の層を
    閉包の小さい順にブローアップすると、例外因子が 1 点で交わるのは
      * 包含関係にある層どうし (鎖), または
      * 包含関係にないが、両者の閉包の交わりに最小比の層が無い
        (= 交わりを先にブローアップしないので、横断的に残る) 層どうし
    である。このような層の集合の最大サイズを m とする。

    chains_only=True で鎖だけ数える (最初の素朴な規則。比較用)。
    """
    P = [tuple(sorted(rp.items())) for rp in tied_profiles]
    P = list(dict.fromkeys(P))
    D = [dict(p) for p in P]
    n = len(D)

    def comparable(i, j):
        return _leq(D[i], D[j]) or _leq(D[j], D[i])

    def compatible(i, j):
        if comparable(i, j):
            return True
        if chains_only:
            return False
        # 両者より下にある最小比の層があれば、それが先に分離する
        return not any(_leq(D[k], D[i]) and _leq(D[k], D[j])
                       for k in range(n) if k not in (i, j))

    best = 1
    for size in range(2, n + 1):
        found = any(all(compatible(i, j) for i, j in itertools.combinations(T, 2))
                    for T in itertools.combinations(range(n), size))
        if not found:
            break
        best = size
    return best


# ----------------------------------------------------------------------
# r > 0: (L4) で正則部分を剥がす
# ----------------------------------------------------------------------
def jacobian_rank_at(H: Sequence[int], r: int) -> int:
    """θ* (各 A^(s) の左上 r×r が単位) での積写像の Jacobian の厳密な階数。"""
    L = len(H) - 1
    stars = []
    for s in range(L):
        M = sp.zeros(H[s], H[s + 1])
        for i in range(min(r, H[s], H[s + 1])):
            M[i, i] = 1
        stars.append(M)
    cols = []
    for s in range(L):
        for i in range(H[s]):
            for j in range(H[s + 1]):
                E = sp.zeros(H[s], H[s + 1])
                E[i, j] = 1
                mats = stars[:s] + [E] + stars[s + 1:]
                P = mats[0]
                for Mx in mats[1:]:
                    P = P * Mx
                cols.append(list(P))
    return sp.Matrix(cols).rank()


def lambda_dln_strata(H: Sequence[int], r: int = 0, *, seed: int = 0):
    """層の公式による (λ, m, 詳細)。r > 0 は (L4) で r = 0 に帰着。"""
    H = [int(h) for h in H]
    rho = jacobian_rank_at(H, r) if r else 0
    Hr = [h - r for h in H]
    if min(Hr) == 0:
        return Fraction(rho, 2), 1, dict(rho=rho, strata=[], tied=[])
    st = dln_strata(Hr, seed=seed)
    lam0, m, tied = strata_bound(st)
    return Fraction(rho, 2) + lam0, m, dict(rho=rho, strata=st, tied=tied)


# ----------------------------------------------------------------------
# 一般のイデアル: 対称性から作った線形部分空間の層での上界 (L5)
# ----------------------------------------------------------------------
def linear_strata_bound(gens, variables, forms, *, seed: int = 0,
                        tries: int = 3):
    r"""線形形式の族 forms の部分集合の共通零点 S (原点を通る線形部分空間)
    をすべて作り、min codim S / (2 ord_S I) を返す。(L5) により **常に上界**。

    forms には対称性の固定点集合を張る超平面を入れる
    (死んだユニット a_h = 0, b_h = 0, 衝突 b_i = b_j, 打ち消し a_i + a_j = 0 など)。
    返り値: (上界, それを与えた形式の部分集合, codim, ord_I)
    """
    rng = random.Random(seed)
    X = list(variables)
    t = sp.Symbol("t")
    lin = [sp.Matrix([[sp.diff(f, x) for x in X]]) for f in forms]
    seen = {}
    for k in range(1, len(forms) + 1):
        for T in itertools.combinations(range(len(forms)), k):
            A = sp.Matrix.vstack(*[lin[i] for i in T])
            key = tuple(sorted(tuple(r) for r in A.rref()[0].tolist()
                               if any(r)))
            if key in seen:
                continue
            cod = A.rank()
            basis = A.nullspace()           # S の基底
            best = None
            for _ in range(tries):
                p = sp.zeros(len(X), 1)
                for v in basis:
                    p += rng.randint(-29, 29) * v
                d = sp.Matrix([rng.choice([-1, 1]) * rng.randint(1, 29)
                               for _ in X])
                sub = {x: p[i] + t * d[i] for i, x in enumerate(X)}
                o = None
                for g in gens:
                    e = sp.expand(g.subs(sub, simultaneous=True))
                    if e == 0:
                        continue
                    kk = min(m[0] for m in sp.Poly(e, t).monoms())
                    o = kk if o is None else min(o, kk)
                if o is not None:
                    best = o if best is None else min(best, o)
            seen[key] = (cod, best, T)
    cand = [(Fraction(c, 2 * o), T, c, o) for c, o, T in seen.values()
            if o and o >= 1]
    return min(cand, key=lambda x: (x[0], x[2]))
