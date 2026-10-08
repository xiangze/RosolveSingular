"""
lc_filter.py — 「λ = codim/2」フィルタ (層別 → lci・lc 判定 → λ = c/2) を、
このツール群の **健全な上界・下界** として組み込む。

f = Σ g_i² (生成元 g_i, 原点で 0) の原点での実 RLCT λ について、ブローアップ
なしで次を出す。λ はイデアル <g_i> だけで決まる (Aoyagi Lemma 1(2))。

上界 (健全・厳密計算のみ)
  U1  有理点 q ∈ X = V(g), q ≠ 0 で rank J(q) = r (Q 上)。r 本の生成元 S を
      J(q) の独立な行に取り、残りの各 g_j について局所所属 g_j ∈ (S)_q を
      「商イデアル (S) : g_j が q で消えない元を含む」で厳密に判定する。
      成り立てば K ≍ Σ_S g² が q の近傍で成り立つので λ_q = r/2。
      イデアルが斉次なら λ_{sq} = λ_q (s ≠ 0) なので s → 0 として
      λ_0 = (原点の近傍での min) ≤ r/2。
      r の最小値は c = codim X (top 成分上の点) で、上界 c/2 が得られる。
  q の探し方: (a) 呼び出し側が渡す witness (族の構造から作れる。rrr_problem 参照)
              (b) 多重線形性を使った探索: どの単項式にも 2 個以上現れない変数の組
                  U を取り、残りを乱数 (一部は 0) に固定すると g は U について
                  一次になるので、Q 上の連立一次方程式として解く。

下界 (健全)
  L1  rank J(0) = r0 ⇒ λ ≥ r0/2   (部分生成元の単調性 Aoyagi Lemma 1(1))
  L2  λ ≥ lct_C(I)/2,  lct_C(I) ≥ k·lct_C(h_1⋯h_k)  (h_i ∈ I なら何でもよい)
      Singular の bfct (b 関数, bfun.lib) で lct = -(最大根)。b 関数は大域的
      なので lct_大域 ≤ lct_0、h_i が一般でなければ値は小さくなるだけ。
      どちらに転んでも下界として健全。h_i が一般で k > lct(I) なら等号。
      **変数 3 個程度まで** (4 変数で bfct が止まる)。
  L3  λ ≥ lct_C(f)   (f = Σg² 自体の b 関数。≤ 1 なので弱い)

数値 (証拠。区間には入れない)
  c (斉次なら sympy.groebner で厳密)。MCMC は rlct_filter.py (別リポジトリ) 側。

上下が一致すれば λ が確定する (経路名 'lc_filter')。L2 は Singular の出力を
信頼する点で、Varchenko の定理を前提にする 'newton' 経路と同じ種類の前提を置く。
"""
from __future__ import annotations

import itertools
import random
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import sympy as sp

from invariants import _singular_poly, singular_available


# ----------------------------------------------------------------------
# 小道具
# ----------------------------------------------------------------------
def _is_homogeneous_ideal(gens, X) -> bool:
    return all(sp.Poly(g, *X).is_homogeneous for g in gens)


def _independent_rows(J: sp.Matrix) -> List[int]:
    rows: List[int] = []
    for i in range(J.rows):
        if sp.Matrix([J.row(j) for j in rows + [i]]).rank() == len(rows) + 1:
            rows.append(i)
    return rows


def _local_member(g, S, X, q: Dict) -> bool:
    """g ∈ (S) を q での局所環で厳密に判定する (十分条件ではなく同値)。

    (S) : g = ((S) ∩ (g)) / g を lex のグレブナー基底で作り、q で消えない元が
    あれば所属。(S) ∩ (g) は T·S + (1-T)·g から T を消去して得る。
    """
    T = sp.Dummy("T")
    G = sp.groebner([T * s for s in S] + [(1 - T) * g], T, *X, order="lex")
    for p in G.exprs:
        if p.has(T):
            continue
        c = sp.cancel(p / g)
        if sp.expand(c.subs(q)) != 0:
            return True
    return False


# ----------------------------------------------------------------------
# U1: 有理点での上界
# ----------------------------------------------------------------------
@dataclass
class PointBound:
    q: Dict
    r: int
    ok: bool                 # 局所所属まで確認できた
    why: str = ""


def upper_bound_at_point(gens, X, q: Dict) -> PointBound:
    gens = [sp.expand(g) for g in gens]
    if any(sp.expand(g.subs(q)) != 0 for g in gens):
        return PointBound(q, -1, False, "q は X 上にない")
    if all(sp.nsimplify(v) == 0 for v in q.values()):
        return PointBound(q, -1, False, "q = 0")
    J = sp.Matrix(gens).jacobian(X).subs(q)
    S_idx = _independent_rows(J)
    r = len(S_idx)
    S = [gens[i] for i in S_idx]
    for j, g in enumerate(gens):
        if j in S_idx:
            continue
        if not _local_member(g, S, X, q):
            return PointBound(q, r, False, f"g_{j} ∉ (S)_q")
    return PointBound(q, r, True)


def _multilinear_blocks(gens, X, rng, n_tries=8):
    """どの単項式にも 2 個以上 (重複を含む) 現れない変数の組 U を乱択で作る。"""
    conflict = {v: set() for v in X}
    for g in gens:
        for mon in sp.Poly(g, *X).monoms():
            vs = [X[i] for i, e in enumerate(mon) if e > 0]
            for i, e in enumerate(mon):
                if e >= 2:
                    conflict[X[i]].add(X[i])
            for a, b in itertools.combinations(vs, 2):
                conflict[a].add(b); conflict[b].add(a)
    out = []
    for _ in range(n_tries):
        order = list(X); rng.shuffle(order)
        U = []
        for v in order:
            if v in conflict[v]:
                continue
            if all(u not in conflict[v] for u in U):
                U.append(v)
        out.append(U)
    return out


def search_rational_points(gens, X, *, tries: int = 30, seed: int = 0,
                           zero_prob=(0.0, 0.3, 0.6)) -> List[Dict]:
    """X 上の 0 でない有理点を探す (多重線形ブロックを固定して一次方程式を解く)。"""
    rng = random.Random(seed)
    pts: List[Dict] = []
    blocks = _multilinear_blocks(gens, X, rng)
    for t in range(tries):
        U = blocks[t % len(blocks)]
        if not U:
            continue
        p0 = zero_prob[t % len(zero_prob)]
        fix = {v: (0 if rng.random() < p0 else sp.Integer(rng.randint(-3, 3)))
               for v in X if v not in U}
        eqs = [sp.expand(g.subs(fix)) for g in gens]
        A, b = sp.linear_eq_to_matrix(eqs, U)
        try:
            sol, params = A.gauss_jordan_solve(b)
        except ValueError:
            continue                                   # 解なし
        subs_par = {p: sp.Integer(rng.randint(-3, 3)) for p in params}
        val = sol.subs(subs_par)
        q = dict(fix); q.update({u: sp.nsimplify(x) for u, x in zip(U, val)})
        if all(v == 0 for v in q.values()):
            continue
        pts.append(q)
    return pts


# ----------------------------------------------------------------------
# L2/L3: Singular の b 関数による lct
# ----------------------------------------------------------------------
def _singular_lct(poly, X, timeout: int = 60) -> Optional[sp.Rational]:
    """-(b_f の最大根) = lct_C(f) (大域)。失敗したら None。"""
    if not singular_available():
        return None
    names = ",".join(str(v) for v in X)
    script = (
        'LIB "bfun.lib";\n'
        f"ring r=0,({names}),dp;\n"
        f"poly f={_singular_poly(poly, X)};\n"
        "list L = bfct(f, 1);\n"          # eng=1 (std): 既定の slimgb は止まることがある
        "ideal rt = L[1]; int i; number mx = leadcoef(rt[1]);\n"
        "for (i=2; i<=size(rt); i++) { if (leadcoef(rt[i]) > mx) { mx = leadcoef(rt[i]); } }\n"
        'printf("LCT %s", string(-mx));\n'
        "quit;\n"
    )
    try:
        pr = subprocess.run(["Singular", "-q"], input=script, text=True,
                            capture_output=True, timeout=timeout)
    except Exception:
        return None
    for line in pr.stdout.splitlines():
        if line.strip().startswith("LCT "):
            try:
                return sp.Rational(line.strip()[4:])
            except Exception:
                return None
    return None


def lct_ideal_lower(gens, X, k: int, *, seed: int = 0, timeout: int = 60):
    """lct_C(I) ≥ k·lct_C(h_1⋯h_k),  h_i = gens の乱数一次結合。常に健全。"""
    rng = random.Random(seed)
    if len(gens) == 1:
        return _singular_lct(gens[0], X, timeout)
    prod = sp.Integer(1)
    for _ in range(k):
        prod *= sum(rng.randint(1, 97) * g for g in gens)
    v = _singular_lct(sp.expand(prod), X, timeout)
    return None if v is None else k * v


# ----------------------------------------------------------------------
# 本体
# ----------------------------------------------------------------------
@dataclass
class LCBounds:
    lo: sp.Rational
    hi: Optional[sp.Rational]
    sources: Dict[str, str] = field(default_factory=dict)
    c: Optional[int] = None              # 余次元 (斉次なら厳密)
    notes: List[str] = field(default_factory=list)

    @property
    def exact(self) -> Optional[sp.Rational]:
        return self.lo if self.hi is not None and self.lo == self.hi else None


def lc_bounds(gens, X, *, witnesses: Sequence[Dict] = (), search: bool = True,
              singular: bool = True, lct_max_vars: int = 3,
              singular_timeout: int = 10, seed: int = 0) -> LCBounds:
    """f = Σ g² の原点での λ に対する健全な [lo, hi] (hi は None もありうる)。"""
    X = list(X)
    gens = [sp.expand(g) for g in gens if sp.expand(g) != 0]
    n = len(X)
    src: Dict[str, str] = {}
    notes: List[str] = []
    homog = _is_homogeneous_ideal(gens, X)

    # --- c (余次元): 斉次なら大域次元 = 原点での局所次元 ----------------
    c = None
    if homog and n <= 10:
        try:
            c = n - _gb_dimension(gens, X)
            src["余次元"] = f"c = {c} (groebner、斉次なので原点での値)"
        except Exception:
            pass

    # --- L1 ------------------------------------------------------------
    J0 = sp.Matrix(gens).jacobian(X).subs({v: 0 for v in X})
    r0 = J0.rank()
    lo = sp.Rational(r0, 2)
    src["L1 rank J(0)"] = f"λ ≥ {lo}"

    # --- U1 ------------------------------------------------------------
    hi = None
    if homog:
        cand = list(witnesses)
        if search:
            cand += search_rational_points(gens, X, seed=seed)
        best: Optional[PointBound] = None
        seen = set()
        for q in cand:
            key = tuple(sp.nsimplify(q.get(v, 0)) for v in X)
            if key in seen:
                continue
            seen.add(key)
            # 局所所属の判定は重いので、階数で先に絞る
            Jq = sp.Matrix(gens).jacobian(X).subs(q)
            rq = Jq.rank()
            if best is not None and rq >= best.r:
                continue
            if c is not None and rq < c:
                continue          # 起こらないはず (rank ≤ codim_q, codim_q ≥ c)
            pb = upper_bound_at_point(gens, X, q)
            if pb.ok and (best is None or pb.r < best.r):
                best = pb
                if c is not None and pb.r == c:
                    break
        if best is not None:
            hi = sp.Rational(best.r, 2)
            src["U1 有理点"] = (f"λ ≤ {hi}  (q = {[best.q[v] for v in X]}, rank J(q) = {best.r}, "
                               "局所所属を商イデアルで確認、斉次なので q→0)")
        else:
            notes.append("U1: 局所所属まで確認できる有理点が見つからなかった")
    else:
        notes.append("U1: イデアルが斉次でないので未実施 (擬斉次なら重み付きで拡張可)")

    # --- L2 / L3 (Singular) -------------------------------------------
    # pay-when-needed: b 関数は高い (一般結合の積で 60s のタイムアウトに達する例
    # がある)。上界が出ていて、下界がそれに届いていないときだけ呼ぶ。
    want_l2 = hi is not None and lo < hi
    if singular and want_l2 and singular_available() and n <= lct_max_vars:
        k = int(2 * hi) + 1 if hi is not None else n
        try:
            l2 = lct_ideal_lower(gens, X, k, seed=seed, timeout=singular_timeout)
        except Exception:
            l2 = None
        if l2 is not None and l2 / 2 > lo:
            lo = l2 / 2
            src["L2 lct_C(I)/2"] = f"λ ≥ {lo} (Singular bfct, k = {k})"
        f = sp.expand(sum(g ** 2 for g in gens))
        l3 = _singular_lct(f, X, singular_timeout)
        if l3 is not None and l3 > lo:
            lo = l3
            src["L3 lct_C(f)"] = f"λ ≥ {lo}"
    elif want_l2 and n > lct_max_vars:
        notes.append(f"L2: n = {n} > {lct_max_vars} なので b 関数は呼ばない")

    if hi is not None and lo > hi:
        notes.append(f"** 矛盾: lo = {lo} > hi = {hi} (どちらかの実装に誤り) **")
    return LCBounds(lo, hi, src, c, notes)


def _gb_dimension(gens, X) -> int:
    G = sp.groebner(gens, *X, order="grevlex")
    n = len(X)
    sup = []
    for g in G.exprs:
        lm = sp.Poly(g, *X).monoms(order="grevlex")[0]
        sup.append(frozenset(i for i, e in enumerate(lm) if e > 0))
    if any(len(s) == 0 for s in sup):
        return -1
    for size in range(n, -1, -1):
        for S in itertools.combinations(range(n), size):
            S = frozenset(S)
            if not any(s <= S for s in sup):
                return size
    return 0


# ----------------------------------------------------------------------
# 縮小ランク回帰 (2 層線形ネット): 真値と witness 付きの問題
# ----------------------------------------------------------------------
def aoyagi_rrr(M, N, H, r):
    """Aoyagi & Watanabe, Neural Networks 18 (2005) 924-933。(λ, m)。"""
    if M + H < N + r:
        return sp.Rational(H * M - H * r + N * r, 2), 1
    if N + H < M + r:
        return sp.Rational(H * N - H * r + M * r, 2), 1
    if M + N < H + r:
        return sp.Rational(M * N, 2), 1
    lam = 2 * (H + r) * (M + N) - (M - N) ** 2 - (H + r) ** 2
    if (M + H + N + r) % 2:
        return sp.Rational(lam + 1, 8), 2
    return sp.Rational(lam, 8), 1


def rrr_problem(M, N, H, r=0, seed: int = 12345):
    """y = BAx, A: HxM, B: NxH。真の B*A* は階数 r。原点 = 真のパラメータ。

    Returns (gens, X, witness, (λ, m)).  witness は top 次元のランク層の
    一般点 (r = 0 のときだけ。r > 0 は斉次でないので U1 の対象外)。
    """
    a = sp.symbols(f"a1:{H*M+1}", real=True)
    b = sp.symbols(f"b1:{N*H+1}", real=True)
    X = list(a) + list(b)
    As = sp.zeros(H, M); Bs = sp.zeros(N, H)
    for i in range(r):
        As[i, i] = 1; Bs[i, i] = 1
    A = As + sp.Matrix(H, M, a); B = Bs + sp.Matrix(N, H, b)
    gens = [e for e in sp.expand(B * A - Bs * As) if e != 0]
    wit = []
    if r == 0:
        k = max(range(min(H, M) + 1),
                key=lambda k: (k * (H + M - k) + N * (H - k), -k))
        rng = random.Random(seed)
        q = {v: sp.Integer(0) for v in X}
        for i in range(k):
            q[a[i * M + i]] = sp.Integer(1)                 # A0 = [I_k 0; 0 0]
        for i in range(N):
            for j in range(k, H):
                q[b[i * H + j]] = sp.Integer(rng.randint(1, 9))   # B0 = [0 | 乱数]
        wit.append(q)
    return gens, X, wit, aoyagi_rrr(M, N, H, r)


if __name__ == "__main__":
    for case in [(1, 1, 1, 0), (2, 1, 1, 0), (2, 2, 1, 0), (2, 2, 2, 0), (3, 3, 1, 0)]:
        gens, X, wit, truth = rrr_problem(*case)
        b1 = lc_bounds(gens, X, witnesses=wit)
        b2 = lc_bounds(gens, X, witnesses=(), search=True)
        print(case, "真値", truth, "| witness:", f"[{b1.lo}, {b1.hi}]",
              "| 探索のみ:", f"[{b2.lo}, {b2.hi}]", "|", b1.notes)
