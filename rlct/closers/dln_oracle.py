r"""
dln_oracle.py — **深層線形網 (DLN) の閉形式オラクル**

M. Aoyagi, *Consideration on the learning efficiency of multiple-layered
neural networks with linear units*, Neural Networks **172**:106132 (2024).
著者公開 PDF:
https://www2.math.cst.nihon-u.ac.jp/aoyagi/wp/wp-content/uploads/2025/02/neuralnet.pdf

深層線形網のファイバーイデアルは変数を増やすと解消が指数的に重くなるが、
この族には λ と θ の閉形式が知られている。イデアルが DLN のものだと
**確認できた**場合に限り、解消を一切せずに値を返す。

------------------------------------------------------------------
モデル
------------------------------------------------------------------
w = {A^(s)}_{s=1..L},  A^(s) は H^(s) × H^(s+1) 行列
H^(1) が出力ユニット数、H^(L+1) が入力ユニット数、間が隠れ層。
r は真のパラメータ ∏ A*^(s) の階数。ファイバーイデアルは
(∏ A^(s) − ∏ A*^(s)) の成分。

------------------------------------------------------------------
健全性の設計 — detect -> reconstruct -> compare
------------------------------------------------------------------
閉形式は「入力が DLN のイデアルである」ことに全面的に依存する。
似て非なるイデアルに適用すれば黙って誤った λ を返すので、

  1. **detect**     : 生成元の形から (H, r) と「変数 -> 行列成分」の
                      対応を推測する。ここは当たらなくてもよい
                      (外せば単に通常の解消に回るだけ)。
  2. **reconstruct**: 推測した (H, r) と対応から、DLN のイデアルを
                      **こちらで組み直す**。
  3. **compare**    : 組み直したイデアルが入力と**イデアルとして一致**
                      するかを厳密に確かめる。一致しなければ発火しない。

3 を通らないものは決して `proved` にしない。したがって誤検出による
誤った λ は構造上ありえず、失敗の仕方は「検出し損なう」(= 遅いだけ) の
一方向に限られる。

比較は生成元の集合としての一致 (並べ替えと符号を許す) で行う。DLN の
生成元は行列成分そのものなので、同じイデアルなら成分が 1 対 1 に対応する。

------------------------------------------------------------------
検出できる範囲 (正直な限界)
------------------------------------------------------------------
* r = 0 (θ* = 0) で、生成元が変数行列の積の成分そのものの場合:
  L = 2, 3 は生成元の構造だけから検出する (変数名を見ない)。
* L ≥ 4、あるいは r > 0 の場合は、呼び出し側が `(H, r)` を宣言する
  必要がある (`verify_dln`)。宣言は 2–3 で検証されるので、誤った宣言で
  誤った λ が出ることはない。
* 変数名が我々の構成器の規約 (`A{s}_{i}_{j}`) に従っていれば、任意の
  L・r=0 でその名前を**手がかりとして**使う。名前は候補を作るためだけに
  使い、正しさは 3 の厳密比較だけに依っている。
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

__all__ = [
    "ShatInfo",
    "lambda_dln",
    "DLNMatch",
    "detect_dln",
    "verify_dln",
    "dln_route",
    "self_check",
]


# ----------------------------------------------------------------------
# Aoyagi の閉形式 (Theorem 1 / Remark 1)
# ----------------------------------------------------------------------
class ShatInfo(tuple):
    """(Shat, ell, a, Mbar, degenerate) を名前つきで持つだけの入れ物。"""

    @property
    def shat(self):
        return self[0]

    @property
    def ell(self):
        return self[1]

    @property
    def a(self):
        return self[2]

    @property
    def mbar(self):
        return self[3]

    @property
    def degenerate(self):
        return self[4]


def _find_shat(M: Sequence[int]) -> ShatInfo:
    r"""条件 (i)–(iii) を満たす Ŝ を求める。

    (i) より Ŝ は M を昇順に並べたときの**同値類まで含んだ先頭部分**に限る
    (同じ値の s を一部だけ入れると M^(S_j) < M^(s) が strict にならない)。
    その候補を小さい順に試し、(ii) と (iii) を満たすものを返す。
    """
    n = len(M)
    idx = sorted(range(n), key=lambda i: M[i])
    zeros = [i for i in range(n) if M[i] == 0]

    # --- Remark 1: M^(s) = 0 が 1 個以上ある場合 ---
    if len(zeros) > 1:
        return ShatInfo((tuple(sorted(zeros)), len(zeros) - 1, None, None,
                         True))
    if len(zeros) == 1:
        pos = [M[i] for i in range(n) if M[i] > 0]
        if not pos:
            return ShatInfo((tuple(zeros), 0, None, None, True))
        mn = min(pos)
        S = tuple(sorted(zeros + [i for i in range(n) if M[i] == mn]))
        return ShatInfo((S, len(S) - 1, None, None, True))

    # --- 一般の場合 ---
    for k in range(2, n + 1):
        S = tuple(sorted(idx[:k]))
        out = [i for i in idx[k:]]
        # (i) 同値類の切れ目でなければ候補にならない
        if out and M[idx[k - 1]] >= M[idx[k]]:
            continue
        ell = k - 1
        tot = sum(M[i] for i in S)
        if any(tot < ell * M[i] for i in S):
            continue
        if any(tot >= ell * M[i] for i in out):
            continue
        # M̄ − 1 < tot/ℓ ≤ M̄
        mbar = int(sp.ceiling(sp.Rational(tot, ell)))
        a = tot - (mbar - 1) * ell
        return ShatInfo((S, ell, a, mbar, False))
    raise ValueError(f"Ŝ が見つかりません: M = {list(M)}")


def lambda_dln(H: Sequence[int], r: int, *, info: bool = False):
    r"""深層線形網の (λ, θ)。H = (H^(1), H^(2), …, H^(L+1))、r は真の階数。

    H^(1) が出力ユニット数、H^(L+1) が入力ユニット数、間が隠れ層。
    L = len(H) − 1 が重み行列の枚数。

    返り値は (λ, θ)。`info=True` なら (λ, θ, ShatInfo) を返す。
    """
    H = [int(h) for h in H]
    if len(H) < 3:
        raise ValueError("H は少なくとも 3 つ (L ≥ 2) 必要です")
    r = int(r)
    if r < 0 or r > min(H):
        raise ValueError(f"階数 r={r} が H={H} と整合しません")
    M = [h - r for h in H]
    si = _find_shat(M)
    base = (-sp.Rational(r ** 2, 2)
            + sp.Rational(r * (H[0] + H[-1]), 2))
    if si.degenerate:
        lam, th = base, 1
    else:
        S, ell, a = si.shat, si.ell, si.a
        tot = sum(M[i] for i in S)
        lam = (base
               + sp.Rational(a * (ell - a), 4 * ell)
               - sp.Rational(ell * (ell - 1), 4) * sp.Rational(tot, ell) ** 2
               + sp.Rational(1, 2) * sum(
                   M[i] * M[j] for i, j in itertools.combinations(S, 2)))
        th = a * (ell - a) + 1
    lam = sp.nsimplify(lam)
    return (lam, th, si) if info else (lam, th)


def _theorem2(H1: int, H2: int, H3: int, r: int):
    """Theorem 2 の場合分け。`lambda_dln` の検算にだけ使う。

    case 3,4,5 は λ が閉じた式で与えられているので直接比べられる。
    case 1,2 は λ が与えられていないので (ℓ, a, θ) だけ比べる。
    """
    if H1 + H2 < H3 + r:
        return dict(case=3, ell=1, a=1, theta=1,
                    lam=sp.Rational(H2 * H1 - H2 * r + H3 * r, 2))
    if H2 + H3 < H1 + r:
        return dict(case=4, ell=1, a=1, theta=1,
                    lam=sp.Rational(H2 * H3 - H2 * r + H1 * r, 2))
    if H1 + H3 < H2 + r:
        return dict(case=5, ell=1, a=1, theta=1,
                    lam=sp.Rational(H1 * H3, 2))
    if (H1 + H2 + H3 + r) % 2 == 0:
        return dict(case=1, ell=2, a=2, theta=1, lam=None)
    return dict(case=2, ell=2, a=1, theta=2, lam=None)


# ----------------------------------------------------------------------
# reconstruct: (H, r) と変数割り当てから DLN のイデアルを組み直す
# ----------------------------------------------------------------------
@dataclass
class DLNMatch:
    """検出・検証に成功した DLN の素性。"""
    H: Tuple[int, ...]
    r: int
    layers: Tuple[Tuple[Tuple[sp.Symbol, ...], ...], ...]   # layers[s][i][j]
    how: str                 # 'structure' / 'names' / 'declared'

    @property
    def L(self) -> int:
        return len(self.H) - 1

    def lam_theta(self):
        return lambda_dln(self.H, self.r)


def _star_matrices(H: Sequence[int], r: int) -> List[sp.Matrix]:
    """階数 r の θ*: A*^(1) の左上 r×r を単位、他は 0。

    λ は θ* の取り方に依らず階数だけで決まる (Aoyagi)。比較対象の
    イデアルをこの θ* で組むので、呼び出し側のイデアルも同じ θ* から
    来ている必要があり、それは compare 段で確かめられる。
    """
    out = []
    for s in range(len(H) - 1):
        rows, cols = H[s], H[s + 1]
        Ms = sp.zeros(rows, cols)
        for i in range(min(r, rows, cols)):
            Ms[i, i] = 1
        out.append(Ms)
    return out


def _reconstruct(match: DLNMatch) -> List[sp.Expr]:
    """match の割り当てで DLN のファイバー生成元を組み直す。

    A^(s) = A*^(s) + U^(s) (U が match.layers の変数) として
    ∏ A^(s) − ∏ A*^(s) の成分を返す。
    """
    H, r = match.H, match.r
    stars = _star_matrices(H, r)
    mats = []
    for s in range(match.L):
        rows, cols = H[s], H[s + 1]
        U = sp.Matrix(rows, cols,
                      lambda i, j: match.layers[s][i][j])
        mats.append(stars[s] + U)
    P = mats[0]
    for Mx in mats[1:]:
        P = P * Mx
    P0 = stars[0]
    for Mx in stars[1:]:
        P0 = P0 * Mx
    D = sp.expand(P - P0)
    return [D[i, j] for i in range(H[0]) for j in range(H[-1])]


# ----------------------------------------------------------------------
# compare: 生成元の集合としての厳密一致
# ----------------------------------------------------------------------
def _same_generator_set(a: Sequence[sp.Expr], b: Sequence[sp.Expr]) -> bool:
    """2 つの生成元列が、並べ替えと全体の符号を除いて一致するか。

    DLN の生成元は行列成分そのものなので、同じ族なら成分が 1 対 1 に
    対応する。ここでは厳密な多項式の同一性だけを見る (数値許容なし)。
    """
    A = [sp.expand(x) for x in a]
    B = [sp.expand(x) for x in b]
    A = [x for x in A if x != 0]
    B = [x for x in B if x != 0]
    if len(A) != len(B):
        return False
    key = lambda x: sp.srepr(x)
    remaining = list(B)
    for x in A:
        hit = None
        for i, y in enumerate(remaining):
            if sp.expand(x - y) == 0 or sp.expand(x + y) == 0:
                hit = i
                break
        if hit is None:
            return False
        remaining.pop(hit)
    return not remaining


def verify_dln(generators: Sequence[sp.Expr],
               variables: Sequence[sp.Symbol],
               H: Sequence[int], r: int,
               *, layers=None, how: str = "declared") -> Optional[DLNMatch]:
    """宣言された (H, r) が本当にこのイデアルを与えるかを確かめる。

    `layers` を渡さない場合は `variables` を層ごとに行優先で並べた
    順序を仮定する (Σ H_s H_{s+1} == len(variables) が必要)。

    一致すれば DLNMatch、しなければ None。**None の側に倒すのが安全側**。
    """
    H = tuple(int(h) for h in H)
    if len(H) < 3:
        return None
    r = int(r)
    if r < 0 or r > min(H):
        return None
    variables = list(variables)
    need = sum(H[s] * H[s + 1] for s in range(len(H) - 1))
    if layers is None:
        if need != len(variables):
            return None
        layers, pos = [], 0
        for s in range(len(H) - 1):
            rows, cols = H[s], H[s + 1]
            layers.append(tuple(
                tuple(variables[pos + i * cols + j] for j in range(cols))
                for i in range(rows)))
            pos += rows * cols
        layers = tuple(layers)
    match = DLNMatch(H=H, r=r, layers=tuple(layers), how=how)
    try:
        rebuilt = _reconstruct(match)
    except Exception:
        return None
    if not _same_generator_set(generators, rebuilt):
        return None
    return match


# ----------------------------------------------------------------------
# detect: 生成元の形から (H, r) と割り当てを推測する
# ----------------------------------------------------------------------
_NAME_RE = re.compile(r"^A(\d+)_(\d+)_(\d+)$")


def _layers_from_names(variables: Sequence[sp.Symbol]):
    """変数名が `A{s}_{i}_{j}` 規約なら層構造を読む。

    名前は**候補を作る手がかり**にすぎない。返した候補は必ず
    reconstruct -> compare を通すので、名前が嘘でも誤った λ は出ない。
    """
    cells: Dict[int, Dict[Tuple[int, int], sp.Symbol]] = {}
    for v in variables:
        m = _NAME_RE.match(getattr(v, "name", ""))
        if not m:
            return None
        s, i, j = (int(x) for x in m.groups())
        cells.setdefault(s, {})[(i, j)] = v
    if not cells:
        return None
    ss = sorted(cells)
    if ss != list(range(ss[0], ss[0] + len(ss))):
        return None
    layers, shapes = [], []
    for s in ss:
        cell = cells[s]
        rows = max(i for i, _ in cell) + 1
        cols = max(j for _, j in cell) + 1
        if len(cell) != rows * cols:
            return None
        layers.append(tuple(tuple(cell[(i, j)] for j in range(cols))
                            for i in range(rows)))
        shapes.append((rows, cols))
    for s in range(len(shapes) - 1):
        if shapes[s][1] != shapes[s + 1][0]:
            return None                      # 積が組めない
    H = tuple([shapes[0][0]] + [c for _, c in shapes])
    return H, tuple(layers)


def _grid_from_signatures(gens: Sequence[sp.Expr],
                          variables: Sequence[sp.Symbol]):
    """生成元の添字格子と、第 1 層・第 L 層の変数を構造から読む。

    変数 v = u^(s)_{ij} が生成元 g_{pq} に現れるかは

      s = 1   : p = i のときだけ        -> S(v) は「行」
      s = L   : q = j のときだけ        -> S(v) は「列」
      1<s<L   : 常に現れる              -> S(v) は全体

    なので、S(v) の形から第 1 層・第 L 層・中間層が分かれる。行の族と
    列の族は「各行が各列とちょうど 1 点で交わる」ことで格子を復元できる。

    戻り値 (rows, cols, sig) — rows/cols は生成元添字の分割、
    sig は変数 -> S(v)。復元できなければ None。
    """
    G = len(gens)
    sig: Dict[sp.Symbol, frozenset] = {}
    for v in variables:
        s = frozenset(i for i, g in enumerate(gens)
                      if sp.expand(sp.diff(g, v)) != 0)
        if not s:
            return None                       # 現れない変数がある
        sig[v] = s
    full = frozenset(range(G))
    parts = {s for s in sig.values() if s != full}
    if not parts:
        return None
    fams: List[List[frozenset]] = []
    for p in sorted(parts, key=lambda s: (len(s), sorted(s))):
        placed = False
        for fam in fams:
            if all(q == p or not (q & p) for q in fam):
                fam.append(p)
                placed = True
                break
        if not placed:
            fams.append([p])
    fams = [f for f in fams if sum(len(x) for x in f) == G]
    if len(fams) != 2:
        return None
    A, B = fams
    if len({len(x) for x in A}) != 1 or len({len(x) for x in B}) != 1:
        return None
    if len(A) * len(B) != G:
        return None
    for x in A:
        for y in B:
            if len(x & y) != 1:
                return None
    return A, B, sig


def _detect_r0(gens: Sequence[sp.Expr], variables: Sequence[sp.Symbol]):
    """θ* = 0 (r = 0) の DLN 候補を生成元の構造から作る。L = 2, 3 のみ。

    L ≥ 4 では中間層が 2 枚以上あり、S(v) だけでは層の順序が決まらない
    ので候補を作らない (名前の手がかり側に任せる)。
    """
    gens = [sp.expand(g) for g in gens]
    if not gens:
        return None
    # 全生成元が同じ次数の多重線形 (各単項式が相異なる L 変数の積)
    L = None
    for g in gens:
        try:
            p = sp.Poly(g, *variables)
        except Exception:
            return None
        for m in p.monoms():
            if any(e > 1 for e in m):
                return None
            d = sum(m)
            if L is None:
                L = d
            elif d != L:
                return None
    if L is None or L < 2 or L > 3:
        return None
    grid = _grid_from_signatures(gens, variables)
    if grid is None:
        return None
    A, B, sig = grid
    full = frozenset(range(len(gens)))
    mid = [v for v in variables if sig[v] == full]
    first = [v for v in variables if any(sig[v] == x for x in A)]
    last = [v for v in variables if any(sig[v] == x for x in B)]
    if len(first) + len(last) + len(mid) != len(variables):
        return None
    H1, Hend = len(A), len(B)
    out = []
    # A を第 1 層側に割り当てる場合と、逆にする場合の両方を候補にする
    for h1, hend, fv, lv in ((H1, Hend, first, last), (Hend, H1, last, first)):
        if h1 == 0 or hend == 0:
            continue
        if L == 2:
            if len(fv) % h1 or len(lv) % hend or mid:
                continue
            h2 = len(fv) // h1
            if len(lv) // hend != h2:
                continue
            H = (h1, h2, hend)
        else:                                # L = 3
            if len(fv) % h1 or len(lv) % hend:
                continue
            h2, h3 = len(fv) // h1, len(lv) // hend
            if h2 * h3 != len(mid):
                continue
            H = (h1, h2, h3, hend)
        shapes = [(H[s], H[s + 1]) for s in range(len(H) - 1)]
        pools = [fv] + ([mid] if L == 3 else []) + [lv]
        if any(len(pool) != a * b for pool, (a, b) in zip(pools, shapes)):
            continue
        # 層内の並べ方は、compare が通る並びを総当りで探す。層のサイズは
        # 小さい (実用上 <= 9) ので、行ごとの割り当てだけを試す。
        out.append((H, pools))
    return out or None


def _try_assignments(gens, H, pools, max_perm: int = 50000):
    """層ごとの変数プールから、compare を通す割り当てを探す。

    層内の並べ方は本質的に行・列の番号付けの自由度でしかないので、
    小さい層では総当りで足りる。見つからなければ None。
    """
    shapes = [(H[s], H[s + 1]) for s in range(len(H) - 1)]
    total = 1
    for pool in pools:
        f = 1
        for k in range(2, len(pool) + 1):
            f *= k
        total *= f
        if total > max_perm:
            return None
    for combo in itertools.product(*[itertools.permutations(p)
                                     for p in pools]):
        layers = []
        for order, (a, b) in zip(combo, shapes):
            layers.append(tuple(tuple(order[i * b + j] for j in range(b))
                                for i in range(a)))
        m = verify_dln(gens, [v for p in pools for v in p], H, 0,
                       layers=tuple(layers), how="structure")
        if m is not None:
            return m
    return None


def detect_dln(generators: Sequence[sp.Expr],
               variables: Sequence[sp.Symbol]) -> Optional[DLNMatch]:
    """イデアルが DLN のものかを調べ、確認できたら DLNMatch を返す。

    検出 -> 組み直し -> 厳密比較。比較を通らない限り None を返すので、
    誤検出で誤った λ が出ることはない。
    """
    gens = [sp.expand(g) for g in generators if sp.expand(g) != 0]
    variables = list(variables)
    if not gens or not variables:
        return None

    # (a) 変数名の規約が使える場合 (任意の L、r = 0)
    named = _layers_from_names(variables)
    if named is not None:
        H, layers = named
        m = verify_dln(gens, variables, H, 0, layers=layers, how="names")
        if m is not None:
            return m

    # (b) 構造からの検出 (L = 2, 3、r = 0)
    cands = _detect_r0(gens, variables)
    if cands:
        for H, pools in cands:
            m = _try_assignments(gens, H, pools)
            if m is not None:
                return m
    return None


def dln_route(generators: Sequence[sp.Expr],
              variables: Sequence[sp.Symbol],
              *, hint: Optional[Tuple[Sequence[int], int]] = None):
    """DLN オラクル経路。(λ, θ, DLNMatch) か None を返す。

    `hint=(H, r)` を渡すと、その宣言を検証してから使う (r > 0 や L ≥ 4 は
    この経路でのみ扱える)。検証を通らなければ None。
    """
    if hint is not None:
        H, r = hint
        m = verify_dln(generators, variables, H, r, how="declared")
        if m is None:
            return None
        lam, th = m.lam_theta()
        return lam, th, m
    m = detect_dln(generators, variables)
    if m is None:
        return None
    lam, th = m.lam_theta()
    return lam, th, m


# ----------------------------------------------------------------------
# 自己検査
# ----------------------------------------------------------------------
def _build_ideal(H, r):
    """テスト用に DLN のイデアルを素直に組む (変数名は規約どおり)。"""
    L = len(H) - 1
    mats, variables = [], []
    for s in range(L):
        rows, cols = H[s], H[s + 1]
        M = sp.Matrix(rows, cols, lambda i, j:
                      sp.Symbol(f"A{s + 1}_{i}_{j}", real=True))
        mats.append(M)
        variables += [M[i, j] for i in range(rows) for j in range(cols)]
    stars = _star_matrices(H, r)
    P = mats[0] + stars[0]
    for s in range(1, L):
        P = P * (mats[s] + stars[s])
    P0 = stars[0]
    for Mx in stars[1:]:
        P0 = P0 * Mx
    D = sp.expand(P - P0)
    gens = [D[i, j] for i in range(H[0]) for j in range(H[-1])]
    return [g for g in gens if g != 0], variables


def self_check(verbose: bool = True) -> List[str]:
    """オラクルが (1) 公式として正しく (2) 誤検出しないことを確かめる。"""
    bad: List[str] = []

    def chk(cond, msg):
        if not cond:
            bad.append(msg)
            if verbose:
                print("  NG", msg)

    # (1) Theorem 2 との突き合わせ (L = 2)
    for H1, H2, H3, r in itertools.product(range(1, 5), range(1, 5),
                                           range(1, 5), range(0, 4)):
        if r > min(H1, H2, H3):
            continue
        lam, th, si = lambda_dln((H1, H2, H3), r, info=True)
        t2 = _theorem2(H1, H2, H3, r)
        if t2["lam"] is not None:
            chk(lam == t2["lam"],
                f"Thm2 case{t2['case']} H=({H1},{H2},{H3}) r={r}: "
                f"{lam} != {t2['lam']}")
        if not si.degenerate:
            chk(th == t2["theta"],
                f"Thm2 case{t2['case']} θ H=({H1},{H2},{H3}) r={r}: "
                f"{th} != {t2['theta']}")

    # (2) 検出が正しい (H, r) を復元する
    for H, r in [((1, 1, 1), 0), ((2, 1, 2), 0), ((1, 2, 1), 0),
                 ((2, 2, 1), 0), ((2, 2, 2), 0), ((1, 1, 1, 1), 0),
                 ((2, 1, 2), 1), ((2, 2, 2), 1)]:
        gens, V = _build_ideal(H, r)
        if r == 0:
            m = detect_dln(gens, V)
            chk(m is not None, f"検出に失敗 H={H} r={r}")
            if m is not None:
                chk(tuple(m.H) == tuple(H) or tuple(m.H) == tuple(reversed(H)),
                    f"H を誤認 {m.H} != {H}")
                chk(m.lam_theta() == lambda_dln(H, r),
                    f"λ が不一致 H={H}: {m.lam_theta()} != {lambda_dln(H, r)}")
        got = dln_route(gens, V, hint=(H, r))
        chk(got is not None, f"宣言経路が通らない H={H} r={r}")
        if got is not None:
            chk((got[0], got[1]) == lambda_dln(H, r),
                f"宣言経路の λ が不一致 H={H} r={r}")

    # (3) **誤検出しない**: DLN でないイデアルで発火してはいけない
    a, b, c, d = sp.symbols("A1_0_0 A2_0_0 A3_0_0 A4_0_0", real=True)
    negatives = [
        ([a * b + a], [a, b]),                       # 余分な線形項
        ([a ** 2 * b], [a, b]),                      # 多重線形でない
        ([a * b, a * c], [a, b, c]),                 # 格子が組めない
        ([a * b * c + a * b], [a, b, c]),
        ([a + b], [a, b]),                           # 正則
        ([a * b - c * d], [a, b, c, d]),             # 行列積ではない
    ]
    for gens, V in negatives:
        m = detect_dln(gens, V)
        chk(m is None, f"誤検出: {gens} を DLN と判定した ({m})")
        # 宣言経路も、宣言が嘘なら通してはいけない
        for H in [(1, 1, 1), (1, 1, 1, 1), (2, 1, 2)]:
            for r in (0, 1):
                if r > min(H):
                    continue
                g2 = dln_route(gens, V, hint=(H, r))
                chk(g2 is None,
                    f"誤検証: {gens} を H={H} r={r} の DLN と認めた")

    # (4) 宣言が構造と合っていても r が違えば通らない
    gens, V = _build_ideal((2, 2, 2), 0)
    chk(dln_route(gens, V, hint=((2, 2, 2), 1)) is None,
        "r を 0 -> 1 と偽った宣言が通った")

    if verbose:
        print(f"dln_oracle.self_check: "
              f"{'全て OK' if not bad else f'{len(bad)} 件 NG'}")
    return bad


if __name__ == "__main__":        # pragma: no cover
    import sys
    sys.exit(1 if self_check() else 0)
