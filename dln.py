r"""
bench/dln.py — **深層線形網の厳密な λ と θ** (Aoyagi の閉形式)

M. Aoyagi, *Consideration on the learning efficiency of multiple-layered
neural networks with linear units*, Neural Networks **172**:106132 (2024).
著者公開 PDF: https://www2.math.cst.nihon-u.ac.jp/aoyagi/wp/wp-content/uploads/2025/02/neuralnet.pdf

LLC の数値推定の論文 (Lau, Furman, Wang, Murfet & Wei, AISTATS 2025) が
**校正に使っている唯一の族**がこれなので、我々の厳密計算をこの公式と
突き合わせるのが最も説得力のある検証になる。

------------------------------------------------------------------
モデル
------------------------------------------------------------------
w = {A^(s)}_{s=1..L},  A^(s) は H^(s) × H^(s+1) 行列
入力 X ∈ R^{H^(L+1)}、出力 Y = (∏_s A^(s)) X + 雑音

  H^(L+1) : 入力ユニット数
  H^(1)   : 出力ユニット数
  H^(s)   : 第 s 隠れ層のユニット数
  r       : 真のパラメータ ∏ A*^(s) の階数

------------------------------------------------------------------
公式 (Theorem 1)
------------------------------------------------------------------
    M^(s) = H^(s) − r        (s = 1, …, L+1)

Ŝ ⊂ {1,…,L+1} を次で定める。ℓ = Card(Ŝ) − 1、Ŝ = {S_1,…,S_{ℓ+1}}:

    (i)   M^(S_j) < M^(s)            (S_j ∈ Ŝ, s ∉ Ŝ)
    (ii)  Σ_k M^(S_k) ≥ ℓ · M^(s)    (s ∈ Ŝ)
    (iii) Σ_k M^(S_k) < ℓ · M^(s)    (s ∉ Ŝ)

M̄ は  M̄ − 1 < (Σ_j M^(S_j))/ℓ ≤ M̄  を満たす整数、
a = Σ_k M^(S_k) − (M̄ − 1)ℓ。このとき

    λ = −r²/2 + r(H^(1) + H^(L+1))/2
        + a(ℓ−a)/(4ℓ)
        − (ℓ(ℓ−1)/4) · (Σ_j M^(S_j)/ℓ)²
        + (1/2) Σ_{1≤i<j≤ℓ+1} M^(S_i) M^(S_j)

    θ = a(ℓ−a) + 1

**Remark 1 の退化した場合**: M^(s) = 0 となる s が
  1 個なら Ŝ = {s : M^(s) = min{M^(s') : M^(s') > 0}} ∪ {s : M^(s) = 0}
  2 個以上なら Ŝ = {s : M^(s) = 0}
いずれも λ = −r²/2 + r(H^(1)+H^(L+1))/2、θ = 1。

------------------------------------------------------------------
実装を信用してよい根拠 (`self_check()` が毎回検査する)
------------------------------------------------------------------
PDF からの抽出では M̄ の上線や r² の指数が落ちている可能性があったので、
**独立に分かっている値 4 つと突き合わせて確認した**:

1. 同論文 Theorem 2 (L=2, 縮約ランク回帰) の case 5:
   H^(1)+H^(3) < H^(2)+r なら λ = H^(1)H^(3)/2。
   上の式に Ŝ={1,3}, ℓ=1, a=1 を入れると
       −r²/2 + r(H1+H3)/2 + (1/2)(H1−r)(H3−r) = H1H3/2  ✓
2. 同 case 3: H^(1)+H^(2) < H^(3)+r なら λ = (H1H2 − rH2 + rH3)/2。
   Ŝ={1,2}, ℓ=1, a=1 を入れると一致  ✓
3. 同 case 1 と 2 の差 (+1/8): a(ℓ−a)/(4ℓ) が ℓ=2 で a=2 なら 0、
   a=1 なら 1/8  ✓
4. **我々の厳密計算との一致**: H=(1,1,1), r=0 (= y=b(ax), イデアル ⟨ab⟩)
   で λ=1/2, θ=2。H=(1,1,1,1), r=0 (= ⟨abc⟩) で λ=1/2, θ=3。
   どちらも `torch_rlct` の記号計算が返した値と**位数まで一致**  ✓

4 は ℓ(ℓ−1) の項が 0 でない場合を含むので、その項も検証できている。
"""
from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple

import sympy as sp

__all__ = ["lambda_dln", "dln_cases", "self_check", "ShatInfo"]


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
               + sp.Rational(1, 2) * sum(M[i] * M[j]
                                         for i, j in itertools.combinations(S, 2)))
        th = a * (ell - a) + 1
    lam = sp.nsimplify(lam)
    return (lam, th, si) if info else (lam, th)


# ----------------------------------------------------------------------
# Theorem 2 (L = 2, 縮約ランク回帰) — 独立な経路
# ----------------------------------------------------------------------
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


def self_check(verbose: bool = True) -> List[Tuple[str, bool, str]]:
    r"""公式の実装を独立な経路と突き合わせる。

    (A) Theorem 2 の case 3,4,5 の λ と一致するか (L=2 を総当り)
    (B) Theorem 2 の (ℓ, a, θ) と一致するか (case 1,2 を含む)
    (C) 我々の厳密計算が出した値と一致するか
        H=(1,1,1), r=0 -> (1/2, 2) と H=(1,1,1,1), r=0 -> (1/2, 3)
    """
    out: List[Tuple[str, bool, str]] = []
    # (A)(B)
    bad_lam, bad_sh, n_tested = [], [], 0
    for H1 in range(1, 7):
        for H2 in range(1, 7):
            for H3 in range(1, 7):
                for r in range(0, min(H1, H2, H3) + 1):
                    try:
                        lam, th, si = lambda_dln((H1, H2, H3), r, info=True)
                    except Exception as e:                # noqa: BLE001
                        bad_sh.append(f"H=({H1},{H2},{H3}) r={r}: {e}")
                        continue
                    t2 = _theorem2(H1, H2, H3, r)
                    n_tested += 1
                    if t2["lam"] is not None and lam != t2["lam"]:
                        bad_lam.append(f"H=({H1},{H2},{H3}) r={r} "
                                       f"case{t2['case']}: {lam} 対 {t2['lam']}")
                    if not si.degenerate:
                        if (si.ell, si.a, th) != (t2["ell"], t2["a"],
                                                  t2["theta"]):
                            bad_sh.append(
                                f"H=({H1},{H2},{H3}) r={r} case{t2['case']}: "
                                f"(ℓ,a,θ)=({si.ell},{si.a},{th}) 対 "
                                f"({t2['ell']},{t2['a']},{t2['theta']})")
    out.append(("Theorem 2 の λ と一致 (case 3,4,5)", not bad_lam,
                f"{n_tested} 件中 不一致 {len(bad_lam)}"
                + (f" 例 {bad_lam[0]}" if bad_lam else "")))
    out.append(("Theorem 2 の (ℓ,a,θ) と一致", not bad_sh,
                f"不一致 {len(bad_sh)}"
                + (f" 例 {bad_sh[0]}" if bad_sh else "")))
    # (C)
    known = [((1, 1, 1), 0, sp.Rational(1, 2), 2),
             ((1, 1, 1, 1), 0, sp.Rational(1, 2), 3)]
    bad_known = []
    for H, r, lam0, th0 in known:
        lam, th = lambda_dln(H, r)
        if (lam, th) != (lam0, th0):
            bad_known.append(f"H={H} r={r}: ({lam},{th}) 対 ({lam0},{th0})")
    out.append(("厳密計算が出した値と一致 (⟨ab⟩, ⟨abc⟩)", not bad_known,
                "λ=1/2 θ=2 と λ=1/2 θ=3"
                if not bad_known else "; ".join(bad_known)))
    if verbose:
        for name, ok, note in out:
            print(f"  [{'OK' if ok else 'FAIL'}] {name}  {note}")
    return out


# ----------------------------------------------------------------------
# ベンチ用のケース
# ----------------------------------------------------------------------
def dln_cases(max_vars: int = 12) -> List[Tuple]:
    r"""深層線形網のファイバーイデアルと真値の列。

    真のパラメータは階数 r の対角行列に取る (一般性を失わない:
    λ は直交変換で不変で、Aoyagi の公式も階数だけで決まる)。
    生成元は (∏A^(s) − ∏A*^(s)) の成分。

    Returns [(名前, f, 変数, 族, 真値 λ, 真値 θ)]
    """
    out = []
    specs = [((1, 1, 1), 0), ((1, 1, 1, 1), 0), ((1, 1, 1, 1, 1), 0),
             ((2, 1, 2), 0), ((2, 2, 2), 1), ((1, 2, 1), 0),
             ((2, 1, 1), 0), ((1, 1, 2), 0), ((2, 2, 1), 0),
             ((3, 1, 3), 0), ((2, 1, 2), 1)]
    for H, r in specs:
        L = len(H) - 1
        mats = []
        nv = 0
        for s in range(L):
            rows, cols = H[s], H[s + 1]
            nv += rows * cols
        if nv > max_vars:
            continue
        for s in range(L):
            rows, cols = H[s], H[s + 1]
            mats.append(sp.Matrix(rows, cols, lambda i, j:
                                  sp.Symbol(f"A{s+1}_{i}_{j}", real=True)))
        P = mats[0]
        for Mx in mats[1:]:
            P = P * Mx
        # 真のパラメータ: 階数 r の対角
        star = sp.zeros(H[0], H[-1])
        for i in range(r):
            star[i, i] = 1
        # θ* の値そのものは λ に効かないが、生成元が θ* で消えるように
        # A*^(s) を具体的に取る: A*^(1) の左上 r×r を単位、他を 0
        stars = []
        for s in range(L):
            rows, cols = H[s], H[s + 1]
            Ms = sp.zeros(rows, cols)
            for i in range(min(r, rows, cols)):
                Ms[i, i] = 1
            stars.append(Ms)
        Pstar = stars[0]
        for Mx in stars[1:]:
            Pstar = Pstar * Mx
        gens = [sp.expand(P[i, j] - Pstar[i, j])
                for i in range(H[0]) for j in range(H[-1])]
        # 変数は θ* まわりにずらす: A = A* + u
        subs = {}
        variables = []
        for s in range(L):
            rows, cols = H[s], H[s + 1]
            for i in range(rows):
                for j in range(cols):
                    u = sp.Symbol(f"u{s+1}_{i}_{j}", real=True)
                    variables.append(u)
                    subs[sp.Symbol(f"A{s+1}_{i}_{j}", real=True)] = \
                        stars[s][i, j] + u
        gens = [sp.expand(g.subs(subs, simultaneous=True)) for g in gens]
        f = sp.expand(sum(g ** 2 for g in gens))
        if f == 0 or f.is_number:
            continue
        lam, th = lambda_dln(H, r)
        out.append((f"DLN H={H} r={r}", f, tuple(variables), "dln", lam, th))
    return out


def _demo() -> int:
    print("===== Aoyagi の深層線形網の公式: 実装の検算 =====")
    res = self_check()
    print("\n===== いくつかの値 =====")
    for H, r in [((1, 1, 1), 0), ((1, 1, 1, 1), 0), ((2, 2, 2), 0),
                 ((3, 2, 3), 1), ((10, 5, 10), 2), ((100, 100, 100), 0)]:
        lam, th, si = lambda_dln(H, r, info=True)
        print(f"  H={str(H):<16} r={r}  λ={str(lam):<10} θ={th:<3} "
              f"Ŝ={si.shat} ℓ={si.ell} a={si.a}"
              + ("  (Remark 1)" if si.degenerate else ""))
    return 0 if all(ok for _, ok, _ in res) else 1


if __name__ == "__main__":
    raise SystemExit(_demo())
