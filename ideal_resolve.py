r"""
ideal_resolve.py
================

イデアルを運ぶ解消器 (実験版)。

resolve_singularity.py は多項式 f = sum g_i^2 を運ぶので、途中の節点で
「イデアルの生成元を座標に取る」操作ができない。lambda はイデアルだけで
決まる (Aoyagi, Entropy 2019, Lemma 1(2)) のに、多項式に潰した時点で
その自由度を失っている。実際 Vandermonde 型の chart 内で座標依存が出る
(chart 座標で 3/2、G_1 を座標に取ると 7/6)。

ここでは各節点で **生成元の組** を運び、次を順に試す:

  1. 定数係数による線形簡約           (イデアルは不変)
  2. 共通単項式の括り出し             (k に繰り入れる。f = x^k ... なので 2c)
  3. 正則方向の消去                   (生成元 g に原点で dg != 0 があれば
                                       その変数を消し lambda += 1/2)
  4. 停止判定: 全生成元が単項式 x 単元 -> イデアルは単項式イデアル。
     その局所 lambda は Howald の線形計画で **厳密に** 決まる
  5. ブローアップ                     (中心は最小ヒッティング集合)

3 が本質的な追加で、resolve_singularity では根で一度しか行われていない
(nn_rlct.local_rlct_from_ideal の前処理)。ブローアップのたびに新しい正則
方向が生まれるので、節点ごとに繰り返す必要がある。

4 も改善になっている。従来の停止条件は「f_rest が原点で単元」= 単一の
単項式になるまで掘る、だったが、単項式イデアルになった時点で LP で
厳密に閉じられる。sum_i y^{2a_i} u_i^2 は u_i^2 >= 0 なので符号の相殺が
なく、max_i y^{2a_i} と同値であることを使う。

------------------------------------------------------------------
未実装 (実験版の限界)
------------------------------------------------------------------
* phi を追跡していないので certify.py / lean_export.py に渡せない。
  値の比較のための実験用。
* 中心の付け替えと滑らかな因子の座標化は入れていない (3 がその多くを
  代替する)。
* 重み付きブローアップは入れていない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

from resolve_singularity import (ResolutionFailure, _min_center, newton_rlct)

__all__ = ["resolve_ideal", "IdealResolution"]


# ----------------------------------------------------------------------
# 補助
# ----------------------------------------------------------------------
def _linear_reduce(gens: Sequence[sp.Expr], vars_) -> List[sp.Expr]:
    """定数係数の線形結合で生成元を簡約する (イデアルは変わらない)。"""
    polys = [sp.expand(g) for g in gens if sp.expand(g) != 0]
    if not polys:
        return []
    monoms = sorted({m for p in polys for m in p.as_coefficients_dict()},
                    key=sp.default_sort_key)
    M = sp.Matrix([[sp.nsimplify(p.as_coefficients_dict().get(m, 0))
                    for m in monoms] for p in polys])
    R, _ = M.rref()
    out = []
    for i in range(R.rows):
        row = [R[i, j] for j in range(len(monoms))]
        if any(v != 0 for v in row):
            out.append(sp.expand(sum(row[j] * monoms[j]
                                     for j in range(len(monoms)))))
    return out


def _mono_content(g, vars_) -> Tuple[List[int], sp.Expr]:
    """g から単項式の最大公約因子を括り出す。"""
    p = sp.Poly(sp.expand(g), *vars_)
    ms = p.monoms()
    mins = [min(m[i] for m in ms) for i in range(len(vars_))]
    if not any(mins):
        return mins, p.as_expr()
    d = {tuple(e - a for e, a in zip(mon, mins)): c
         for mon, c in zip(ms, p.coeffs())}
    return mins, sp.Poly.from_dict(d, *vars_).as_expr()


def _is_monomial_times_unit(g, vars_) -> Optional[List[int]]:
    """g = x^a * (原点で非零) なら a を返す。そうでなければ None。"""
    a, rest = _mono_content(g, vars_)
    if rest.subs({v: 0 for v in vars_}) != 0:
        return a
    return None


def _blowup_gen(g, vars_, i: int, J: Sequence[int]) -> sp.Expr:
    """chart i のブローアップ x_j -> x_i x_j を指数ベクトル上で行う。"""
    p = sp.Poly(sp.expand(g), *vars_)
    acc: Dict[Tuple[int, ...], sp.Expr] = {}
    shift = [j for j in J if j != i]
    for mon, c in zip(p.monoms(), p.coeffs()):
        e = list(mon)
        e[i] = mon[i] + sum(mon[j] for j in shift)
        key = tuple(e)
        acc[key] = acc.get(key, sp.S.Zero) + c
    acc = {kk: sp.simplify(vv) for kk, vv in acc.items()}
    acc = {kk: vv for kk, vv in acc.items() if vv != 0}
    if not acc:
        return sp.S.Zero
    return sp.Poly.from_dict(acc, *vars_).as_expr()


def _support_indices_list(node) -> List[int]:
    """生成元に現れる変数の添字。"""
    used = set()
    for g in node.gens:
        used |= (g.free_symbols & set(node.vars))
    return [i for i, v in enumerate(node.vars) if v in used]


def _push(e: Sequence[int], i: int, J: Sequence[int]) -> List[int]:
    ne = list(e)
    ne[i] = e[i] + sum(e[j] for j in J if j != i)
    return ne


# ----------------------------------------------------------------------
# 節点
# ----------------------------------------------------------------------
@dataclass
class INode:
    name: str
    gens: List[sp.Expr]
    vars: Tuple[sp.Symbol, ...]
    k: List[int]
    h: List[int]
    lam_reg: sp.Rational
    depth: int
    used: frozenset = frozenset()     # 既に座標に取った変数
    note: str = ""


@dataclass
class IdealResolution:
    rlct: sp.Rational
    multiplicity: int
    leaves: List[Tuple[str, sp.Rational, int, str]]
    n_nodes: int
    n_regular: int   # 座標に取った回数

    def report(self) -> str:
        L = [f"lambda = {self.rlct}, m = {self.multiplicity} "
             f"(葉 {len(self.leaves)}, 節点 {self.n_nodes}, "
             f"正則方向の消去 {self.n_regular} 回)"]
        best = [x for x in self.leaves if x[1] == self.rlct]
        for name, lam, m, note in best[:5]:
            L.append(f"  最小を与える葉 {name}: {lam} (m={m}) {note}")
        return "\n".join(L)


# ----------------------------------------------------------------------
# 各ステップ
# ----------------------------------------------------------------------
def _terminal_value(node: INode):
    """全生成元が単項式 x 単元なら、Howald の LP で厳密な lambda を返す。"""
    exps = []
    for g in node.gens:
        a = _is_monomial_times_unit(g, node.vars)
        if a is None:
            return None
        exps.append(a)
    if not exps:
        return (sp.oo, 0)          # イデアルが 0 -> 極なし
    # f = x^k * sum (x^{a_i} u_i)^2 ~ x^k * max_i x^{2a_i}
    monoms = [tuple(kk + 2 * ai for kk, ai in zip(node.k, a)) for a in exps]
    P = sum(sp.prod([v ** e for v, e in zip(node.vars, m)]) for m in monoms)
    P = sp.expand(P)
    if P == 0 or P.is_number:
        return (sp.oo, 0)
    lam = newton_rlct(P, node.vars, h=list(node.h))
    try:
        from certify import newton_multiplicity
        m = newton_multiplicity(P, node.vars, h=list(node.h)) or 1
    except Exception:
        m = 1
    return (lam, m)


def _coordinate_step(node: INode) -> Optional[INode]:
    """生成元 g に原点で dg/dv != 0 があれば、g 自身を座標 v に取る。

    これが resolve_singularity に無かった操作。多項式の *因子* ではなく
    イデアルの *生成元* を座標にするので、Vandermonde 型の chart 内の
    G_1 = a_1 + a_2 b_2 + a_3 b_3 のような「和の項」も使える。

    条件は v が例外因子に使われていないこと (k_v = h_v = 0)。そうでないと
    x^k が新座標で単項式でなくなる。

    注意: 「lambda += 1/2 して変数を消す」という分離は k = h = 0 の
    ときしか使えない (f = x^k (v^2 + ...) は v について分離しない)。
    ここでは分離せず座標変換だけを行い、v を生成元として残す。単項式
    生成元は終端判定の Howald LP がそのまま扱えるので、それで十分。
    """
    vars_ = list(node.vars)
    zeros = {v: 0 for v in vars_}
    free = {v for v, kv, hv in zip(vars_, node.k, node.h)
            if kv == 0 and hv == 0}
    for gi, g in enumerate(node.gens):
        for v in vars_:
            if v not in free or v in node.used or v not in g.free_symbols:
                continue
            pg = sp.Poly(g, v)
            if pg.degree() != 1:
                continue
            A = pg.coeff_monomial(v)
            B = pg.coeff_monomial(1)
            if sp.simplify(A.subs(zeros)) == 0:
                continue
            sol = sp.cancel(-B / A)
            new_gens = []
            ok = True
            for gj, hgen in enumerate(node.gens):
                if gj == gi:
                    continue
                hh = sp.cancel(sp.together(hgen.subs(v, sol)))
                num, den = sp.fraction(hh)
                if sp.simplify(den.subs(zeros)) == 0:
                    ok = False
                    break
                num = sp.expand(num)
                if num != 0:
                    new_gens.append(num)
            if not ok:
                continue
            return INode(name=node.name + "c",
                         gens=_linear_reduce([v] + new_gens, node.vars),
                         vars=node.vars,
                         k=list(node.k), h=list(node.h),
                         lam_reg=node.lam_reg,
                         depth=node.depth + 1,
                         used=node.used | {v},
                         note=f"{sp.sstr(g)[:30]} を座標 {v} に取る")
    return None


def _normalize(node: INode) -> INode:
    """線形簡約 + 共通単項式の括り出し。"""
    node.gens = _linear_reduce(node.gens, node.vars)
    if not node.gens:
        return node
    conts = [_mono_content(g, node.vars)[0] for g in node.gens]
    common = [min(c[i] for c in conts) for i in range(len(node.vars))]
    if any(common):
        mono = sp.prod([v ** e for v, e in zip(node.vars, common)])
        node.gens = [sp.expand(sp.cancel(g / mono)) for g in node.gens]
        # f = sum g^2 なので、共通単項式は 2 乗で k に入る
        node.k = [kk + 2 * c for kk, c in zip(node.k, common)]
    # 生成元に単元があればイデアルは全体環。f = x^k * (単元) なので即終了。
    z = {v: 0 for v in node.vars}
    if any(sp.simplify(sp.expand(g).subs(z)) != 0 for g in node.gens):
        node.gens = [sp.Integer(1)]
    # どの生成元にも現れない変数は自由方向 (lambda に影響しない)
    return node


# ----------------------------------------------------------------------
# 本体
# ----------------------------------------------------------------------
def resolve_ideal(generators: Sequence[sp.Expr], gens, *, max_depth: int = 14,
                  max_nodes: int = 4000, verbose: bool = False
                  ) -> IdealResolution:
    """イデアルを運ぶ解消。lambda と位数を返す。

    generators : 原点で消える多項式の列 (f = sum g_i^2 に対応)
    """
    gens = tuple(gens)
    n = len(gens)
    root = INode("C0", [sp.expand(g) for g in generators if sp.expand(g) != 0],
                 gens, [0] * n, [0] * n, sp.Integer(0), 0)
    stack = [root]
    leaves: List[Tuple[str, sp.Rational, int, str]] = []
    n_nodes = 0
    n_regular = 0
    seen = set()

    while stack:
        node = stack.pop()
        n_nodes += 1
        if n_nodes > max_nodes:
            raise ResolutionFailure(f"節点数が上限 {max_nodes} を超えました",
                                    reason="max_charts")
        node = _normalize(node)
        if not node.gens:
            # イデアルが 0 = 残りの変数方向には完全に平坦。
            # 消去した正則方向の寄与だけが残る (lambda_reg)。
            # oo にすると (x-y)^2 のように 1 本消せば終わる例を落とす。
            lam = node.lam_reg if node.lam_reg > 0 else sp.oo
            leaves.append((node.name, lam, 1, "イデアルが 0 (平坦)"))
            continue

        term = _terminal_value(node)
        if term is not None:
            lam, m = term
            lam = node.lam_reg + lam if lam is not sp.oo else sp.oo
            leaves.append((node.name, lam, m, node.note))
            if verbose:
                print(f"  [終了] {node.name}: {lam} (m={m})")
            continue

        # --- 生成元を座標に取る (毎節点で試すのがこの実装の要点) -----
        el = _coordinate_step(node)
        if el is not None:
            n_regular += 1
            stack.append(el)
            if verbose:
                print(f"  [座標] {node.name}: {el.note}")
            continue

        if node.depth >= max_depth:
            raise ResolutionFailure(
                f"節点 {node.name} が {max_depth} 回で終わりませんでした: "
                f"{[sp.sstr(g) for g in node.gens][:2]}",
                reason="max_depth")

        key = (node.name.count("-"), tuple(sp.sstr(g) for g in node.gens),
               tuple(node.k), tuple(node.h))
        if key in seen:
            raise ResolutionFailure("同じ状態が再出現しました",
                                    reason="no-progress")
        seen.add(key)

        # --- ブローアップ ------------------------------------------
        P = sp.expand(sum(g ** 2 for g in node.gens))
        J = _min_center(sp.Poly(P, *node.vars))
        if len(J) < 2:
            # 1 変数で全単項式を hit できる = その変数が全生成元を割る。
            # 共通単項式の括り出しで消えるはずなので、消えないのは
            # 生成元ごとの内容が残っているとき。個別に括り出して再挑戦する
            # (イデアルは変わるが <x^{a_i} h_i> の形は保たれる)。
            changed = False
            new_gens = []
            for g in node.gens:
                a, rest = _mono_content(g, node.vars)
                if any(a) and rest != g:
                    changed = True
                    new_gens.append(sp.expand(g))
                else:
                    new_gens.append(g)
            if not changed:
                raise ResolutionFailure(
                    f"中心を取れません: {node.name} "
                    f"gens={[sp.sstr(g)[:40] for g in node.gens][:3]}",
                    reason="degenerate-center")
            J = sorted(set(J) | set(_support_indices_list(node)))
            if len(J) < 2:
                raise ResolutionFailure(f"中心を取れません: {node.name}",
                                        reason="degenerate-center")
        for i in J:
            new_gens = [_blowup_gen(g, node.vars, i, J) for g in node.gens]
            new_gens = [g for g in new_gens if g != 0]
            nk = _push(node.k, i, J)
            nh = _push(node.h, i, J)
            nh[i] += len(J) - 1
            stack.append(INode(f"{node.name}-{node.vars[i]}", new_gens,
                               node.vars, nk, nh, node.lam_reg,
                               node.depth + 1, used=frozenset()))

    lam = sp.oo
    m = 0
    for _, l, mm, _n in leaves:
        if l is sp.oo:
            continue
        if l < lam:
            lam, m = l, mm
        elif l == lam:
            m = max(m, mm)
    return IdealResolution(lam, m, leaves, n_nodes, n_regular)


# ----------------------------------------------------------------------
# 実験
# ----------------------------------------------------------------------
def _demo():
    import time

    from resolve_singularity import resolve_singularities

    x, y, z = sp.symbols("x y z", real=True)

    print("=" * 78)
    print("既知の値での回帰 (多項式版と比較)")
    print("=" * 78)
    cases = [("x^2+y^2", [x, y], (x, y), sp.Integer(1)),
             ("x^2*y^2", [x * y], (x, y), sp.Rational(1, 2)),
             ("(x-y)^2", [x - y], (x, y), sp.Rational(1, 2)),
             ("x^2+y^3", [x, y * sp.sqrt(y)] if False else None, None, None)]
    cases = [c for c in cases if c[1] is not None]
    cases += [("<x^2, y^3>", [x**2, y**3], (x, y), None),
              ("<x, y, z>", [x, y, z], (x, y, z), sp.Rational(3, 2)),
              ("<xy, yz, zx>", [x * y, y * z, z * x], (x, y, z), None),
              ("<x^2+y^3>", [x**2 + y**3], (x, y), sp.Rational(5, 6))]
    for label, G, g, truth in cases:
        f = sp.expand(sum(t**2 for t in G))
        t0 = time.time()
        try:
            r = resolve_ideal(G, g, max_depth=12)
            got = (r.rlct, r.multiplicity)
        except ResolutionFailure as e:
            got = ("FAIL", e.reason)
        try:
            r2 = resolve_singularities(f, g, prune="ties", weighted=True,
                                       max_depth=14)
            ref = (r2.rlct, r2.multiplicity)
        except ResolutionFailure as e:
            ref = ("FAIL", e.reason)
        mark = "" if truth is None else (
            " OK" if got[0] == truth else f" !! 真値 {truth}")
        print(f"  {label:<18} イデアル版 {str(got):<14} 多項式版 {str(ref):<14}"
              f" {time.time()-t0:5.1f}s{mark}")

    print("\n" + "=" * 78)
    print("Vandermonde 型 (Aoyagi の真値と比較)")
    print("=" * 78)
    import sys
    sys.path.insert(0, "bench")
    try:
        from vandermonde import lambda_N1, vandermonde_ideal
    except Exception:
        print("  bench/vandermonde.py が見つかりません")
        return
    for (M, N, H, Q) in [(1, 1, 1, 1), (1, 1, 2, 1), (1, 1, 2, 2),
                         (1, 1, 3, 1), (2, 1, 2, 1), (1, 2, 2, 1)]:
        G, v = vandermonde_ideal(M, N, H, Q)
        truth = lambda_N1(M, H, Q) if N == 1 else (None, None)
        f = sp.expand(sum(t**2 for t in G))
        t0 = time.time()
        try:
            r = resolve_ideal(G, v, max_depth=12)
            got = f"{r.rlct} (m={r.multiplicity}, 正則{r.n_regular})"
            ok = (truth[0] is None) or (r.rlct == truth[0])
        except ResolutionFailure as e:
            got, ok = f"FAIL[{e.reason}]", False
        try:
            ref = str(resolve_singularities(f, v, prune="ties", weighted=True,
                                            max_depth=12).rlct)
        except ResolutionFailure as e:
            ref = f"FAIL[{e.reason}]"
        print(f"  M={M} N={N} H={H} Q={Q}  真値 {str(truth[0]):<5} "
              f"イデアル版 {got:<26} 多項式版 {ref:<8} {time.time()-t0:5.1f}s"
              f" {'OK' if ok else '<- 不一致'}")


if __name__ == "__main__":
    _demo()
