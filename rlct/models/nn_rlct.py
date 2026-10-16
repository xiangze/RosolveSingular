r"""
nn_rlct.py
==========

ニューラルネット (特に ReLU) の局所 RLCT を、次の順序で計算する。

  (i)   theta* を含む活性化セルを 1 つ固定し、その上でモデルを多項式化する
  (ii)  連続対称性 (スケール軌道) をゲージ固定で商にし、離散対称性 (置換) と
        自由方向 (死んだユニット等) を検出して落とす
  (iii) 残った低次元の fiber ideal を resolve_singularity.py に渡す

------------------------------------------------------------------
理論的な根拠
------------------------------------------------------------------
* fiber ideal
    I = < f(x_i; theta) - f(x_i; theta*) >_i
  に対し K = sum_i g_i^2 とすると、lambda = RLCT_0(K)。RLCT はイデアルの
  不変量なので、生成元を単元倍したり、イデアルとして等価な変形をしてよい。

* セルの固定 (i)
  ReLU ネットは theta について区分多項式。データ点 x_i とユニット j の
  符号 sign(a_j . x_i + b_j) を theta* で固定すると、そのセルの上では
  出力は theta の多項式になる。theta* がセルの内部 (すべての前活性化が
  非零) なら、theta* の近傍はそのセルに含まれるので、通常の (領域制限の
  ない) 局所 RLCT の計算になり厳密。
  theta* がセルの境界にある場合は、近傍が複数のセルに分かれるので
      lambda_true = min over 接するセル (錐に制限した RLCT)
  であり、錐への制限は積分領域を狭めるので lambda を下げない。すなわち
  本コードが返す「制限なしの値」は lambda_true の下界になる。

* 連続対称性の商 (ii)
  群 G が theta* の近傍に自由に作用し K が G 不変なら、局所座標を
  (軌道方向 t, スライス s) に分けて K = K(s) と書けるので
      zeta(z) = (t 方向の有限体積) x int K(s)^z ds
  となり、lambda と位数 m は不変で、次元だけ dim G 下がる。
  ReLU の正斉次性 (a_j, b_j, c_j) -> (t a_j, t b_j, c_j / t) はユニット
  ごとに 1 次元の対称性を与える。theta* での接ベクトルは
  (a_j*, b_j*, -c_j*) なので、これが非零のユニットについて、その成分が
  非零の座標を 1 つ theta* の値に凍結すればスライスが取れる。
  (a_j*, b_j*, c_j*) = 0 のユニットは作用が自由でないので商にできない。

  置換対称性は有限群なので次元は落ちず lambda も変わらない。ただし
  theta* が置換で固定される (同一のユニットが複数ある) ことは特異性が
  深いことの目印なので検出して報告する。

* 正則方向の消去 (ii')
  イデアル I の生成元 g のうち原点で dg != 0 のものがあれば、g 自身を
  座標に取れて I = <g, ...> は <v, h_1(w), ..., h_r(w)> の形になり
      K = v^2 + sum h_i(w)^2
  と分離する。よって
      lambda = 1/2 + lambda(<h_i|_{g=0}>),   m = m(<h_i|_{g=0}>)
  これを繰り返すと「本質的に特異な部分 (コアイデアル)」だけが残る。
  ニューラルネットでは正則方向が大量にあるので、この段階で変数が劇的に
  減る。

* 自由方向
  どの生成元にも現れない変数は K に現れないので、zeta には体積の因子と
  してしか効かず lambda を変えない。死んだ ReLU ユニットのパラメータが
  典型例。落として次元を下げる。
  なお生成元が全て 0 になった場合 (セル全体が真のパラメータ集合) は
  K == 0 で、自由エネルギーに log n 項が出ない。lambda = 0 と報告する。

------------------------------------------------------------------
使い方
------------------------------------------------------------------
    from nn_rlct import relu_local_rlct
    X = [[-1.0], [0.3], [1.0], [2.0]]          # 入力データ
    A = [[1.0], [0.0]]; b = [0.5, 2.0]         # theta*: 隠れ層
    c = [1.0, 0.0]; d = 0.0                    # theta*: 出力層
    rep = relu_local_rlct(X, A, b, c, d)
    rep.print_report()

一般のモデルには local_rlct_from_ideal(生成元, 変数, 対称ベクトル) を使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

from rlct.core.resolve_singularity import Resolution, ResolutionFailure, resolve_singularities
from rlct.core.newton import newton_rlct

try:
    from rlct.core.timing import timed as _timed
except Exception:      # pragma: no cover
    from contextlib import contextmanager as _cm

    @_cm
    def _timed(_name):
        yield


__all__ = [
    "LocalRLCT",
    "local_rlct_from_ideal",
    "relu_network_symbols",
    "relu_local_rlct",
]


# ----------------------------------------------------------------------
# 結果
# ----------------------------------------------------------------------
@dataclass
class LocalRLCT:
    rlct: sp.Rational
    multiplicity: int
    n_vars_initial: int
    free_vars: List[sp.Symbol] = field(default_factory=list)       # どの生成元にも現れない
    gauge_fixed: List[sp.Symbol] = field(default_factory=list)     # 連続対称性で凍結
    regular_vars: List[sp.Symbol] = field(default_factory=list)    # 正則方向として消去
    core_vars: List[sp.Symbol] = field(default_factory=list)
    core_gens: List[sp.Expr] = field(default_factory=list)
    lam_regular: sp.Rational = sp.Integer(0)
    lam_core: sp.Rational = sp.Integer(0)
    m_core: int = 1
    resolution: Optional[Resolution] = None
    core_route: str = ""      # 'resolve' | 'newton' | 'empty'
    notes: List[str] = field(default_factory=list)

    def report(self) -> str:
        L = []
        L.append("############ 局所 RLCT ############")
        L.append(f"  lambda = {self.rlct}"
                 + (f"  (~ {float(self.rlct):.6f})" if self.rlct.is_Number else "")
                 + f",  位数 m = {self.multiplicity}")
        L.append(f"  内訳: 正則方向 {len(self.regular_vars)} 本 -> {self.lam_regular}"
                 f"  +  コア -> {self.lam_core} (m={self.m_core})")
        L.append(f"  変数: 初期 {self.n_vars_initial}"
                 f" -> 自由 {len(self.free_vars)} 除去"
                 f" -> ゲージ固定 {len(self.gauge_fixed)}"
                 f" -> 正則消去 {len(self.regular_vars)}"
                 f" -> コア {len(self.core_vars)}")
        if self.free_vars:
            L.append(f"    自由方向 (K に現れない): {[str(v) for v in self.free_vars]}")
        if self.gauge_fixed:
            L.append(f"    ゲージ固定 (連続対称性): {[str(v) for v in self.gauge_fixed]}")
        if self.regular_vars:
            L.append(f"    正則方向: {[str(v) for v in self.regular_vars]}")
        L.append(f"    コア変数: {[str(v) for v in self.core_vars]}")
        L.append(f"    コア生成元: {[sp.sstr(g) for g in self.core_gens]}")
        for n in self.notes:
            L.append(f"  [note] {n}")
        return "\n".join(L)

    def print_report(self, with_history: bool = False) -> None:
        print(self.report())
        if with_history and self.resolution is not None:
            print(self.resolution.report(only_minimal=True))

    # --- Graphviz (コアイデアルの解消過程) ---------------------------
    def to_dot(self, **kw) -> str:
        """コアイデアルの解消の場合分けを DOT 形式で返す。"""
        if self.resolution is None:
            raise ValueError("コアイデアルが空なので解消の木はありません "
                             "(lambda は正則方向だけで決まっています)。")
        return self.resolution.to_dot(**kw)

    def render_tree(self, path: str = "core_resolution", fmt: str = "png", **kw):
        """コアイデアルの解消過程を Graphviz で描画する。"""
        if self.resolution is None:
            raise ValueError("コアイデアルが空なので解消の木はありません。")
        return self.resolution.render_tree(path=path, fmt=fmt, **kw)


# ----------------------------------------------------------------------
# (ii) 対称性の商 / 自由方向 / 正則方向
# ----------------------------------------------------------------------
def _at_origin(expr, variables) -> sp.Expr:
    return sp.simplify(expr.subs({v: 0 for v in variables}))


def _linear_reduce(gens: Sequence[sp.Expr], vars_=None) -> List[sp.Expr]:
    """生成元を定数係数の線形結合で簡約する (イデアルは変わらない)。

    実装は sympy の Matrix.rref ではなく、単項式を列とみなした疎な
    掃き出しを Fraction で行う。rref は Expr 上の演算が重く、tanh の
    展開のように項が増えると支配的なコストになるため。
    """
    from fractions import Fraction

    exprs = [g for g in gens if g != 0]
    if not exprs:
        return []
    if vars_ is None:
        vs = set()
        for g in exprs:
            vs |= g.free_symbols
        vars_ = tuple(sorted(vs, key=str))
    vars_ = tuple(vars_)

    rows = []
    for g in exprs:
        try:
            p = sp.Poly(sp.expand(g), *vars_)
        except sp.PolynomialError:
            return _linear_reduce_slow(exprs)
        if p.is_zero:
            continue
        d = {}
        try:
            for mon, c in zip(p.monoms(), p.coeffs()):
                r = sp.Rational(c)
                d[tuple(mon)] = Fraction(int(r.p), int(r.q))
        except (TypeError, ValueError):
            # 係数が有理数でない (tanh(5/4) のような記号定数を含む) 場合は
            # 従来の実装に落とす。有理数化しておくほうが桁で速い。
            return _linear_reduce_slow(exprs)
        rows.append(d)
    if not rows:
        return []

    cols = sorted({m for d in rows for m in d})
    order = {m: i for i, m in enumerate(cols)}

    basis: Dict[int, dict] = {}
    for r in rows:
        r = {m: v for m, v in r.items() if v}
        while r:
            pi = min(order[m] for m in r)
            pm = cols[pi]
            if pi in basis:
                b = basis[pi]
                f = r[pm] / b[pm]
                for m, v in b.items():
                    nv = r.get(m, Fraction(0)) - f * v
                    if nv:
                        r[m] = nv
                    else:
                        r.pop(m, None)
            else:
                basis[pi] = r
                break

    out = []
    for pi in sorted(basis):
        r = basis[pi]
        d = {m: sp.Rational(v.numerator, v.denominator)
             for m, v in r.items() if v}
        if not d:
            continue
        # 実測では Poly.from_dict().as_expr() より手で積み上げるほうが
        # 速かった (expr_from_dict がプロファイル上で最も重い)。
        e = sp.Integer(0)
        for m, v in d.items():
            e += v * sp.prod([x ** k for x, k in zip(vars_, m) if k])
        out.append(e)
    return out


def _linear_reduce_slow(gens):
    """多項式にできない場合のための従来実装 (sympy の rref)。"""
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


def _used_vars(gens, variables) -> List[sp.Symbol]:
    used = set()
    for g in gens:
        used |= g.free_symbols
    return [v for v in variables if v in used]


def _gauge_fix(symmetry_vectors, variables):
    """対称性の接ベクトルに対し、凍結する変数 (ピボット) を選ぶ。

    行列 S (対称性の本数 x 変数の数) を掃き出し、各対称性ごとに独立な
    ピボット列を 1 本ずつ取る。ピボット変数を theta* の値に凍結すると、
    軌道に横断的なスライスが得られる。
    """
    rows = []
    for vec in symmetry_vectors:
        rows.append([sp.nsimplify(vec.get(v, 0)) for v in variables])
    if not rows:
        return [], []
    M = sp.Matrix(rows)
    rref, pivots = M.rref()
    used_rows = min(M.rank(), len(rows))
    pivot_vars = [variables[j] for j in pivots[:used_rows]]
    return pivot_vars, list(pivots[:used_rows])


def _regular_candidates(gens, variables):
    """原点で正則な (生成元, 変数) の候補を、安い順に並べて返す。

    g = A*v + B (A, B は v を含まない) と書け、**A が原点で非零**なら、
    局所環で A は単元なので v を消去できる。候補ごとの消去コストは
    A, B の大きさでほぼ決まるので、

      (1) A が定数 (このとき sol = -B/A は多項式で、A の冪が要らない)
      (2) A, B の項数が小さい

    の順で並べる。順序を選ぶだけで中間式の膨張がまったく違う。

    Returns [(score, gi, v, A, B), ...] (score の昇順)
    """
    out = []
    zero = {w: 0 for w in variables}
    idx = {v: i for i, v in enumerate(variables)}
    for gi, g in enumerate(gens):
        try:
            P = sp.Poly(g, *variables)
        except sp.PolynomialError:
            continue
        monoms, coeffs = P.monoms(), P.coeffs()
        for v in variables:
            j = idx[v]
            if P.coeff_monomial(v) == 0:
                continue                 # 原点で v について 1 次の項が無い
            if max(m[j] for m in monoms) != 1:
                continue                 # v について 1 次でない
            # A = d g / d v (v を含まない), B = g|_{v=0}
            At, Bt = [], []
            for m, c in zip(monoms, coeffs):
                term = c * sp.prod([w ** e for w, e in zip(variables, m)
                                    if w is not v])
                (At if m[j] == 1 else Bt).append(term)
            A = sp.expand(sp.Add(*At)) if At else sp.S.Zero
            if A == 0 or sp.expand(A.subs(zero)) == 0:
                continue                 # A(0) = 0 -> 原点で正則でない
            B = sp.expand(sp.Add(*Bt)) if Bt else sp.S.Zero
            na = 1 if A.is_number else len(sp.Add.make_args(A))
            nb = 1 if B.is_number else len(sp.Add.make_args(B))
            out.append(((0 if A.is_number else 1, na * nb, na + nb), gi, v, A, B))
    out.sort(key=lambda t: t[0])
    return out


def _pseudo_substitute(h, v, A, B, variables):
    """v = -B/A を h に代入し、分母を払った多項式 (擬剰余) を返す。

    h = sum_e c_e(others) v^e に対し

        A^d * h(-B/A) = sum_e c_e (-B)^e A^{d-e}         (d = deg_v h)

    を Horner 法で組む。**有理式を経由しない**ので sp.cancel /
    sp.together が要らない。A は原点で単元なので A^d を掛けても局所環では
    イデアルは変わらず、RLCT も変わらない。

    仕上げに整数の content を割る (単元倍なのでイデアルは不変)。
    代入を有理式で行うと係数が 28 桁まで膨らみ、cancel が支配的コストに
    なっていた (実測 19.7 秒のうち cancel が 10.7 秒)。
    """
    others = [w for w in variables if w is not v]
    try:
        hp = sp.Poly(h, v)
    except sp.PolynomialError:
        return None
    d = hp.degree()
    if d <= 0:
        return sp.expand(h)
    # 係数 c_d, ..., c_0 (v を含まない)
    cs = [sp.S.Zero] * (d + 1)
    for (e,), c in zip(hp.monoms(), hp.coeffs()):
        cs[e] = c
    if others:
        R = lambda e: sp.Poly(sp.expand(e), *others)
        Ap, Bp = R(A), R(-B)
        acc = R(cs[d])
        pw = sp.Poly(1, *others)
        for e in range(d - 1, -1, -1):
            pw = pw * Ap
            acc = acc * Bp + R(cs[e]) * pw
        num = acc.as_expr()
    else:
        acc = cs[d]
        pw = sp.S.One
        for e in range(d - 1, -1, -1):
            pw = pw * A
            acc = sp.expand(acc * (-B) + cs[e] * pw)
        num = sp.expand(acc)
    num = sp.expand(num)
    if num == 0:
        return sp.S.Zero
    # 整数 content を割る (非零の有理数倍なのでイデアルは変わらない)
    try:
        c, pp = sp.Poly(num, *others).primitive() if others else (1, None)
        if pp is not None and c != 0:
            num = pp.as_expr()
    except Exception:
        pass
    return num


def _eliminate_regular(gens, variables, verbose=False):
    """原点で微分が非零な生成元を使って正則方向を消去する。

    g = A*v + B で A(0) != 0 なら、局所環で A は単元なので v を消去でき、
    lambda に 1/2 を加える (nn_rlct.py 冒頭の「正則方向の消去」を参照)。

    代入は有理式ではなく**擬剰余 (多項式)** で行う (_pseudo_substitute)。
    また候補は「A が定数」「A, B が小さい」順に選ぶ。どちらも値を変えず、
    中間式の膨張だけを抑える。
    """
    gens = [sp.expand(g) for g in gens if g != 0]
    variables = list(variables)
    eliminated: List[sp.Symbol] = []

    while gens and variables:
        used = None
        for _score, gi, v, A, B in _regular_candidates(gens, variables):
            new_gens = []
            ok = True
            for hj, h in enumerate(gens):
                if hj == gi:
                    continue
                num = _pseudo_substitute(h, v, A, B, variables)
                if num is None:
                    ok = False
                    break
                if num != 0:
                    new_gens.append(num)
            if not ok:
                continue
            used = (gi, v, new_gens)
            break
        if used is None:
            break
        gi, v, new_gens = used
        variables = [w for w in variables if w is not v]
        # ループ内で毎回簡約する。外に出すと生成元が増えたまま次の代入が
        # 走り、かえって遅くなる (実測で 15 -> 19 秒)。
        gens = _linear_reduce(new_gens, variables)
        eliminated.append(v)
        if verbose:
            print(f"    [regular] {v} を消去 (残り {len(gens)} 生成元, "
                  f"{len(variables)} 変数)")
    return gens, variables, eliminated


# ----------------------------------------------------------------------
# 一般のモデル用のエントリポイント
# ----------------------------------------------------------------------
def local_rlct_from_ideal(
    generators: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    *,
    symmetry_vectors: Sequence[Dict[sp.Symbol, sp.Expr]] = (),
    verbose: bool = False,
    dln_oracle: bool = True,
    dln_hint: Optional[Tuple[Sequence[int], int]] = None,
    **resolve_kwargs,
) -> LocalRLCT:
    """fiber ideal <generators> の原点における局所 RLCT。

    generators : 原点で 0 になる多項式の列 (残差)
    variables  : theta - theta* に対応する変数
    symmetry_vectors : theta* における連続対称性の接ベクトル
        (変数 -> 成分 の dict)。自由な作用のものだけを渡すこと。
    dln_oracle : 深層線形網の閉形式オラクルを試すか (既定 True)。
        イデアルを DLN のものと**確認できた場合だけ**発火する
        (detect -> reconstruct -> compare; dln_oracle.py 参照)。
    dln_hint : (H, r) を宣言してオラクルを使う。宣言は検証されるので
        誤った宣言で誤った lambda が出ることはない。r > 0 や L >= 4 は
        この経路でのみ扱える。
    resolve_kwargs : resolve_singularities に渡す (prune, max_depth, ...)
    """
    variables = list(variables)
    n0 = len(variables)
    notes: List[str] = []

    # --- 深層線形網の閉形式オラクル -----------------------------------
    # 構造を壊す簡約の前、生成元がそのままの段階で試す。
    if dln_oracle or dln_hint is not None:
        try:
            from rlct.closers.dln_oracle import dln_route as _dln_route
            _hit = _dln_route([sp.expand(g) for g in generators], variables,
                              hint=dln_hint)
        except Exception:
            _hit = None
        if _hit is not None:
            _lam, _th, _m = _hit
            notes.append(
                f"深層線形網として同定 (H={tuple(_m.H)}, r={_m.r}, "
                f"同定経路={_m.how}): Aoyagi の閉形式 "
                "(Neural Networks 172:106132, Thm 1) を適用。"
                "イデアルを組み直して入力と厳密に一致することを確認済み。")
            if verbose:
                print(f"  [dln] H={tuple(_m.H)} r={_m.r} -> "
                      f"lambda={_lam}, m={_th}")
            return LocalRLCT(rlct=sp.nsimplify(_lam), multiplicity=int(_th),
                             n_vars_initial=n0, core_vars=list(variables),
                             core_gens=[sp.expand(g) for g in generators],
                             lam_core=sp.nsimplify(_lam), m_core=int(_th),
                             core_route="dln", notes=notes)

    gens = []
    for g in generators:
        g = sp.expand(g)
        if g == 0:
            continue
        if _at_origin(g, variables) != 0:
            raise ValueError(f"生成元 {g} が原点で 0 になりません "
                             "(theta* が真のパラメータではない)。")
        gens.append(g)

    if not gens:
        notes.append("すべての残差が恒等的に 0: セル全体が真のパラメータ集合。"
                     "K == 0 なので自由エネルギーに log n 項は出ない (lambda = 0)。")
        return LocalRLCT(rlct=sp.Integer(0), multiplicity=0, n_vars_initial=n0,
                         free_vars=variables, notes=notes)

    gens = _linear_reduce(gens, variables)

    # --- 自由方向 (K に現れない変数) を落とす ------------------------
    used = _used_vars(gens, variables)
    free_vars = [v for v in variables if v not in used]
    variables = used

    # --- 連続対称性のゲージ固定 --------------------------------------
    sym = []
    for vec in symmetry_vectors:
        vv = {k: val for k, val in vec.items() if k in variables and val != 0}
        if vv:
            sym.append(vv)
    gauge_vars, _ = _gauge_fix(sym, variables)
    if gauge_vars:
        sub0 = {v: 0 for v in gauge_vars}
        gens = _linear_reduce([sp.expand(g.subs(sub0)) for g in gens], variables)
        variables = [v for v in variables if v not in gauge_vars]
        # 固定後に使われなくなった変数も落とす
        used2 = _used_vars(gens, variables)
        free_vars += [v for v in variables if v not in used2]
        variables = used2

    if not gens:
        notes.append("ゲージ固定後に残差が消えました: スライス上で K == 0。")
        return LocalRLCT(rlct=sp.Integer(0), multiplicity=0, n_vars_initial=n0,
                         free_vars=free_vars, gauge_fixed=gauge_vars, notes=notes)

    # --- 正則方向の消去 ----------------------------------------------
    with _timed("eliminate_regular"):
        core_gens, core_vars, regular = _eliminate_regular(
            gens, variables, verbose=verbose)
        core_gens = _linear_reduce(core_gens, core_vars)
    lam_reg = sp.Rational(len(regular), 2)

    # 消去で使われなくなった変数を整理
    used3 = _used_vars(core_gens, core_vars) if core_gens else []
    free_vars += [v for v in core_vars if v not in used3]
    core_vars = used3

    # --- コアイデアルの解消 ------------------------------------------
    resolution = None
    core_route = "empty"
    if not core_gens:
        lam_core, m_core = sp.Integer(0), 1
        notes.append("コアイデアルは空 (正則な特異点): lambda は正則方向のみで決まる。")
    else:
        K = sp.expand(sum(g ** 2 for g in core_gens))
        # まず K 全体がニュートン非退化かを見る。非退化なら Varchenko の
        # 定理で lambda が厳密に決まるので、ブローアップは要らない
        # (LP + 有限個の Z3 判定だけなので、変数が多いほど効く)。
        lam_core = m_core = None
        try:
            from rlct.core.newton import newton_multiplicity, rlct_via_newton
            with _timed("newton"):
                lam_n, st_n = rlct_via_newton(K, core_vars, timeout_ms=8000)
            if st_n == "proved" and lam_n is not sp.oo:
                m_n = newton_multiplicity(K, core_vars)
                lam_core, m_core = lam_n, (int(m_n) if m_n else 1)
                core_route = "newton"
                notes.append("コアはニュートン非退化: Varchenko の定理で厳密。")
        except Exception:
            lam_core = m_core = None
        # つぎに Aoyagi Lemma 1(1) の挟み込み。部分生成元の lambda は
        # 全体の lambda の厳密な下界で、newton_rlct は常に上界なので、
        # 一致すればブローアップなしで確定する。一致しなくても下界は
        # 分枝限定の早期終了に使える (位数 m は下限になる)。
        lb = None
        if lam_core is None and len(core_gens) >= 2:
            try:
                from rlct.closers.lemmas import squeeze_rlct
                sq = squeeze_rlct(K, core_vars, core_gens)
                if sq.status == "proved":
                    lam_core, m_core = sq.rlct, 1
                    core_route = "squeeze"
                    notes.append(
                        f"挟み込みで確定: lambda ∈ [{sq.lo}, {sq.hi}] が一致 "
                        f"({sq.witness})。位数 m は下限。")
                elif sq.status != "inconsistent":
                    lb = sq.lo
            except Exception:
                lb = None
        # --- lc_filter: コアイデアルの健全な上下界 ---------------------
        # 上界 U1 (ヤコビアンの階数 + 商イデアルの (S) 判定) と下界 L2
        # (b 関数から lct_C、lambda_R >= lct_C) の挟み込み。一致すれば
        # 解消なしで確定する。一致しなくても **上界は健全** なので、
        # 解消の値がそれを超えたら解消側の誤りとして検出できる。
        lcb = None
        try:
            from rlct.closers.lc_filter import lc_bounds
            lcb = lc_bounds(core_gens, core_vars)
        except ImportError:
            lcb = None                 # lc_filter.py が無い環境では飛ばす
        except Exception as e:         # noqa: BLE001
            notes.append(f"lc_filter 失敗 ({str(e)[:60]})")
        if lam_core is None and lcb is not None and lcb.exact is not None:
            lam_core, m_core = lcb.exact, 1
            core_route = "lc_filter"
            notes.append(f"コアは lc_filter で確定: lambda_core = {lcb.exact} "
                         f"({'; '.join(lcb.sources.values())})。"
                         "位数 m は未確定 (下限 1)。")
        if lam_core is None:
            # 上界は解消に渡さない: prune='ties' と組み合わせると同値の
            # 部分木が早く刈られて位数 m が下限に落ちる ((2,2,1,0) で
            # m=2 -> 1)。事後チェックだけに使う。
            kwargs = dict(prune="ties", lower_bound=lb)
            kwargs.update(resolve_kwargs)
            try:
                with _timed("resolve"):
                    resolution = resolve_singularities(K, core_vars, **kwargs)
                lam_core, m_core = resolution.rlct, resolution.multiplicity
                core_route = "resolve"
            except ResolutionFailure:
                # 解消が失敗したときだけ挟み込みに大きな予算を与える
                # (pay-when-needed)。だめなら例外はそのまま伝える。
                ok = False
                if len(core_gens) >= 2:
                    try:
                        from rlct.closers.lemmas import squeeze_rlct
                        sq2 = squeeze_rlct(K, core_vars, core_gens,
                                           max_subset=2, max_calls=24,
                                           time_budget=8.0)
                        if sq2.status == "proved":
                            lam_core, m_core = sq2.rlct, 1
                            core_route = "squeeze(解消失敗後)"
                            notes.append(
                                f"解消は失敗したが挟み込みで確定: "
                                f"lambda ∈ [{sq2.lo}, {sq2.hi}] ({sq2.witness})。"
                                "位数 m は下限。")
                            ok = True
                    except Exception:
                        ok = False
                if not ok:
                    raise
        if lam_core is sp.oo:
            lam_core, m_core = sp.Integer(0), 1
        # --- 解消の値が lc_filter の健全な上界を超えていないか ---------
        # 超えていたら解消側の誤り (certify では refuted になるはず)。
        # 値は健全な上界で置き換え、厳密値ではないことを note に残す。
        if (lcb is not None and lcb.hi is not None
                and core_route not in ("lc_filter",) and lam_core > lcb.hi):
            notes.append(
                f"** 解消の値 {lam_core} が lc_filter の健全な上界 {lcb.hi} を"
                " 超えています。解消側の誤り (certify では refuted になるはず)。"
                " lambda_core は上界で置き換えました (値は上界、厳密値ではない)。"
                " 位数 m は未確定なので 1 (下限) にしています。 **")
            lam_core, m_core = lcb.hi, 1
            core_route += "+lc_filter上界で訂正"

    lam = lam_reg + lam_core
    m = m_core if core_gens else 1

    return LocalRLCT(
        rlct=sp.nsimplify(lam),
        multiplicity=m,
        n_vars_initial=n0,
        free_vars=free_vars,
        gauge_fixed=gauge_vars,
        regular_vars=regular,
        core_vars=core_vars,
        core_route=core_route,
        core_gens=core_gens,
        lam_regular=lam_reg,
        lam_core=lam_core,
        m_core=m_core,
        resolution=resolution,
        notes=notes,
    )


# ----------------------------------------------------------------------
# (i) ReLU ネットのセル固定 -> 多項式化
# ----------------------------------------------------------------------
def relu_network_symbols(n_in: int, n_hidden: int, prefix: str = ""):
    """1 隠れ層 ReLU ネット f(x) = sum_j c_j relu(a_j . x + b_j) + d の記号。"""
    A = sp.Matrix(n_hidden, n_in,
                  lambda j, i: sp.Symbol(f"{prefix}a{j}_{i}", real=True))
    b = [sp.Symbol(f"{prefix}b{j}", real=True) for j in range(n_hidden)]
    c = [sp.Symbol(f"{prefix}c{j}", real=True) for j in range(n_hidden)]
    d = sp.Symbol(f"{prefix}d", real=True)
    return A, b, c, d


@dataclass
class ReLUReport:
    local: LocalRLCT
    pattern: List[List[int]]          # sign(a_j . x_i + b_j) at theta*  (行=データ, 列=ユニット)
    interior: bool
    active_constraints: List[Tuple[int, int]]
    dead_units: List[int]
    duplicate_groups: List[List[int]]
    scaling_units: List[int]
    cells: List[Tuple[Tuple[int, ...], "LocalRLCT"]] = field(default_factory=list)

    def print_report(self, with_history: bool = False) -> None:
        print("############ ReLU ネットの局所 RLCT ############")
        print("  活性化パターン (行=データ点, 列=ユニット, +/-/0):")
        for i, row in enumerate(self.pattern):
            print("    x[%d]: %s" % (i, " ".join("+0-"[1 - s] for s in row)))
        if self.interior:
            print("  theta* はセルの内部 -> 近傍はこのセルに含まれるので厳密")
        else:
            print(f"  theta* はセルの境界 (前活性化が 0: {self.active_constraints})")
            print("    -> 真の lambda は接する各錐に制限した RLCT の最小値。"
                  "領域を狭めると lambda は下がらないので、以下の値は下界。")
            if self.cells:
                from collections import Counter
                cnt = Counter((str(loc.rlct), loc.multiplicity) for _, loc in self.cells)
                print(f"    接するセル {len(self.cells)} 個を列挙 "
                      "((lambda, m) -> 個数):")
                for (lam, m), k in sorted(cnt.items(), key=lambda t: float(sp.Rational(t[0][0]))):
                    print(f"      lambda={lam}, m={m}  ({k} セル)")
                best = min(self.cells, key=lambda t: float(t[1].rlct))
                print("      最小を与えるセル: %s"
                      % "".join("+" if a > 0 else "-" for a in best[0]))
        if self.dead_units:
            print(f"  死んだユニット (全データで不活性): {self.dead_units}"
                  " -> そのパラメータは K に現れない")
        if self.duplicate_groups:
            print(f"  同一のユニット (置換対称性が theta* を固定): {self.duplicate_groups}"
                  " -> 次元は落ちないが特異性が深い目印")
        print(f"  スケール対称性を商にしたユニット: {self.scaling_units}")
        print()
        self.local.print_report(with_history=with_history)


def relu_local_rlct(
    X,
    A_star,
    b_star,
    c_star,
    d_star=0,
    *,
    targets=None,
    boundary: str = "both",
    max_cells: int = 64,
    verbose: bool = False,
    **resolve_kwargs,
) -> ReLUReport:
    """1 隠れ層 ReLU ネットの theta* における局所 RLCT。

    X       : 入力データ (n_data x n_in のリスト)
    A_star, b_star, c_star, d_star : theta* の値
    targets : 真の出力値。None なら realizable (真の関数 = theta* のネット) とする。
    boundary: 前活性化がちょうど 0 の制約の扱い。
              'both' なら接するセルをすべて列挙して lambda の最小値を返す
              (真の lambda はさらに錐への制限で上がりうるので、これは下界)。
              'active' / 'inactive' なら 0 を +/- とみなして 1 セルだけ見る。
    max_cells: 'both' のときに列挙するセル数の上限。
    """
    X = [list(map(sp.nsimplify, xi)) for xi in X]
    n_in = len(X[0])
    n_hidden = len(b_star)
    A, b, c, d = relu_network_symbols(n_in, n_hidden)

    A_star = sp.Matrix([[sp.nsimplify(v) for v in row] for row in A_star])
    b_star = [sp.nsimplify(v) for v in b_star]
    c_star = [sp.nsimplify(v) for v in c_star]
    d_star = sp.nsimplify(d_star)

    # --- (i) 活性化パターン ------------------------------------------
    pattern, active = [], []
    for i, xi in enumerate(X):
        row = []
        for j in range(n_hidden):
            pre = sum(A_star[j, k] * xi[k] for k in range(n_in)) + b_star[j]
            pre = sp.simplify(pre)
            s = 0 if pre == 0 else (1 if pre > 0 else -1)
            if s == 0:
                active.append((i, j))
            row.append(s)
        pattern.append(row)
    interior = not active

    dead_units = [j for j in range(n_hidden)
                  if all(pattern[i][j] <= 0 for i in range(len(X)))]

    # 同一ユニット (置換対称性の固定点)
    dup: List[List[int]] = []
    seen = set()
    for j in range(n_hidden):
        if j in seen:
            continue
        grp = [j]
        for k in range(j + 1, n_hidden):
            same = (list(A_star[j, :]) == list(A_star[k, :])
                    and b_star[j] == b_star[k] and c_star[j] == c_star[k])
            if same:
                grp.append(k)
                seen.add(k)
        if len(grp) > 1:
            dup.append(grp)

    # --- セル上での出力 (多項式) と残差 -------------------------------
    def out(xi, row, sym=True):
        val = d if sym else d_star
        for j in range(n_hidden):
            if row[j] > 0:          # 活性なユニットだけが寄与する
                pre = (sum((A[j, k] if sym else A_star[j, k]) * xi[k]
                           for k in range(n_in))
                       + (b[j] if sym else b_star[j]))
                val += (c[j] if sym else c_star[j]) * pre
        return sp.expand(val)

    def build_gens(pat):
        gg = []
        for i, xi in enumerate(X):
            y = out(xi, pat[i], sym=False) if targets is None else sp.nsimplify(targets[i])
            gg.append(sp.expand(out(xi, pat[i], sym=True) - y))
        return gg

    def resolved_pattern(assign):
        """境界制約 (前活性化 0) に符号 assign を割り当てたパターン。"""
        pat = [row[:] for row in pattern]
        for (i, j), sgn in zip(active, assign):
            pat[i][j] = sgn
        return pat

    if interior:
        assignments = [()]
    elif boundary == "active":
        assignments = [tuple(1 for _ in active)]
    elif boundary == "inactive":
        assignments = [tuple(-1 for _ in active)]
    else:
        import itertools
        assignments = list(itertools.product((1, -1), repeat=len(active)))[:max_cells]

    gens_theta = build_gens(resolved_pattern(assignments[0]))

    # --- theta = theta* + u へのシフト --------------------------------
    uA = sp.Matrix(n_hidden, n_in,
                   lambda j, i: sp.Symbol(f"ua{j}_{i}", real=True))
    ub = [sp.Symbol(f"ub{j}", real=True) for j in range(n_hidden)]
    uc = [sp.Symbol(f"uc{j}", real=True) for j in range(n_hidden)]
    ud = sp.Symbol("ud", real=True)
    shift = {d: d_star + ud}
    for j in range(n_hidden):
        shift[b[j]] = b_star[j] + ub[j]
        shift[c[j]] = c_star[j] + uc[j]
        for k in range(n_in):
            shift[A[j, k]] = A_star[j, k] + uA[j, k]
    gens = [sp.expand(g.subs(shift, simultaneous=True)) for g in gens_theta]

    u_vars = [uA[j, k] for j in range(n_hidden) for k in range(n_in)] \
        + list(ub) + list(uc) + [ud]

    # --- (ii) スケール対称性の接ベクトル ------------------------------
    # (a_j, b_j, c_j) -> (t a_j, t b_j, c_j / t) の t=1 での微分
    sym_vecs, scaling_units = [], []
    for j in range(n_hidden):
        vec = {}
        for k in range(n_in):
            if A_star[j, k] != 0:
                vec[uA[j, k]] = A_star[j, k]
        if b_star[j] != 0:
            vec[ub[j]] = b_star[j]
        if c_star[j] != 0:
            vec[uc[j]] = -c_star[j]
        if vec:
            sym_vecs.append(vec)
            scaling_units.append(j)

    def evaluate(assign):
        gg = [sp.expand(g.subs(shift, simultaneous=True))
              for g in build_gens(resolved_pattern(assign))]
        return local_rlct_from_ideal(gg, u_vars, symmetry_vectors=sym_vecs,
                                     verbose=verbose, **resolve_kwargs)

    cells: List[Tuple[Tuple[int, ...], LocalRLCT]] = []
    if len(assignments) == 1:
        local = evaluate(assignments[0])
    else:
        for assign in assignments:
            cells.append((assign, evaluate(assign)))
        local = min(cells, key=lambda t: (float(t[1].rlct), -t[1].multiplicity))[1]
    if dead_units:
        local.notes.append(
            f"ユニット {dead_units} はこのセルで恒等的に 0 なので、"
            "そのパラメータは K に現れない (自由方向)。")
    if not interior:
        local.notes.append(
            "theta* はセル境界。錐への制限を行っていないので、この lambda は"
            "真の局所 lambda の下界。")
    if dup:
        local.notes.append(
            f"ユニット {dup} が同一。置換対称性が theta* を固定するので "
            "lambda 自体は変わらないが、特異性が深くなっている可能性が高い。")

    return ReLUReport(local=local, pattern=pattern, interior=interior,
                      active_constraints=active, dead_units=dead_units,
                      duplicate_groups=dup, scaling_units=scaling_units,
                      cells=cells)


# ----------------------------------------------------------------------
# デモ
# ----------------------------------------------------------------------
def _demo():
    X = [[-2], [-1], [sp.Rational(1, 3)], [1], [2], [3]]

    def head(t):
        print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)

    head("例 1: 正則な点 (1 ユニット、全データで活性、c* != 0)\n"
         "  セル上ではモデルが x の 1 次式になるので像は 2 次元 -> lambda = 2/2 = 1")
    relu_local_rlct(X, [[1]], [3], [2], 0).print_report()

    head("例 2: 冗長ユニット c*=0, a*=0, b*>0\n"
         "  コアが <ua*ud> になり lambda = 1/2 + 1/2 = 1, m = 2 (行列式型の退化)")
    relu_local_rlct(X, [[0]], [1], [0], 0).print_report()

    head("例 3: 2 ユニットが別々の区間で活性 (ヒンジがデータ範囲の内側)\n"
         "  セル内部では各方向が正則になり lambda = 4/2 = 2")
    relu_local_rlct(X, [[1], [1]], [0, sp.Rational(-3, 2)], [1, 0], 0).print_report()

    head("例 4: 死んだユニット (全データで不活性)")
    relu_local_rlct(X, [[1], [1]], [3, -10], [2, 5], 0).print_report()

    head("例 5: 同一の 2 ユニット (置換対称性が theta* を固定)")
    relu_local_rlct(X, [[1], [1]], [3, 3], [1, 1], 0).print_report()

    head("例 6: 完全に 0 のユニット -> セル境界。接するセルを全列挙して最小を取る")
    relu_local_rlct(X, [[1], [0]], [0, 0], [1, 0], 0).print_report()

    head("例 7: 2 次元入力、ヒンジがデータ点をちょうど通る (境界)")
    X2 = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]]
    relu_local_rlct(X2, [[1, -1], [1, 1]], [0, 0], [1, 1], 0).print_report()

    head("例 8: ReLU 以外 — fiber ideal を直接渡す (低ランク回帰型)")
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)
    local_rlct_from_ideal([a1 * b1 + a2 * b2], [a1, a2, b1, b2]).print_report()

    head("例 9: 履歴つきで表示 (例 2 のコアの解消過程)")
    relu_local_rlct(X, [[0]], [1], [0], 0).local.print_report(with_history=True)


if __name__ == "__main__":
    _demo()
