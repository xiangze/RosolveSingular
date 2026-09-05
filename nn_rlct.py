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
  に対し K = sum_i g_i^2 とすると、lambda = RLCT_0(K)。RLCT はイデアルの不変量なので、生成元を単元倍したり、イデアルとして等価な変形をしてよい。

* セルの固定 (i)
  ReLU ネットは theta について区分多項式。データ点 x_i とユニット j の 符号 sign(a_j . x_i + b_j) を theta* で固定すると、そのセルの上では
  出力は theta の多項式になる。theta* がセルの内部 (すべての前活性化が非零) なら、theta* の近傍はそのセルに含まれるので、通常の (領域制限のない) 局所 RLCT の計算になり厳密。
  theta* がセルの境界にある場合は、近傍が複数のセルに分かれるので
      lambda_true = min over 接するセル (錐に制限した RLCT)
  であり、錐への制限は積分領域を狭めるので lambda を下げない。すなわち本コードが返す「制限なしの値」は lambda_true の下界になる。

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

from resolve_singularity import (
    Resolution,
    ResolutionFailure,
    newton_rlct,
    resolve_singularities,
)

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


# ----------------------------------------------------------------------
# (ii) 対称性の商 / 自由方向 / 正則方向
# ----------------------------------------------------------------------
def _at_origin(expr, variables) -> sp.Expr:
    return sp.simplify(expr.subs({v: 0 for v in variables}))


def _linear_reduce(gens: Sequence[sp.Expr]) -> List[sp.Expr]:
    """生成元を定数係数の線形結合で簡約する (イデアルは変わらない)。

    データ点ごとの残差は互いに一次従属なことが多い (モデルの像の次元しか
    独立な生成元がない)。単項式を基底とみなして行簡約すると、生成元の
    本数がモデルの「関数空間の次元」まで落ちる。
    """
    polys = [sp.expand(g) for g in gens if g != 0]
    if not polys:
        return []
    monoms = sorted({m for p in polys for m in p.as_coefficients_dict()},
                    key=sp.default_sort_key)
    M = sp.Matrix([[sp.nsimplify(p.as_coefficients_dict().get(m, 0)) for m in monoms]
                   for p in polys])
    R, _ = M.rref()
    out = []
    for i in range(R.rows):
        row = [R[i, j] for j in range(len(monoms))]
        if any(v != 0 for v in row):
            out.append(sp.expand(sum(row[j] * monoms[j] for j in range(len(monoms)))))
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


def _eliminate_regular(gens, variables, verbose=False):
    """原点で微分が非零な生成元を使って正則方向を消去する。

    g が変数 v について 1 次で、その係数が原点で非零なら v = sol を他の
    生成元に代入し、v と g を落として lambda に 1/2 を加える。
    """
    gens = [sp.expand(g) for g in gens if g != 0]
    variables = list(variables)
    eliminated: List[sp.Symbol] = []
    progress = True
    while progress and gens:
        progress = False
        for gi, g in enumerate(gens):
            zeros = {v: 0 for v in variables}
            lin = {v: sp.simplify(sp.diff(g, v).subs(zeros)) for v in variables}
            cand = [v for v in variables if lin[v] != 0]
            if not cand:
                continue
            for v in cand:
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
                for hj, h in enumerate(gens):
                    if hj == gi:
                        continue
                    hh = sp.cancel(sp.together(h.subs(v, sol)))
                    num, den = sp.fraction(hh)
                    if sp.simplify(den.subs(zeros)) == 0:
                        ok = False
                        break
                    num = sp.expand(num)          # den は原点で単元なので落としてよい
                    if num != 0:
                        new_gens.append(num)
                if not ok:
                    continue
                gens = _linear_reduce(new_gens)
                variables = [w for w in variables if w != v]
                eliminated.append(v)
                if verbose:
                    print(f"    [regular] {v} = {sol} を消去")
                progress = True
                break
            if progress:
                break
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
    **resolve_kwargs,
) -> LocalRLCT:
    """fiber ideal <generators> の原点における局所 RLCT。

    generators : 原点で 0 になる多項式の列 (残差)
    variables  : theta - theta* に対応する変数
    symmetry_vectors : theta* における連続対称性の接ベクトル
        (変数 -> 成分 の dict)。自由な作用のものだけを渡すこと。
    resolve_kwargs : resolve_singularities に渡す (prune, max_depth, ...)
    """
    variables = list(variables)
    n0 = len(variables)
    notes: List[str] = []

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

    gens = _linear_reduce(gens)

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
        gens = _linear_reduce([sp.expand(g.subs(sub0)) for g in gens])
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
    core_gens, core_vars, regular = _eliminate_regular(gens, variables, verbose=verbose)
    core_gens = _linear_reduce(core_gens)
    lam_reg = sp.Rational(len(regular), 2)

    # 消去で使われなくなった変数を整理
    used3 = _used_vars(core_gens, core_vars) if core_gens else []
    free_vars += [v for v in core_vars if v not in used3]
    core_vars = used3

    # --- コアイデアルの解消 ------------------------------------------
    resolution = None
    if not core_gens:
        lam_core, m_core = sp.Integer(0), 1
        notes.append("コアイデアルは空 (正則な特異点): lambda は正則方向のみで決まる。")
    else:
        K = sp.expand(sum(g ** 2 for g in core_gens))
        kwargs = dict(prune="ties")
        kwargs.update(resolve_kwargs)
        resolution = resolve_singularities(K, core_vars, **kwargs)
        lam_core, m_core = resolution.rlct, resolution.multiplicity
        if lam_core is sp.oo:
            lam_core, m_core = sp.Integer(0), 1

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
