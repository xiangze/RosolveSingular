r"""
resolve_singularity.py
======================

多項式 f(x_1, ..., x_n) の特異点を「ブローアップ（変数変換）+ 正規化
(renormalization: 単項式因子の括り出し)」の反復で解消し、

  * 式変形の履歴 (chart ごとの座標変換の合成 phi と各ステップの記録)
  * 実対数閾値 RLCT (real log canonical threshold) lambda とその位数 m

を計算する。一定回数 (max_depth) 繰り返しても正規交差にできない chart が
残った場合は ResolutionFailure 例外を送出して終了する。

------------------------------------------------------------------
理論的背景
------------------------------------------------------------------
ゼータ関数
        zeta(z) = \int |f(x)|^z phi(x) dx      (phi は原点近傍の台をもつ)
に対し、特異点解消定理により固有写像 g: U -> W が存在して局所座標で

        f(g(u)) = a(u) * u^{k},      |g'(u)| = b(u) * u^{h}
        (a, b は単元, u^{k} = prod_j u_j^{k_j})

と表せる。このとき zeta の最大の極は

        lambda = min_{j : k_j > 0} (h_j + 1) / k_j
        m      = その最小値を達成する j の個数

で与えられる (chart が複数あるときは lambda は chart 上の最小値、
m はそれを達成する chart 上の最大値)。本プログラムは各 chart について
(k, h) を厳密に追跡することでこれを計算する。

------------------------------------------------------------------
アルゴリズム
------------------------------------------------------------------
各 chart は (f_rest, k, h) の三つ組で表される。恒等的に

        f (現在の座標での引き戻し) = x^{k} * f_rest
        ヤコビアン                 = 単元 * x^{h}

が成り立つように更新していく。

1. 正規化 (renormalize)
   (a) f_rest から単項式の最大公約因子 x^{a} を括り出し、k <- k + a。
   (b) 原点で消えない因子 (単元) を因数分解で取り除く。|f|^z の原点での
       特異性に単元は寄与しないので落としてよい。

2. 停止判定
   f_rest(0) != 0 ならば f_rest は原点で単元。すなわち f は正規交差型
   x^{k} * (単元) になっており、この chart は解消済み。

3. 滑らかな因子の座標化 (smooth_coords)
   f_rest が原点で滑らかな因子 g をもち、g がある変数 x_v について 1 次
   (係数 A は原点で非零)、かつ x_v が例外因子に使われていない
   (k_v = h_v = 0) とき、x_v = (x_v' - B)/A という座標変換で g -> x_v'。
   ヤコビアンは単元 1/A。特異点集合が正の次元をもつ場合
   ((x-y)^2, (a1 b1 + a2 b2)^2 など) にブローアップの無限反復を防ぐ。

4. ブローアップ
   中心 {x_j = 0 : j in J} でブローアップする。J は既定 (center='min') では
   「f_rest の全単項式が J の変数を少なくとも 1 つ含む」ような最小の集合
   (最小ヒッティング集合、分枝限定で厳密に計算)。これは中心が零点集合に
   含まれる最大次元の座標部分空間に対応し、点中心のブローアップより
   ずっと停止しやすい。chart i (i in J) は

        x_i = y_i,   x_j = y_i * y_j   (j in J, j != i)

   で与えられ、ヤコビアン因子として y_i^{|J| - 1} が生じる。単項式の
   置換は指数ベクトルの線形変換なので、k, h も同じ規則で押し出す:

        k_i <- k_i + sum_{j in J, j != i} k_j,
        h_i <- h_i + sum_{j in J, j != i} h_j + (|J| - 1)

   多項式の変換も指数ベクトルの操作だけで済むので、数十変数でも高速。

5. 中心の付け替え (recenter, 省略可)
   例外因子 {x_e = 0} 上の原点以外の点でも f_rest が消え、しかも正規交差
   でないことがある。そのような点

        { f_rest = 0 } かつ { grad_{j != e} f_rest = 0 }  on {x_e = 0}

   を sympy.solve で探し、見つかればそこへ平行移動して探索を続ける。
   平行移動で x_j -> x_j + c_j (c_j != 0) となった方向の単項式因子
   (x_j + c_j)^{k_j} は単元に吸収されるので k_j, h_j は 0 に落とす。
   変数が多いと solve が現実的でないため、既定では変数が少ない場合のみ
   有効 (recenter_max_vars)。

6. 停止しない場合
   * 同じ chart 系列で同一の f_rest が再出現 -> 無限反復と判定して例外
   * ブローアップの反復が max_depth を超過 -> 例外
   * chart 数が max_charts を超過 -> 例外
   いずれも ResolutionFailure(reason=...) を送出する。

------------------------------------------------------------------
分枝限定 (prune=True / 'ties')
------------------------------------------------------------------
lambda は「全 chart の最小値」なので、最小化問題として分枝限定が使える。
解消そのもの (Hironaka 流) では被覆全体が必要なため枝刈りできないが、
RLCT だけが欲しい場合は不要な部分木を捨てられる。

* 上界 UB (primal bound) — ニュートン多面体の LP:

      maximize sum_a nu_a  s.t.  sum_a nu_a * a_j <= h_j + 1,  nu >= 0

  ここで {a} は f = x^k * f_rest の指数ベクトル。単項式なら値は
  min_j (h_j+1)/k_j に退化し、一般には真の RLCT の上界 (非退化なら等号)。
  どの節点で計算しても大域最小値の上界になるので incumbent に使える。
  pulp -> CBC、または scipy(HiGHS) で解く。

* 下界 LB (dual bound) — メディアント不等式:

      (h_i'+1)/k_i' = sum_{j in J}(h_j+1) / (sum_{j in J} k_j + a)
                    >= min_j (h_j+1)/(k_j + a)

  つまりブローアップ単体では比は下がらず、下がるのは正規化で括り出される
  order a の分だけ。残り反復で加算されうる order を budget M で抑えれば
  lambda(部分木) >= min_j (h_j+1)/(k_j+M)。Hironaka の解消不変量が
  「停止性を保証する単調量」であるのに対し、ここでは同じ量を
  「枝刈りのための双対限界」として使っていることになる。

* LB > UB なら展開を打ち切る。厳密な不等号なので lambda も位数 m も保存
  される。prune='ties' にすると到達済みの値と同値の部分木も刈る
  (lambda は正しいまま、m は下限になりうる) 代わりに探索が劇的に縮み、
  x^3+y^4+z^5 のように素の探索では停止しない例も 1 chart で片づく。

* 中心の選択 (最小ヒッティング集合) も 0-1 整数計画として pulp に投げる
  (変数や単項式が多い場合)。内部の分枝限定はフォールバック。

------------------------------------------------------------------
注意 (limitation)
------------------------------------------------------------------
* 本実装は「座標部分空間を中心とするブローアップ + 局所的な座標変換」
  であり、Hironaka / Bierstone-Milman の一般的な解消アルゴリズムでは
  ない。例えば x^3 + y^4 + z^5 のように、例外因子と厳密変換の接触が
  何度ブローアップしても解消しない例では停止せず例外になる (仕様)。
* recenter を切った場合、例外因子上の原点以外に非正規交差点が残ると
  lambda を過大評価する (= 真の値より大きい値を返す) 可能性がある。
* ニュートン多面体に関して非退化な f では newton_rlct() (線形計画、
  多変数でも高速) と一致するので、クロスチェックに使うとよい。退化して
  いる場合 (実数体上ではしばしば起こる) はブローアップ側が正しい。
* 枝刈りの下界は「各ステップで括り出される order <= 現在の重複度、かつ
  重複度はブローアップで増えない」という標準的な仮定に依る。完全に網羅
  したい場合は prune=False とする (その分だけ chart 数は増える)。
* 変数が数十個ある場合、chart 数は分岐の積で増えるため max_charts で
  打ち切る。単元因子の除去と最小中心の選択がこの爆発をかなり抑える。

------------------------------------------------------------------
使い方
------------------------------------------------------------------
    import sympy as sp
    from resolve_singularity import resolve_singularities, ResolutionFailure

    x, y = sp.symbols("x y", real=True)
    res = resolve_singularities(x**2 + y**3, (x, y), max_depth=20)
    print(res.rlct, res.multiplicity)   # 5/6 1
    res.print_report()                  # 式変形の履歴

    python resolve_singularity.py       # デモ一式

依存: sympy (必須)、pulp (LP/ILP、pip install pulp で CBC も入る)、
      scipy (探索ループ内の高速な LP に使用、任意)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp


# ----------------------------------------------------------------------
# LP / ILP バックエンド (pulp -> CBC などのソルバー、scipy は高速な代替)
# ----------------------------------------------------------------------
try:
    import pulp as _pulp
    _HAS_PULP = True
except Exception:      # pragma: no cover
    _pulp = None
    _HAS_PULP = False

try:
    from scipy.optimize import linprog as _linprog
    _HAS_SCIPY = True
except Exception:      # pragma: no cover
    _linprog = None
    _HAS_SCIPY = False


def _pick_backend(backend: str, hot: bool = False) -> str:
    """'pulp' / 'scipy' / 'auto' を実際のバックエンド名に解決する。

    hot=True (探索ループ内で何度も呼ばれる用途) では、プロセス起動を伴う
    CBC より in-process の scipy(HiGHS) を優先する。
    """
    if backend == "pulp":
        if not _HAS_PULP:
            raise RuntimeError("pulp がインストールされていません (pip install pulp)")
        return "pulp"
    if backend == "scipy":
        if not _HAS_SCIPY:
            raise RuntimeError("scipy がインストールされていません")
        return "scipy"
    if hot:
        return "scipy" if _HAS_SCIPY else ("pulp" if _HAS_PULP else "none")
    return "pulp" if _HAS_PULP else ("scipy" if _HAS_SCIPY else "none")


def _lp_newton_value(monoms: Sequence[Sequence[int]],
                     hplus: Sequence[int],
                     backend: str = "auto",
                     hot: bool = False) -> float:
    """ニュートン多面体による RLCT の上界を線形計画で求める。

    振幅 x^h をもつ局所データ (f, x^h dx) に対し、Newton(f) の頂点集合を
    {a} とすると

        maximize   sum_a nu_a
        s.t.       sum_a nu_a * a_j <= h_j + 1   (各 j),   nu >= 0

    の最適値が「ニュートン多面体的な lambda」に一致する。単項式
    f = x^k のときは値が min_j (h_j+1)/k_j に退化し、一般には真の RLCT の
    *上界* になる (Varchenko / Lin: 非退化なら等号)。上界なので分枝限定の
    incumbent (primal bound) としてそのまま使える。
    """
    n = len(hplus)
    monoms = [tuple(m) for m in monoms]
    if any(all(e == 0 for e in m) for m in monoms):
        return float("inf")          # 定数項 (原点で消えない) -> 極なし
    N = len(monoms)
    if N == 0:
        return float("inf")
    which = _pick_backend(backend, hot=hot)

    if which == "pulp":
        prob = _pulp.LpProblem("newton_rlct", _pulp.LpMaximize)
        nu = [_pulp.LpVariable(f"nu{a}", lowBound=0) for a in range(N)]
        prob += _pulp.lpSum(nu)
        for j in range(n):
            coeffs = [(nu[a], monoms[a][j]) for a in range(N) if monoms[a][j]]
            if coeffs:
                prob += _pulp.LpAffineExpression(coeffs) <= hplus[j]
        status = prob.solve(_pulp.PULP_CBC_CMD(msg=0))
        if _pulp.LpStatus[status] == "Unbounded":
            return float("inf")
        if _pulp.LpStatus[status] != "Optimal":
            return float("inf")
        return float(_pulp.value(prob.objective))

    if which == "scipy":
        c = [-1.0] * N
        A_ub = [[float(monoms[a][j]) for a in range(N)] for j in range(n)]
        b_ub = [float(v) for v in hplus]
        res = _linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * N)
        if res.status == 3:          # unbounded
            return float("inf")
        if not res.success:
            return float("inf")
        return float(-res.fun)

    return float("inf")              # ソルバーが無ければ上界なし (枝刈りしない)


def _ilp_min_hitting_set(supports: Sequence[frozenset], n: int,
                         backend: str = "auto") -> Optional[List[int]]:
    """最小ヒッティング集合 (集合被覆) を 0-1 整数計画として解く。

        minimize  sum_j z_j
        s.t.      sum_{j in S} z_j >= 1  (各単項式の台 S),  z_j in {0,1}

    pulp (既定では CBC) に投げる。解けなければ None を返し、呼び出し側の
    分枝限定にフォールバックする。
    """
    which = _pick_backend(backend)
    if which != "pulp":
        return None
    prob = _pulp.LpProblem("min_center", _pulp.LpMinimize)
    z = {j: _pulp.LpVariable(f"z{j}", cat="Binary") for j in range(n)}
    prob += _pulp.lpSum(z.values())
    for S in supports:
        prob += _pulp.lpSum(z[j] for j in S) >= 1
    status = prob.solve(_pulp.PULP_CBC_CMD(msg=0))
    if _pulp.LpStatus[status] != "Optimal":
        return None
    return sorted(j for j in range(n) if z[j].value() is not None and z[j].value() > 0.5)


__all__ = [
    "ResolutionFailure",
    "Chart",
    "Resolution",
    "resolve_singularities",
    "newton_rlct",
]


# ----------------------------------------------------------------------
# 例外
# ----------------------------------------------------------------------
class ResolutionFailure(Exception):
    """一定回数の反復では特異点を解消できなかった場合に送出される。"""

    def __init__(self, message: str, *, chart: "Chart" = None, reason: str = "",
                 bound=None, lower=None):
        super().__init__(message)
        self.chart = chart
        self.reason = reason
        self.bound = bound   # 分枝限定で得られている lambda の上界 (あれば)
        self.lower = lower   # 未解消 chart の下界 (あれば)


# ----------------------------------------------------------------------
# 履歴 / chart
# ----------------------------------------------------------------------
@dataclass
class Step:
    """1 回の式変形の記録。"""

    kind: str          # 'init' | 'blowup' | 'normalize' | 'recenter'
    detail: str        # 人間可読な説明
    f_rest: sp.Expr    # 変形後の単元候補
    k: Tuple[int, ...]
    h: Tuple[int, ...]

    def __str__(self) -> str:
        return f"[{self.kind:<9}] {self.detail}\n            f_rest = {self.f_rest}\n            k = {self.k}, h = {self.h}"


@dataclass
class Chart:
    """解消の途中/結果を表す局所座標系。"""

    name: str
    poly: sp.Poly                       # f_rest (現在の座標)
    k: List[int]                        # f から括り出した単項式の指数
    h: List[int]                        # ヤコビアンの単項式の指数
    phi: Dict[sp.Symbol, sp.Expr]       # 現在の座標 -> もとの座標 への写像
    steps: List[Step] = field(default_factory=list)
    depth: int = 0
    resolved: bool = False
    seen: frozenset = frozenset()   # この系列で現れた f_rest (無限反復の検出用)

    # -- 便利メソッド ---------------------------------------------------
    @property
    def gens(self) -> Tuple[sp.Symbol, ...]:
        return self.poly.gens

    def f_rest(self) -> sp.Expr:
        return self.poly.as_expr()

    def monomial(self) -> sp.Expr:
        """f = monomial * f_rest となる単項式 x^k。"""
        return sp.prod([v ** e for v, e in zip(self.gens, self.k)])

    def jacobian(self) -> sp.Expr:
        """ヤコビアンの単項式部分 x^h。"""
        return sp.prod([v ** e for v, e in zip(self.gens, self.h)])

    def local_rlct(self) -> Tuple[sp.Rational, int]:
        """この chart の (lambda, 位数)。k がすべて 0 なら (oo, 0)。"""
        cand = [
            (sp.Rational(hj + 1, kj), j)
            for j, (kj, hj) in enumerate(zip(self.k, self.h))
            if kj > 0
        ]
        if not cand:
            return (sp.oo, 0)
        lam = min(c[0] for c in cand)
        mult = sum(1 for c in cand if c[0] == lam)
        return (lam, mult)

    def report(self) -> str:
        lam, mult = self.local_rlct()
        lines = [f"=== chart {self.name} (depth={self.depth}) ==="]
        for i, s in enumerate(self.steps):
            lines.append(f"  ({i}) {s}")
        lines.append("  -- 座標変換 phi (現在の座標 -> もとの座標) --")
        for v, e in self.phi.items():
            lines.append(f"      {v} = {sp.factor(e)}")
        lines.append(f"  f o phi = ({self.monomial()}) * ({self.f_rest()})")
        lines.append(f"  |det phi'| = (unit) * ({self.jacobian()})")
        lines.append(f"  lambda = {lam}, multiplicity = {mult}")
        return "\n".join(lines)


@dataclass
class Resolution:
    """resolve_singularities の返り値。"""

    f: sp.Expr
    gens: Tuple[sp.Symbol, ...]
    charts: List[Chart]
    rlct: sp.Rational
    multiplicity: int
    n_blowups: int
    n_pruned: int = 0
    n_lp: int = 0
    warnings: List[str] = field(default_factory=list)

    def report(self, only_minimal: bool = False) -> str:
        lines = [
            "#############################################################",
            f"# f = {self.f}",
            f"# 変数: {list(self.gens)}",
            f"# chart 数: {len(self.charts)},  ブローアップ回数: {self.n_blowups},"
            f"  枝刈り: {self.n_pruned},  LP 呼び出し: {self.n_lp}",
            f"# RLCT lambda = {self.rlct}  (= {sp.nsimplify(self.rlct)}"
            + (f" ~ {float(self.rlct):.6f}" if self.rlct.is_Number and self.rlct.is_finite else "")
            + f"),  位数 m = {self.multiplicity}",
            "#############################################################",
        ]
        for c in self.charts:
            if only_minimal and c.local_rlct()[0] != self.rlct:
                continue
            lines.append(c.report())
        for w in self.warnings:
            lines.append(f"[warning] {w}")
        return "\n".join(lines)

    def print_report(self, only_minimal: bool = False) -> None:
        print(self.report(only_minimal=only_minimal))


# ----------------------------------------------------------------------
# 低レベル操作
# ----------------------------------------------------------------------
def _to_poly(f, gens) -> sp.Poly:
    p = sp.Poly(sp.expand(f), *gens)
    if p.is_zero:
        raise ValueError("f が恒等的に 0 です。")
    return p


def _monomial_content(p: sp.Poly) -> Tuple[List[int], sp.Poly]:
    """単項式の最大公約因子 x^a を括り出す (renormalization)。"""
    monoms = p.monoms()
    n = len(p.gens)
    mins = [min(m[i] for m in monoms) for i in range(n)]
    if not any(mins):
        return [0] * n, p
    d = {
        tuple(e - a for e, a in zip(mon, mins)): c
        for mon, c in zip(monoms, p.coeffs())
    }
    return mins, sp.Poly.from_dict(d, *p.gens)


def _is_unit_at_origin(p: sp.Poly) -> bool:
    """原点で 0 にならない (= 単元) か。"""
    return p.coeff_monomial(sp.S.One) != 0


def _support_indices(p: sp.Poly) -> List[int]:
    """実際に現れる変数の添字。"""
    n = len(p.gens)
    used = [False] * n
    for mon in p.monoms():
        for i, e in enumerate(mon):
            if e:
                used[i] = True
    return [i for i in range(n) if used[i]]


def _min_center(p: sp.Poly, max_size: int = 8, backend: str = "auto") -> List[int]:
    """ブローアップの中心 {x_j = 0 : j in J} を選ぶ。

    中心が f_rest の零点集合に含まれる (= f_rest の全単項式が J の変数を
    少なくとも 1 つ含む) ような最小の J を分枝限定で求める。正規化済みなら
    |J| >= 2 が保証される (1 変数で全単項式を割れるなら括り出せているため)。
    最小の中心を取ることで、特異点集合が正の次元をもつ場合の無駄な
    (進展のない) ブローアップを避けられる。
    """
    n = len(p.gens)
    supports = [frozenset(i for i, e in enumerate(m) if e) for m in p.monoms()]
    supports = sorted({s for s in supports if s}, key=lambda t: (len(t), sorted(t)))
    if not supports:
        return []
    # 極小な台だけ残す (S1 <= S2 なら S2 の制約は冗長)
    minimal = []
    for S in supports:
        if not any(T <= S for T in minimal):
            minimal.append(S)
    supports = minimal
    # 規模が大きいときは 0-1 整数計画 (pulp -> CBC) に任せる
    if backend != "internal" and (len(supports) > 12 or n > 12):
        sol = _ilp_min_hitting_set(supports, n, backend=backend)
        if sol:
            return sol
    best: List[List[int]] = [None]

    def rec(rest, chosen):
        if best[0] is not None and len(chosen) >= len(best[0]):
            return
        if not rest:
            best[0] = sorted(chosen)
            return
        if len(chosen) >= max_size:
            return
        target = min(rest, key=len)          # 分枝限定: 最小の集合で分岐
        for v in sorted(target):
            rec([s for s in rest if v not in s], chosen + [v])

    rec(supports, [])
    if best[0] is None:  # 上限に達した場合は現れる変数すべてを中心にする
        return _support_indices(p)
    return best[0]


def _blowup_poly(p: sp.Poly, i: int, J: Sequence[int]) -> sp.Poly:
    """chart i のブローアップ x_j -> x_i x_j (j in J, j != i)。

    単項式の置換は指数の線形変換なので、多項式展開なしに実行できる。
    """
    shift = [j for j in J if j != i]
    acc: Dict[Tuple[int, ...], sp.Expr] = {}
    for mon, c in zip(p.monoms(), p.coeffs()):
        e = list(mon)
        e[i] = mon[i] + sum(mon[j] for j in shift)
        key = tuple(e)
        acc[key] = acc.get(key, sp.S.Zero) + c
    acc = {kk: sp.simplify(vv) for kk, vv in acc.items()}
    acc = {kk: vv for kk, vv in acc.items() if vv != 0}
    if not acc:  # 単項式変換は torus 上単射なので通常起こらない
        raise ResolutionFailure("ブローアップ後に f が消えました (異常)。")
    return sp.Poly.from_dict(acc, *p.gens)


def _pushforward_exponents(e: Sequence[int], i: int, J: Sequence[int]) -> List[int]:
    """指数ベクトル x^e を chart i のブローアップで押し出す。"""
    ne = list(e)
    ne[i] = e[i] + sum(e[j] for j in J if j != i)
    return ne


# ----------------------------------------------------------------------
# 分枝限定のための下界 (LB) と上界 (UB)
# ----------------------------------------------------------------------
def _poly_order(p: sp.Poly) -> int:
    """原点における重複度 (最低次数)。"""
    return min(sum(m) for m in p.monoms())


def _subtree_lower_bound(k: Sequence[int], h: Sequence[int], budget: int,
                         recenter_possible: bool) -> sp.Rational:
    """この chart の子孫すべてが満たす lambda の下界。

    ブローアップ chart i では
        k_i' = sum_{j in J} k_j + a,   h_i' + 1 = sum_{j in J} (h_j + 1)
    となるので、メディアント不等式より

        (h_i'+1)/k_i' = sum(h_j+1) / (sum k_j + a) >= min_j (h_j+1)/(k_j + a)

    が成り立つ。つまりブローアップ単体では比は下がらず、下がるのは正規化で
    括り出される order a の分だけ。よって残りの反復で加算されうる order の
    総量を budget M で抑えれば

        lambda(部分木) >= min_j (h_j + 1) / (k_j + M)

    が下界になる (j は k_j + M > 0 のもの)。中心の付け替え (recenter) は
    k_j, h_j を 0 に落とすので、その可能性がある場合は床 1/M も加える。

    注: budget は「各ステップで括り出される order <= 現在の重複度」かつ
    「重複度がブローアップで増えない」(標数 0 の permissible な中心では標準的)
    という前提で見積もっている。厳密な網羅が必要なら prune=False とする。
    """
    cands = []
    for kj, hj in zip(k, h):
        d = kj + budget
        if d > 0:
            cands.append(sp.Rational(hj + 1, d))
    if recenter_possible and budget > 0:
        cands.append(sp.Rational(1, budget))
    return min(cands) if cands else sp.oo


def _chart_upper_bound(k: Sequence[int], p: sp.Poly, h: Sequence[int],
                       backend: str, cache: dict) -> float:
    """この chart の局所データ (x^k * f_rest, 振幅 x^h) のニュートン上界。

    f = x^k * f_rest の Newton 多面体は k + Newton(f_rest) なので、単項式を
    k だけ平行移動して LP を解く。値は真の RLCT の上界であり、大域最小値の
    上界でもあるため、分枝限定の incumbent として使える。
    """
    key = (tuple(k), tuple(h), tuple(p.monoms()))
    if key in cache:
        return cache[key]
    monoms = [tuple(ki + ei for ki, ei in zip(k, m)) for m in p.monoms()]
    val = _lp_newton_value(monoms, [hj + 1 for hj in h], backend=backend, hot=True)
    # 外部ソルバー (CBC) は有効数字 8 桁程度で値を返すことがあり、真値を
    # わずかに下回ると最適枝を刈ってしまう。安全側に緩めておく。
    if val != float("inf"):
        val = val * (1 + 1e-6) + 1e-9
    cache[key] = val
    return val


# ----------------------------------------------------------------------
# 例外因子上の非正規交差点の探索 (中心の付け替え)
# ----------------------------------------------------------------------
def _offorigin_centers(chart: Chart, max_solutions: int = 4):
    """例外因子 {x_e = 0} 上で f_rest が非正規交差になる点を探す。

    条件: f_rest = 0 かつ (x_e 以外の) 勾配 = 0。
    見つかった点の座標 dict のリストを返す (原点は除く)。
    """
    gens = chart.gens
    f = chart.f_rest()
    exc = [
        v
        for v, kv, hv in zip(gens, chart.k, chart.h)
        if kv > 0 or hv > 0
    ]
    found = []
    for e in exc:
        others = [v for v in gens if v != e]
        g = sp.expand(f.subs({e: 0}))
        if g == 0 or not g.free_symbols:
            continue
        eqs = [g] + [sp.diff(g, v) for v in others]
        try:
            sols = sp.solve(eqs, others, dict=True)
        except Exception:
            continue
        for s in sols[:max_solutions]:
            pt = {e: sp.S.Zero}
            ok = True
            for v in others:
                val = sp.nsimplify(s.get(v, sp.S.Zero))
                if val.free_symbols or val.is_real is False:
                    ok = False
                    break
                pt[v] = val
            if ok and any(pt[v] != 0 for v in others):
                if pt not in found:
                    found.append(pt)
    return found


def _drop_unit_factors(p: sp.Poly) -> Tuple[sp.Poly, sp.Expr]:
    """原点で消えない因子 (単元) を f_rest から取り除く。

    |f|^z の原点における特異性には単元因子は寄与しないので落としてよい。
    これによりブローアップの中心を特異点集合に近づけられ、
    (a1^2+a2^2)(b1^2+b2^2) のような例での無限反復を防げる。
    戻り値は (単元を除いた多項式, 除いた単元)。
    """
    gens = p.gens
    zero = {v: 0 for v in gens}
    try:
        c, facs = sp.factor_list(p.as_expr())
    except Exception:
        return p, sp.S.One
    keep, unit = [], sp.Integer(1) * c
    for g, d in facs:
        if g.subs(zero) != 0:
            unit *= g ** d
        else:
            keep.append((g, d))
    if unit == 1 or not keep:
        return p, sp.S.One
    new = sp.expand(sp.prod([g ** d for g, d in keep]))
    return sp.Poly(new, *gens), unit


def _smooth_factor_change(chart: Chart, name: str) -> Optional["Chart"]:
    """f_rest の因子 g が原点で滑らかなら、g 自身を新しい座標に取る。

    g が変数 x_v について 1 次で、その係数 A が原点で非零 (単元)、かつ
    x_v が例外因子に使われていない (k_v = h_v = 0) とき、

        x_v = (x_v' - B) / A        (g = A x_v + B)

    は原点近傍の座標変換で、ヤコビアンは単元 1/A。この変換で g -> x_v'
    となるので、続く正規化で単項式として括り出せる。
    (x - y)^2 や (a1 b1 + a2 b2)^2 のように特異点が正の次元をもつ場合に
    ブローアップの無限反復を回避できる。
    """
    gens = chart.gens
    free = {v for v, kv, hv in zip(gens, chart.k, chart.h) if kv == 0 and hv == 0}
    if not free:
        return None
    _, facs = sp.factor_list(chart.f_rest())
    zero = {v: 0 for v in gens}
    for g, d in facs:
        if g.subs(zero) != 0:          # 単元因子は無視
            continue
        for v in gens:
            if v not in free or v not in g.free_symbols:
                continue
            pg = sp.Poly(g, v)
            if pg.degree() != 1:
                continue
            A = pg.coeff_monomial(v)
            B = pg.coeff_monomial(1)
            if sp.simplify(A.subs(zero)) == 0:
                continue               # A が原点で 0 なら単元でない
            sub = {v: (v - B) / A}
            new_expr = sp.cancel(sp.together(chart.f_rest().subs(sub, simultaneous=True)))
            num, den = sp.fraction(new_expr)
            if sp.simplify(den.subs(zero)) == 0:
                continue
            new_poly = sp.Poly(sp.expand(num), *gens)   # den は単元なので落とす
            new_phi = {
                w: sp.cancel(sp.together(e.subs(sub, simultaneous=True)))
                for w, e in chart.phi.items()
            }
            detail = (f"滑らかな因子の座標化: {v} -> ({v} - ({B}))/({A})"
                      f"  [因子 ({g})^{d} を {v}^{d} に]")
            step = Step("coordchg", detail, new_poly.as_expr(),
                        tuple(chart.k), tuple(chart.h))
            return Chart(
                name=name,
                poly=new_poly,
                k=list(chart.k),
                h=list(chart.h),
                phi=new_phi,
                steps=chart.steps + [step],
                depth=chart.depth + 1,
            )
    return None


def _recenter(chart: Chart, point: Dict[sp.Symbol, sp.Expr], name: str) -> Chart:
    """point を新しい原点とする平行移動を行った chart を作る。"""
    gens = chart.gens
    shift = {v: v + point[v] for v in gens if point.get(v, 0) != 0}
    new_f = sp.expand(chart.f_rest().subs(shift, simultaneous=True))
    new_poly = sp.Poly(new_f, *gens)
    # (x_j + c_j)^{k_j} は c_j != 0 のとき新しい原点で単元 -> k_j, h_j を落とす
    new_k = [0 if point.get(v, 0) != 0 else kj for v, kj in zip(gens, chart.k)]
    new_h = [0 if point.get(v, 0) != 0 else hj for v, hj in zip(gens, chart.h)]
    new_phi = {v: sp.expand(expr.subs(shift, simultaneous=True)) for v, expr in chart.phi.items()}
    detail = "中心の付け替え: " + ", ".join(f"{v} -> {v} + {c}" for v, c in point.items() if c != 0)
    step = Step("recenter", detail, new_poly.as_expr(), tuple(new_k), tuple(new_h))
    return Chart(
        name=name,
        poly=new_poly,
        k=new_k,
        h=new_h,
        phi=new_phi,
        steps=chart.steps + [step],
        depth=chart.depth + 1,
    )


# ----------------------------------------------------------------------
# 本体
# ----------------------------------------------------------------------
def resolve_singularities(
    f,
    gens: Optional[Sequence[sp.Symbol]] = None,
    *,
    max_depth: int = 20,
    max_charts: int = 5000,
    recenter: Optional[bool] = None,
    recenter_max_vars: int = 4,
    smooth_coords: bool = True,
    drop_unit_factors: bool = True,
    center: str = "min",
    prune=True,
    lp_backend: str = "auto",
    verbose: bool = False,
) -> Resolution:
    """多項式 f の原点における特異点をブローアップの反復で解消する。

    Parameters
    ----------
    f : sympy expr
        対象の多項式 (実係数)。原点を含む近傍での局所的な RLCT を計算する。
    gens : list of Symbol, optional
        変数リスト。省略時は sorted(f.free_symbols)。
    max_depth : int
        1 つの chart あたりのブローアップ/付け替えの最大反復回数。
        これを超えても解消できない chart があれば ResolutionFailure。
    max_charts : int
        生成する chart 数の上限 (組合せ爆発への保険)。超えたら例外。
    center : {'min', 'support'}
        ブローアップの中心の選び方。'min' は零点集合に含まれる最小の
        座標部分空間 (最小ヒッティング集合)、'support' は f_rest に現れる
        変数全体。'min' のほうが正の次元の特異点集合に強い。
    drop_unit_factors : bool
        原点で消えない因子 (単元) を f_rest から落とす。ブローアップの
        中心が特異点集合に近づき、反復回数と chart 数が減る。
    smooth_coords : bool
        f_rest が原点で滑らかな因子をもつとき、その因子自身を座標に取る
        変数変換を試みる (正の次元の特異点集合に対する無限反復を回避)。
    prune : bool or 'ties'
        分枝限定による枝刈りを行う。'ties' にすると、到達済みの最小値と
        同値の部分木も刈る (lambda は正しいまま、位数 m は下限になりうる)。各 chart で
          * 上界 UB: ニュートン多面体 LP (pulp/CBC または scipy) による
            局所 RLCT の上界。大域最小値の上界でもある。
          * 下界 LB: メディアント不等式による部分木の lambda の下界。
        を計算し、LB > (これまでの最良値と UB の最小) なら展開を打ち切る。
        lambda と位数 m は保存される (厳密に > のときだけ枝刈りするため)。
        LB は「各ステップで括り出される order <= 現在の重複度、かつ重複度は
        増えない」という標準的な仮定に依る。完全な網羅が必要なら False。
    lp_backend : {'auto', 'pulp', 'scipy'}
        LP/ILP のバックエンド。'pulp' は CBC などの外部ソルバーを呼ぶ。
        'auto' は探索ループ内では in-process の scipy(HiGHS)、単発の
        呼び出し (newton_rlct, 中心の整数計画) では pulp を優先する。
    recenter : bool or None
        例外因子上の原点以外の非正規交差点を探して中心を付け替えるか。
        None なら変数が recenter_max_vars 以下のときのみ有効。
    verbose : bool
        進行状況を表示する。

    Returns
    -------
    Resolution
    """
    if gens is None:
        gens = sorted(f.free_symbols, key=lambda s: s.name)
    gens = tuple(gens)
    if not gens:
        raise ValueError("変数が指定されていません。")
    n = len(gens)

    if recenter is None:
        recenter = n <= recenter_max_vars

    p0 = _to_poly(f, gens)
    root = Chart(
        name="C0",
        poly=p0,
        k=[0] * n,
        h=[0] * n,
        phi={v: v for v in gens},
        steps=[Step("init", f"f = {sp.expand(f)}", p0.as_expr(), (0,) * n, (0,) * n)],
        depth=0,
    )

    stack: List[Chart] = [root]
    done: List[Chart] = []
    warnings: List[str] = []
    n_blowups = 0
    n_created = 1
    n_pruned = 0
    n_lp = 0
    ub_cache: dict = {}
    bound = float("inf")      # 大域 lambda の上界 (LP 上界 or 到達値)
    achieved = float("inf")   # 実際に解消済み chart が到達した最小値
    ties_pruned = False
    prune_ties = (prune == "ties")
    prune = bool(prune)
    if prune and _pick_backend(lp_backend, hot=True) == "none":
        prune = False
        warnings.append("LP ソルバー (pulp / scipy) が無いため枝刈りを無効化しました。")

    while stack:
        ch = stack.pop()

        # ---- 1. 正規化 (単項式因子の括り出し) ----------------------
        a, q = _monomial_content(ch.poly)
        if any(a):
            ch.poly = q
            ch.k = [kj + aj for kj, aj in zip(ch.k, a)]
            mono = sp.prod([v ** e for v, e in zip(gens, a)])
            ch.steps.append(
                Step("normalize", f"単項式 {mono} を括り出し", q.as_expr(),
                     tuple(ch.k), tuple(ch.h))
            )

        unit = _is_unit_at_origin(ch.poly)

        # ---- 1.2 単元因子の除去 -----------------------------------
        if not unit and drop_unit_factors:
            q2, u = _drop_unit_factors(ch.poly)
            if u != 1:
                ch.poly = q2
                ch.steps.append(
                    Step("normalize", f"単元因子 ({u}) を除去", q2.as_expr(),
                         tuple(ch.k), tuple(ch.h))
                )
                unit = _is_unit_at_origin(ch.poly)

        # ---- 1.4 分枝限定: 上界の更新と枝刈り ---------------------
        if prune:
            ub = _chart_upper_bound(ch.k, ch.poly, ch.h, lp_backend, ub_cache)
            n_lp += 1
            if ub < bound:
                bound = ub
            budget = _poly_order(ch.poly) * max(max_depth - ch.depth, 0)
            lb = _subtree_lower_bound(ch.k, ch.h, budget, recenter)
            if lb is not sp.oo:
                lbf = float(lb)
                # (a) 厳密な枝刈り: 上界より真に大きい -> lambda も m も保存
                cut = lbf > bound * (1 + 1e-9) + 1e-12
                # (b) 同値の枝刈り: 到達済みの値以上 -> lambda は保存、m は下限に
                tie = (not cut) and prune_ties and lbf >= achieved * (1 - 1e-9) - 1e-12
                if cut or tie:
                    n_pruned += 1
                    ties_pruned = ties_pruned or tie
                    if verbose:
                        print(f"  [pruned]   {ch.name}: LB={lb} "
                              f"{'>' if cut else '>='} {bound if cut else achieved:.6f}")
                    continue

        # ---- 1.5 滑らかな因子を座標に取る -------------------------
        if not unit and smooth_coords and ch.depth < max_depth:
            nc = _smooth_factor_change(ch, f"{ch.name}c")
            if nc is not None:
                n_created += 1
                stack.append(nc)
                if verbose:
                    print(f"  [coordchg] {ch.name}: {nc.steps[-1].detail}")
                continue

        # ---- 2. 中心の付け替え候補 --------------------------------
        recentered = []
        if recenter and n <= recenter_max_vars:
            for idx, pt in enumerate(_offorigin_centers(ch)):
                recentered.append(_recenter(ch, pt, f"{ch.name}r{idx}"))

        # ---- 3. 原点で単元 かつ 付け替え不要 -> 解消済み -----------
        if unit and not recentered:
            ch.resolved = True
            done.append(ch)
            lam_c, m_c = ch.local_rlct()
            if lam_c is not sp.oo:
                achieved = min(achieved, float(lam_c))
                bound = min(bound, float(lam_c))   # 到達値で incumbent を更新
            if verbose:
                print(f"  [resolved] {ch.name}: lambda={lam_c}, m={m_c}")
            continue

        # ---- 4. 反復回数 / 無限反復のチェック ----------------------
        key = sp.srepr(ch.poly.as_expr())
        if key in ch.seen:
            raise ResolutionFailure(
                f"chart {ch.name} で同じ残差 f_rest = {ch.f_rest()} が再び現れました。"
                " 原点中心のブローアップでは解消できません "
                "(中心の取り方を変えるか、newton_rlct で検算してください)。",
                chart=ch,
                reason="no-progress",
            )
        ch.seen = ch.seen | {key}

        if ch.depth >= max_depth:
            lb_now = _subtree_lower_bound(ch.k, ch.h, 0, False)
            msg = (f"chart {ch.name} は {max_depth} 回の反復では解消できませんでした。"
                   f" 残差 f_rest = {ch.f_rest()}  (k={tuple(ch.k)}, h={tuple(ch.h)})")
            if bound != float("inf"):
                msg += (f" / 現時点で得られている評価: lambda <= {bound:.6g}"
                        f" (この chart の下界 {lb_now})")
            msg += " / max_depth を増やすか、center を変えるか、newton_rlct() で検算してください。"
            raise ResolutionFailure(msg, chart=ch, reason="max_depth",
                                    bound=(None if bound == float("inf") else bound),
                                    lower=lb_now)

        # 付け替え chart を積む
        for rc in recentered:
            n_created += 1
            stack.append(rc)
        if unit:
            # 原点自身は解消済みだが、他の点のために付け替え chart のみ続行
            continue

        # ---- 5. ブローアップ --------------------------------------
        J = (_min_center(ch.poly, backend=lp_backend) if center == "min"
             else _support_indices(ch.poly))
        if len(J) <= 1:
            # 1 変数しか現れないのに単元でない -> 正規化で単項式化済みのはず
            raise ResolutionFailure(
                f"chart {ch.name} でブローアップの中心を取れません: f_rest = {ch.f_rest()}",
                chart=ch,
                reason="degenerate-center",
            )

        children: List[Tuple[sp.Rational, Chart]] = []
        for i in J:
            n_created += 1
            if n_created > max_charts:
                raise ResolutionFailure(
                    f"chart 数が上限 {max_charts} を超えました "
                    f"(変数 {n} 個の組合せ爆発)。max_charts を上げるか変数を減らしてください。",
                    chart=ch,
                    reason="max_charts",
                )
            new_poly = _blowup_poly(ch.poly, i, J)
            new_k = _pushforward_exponents(ch.k, i, J)
            new_h = _pushforward_exponents(ch.h, i, J)
            new_h[i] += len(J) - 1  # ヤコビアン y_i^{|J|-1}
            sub = {gens[j]: gens[i] * gens[j] for j in J if j != i}
            new_phi = {v: sp.expand(e.subs(sub, simultaneous=True)) for v, e in ch.phi.items()}
            detail = (
                f"blow-up 中心 {{{', '.join(str(gens[j]) for j in J)} = 0}}, "
                f"chart {gens[i]}: " + ", ".join(f"{gens[j]} -> {gens[i]}*{gens[j]}" for j in J if j != i)
            )
            child = Chart(
                name=f"{ch.name}-{gens[i]}",
                poly=new_poly,
                k=new_k,
                h=new_h,
                phi=new_phi,
                steps=ch.steps + [Step("blowup", detail, new_poly.as_expr(),
                                       tuple(new_k), tuple(new_h))],
                depth=ch.depth + 1,
                seen=ch.seen,      # ブローアップの系列でのみ無限反復を検出
            )
            # 見込み値 (小さいほど大域最小を早く更新できる) で並べ替える
            est = _subtree_lower_bound(new_k, new_h, 0, False)
            children.append((est if est is not sp.oo else sp.Integer(10) ** 9, child))
        # DFS スタックなので、有望なものが最後に push されるよう降順で積む
        for _, child in sorted(children, key=lambda t: -t[0]):
            stack.append(child)
        n_blowups += 1
        if verbose:
            print(f"  [blowup {n_blowups}] {ch.name} depth={ch.depth} "
                  f"center={[str(gens[j]) for j in J]} f_rest={ch.f_rest()}")

    # ---- RLCT の集約 ---------------------------------------------
    lam = sp.oo
    mult = 0
    for c in done:
        l, m = c.local_rlct()
        if l < lam:
            lam, mult = l, m
        elif l == lam:
            mult = max(mult, m)
    if lam is sp.oo:
        warnings.append("f は原点で消えていません (lambda = oo)。")

    if ties_pruned:
        warnings.append(
            "prune='ties': 同値の部分木を枝刈りしました。lambda は正しいですが、"
            "位数 m は下限 (真の値以下) の可能性があります。"
        )
    if not recenter:
        warnings.append(
            "recenter=False: 例外因子上の原点以外に非正規交差点が残る場合、"
            "lambda を過大評価している可能性があります。"
        )

    return Resolution(
        f=sp.expand(f),
        gens=gens,
        charts=done,
        rlct=lam,
        multiplicity=mult,
        n_blowups=n_blowups,
        n_pruned=n_pruned,
        n_lp=n_lp,
        warnings=warnings,
    )


# ----------------------------------------------------------------------
# ニュートン多面体によるクロスチェック (非退化な f に対して厳密)
# ----------------------------------------------------------------------
def newton_rlct(f, gens: Optional[Sequence[sp.Symbol]] = None,
                h: Optional[Sequence[int]] = None,
                backend: str = "auto",
                max_denominator: int = 10 ** 6):
    """ニュートン多面体による RLCT を線形計画で求める (Varchenko / Lin)。

    振幅 x^h をもつ局所データ (f, x^h dx) に対し、Newton(f) の指数ベクトル
    {a} を使って

        maximize   sum_a nu_a
        s.t.       sum_a nu_a * a_j <= h_j + 1  (各 j),   nu >= 0

    を解く。h を省略すると h = 0、すなわち通常の
    「対角線がニュートン多面体に到達する距離 t の逆数 1/t」に一致する。

    f が (実数体上) ニュートン多面体に関して非退化なら真の RLCT と一致し、
    一般には *上界* を与える。多変数でも LP なので高速で、
    resolve_singularities の枝刈り (incumbent) と検算の両方に使われる。

    backend : 'auto' | 'pulp' | 'scipy'
        'pulp' は CBC などの外部ソルバーを呼ぶ。'auto' は pulp を優先。
    """
    if gens is None:
        gens = sorted(f.free_symbols, key=lambda s: s.name)
    gens = tuple(gens)
    p = sp.Poly(sp.expand(f), *gens)
    if h is None:
        h = [0] * len(gens)
    val = _lp_newton_value(p.monoms(), [hj + 1 for hj in h], backend=backend)
    if val == float("inf"):
        return sp.oo
    return sp.Rational(val).limit_denominator(max_denominator)


# ----------------------------------------------------------------------
# デモ
# ----------------------------------------------------------------------
def _demo():
    import time

    x, y, z = sp.symbols("x y z", real=True)
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)
    w = sp.symbols("w0:4", real=True)
    R = sp.Rational

    examples = [
        ("x^2 + y^2", x**2 + y**2, (x, y), R(1)),
        ("x^2 y^2", x**2 * y**2, (x, y), R(1, 2)),
        ("(x - y)^2", (x - y) ** 2, (x, y), R(1, 2)),
        ("x^2 + y^3", x**2 + y**3, (x, y), R(5, 6)),
        ("(x^2 - y^3)^2", (x**2 - y**3) ** 2, (x, y), R(5, 12)),
        ("x^2 + y^3 + z^4", x**2 + y**3 + z**4, (x, y, z), R(1,2)+R(1,3)+R(1,4)),
        ("x^2y^2 + y^2z^2 + z^2x^2", x**2*y**2 + y**2*z**2 + z**2*x**2, (x, y, z), R(3, 4)),
        ("(xy + z^2)^2", (x * y + z**2) ** 2, (x, y, z), R(1, 2)),
        ("(a1b1 + a2b2)^2", (a1*b1 + a2*b2) ** 2, (a1, a2, b1, b2), R(1, 2)),
        ("||a b^T||^2 (2x2)",
         sum((ai * bj) ** 2 for ai in (a1, a2) for bj in (b1, b2)),
         (a1, a2, b1, b2), R(1)),
    ]
    print("=== 例題 (lambda / 位数 m / chart 数 / 枝刈り数) " + "=" * 24)
    for label, f, g, expect in examples:
        res = resolve_singularities(f, g)
        mark = "OK" if res.rlct == expect else f"期待値 {expect} と不一致!"
        print(f"  {label:<26} lambda={str(res.rlct):<7} m={res.multiplicity}"
              f"  charts={len(res.charts):<3} pruned={res.n_pruned:<4} {mark}")

    print("\n=== 分枝限定の効果 " + "=" * 41)
    bench = [
        ("x^2+y^3+z^7", x**2 + y**3 + z**7, (x, y, z), 30),
        ("x^3+y^4+z^5", x**3 + y**4 + z**5, (x, y, z), 30),
        ("w0^2+w1^3+w2^4+w3^5", sum(v ** (i + 2) for i, v in enumerate(w)), w, 30),
    ]
    for label, f, g, md in bench:
        print(f"  {label}")
        for mode in (False, True, "ties"):
            st = time.time()
            try:
                r = resolve_singularities(f, g, max_depth=md, prune=mode)
                print(f"    prune={str(mode):<5}: lambda={str(r.rlct):<7} m={r.multiplicity}"
                      f" charts={len(r.charts):<4} pruned={r.n_pruned:<5}"
                      f" LP={r.n_lp:<5} {time.time()-st:5.1f}s")
            except ResolutionFailure as e:
                print(f"    prune={str(mode):<5}: ResolutionFailure({e.reason})"
                      f"  暫定上界 lambda <= {e.bound}  この chart の下界 {e.lower}"
                      f"  {time.time()-st:5.1f}s")

    print("\n=== LP バックエンドの比較 (newton_rlct) " + "=" * 21)
    for label, f, g in [("x^2+y^3", x**2 + y**3, (x, y)),
                        ("x^3+y^4+z^5", x**3 + y**4 + z**5, (x, y, z)),
                        ("(xy+z^2)^2", (x*y + z**2) ** 2, (x, y, z))]:
        vals = {}
        for be in ("pulp", "scipy"):
            try:
                vals[be] = newton_rlct(f, g, backend=be)
            except Exception as e:
                vals[be] = f"(不可: {e})"
        blow = resolve_singularities(f, g, prune="ties").rlct
        note = "" if blow == vals.get("pulp") else "   <- 実数体上で退化 (blow-up 側が正しい)"
        print(f"  {label:<14} pulp/CBC: {str(vals['pulp']):<7} scipy: {str(vals['scipy']):<7}"
              f" blow-up: {blow}{note}")
    xs = sp.symbols("x0:40", real=True)
    st = time.time()
    v = newton_rlct(sum(vv ** 2 for vv in xs), xs)
    print(f"  40 変数 sum x_i^2: newton_rlct = {v} ({time.time()-st:.2f}s, LP なので高速)")

    print("\n=== 中心の選択 (最小ヒッティング集合 = 0-1 整数計画) " + "=" * 12)
    ys = sp.symbols("y0:20", real=True)
    st = time.time()
    r = resolve_singularities(sum(vv ** 2 for vv in ys), ys, prune=True)
    print(f"  20 変数 sum y_i^2: lambda={r.rlct} charts={len(r.charts)} "
          f"pruned={r.n_pruned} {time.time()-st:.2f}s")

    print("\n=== 履歴の例: f = (x*y + z^2)^2 " + "=" * 29)
    resolve_singularities((x * y + z**2) ** 2, (x, y, z)).print_report(only_minimal=True)

    print("\n=== 解消できない場合は例外 " + "=" * 34)
    try:
        resolve_singularities(x**2 + y**7, (x, y), max_depth=2)
    except ResolutionFailure as e:
        print(f"  x^2+y^7 (max_depth=2) -> ResolutionFailure(reason={e.reason!r})")
        print(f"    {str(e)[:180]}...")


if __name__ == "__main__":
    _demo()
