r"""
certify.py
==========

解消の結果に「証明書」を付ける。目的は *誤った lambda を返さないこと* で、
確かめられない部分があれば数値ではなく unknown を返す。

------------------------------------------------------------------
なぜ必要か
------------------------------------------------------------------
lambda = min_a min_j (h_j+1)/k_j が正しいためには、chart ごとに

    f(phi_a(y)) = y^k * u_a(y),   det D phi_a = y^h * v_a(y)

が成り立つだけでなく、**f = y^k u_a が V_a 全体で例外因子と正規交差**
になっていることが要る。原点で u_a(0) != 0 を確かめるだけでは足りない。
u_a が消えること自体は構わない (零点が滑らかで例外因子と横断的なら正規交差の
まま) が、横断性が壊れる点があると破綻する。典型例:

    f = (x - y)^2 を原点でブローアップすると chart x で
    f = x^2 * (1 - y)^2。 (1-y)^2 は原点では単元だが y = 1 で二重に消えるので
    (零点が滑らかでない)、その近傍では正規交差になっていない。ここを
    見落とすと lambda = 1 と誤る (真値 1/2)。

この「V_a 上で u_a != 0」は実代数の存在問題

    exists y in V_a. u_a(y) = 0

の否定なので、Z3 (nlsat) のような SMT ソルバで決定できる。UNSAT なら証明、
SAT なら反例 (= 中心の付け替え先)、unknown なら未決定として伝播させる。

------------------------------------------------------------------
定義域 V_a の追跡
------------------------------------------------------------------
根: V = {|x_j| <= eps for all j}

ブローアップ chart i (中心 J): 被覆補題が与える y は
    y_i = x_i,  |y_j| <= 1 (j in J, j != i),  y_k = x_k (k not in J)
なので子の箱は  b'_i = b_i,  b'_j = 1 (j in J\{i}),  b'_k = b_k。
逆に像の側は |x_j| = |y_i y_j| <= b_i <= b_j なので親の箱に収まる ((d) 側)。

座標変換 x_v = (y_v - B)/A: 逆に y_v = A x_v + B なので、親の箱の上での
A, B の区間評価から b'_v = |A|_max * b_v + |B|_max。
平行移動 x_j = y_j + c_j: b'_j = b_j + |c_j|。

いずれも区間演算で健全な (十分大きい) 箱を取る。

------------------------------------------------------------------
三値の結果
------------------------------------------------------------------
  proved   : すべての葉で u_a != 0, v_a != 0 が V_a 上で証明され、
             被覆も確立している。lambda を返す。
  unknown  : どれかが未決定。lambda は返さない (None)。
             ただし「これまでに到達した最小値」は下界情報として残す。
  refuted  : 反例が見つかった。その chart は正規交差になっていない。

------------------------------------------------------------------
証明していないこと
------------------------------------------------------------------
* ゼータ関数の極が min_j (h_j+1)/k_j にあるという解析的な定理 (Watanabe)。
  これは前提として置く。
* 枝刈りの下界の妥当性。証明モードでは prune=False を推奨する。
* 台の関数 psi の正値性。

使い方
------
    res = resolve_singularities(f, gens, prune=False)
    cert = certify(res, eps=1)
    cert.print_report()
    cert.rlct            # proved のときだけ値、そうでなければ None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

__all__ = ["Certificate", "LeafCertificate", "certify", "Box",
           "nonzero_on_box", "normal_crossing_on_box",
           "newton_nondegenerate", "rlct_via_newton", "newton_faces"]

try:
    import z3 as _z3
    _HAS_Z3 = True
except Exception:                                     # pragma: no cover
    _z3 = None
    _HAS_Z3 = False


# ----------------------------------------------------------------------
# 箱 (定義域)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    """原点中心の箱 {|y_j| <= bound_j}。"""

    bounds: Tuple[sp.Rational, ...]

    def __str__(self) -> str:
        return "{" + ", ".join(f"|y{j}|<={b}" for j, b in enumerate(self.bounds)) + "}"


def _interval_eval(expr, gens, box: Box) -> Tuple[sp.Rational, sp.Rational]:
    """多項式を箱の上で区間評価する (健全な外側評価)。"""
    p = sp.Poly(sp.expand(expr), *gens) if expr.free_symbols else None
    if p is None:
        v = sp.nsimplify(expr)
        return (v, v)
    lo = hi = sp.Integer(0)
    for mon, c in zip(p.monoms(), p.coeffs()):
        mag = sp.Integer(1)
        for e, b in zip(mon, box.bounds):
            mag *= b ** e
        term = sp.Abs(c) * mag
        lo -= term
        hi += term
    return (lo, hi)


def _abs_max(expr, gens, box: Box) -> sp.Rational:
    lo, hi = _interval_eval(expr, gens, box)
    return max(abs(lo), abs(hi))


# ----------------------------------------------------------------------
# sympy -> z3
# ----------------------------------------------------------------------
def _to_z3(expr, zvars: Dict[sp.Symbol, object]):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Symbol):
        return zvars[expr]
    if expr.is_Integer:
        return _z3.RealVal(int(expr))
    if expr.is_Rational:
        return _z3.RealVal(f"{expr.p}/{expr.q}")
    if expr.is_Float:
        r = sp.Rational(expr).limit_denominator(10 ** 9)
        return _z3.RealVal(f"{r.p}/{r.q}")
    if expr.is_Add:
        out = _to_z3(expr.args[0], zvars)
        for a in expr.args[1:]:
            out = out + _to_z3(a, zvars)
        return out
    if expr.is_Mul:
        out = _to_z3(expr.args[0], zvars)
        for a in expr.args[1:]:
            out = out * _to_z3(a, zvars)
        return out
    if expr.is_Pow:
        b, e = expr.args
        if e.is_Integer and e >= 0:
            return _to_z3(b, zvars) ** int(e)
    raise ValueError(f"z3 に落とせない式: {expr}")


def nonzero_on_box(expr, gens, box: Box, *, exclusions: Sequence[Dict] = (),
                   fiber: Sequence = (),
                   delta=sp.Rational(1, 100), timeout_ms: int = 10000):
    """expr が箱 V の上で (除外球の外で) 消えないかを判定する。

    Returns (status, witness)
      status : 'proved'  -- V 上 (除外球の外) で expr != 0
               'refuted' -- 零点が存在する (witness にその点)
               'unknown' -- ソルバが決定できなかった
    """
    if not _HAS_Z3:
        return ("unknown", None)
    expr = sp.expand(expr)
    if not expr.free_symbols:
        return ("proved" if expr != 0 else "refuted", None)
    zvars = {v: _z3.Real(f"y{i}") for i, v in enumerate(gens)}
    s = _z3.Solver()
    s.set("timeout", timeout_ms)
    for v, b in zip(gens, box.bounds):
        zv = zvars[v]
        bb = _z3.RealVal(f"{sp.Rational(b).p}/{sp.Rational(b).q}")
        s.add(zv >= -bb, zv <= bb)
    try:
        s.add(_to_z3(expr, zvars) == 0)
        for e in fiber:                       # phi(p) = 0 (もとの原点に写る点だけ)
            s.add(_to_z3(sp.expand(e), zvars) == 0)
    except ValueError:
        return ("unknown", None)
    d = _z3.RealVal(f"{sp.Rational(delta).p}/{sp.Rational(delta).q}")
    for pt in exclusions:
        # 無限ノルムで delta 以上離れている (= 除外球の外)
        far = []
        for v in gens:
            c = sp.nsimplify(pt.get(v, 0))
            cz = _to_z3(c, zvars)
            far.append(_z3.Or(zvars[v] - cz >= d, cz - zvars[v] >= d))
        s.add(_z3.Or(*far))
    r = s.check()
    if r == _z3.unsat:
        return ("proved", None)
    if r == _z3.sat:
        m = s.model()
        w = {}
        for v in gens:
            val = m.eval(zvars[v], model_completion=True)
            try:
                w[v] = sp.Rational(str(val.as_fraction()))
            except Exception:
                w[v] = sp.nsimplify(str(val))
        return ("refuted", w)
    return ("unknown", None)


def normal_crossing_on_box(u, k, h, gens, box: Box, *,
                           exclusions: Sequence[Dict] = (),
                           fiber: Sequence = (),
                           delta=sp.Rational(1, 100), timeout_ms: int = 10000):
    """f = y^k * u が箱 V の上で例外因子と正規交差になっているかを判定する。

    u が消えること自体は問題ではない。u = 0 の超曲面が滑らかで、かつその点を
    通る例外因子 {y_j = 0} (j in E = supp(k) 和 supp(h)) と横断的であればよい。
    点 p で正規交差が壊れる条件は「grad u(p) が {e_j : y_j(p) = 0, j in E} の
    張る空間に入る」ことなので、量化子なしの論理式として

        u = 0
        and  (j not in E について)  du/dy_j = 0
        and  (j in E について)      y_j = 0 or du/dy_j = 0

    と書ける。これが UNSAT なら箱全体で正規交差であることの証明になる。

    Returns (status, witness) — 'proved' / 'refuted' / 'unknown'
    """
    if not _HAS_Z3:
        return ("unknown", None)
    u = sp.expand(u)
    if not u.free_symbols:
        return ("proved" if u != 0 else "refuted", None)
    E = {i for i, (ki, hi) in enumerate(zip(k, h)) if ki > 0 or hi > 0}
    zvars = {v: _z3.Real(f"y{i}") for i, v in enumerate(gens)}
    s = _z3.Solver()
    s.set("timeout", timeout_ms)
    for v, b in zip(gens, box.bounds):
        bb = _z3.RealVal(f"{sp.Rational(b).p}/{sp.Rational(b).q}")
        s.add(zvars[v] >= -bb, zvars[v] <= bb)
    try:
        s.add(_to_z3(u, zvars) == 0)
        for e in fiber:                       # phi(p) = 0 に限定する
            s.add(_to_z3(sp.expand(e), zvars) == 0)
        for i, v in enumerate(gens):
            du = sp.expand(sp.diff(u, v))
            dz = _to_z3(du, zvars) if du.free_symbols or du != 0 else _z3.RealVal(0)
            if i in E:
                s.add(_z3.Or(zvars[v] == 0, dz == 0))
            else:
                s.add(dz == 0)
    except ValueError:
        return ("unknown", None)
    d = _z3.RealVal(f"{sp.Rational(delta).p}/{sp.Rational(delta).q}")
    for pt in exclusions:
        far = []
        for v in gens:
            cz = _to_z3(sp.nsimplify(pt.get(v, 0)), zvars)
            far.append(_z3.Or(zvars[v] - cz >= d, cz - zvars[v] >= d))
        s.add(_z3.Or(*far))
    r = s.check()
    if r == _z3.unsat:
        return ("proved", None)
    if r == _z3.sat:
        m = s.model()
        w = {}
        for v in gens:
            val = m.eval(zvars[v], model_completion=True)
            try:
                w[v] = sp.Rational(str(val.as_fraction()))
            except Exception:
                w[v] = sp.nsimplify(str(val))
        return ("refuted", w)
    return ("unknown", None)


# ----------------------------------------------------------------------
# ニュートン多面体の非退化性 (実数体上) と、その場合の厳密な RLCT
# ----------------------------------------------------------------------
def newton_facet_normals(monoms, max_facets: int = 14):
    """Newton(f) = conv(A) + R^n_+ のコンパクトなファセットの法線 w > 0。

    A に各座標方向への十分長いオフセットを足した点集合の凸包を取り、
    外向き法線が全成分負のファセットだけを拾う。
    """
    try:
        import numpy as np
        from scipy.spatial import ConvexHull
    except Exception:
        return None
    A = np.array([list(m) for m in monoms], dtype=float)
    n = A.shape[1]
    if n == 1:
        return [[1]]
    M = float(A.max() + 1) * (n + 2) * 4
    pts = [A]
    for i in range(n):
        off = np.zeros(n)
        off[i] = M
        pts.append(A + off)
    P = np.unique(np.vstack(pts), axis=0)
    try:
        hull = ConvexHull(P, qhull_options="Qx")
    except Exception:
        return None
    out = []
    for eq in hull.equations:
        nu = eq[:n]
        if np.all(nu < -1e-9):                    # 外向き法線が全成分負
            w = -nu / np.abs(nu).max()
            r = [sp.Rational(float(v)).limit_denominator(400) for v in w]
            if any(v <= 0 for v in r):
                continue
            lcm = sp.ilcm(*[v.q for v in r]) if n > 1 else r[0].q
            wi = [int(v * lcm) for v in r]
            g = 0
            for v in wi:
                g = sp.igcd(g, v)
            wi = [v // max(g, 1) for v in wi]
            if wi not in out:
                out.append(wi)
    return out[:max_facets]


def newton_faces(monoms, max_facets: int = 12):
    """コンパクトな面 (facet の交わり) をすべて列挙する。

    法線扇の錐は facet 法線の正結合なので、facet 法線の部分集合の和を
    重みに取って argmin を見れば全ての面が得られる。
    """
    normals = newton_facet_normals(monoms, max_facets)
    if not normals:
        return None
    monoms = [tuple(m) for m in monoms]
    faces = []
    import itertools
    r = min(len(normals), max_facets)
    for size in range(1, r + 1):
        for S in itertools.combinations(range(r), size):
            w = [sum(normals[i][j] for i in S) for j in range(len(monoms[0]))]
            d = min(sum(wj * e for wj, e in zip(w, m)) for m in monoms)
            face = tuple(m for m in monoms
                         if sum(wj * e for wj, e in zip(w, m)) == d)
            if face not in faces:
                faces.append(face)
    return faces


def newton_nondegenerate(f, gens, *, timeout_ms: int = 10000,
                         max_facets: int = 12):
    """f が (実数体上) ニュートン多面体に関して非退化かを判定する。

    各コンパクト面 sigma について f_sigma = sum_{a in sigma} c_a x^a を取り、
        exists x, (全ての x_i != 0) かつ grad f_sigma(x) = 0
    が充足不能であることを Z3 で確かめる。擬斉次性 (Euler の関係式) から
    grad = 0 なら f_sigma = 0 も従うので、勾配だけ見ればよい。

    * 実数体上で直接判定している点が重要。複素数体で退化していても実数上は
      非退化ということがあり、RLCT に必要なのは実数体上の非退化性である。

    Returns (status, failing_face)
    """
    gens = tuple(gens)
    p = sp.Poly(sp.expand(f), *gens)
    faces = newton_faces(p.monoms(), max_facets)
    if faces is None:
        return ("unknown", None)
    if not _HAS_Z3:
        return ("unknown", None)
    coeff = {tuple(m): c for m, c in zip(p.monoms(), p.coeffs())}
    for face in faces:
        if len(face) < 2:
            continue                       # 単項式の面は自動的に非退化
        fs = sum(coeff[m] * sp.prod([v ** e for v, e in zip(gens, m)])
                 for m in face)
        zvars = {v: _z3.Real(f"t{i}") for i, v in enumerate(gens)}
        s = _z3.Solver()
        s.set("timeout", timeout_ms)
        try:
            for v in gens:
                s.add(zvars[v] != 0)
            for v in gens:
                s.add(_to_z3(sp.expand(sp.diff(fs, v)), zvars) == 0)
        except ValueError:
            return ("unknown", face)
        r = s.check()
        if r == _z3.sat:
            return ("refuted", face)
        if r != _z3.unsat:
            return ("unknown", face)
    return ("proved", None)


def rlct_via_newton(f, gens, *, timeout_ms: int = 10000):
    """非退化なら Varchenko の定理で lambda を確定させる高速パス。

    Returns (lambda, status)
      status='proved'  : 非退化が証明され、値は厳密
      'unknown'/'refuted' : 値は上界としてのみ有効 (ブローアップに回すべき)
    """
    from resolve_singularity import newton_rlct
    st, face = newton_nondegenerate(f, gens, timeout_ms=timeout_ms)
    lam = newton_rlct(f, gens)
    return (lam, st)


# ----------------------------------------------------------------------
# 定義域の伝播
# ----------------------------------------------------------------------
def compute_domains(resolution, eps=sp.Integer(1),
                    delta=sp.Rational(1, 100)) -> Dict[str, Box]:
    """各 chart の定義域 (箱) を根から伝播して求める。

    中心を付け替えた chart は「親の箱のうち付け替え点の delta 球」だけを
    担当するので、その箱は半径 delta の立方体になる。親の側はその球を除いた
    領域を担当する (exclusions として同じ delta を使う)。両者を合わせると
    親の箱が過不足なく覆われる。
    """
    gens = resolution.gens
    n = len(gens)
    eps = sp.nsimplify(eps)
    domains: Dict[str, Box] = {}
    nodes = resolution.nodes
    root = next((c for c in nodes.values() if c.parent is None), None)
    if root is None:
        return domains
    domains[root.name] = Box(tuple(eps for _ in range(n)))

    # 深さ順に処理する
    for name in sorted(nodes, key=lambda t: nodes[t].depth):
        ch = nodes[name]
        if name in domains:
            continue
        if ch.parent not in domains:
            continue
        pb = domains[ch.parent]
        st = next((t for t in ch.own_steps()
                   if t.kind in ("blowup", "coordchg", "recenter")), None)
        if st is None:
            domains[name] = pb
            continue
        b = list(pb.bounds)
        if st.kind == "blowup":
            J = [gens.index(v) for v in st.center]
            i = None
            for v, e in st.subs:                       # x_j -> y_i * y_j
                factors = sp.Mul.make_args(e)
                cand = [w for w in factors if w in gens and w != v]
                if cand:
                    i = gens.index(cand[0])
                    break
            if i is None:
                i = J[0]
            for j in J:
                if j != i:
                    b[j] = sp.Integer(1)               # 被覆補題の与える境界
        elif st.kind == "recenter":
            # 付け替え点のまわりの delta 球だけを担当する
            b = [sp.nsimplify(delta) for _ in range(n)]
        elif st.kind == "coordchg":
            for v, e in st.subs:                       # x_v = (y_v - B)/A
                num, den = sp.fraction(sp.together(e))
                A = sp.cancel(den)
                B = sp.cancel(-(sp.expand(num) - v))
                amax = _abs_max(A, gens, pb) if A.free_symbols else abs(sp.nsimplify(A))
                bmax = _abs_max(B, gens, pb) if B.free_symbols else abs(sp.nsimplify(B))
                b[gens.index(v)] = amax * pb.bounds[gens.index(v)] + bmax
        domains[name] = Box(tuple(b))
    return domains


# ----------------------------------------------------------------------
# 証明書
# ----------------------------------------------------------------------
@dataclass
class LeafCertificate:
    name: str
    box: Box
    lam: sp.Rational
    mult: int
    unit_status: str                # 'proved' | 'refuted' | 'unknown'
    jac_status: str
    witness: Optional[Dict] = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.unit_status == "proved" and self.jac_status == "proved"

    def __str__(self) -> str:
        mark = {"proved": "OK", "refuted": "NG", "unknown": "??"}
        return (f"[{mark[self.unit_status]}/{mark[self.jac_status]}] "
                f"{self.name:<22} lambda={str(self.lam):<7} m={self.mult} "
                f"V={self.box}" + (f"  反例 {self.witness}" if self.witness else "")
                + (f"  {self.note}" if self.note else ""))


@dataclass
class Certificate:
    status: str                     # 'proved' | 'unknown' | 'refuted'
    rlct: Optional[sp.Rational]     # proved のときだけ値が入る
    best_known: Optional[sp.Rational]
    multiplicity: Optional[int]
    leaves: List[LeafCertificate] = field(default_factory=list)
    covering: List[Tuple[str, str, str]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def report(self) -> str:
        L = [f"############ 証明書: {self.status} ############"]
        if self.status == "proved":
            L.append(f"  lambda = {self.rlct}, m = {self.multiplicity} "
                     "(全 chart で単元条件と被覆が確立)")
        else:
            L.append(f"  lambda は返しません (到達した最小値は {self.best_known} "
                     "ですが、正しさが確認できていません)")
        L.append(f"  葉 {len(self.leaves)} 件 [単元/ヤコビアン]:")
        for lf in self.leaves:
            L.append("    " + str(lf))
        if self.covering:
            L.append("  被覆の状態:")
            for name, st, why in self.covering:
                L.append(f"    [{st}] {name}  {why}")
        for r in self.reasons:
            L.append(f"  [理由] {r}")
        return "\n".join(L)

    def print_report(self) -> None:
        print(self.report())


def certify(resolution, *, eps=sp.Integer(1), delta=sp.Rational(1, 100),
            timeout_ms: int = 10000, check_jacobian: bool = True,
            localize: bool = True) -> Certificate:
    """解消の結果を検証し、三値の証明書を返す。

    eps      : 対象とする原点近傍 {|x_j| <= eps}
    delta    : 中心を付け替えた点のまわりで除外する半径 (無限ノルム)
    localize : 検査を phi(p) = 0 を満たす点 (もとの原点のファイバー) に限る。
               台の関数 psi が原点近傍にしか台を持たないので、引き戻した
               振幅はファイバーの外で消える。正規交差性は開条件で、
               ファイバー ∩ 箱 はコンパクトなので、ファイバー上で成り立てば
               その近傍でも成り立つ。既定で有効。
    """
    gens = resolution.gens
    domains = compute_domains(resolution, eps, delta)
    leaves: List[LeafCertificate] = []
    reasons: List[str] = []
    covering: List[Tuple[str, str, str]] = []

    if not _HAS_Z3:
        reasons.append("z3 が無いため単元条件を検証できません "
                       "(pip install z3-solver)。")

    # 節点ごとに、そこから派生した付け替え点を集めておく
    recenter_pts: Dict[str, List[Dict]] = {}
    for c in resolution.nodes.values():
        st = next((t for t in c.own_steps() if t.kind == "recenter"), None)
        if st and c.parent:
            recenter_pts.setdefault(c.parent, []).append(
                {v: sp.nsimplify(e - v) for v, e in st.subs})

    # --- 葉ごとの単元条件 --------------------------------------------
    for ch in resolution.charts:
        box = domains.get(ch.name)
        lam, m = ch.local_rlct()
        if box is None:
            leaves.append(LeafCertificate(ch.name, Box(()), lam, m,
                                          "unknown", "unknown",
                                          note="定義域を追跡できません"))
            continue
        fib = [ch.phi[v] for v in gens] if localize else ()
        # この葉から中心を付け替えた点は、子の chart が担当するので除外する
        excl = recenter_pts.get(ch.name, [])
        us, w = normal_crossing_on_box(ch.f_rest(), ch.k, ch.h, gens, box,
                                       fiber=fib, exclusions=excl, delta=delta,
                                       timeout_ms=timeout_ms)
        js, jw = "proved", None
        if check_jacobian:
            hmono = sp.prod([v ** e for v, e in zip(gens, ch.h)])
            det = sp.Matrix([[sp.diff(ch.phi[v], w2) for w2 in gens]
                             for v in gens]).det()
            q = sp.cancel(sp.together(sp.expand(det) / hmono)) if hmono != 0 \
                else sp.S.Zero
            num, den = sp.fraction(q)
            if den.free_symbols:
                js, jw = "unknown", None
            else:
                js, jw = nonzero_on_box(num, gens, box, fiber=fib,
                                        exclusions=excl, delta=delta,
                                        timeout_ms=timeout_ms)
        leaves.append(LeafCertificate(ch.name, box, lam, m, us, js,
                                      witness=w or jw))

    # --- 被覆の状態 ---------------------------------------------------
    for name, ch in sorted(resolution.nodes.items()):
        kids = [c for c in resolution.nodes.values() if c.parent == name]
        if not kids:
            continue
        kinds = {next((t.kind for t in c.own_steps()
                       if t.kind in ("blowup", "coordchg", "recenter")), "?")
                 for c in kids}
        if kinds <= {"blowup"}:
            wts = set()
            for c in kids:
                st3 = next((t for t in c.own_steps() if t.kind == "blowup"), None)
                if st3 and st3.weights:
                    wts |= {tuple(st3.weights)}
            if any(any(v != 1 for v in t) for t in wts):
                covering.append((name, "proved",
                                 f"重み付きブローアップの chart 族が全射 "
                                 f"(重み {sorted(wts)}; |x_j|^(1/w_j) が最大の "
                                 "chart と符号を選ぶ。Lean 側は未形式化)"))
            else:
                covering.append((name, "proved", "ブローアップの chart 族が全射"))
        elif kinds <= {"coordchg"}:
            covering.append((name, "proved", "座標変換は全単射"))
        elif "recenter" in kinds:
            # 付け替え点の近傍を除いて u が消えないことを確認する
            box = domains.get(name)
            pts = []
            for c in kids:
                st = next((t for t in c.own_steps() if t.kind == "recenter"), None)
                if st:
                    pts.append({v: sp.nsimplify(e - v) for v, e in st.subs})
            if box is None:
                covering.append((name, "unknown", "定義域を追跡できません"))
                continue
            st2, w2 = normal_crossing_on_box(
                ch.f_rest(), ch.k, ch.h, gens, box, exclusions=pts,
                fiber=([ch.phi[v] for v in gens] if localize else ()),
                delta=delta, timeout_ms=timeout_ms)
            if st2 == "proved":
                covering.append((name, "proved",
                                 f"零点は付け替えた {len(pts)} 点の近傍のみ"))
            else:
                covering.append((name, st2,
                                 "付け替えた点以外にも零点がありえます"
                                 + (f" 例 {w2}" if w2 else "")))
        else:
            covering.append((name, "unknown", f"混在: {kinds}"))

    # --- 総合判定 -----------------------------------------------------
    best = resolution.rlct
    bad = [lf for lf in leaves if lf.unit_status == "refuted"
           or lf.jac_status == "refuted"]
    unk = [lf for lf in leaves if not lf.ok and lf not in bad]
    cov_bad = [c for c in covering if c[1] != "proved"]

    if resolution.n_pruned:
        reasons.append("枝刈りが行われています。証明モードでは prune=False "
                       "を使ってください (下界の仮定が入るため)。")

    if bad:
        status = "refuted"
        reasons.append(f"{len(bad)} 個の chart で u または v の零点が "
                       "定義域内に見つかりました。正規交差になっていません。")
    elif unk or cov_bad or resolution.n_pruned or not _HAS_Z3:
        status = "unknown"
        if unk:
            reasons.append(f"{len(unk)} 個の chart で単元条件が未決定です。")
        if cov_bad:
            reasons.append(f"{len(cov_bad)} 個の節点で被覆が確立していません。")
    else:
        status = "proved"

    return Certificate(
        status=status,
        rlct=(best if status == "proved" else None),
        best_known=best,
        multiplicity=(resolution.multiplicity if status == "proved" else None),
        leaves=leaves, covering=covering, reasons=reasons)


# ----------------------------------------------------------------------
# 高速パス付きの入口
# ----------------------------------------------------------------------
def rlct_certified(f, gens=None, *, timeout_ms: int = 20000,
                   max_depth: int = 25, verbose: bool = False, **kw):
    """lambda を「証明付きで」求める。返り値は (lambda, status, 経路)。

    1. ニュートン多面体に関して実数体上で非退化かを Z3 で判定する。
       非退化なら Varchenko の定理から lambda = 1/(ニュートン距離) が厳密。
       LP だけなので多変数でも速い。
    2. 退化している/判定できない場合は、重み付きブローアップで解消して
       certify() にかける。
    status は 'proved' / 'unknown' / 'refuted'。proved でなければ lambda は
    参考値であり、正しさは保証されない。
    """
    from resolve_singularity import resolve_singularities, ResolutionFailure

    if gens is None:
        gens = sorted(sp.sympify(f).free_symbols, key=lambda s: s.name)
    gens = tuple(gens)

    lam, st = rlct_via_newton(f, gens, timeout_ms=timeout_ms)
    if st == "proved":
        if verbose:
            print(f"  ニュートン非退化 -> lambda = {lam} (厳密)")
        return (lam, "proved", "newton")

    try:
        res = resolve_singularities(f, gens, prune=False, weighted=True,
                                    max_depth=max_depth, **kw)
    except ResolutionFailure as e:
        return (None, "unknown", f"解消できず ({e.reason})")
    cert = certify(res, timeout_ms=timeout_ms)
    return (cert.rlct if cert.status == "proved" else res.rlct,
            cert.status, "blowup+certify")


# ----------------------------------------------------------------------
# デモ
# ----------------------------------------------------------------------
def _demo():
    from resolve_singularity import resolve_singularities

    x, y, z = sp.symbols("x y z", real=True)

    cases = [
        ("x^2 + y^2", x**2 + y**2, (x, y), {}),
        ("x^2 * y^2", x**2 * y**2, (x, y), {}),
        ("(x - y)^2", (x - y) ** 2, (x, y), {}),
        ("(x - y)^2 [付け替え経路]", (x - y) ** 2, (x, y),
         dict(smooth_coords=False)),
        ("x^2 + y^3", x**2 + y**3, (x, y), {}),
        ("(x^2 - y^3)^2", (x**2 - y**3) ** 2, (x, y), {}),
        ("x^2 + y^2 + z^2", x**2 + y**2 + z**2, (x, y, z), {}),
        ("x^2 + y^3 + z^4", x**2 + y**3 + z**4, (x, y, z), {}),
        ("(xy + z^2)^2", (x * y + z**2) ** 2, (x, y, z), {}),
    ]
    print("=" * 72)
    print("三値の証明書 (prune=False が証明モード)")
    print("=" * 72)
    for label, f, g, kw in cases:
        res = resolve_singularities(f, g, prune=False, max_depth=25, **kw)
        cert = certify(res)
        mark = {"proved": "証明", "unknown": "未決定", "refuted": "反例"}[cert.status]
        print(f"  {label:<26} lambda={str(res.rlct):<7} -> {mark}"
              f"  (返す値: {cert.rlct})")
        if cert.status != "proved":
            for lf in cert.leaves:
                if not lf.ok:
                    print("      " + str(lf))
            for nm, st, why in cert.covering:
                if st != "proved":
                    print(f"      [{st}] 被覆 {nm}: {why}")

    print("\n" + "=" * 72)
    print("詳細な報告の例: x^2 + y^3")
    certify(resolve_singularities(x**2 + y**3, (x, y), prune=False)).print_report()

    print("\n" + "=" * 72)
    print("判定の要: 単元が消えること自体は問題ではなく、正規交差が壊れるかどうか")
    B = Box((sp.Integer(1), sp.Integer(1)))
    print("  (1-y)^2, k=(2,0), h=(1,0)  零点 y=1 で二重 -> 正規交差が壊れる:")
    print("     ", normal_crossing_on_box((1 - y) ** 2, [2, 0], [1, 0], (x, y), B))
    print("  y+1, k=(6,2), h=(4,1)  零点は滑らかで例外因子と横断的 -> 問題なし:")
    print("     ", normal_crossing_on_box(y + 1, [6, 2], [4, 1], (x, y), B))
    print("  同じ (1-y)^2 でも、もとの原点に写る点に限れば (localize) 判定が変わる:")
    print("      ファイバー x*y = 0, y = 0 の上:",
          normal_crossing_on_box((1 - y) ** 2, [2, 0], [1, 0], (x, y), B,
                                 fiber=[x * y, y]))

    print("\n" + "=" * 72)
    print("ニュートン多面体の非退化性 (実数体上) の判定と高速パス")
    print("=" * 72)
    for label, f, g in [("x^2 + y^3", x**2 + y**3, (x, y)),
                        ("x^3 + y^4 + z^5", x**3 + y**4 + z**5, (x, y, z)),
                        ("x^2y^2+y^2z^2+z^2x^2",
                         x**2*y**2 + y**2*z**2 + z**2*x**2, (x, y, z)),
                        ("(x - y)^2", (x - y) ** 2, (x, y)),
                        ("(xy + z^2)^2", (x * y + z**2) ** 2, (x, y, z))]:
        st, face = newton_nondegenerate(f, g, timeout_ms=8000)
        lam, status, how = rlct_certified(f, g, timeout_ms=8000)
        print(f"  {label:<22} 非退化: {st:<8} -> lambda={str(lam):<7} "
              f"({status}, {how})")

    print("\n" + "=" * 72)
    print("枝刈りを有効にすると、下界の仮定が入るので unknown になる:")
    res = resolve_singularities(x**2 + y**3, (x, y), prune=True)
    c = certify(res)
    print(f"  {c.status}: {c.reasons}")


if __name__ == "__main__":
    _demo()
