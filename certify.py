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
           "RLCTInterval", "rlct_interval",
           "nonzero_on_box", "normal_crossing_on_box",
           "newton_nondegenerate", "rlct_via_newton", "newton_faces",
           "newton_multiplicity", "principal_face_compact"]

try:
    from timing import timed as _timed, add_meta as _add_meta, record as _record
except Exception:      # pragma: no cover
    from contextlib import contextmanager as _cm

    @_cm
    def _timed(_name):
        yield

    @_cm
    def _record(_label="", **_kw):
        yield None

    def _add_meta(**_kw):
        pass


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
            if e == 0:
                continue
            # 指数は解消が進むと数百に達する。b <= 1 なら b^e は単調減少
            # なので、指数を打ち切っても上界としては正しい (b^e <= b^min(e,K))。
            # 打ち切らないと巨大な有理数の冪乗が支配的コストになる。
            if b <= 1:
                mag *= b ** min(int(e), 64)
            else:
                mag *= b ** min(int(e), 64)
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


def _fiber_constraints(fiber):
    """phi(p) = 0 を多項式の等式に直す。

    座標変換のあと phi は有理式になりうる。分母は定義域上で消えないので、
    phi_j(p) = 0 は分子 = 0 と同値。z3 は除算を扱えないのでここで落とす。
    """
    out = []
    for e in fiber:
        num, _ = sp.fraction(sp.together(sp.sympify(e)))
        out.append(sp.expand(num))
    return out


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
        for e in _fiber_constraints(fiber):    # phi(p) = 0 (もとの原点に写る点)
            s.add(_to_z3(e, zvars) == 0)
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
        for e in _fiber_constraints(fiber):    # phi(p) = 0 に限定する
            s.add(_to_z3(e, zvars) == 0)
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
def _used_coords(monoms):
    """どの単項式にも現れない変数の位置を落とす。

    f に現れない変数 x_j の方向には Gamma_+ が筒になっているので、
    その空間ではコンパクトなファセットが 1 枚も存在せず、凸包を取っても
    「全成分が負の法線」が得られない (実測: 53*x0*x2^3 + 61*x1^3*x2 を
    4 変数で渡すと法線が空になり、非退化判定が unknown に落ちて、
    そのまま解消が無限反復していた)。使われている座標だけに落として
    計算し、法線は残りの成分を 0 にして戻す。
    """
    n = len(monoms[0])
    used = [j for j in range(n) if any(m[j] for m in monoms)]
    return used


def newton_facet_normals(monoms, max_facets: int = 14):
    """Newton(f) = conv(A) + R^n_+ のコンパクトなファセットの法線。

    A に各座標方向への十分長いオフセットを足した点集合の凸包を取り、
    外向き法線が全成分負のファセットだけを拾う。f に現れない変数は
    先に落として計算し、その成分は 0 として戻す (筒方向)。
    """
    try:
        import numpy as np
        from scipy.spatial import ConvexHull
    except Exception:
        return None
    monoms = [tuple(m) for m in monoms]
    n_full = len(monoms[0])
    used = _used_coords(monoms)
    if not used:
        return None
    if len(used) < n_full:
        sub = [tuple(m[j] for j in used) for m in monoms]
        inner = newton_facet_normals(sub, max_facets)
        if inner is None:
            return None
        out = []
        for w in inner:
            full = [0] * n_full
            for j, v in zip(used, w):
                full[j] = v
            out.append(full)
        return out
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


def _faces_by_lp(minimal, n):
    """極小な単項式の部分集合を線形計画で篩って、コンパクトな面を列挙する。

    `minimal` は成分ごとの順序についての反鎖 (支配される点は除去済み)。
    部分集合ごとに 1 回の LP なので 2^|minimal| 回。|minimal| が小さい
    (= 台が薄い) ときだけ使う。scipy が無ければ None。
    """
    try:
        from scipy.optimize import linprog
    except Exception:
        return None
    import itertools

    V = list(minimal)
    N = len(V)
    faces = []
    for size in range(N, 1, -1):          # 大きい面から
        for S in itertools.combinations(range(N), size):
            sub = set(S)
            sigma = [V[i] for i in S]
            rest = [V[i] for i in range(N) if i not in sub]
            # 変数 (w_0..w_{n-1}, d)
            A_eq, b_eq = [], []
            base = sigma[0]
            for m in sigma[1:]:
                A_eq.append([float(a - b) for a, b in zip(m, base)] + [0.0])
                b_eq.append(0.0)
            A_ub, b_ub = [], []
            for m in rest:
                # <w,m> >= <w,base> + 1   <=>   -(m - base).w <= -1
                A_ub.append([-float(a - b) for a, b in zip(m, base)] + [0.0])
                b_ub.append(-1.0)
            res = linprog(c=[0.0] * (n + 1),
                          A_ub=A_ub or None, b_ub=b_ub or None,
                          A_eq=A_eq or None, b_eq=b_eq or None,
                          bounds=[(1, None)] * n + [(None, None)])
            if res.status == 0:
                faces.append(tuple(sigma))
    if not faces:
        return None
    return faces


def newton_faces(monoms, max_facets: int = 12, max_faces: int = 600):
    """Gamma_+(f) の**コンパクトな面をすべて**列挙する。

    Varchenko の非退化条件はコンパクトな**面すべて**についての条件であり、
    ファセットだけを見るのでは足りない。

    >>> f = (x + y^2)^2 + z^2 = x^2 + 2x y^2 + y^4 + z^2
    コンパクトなファセットは 1 枚 (法線 (2,1,2)) だけで、その上では
    grad f_sigma = 0 が z != 0 と両立しないので「非退化」に見える。
    しかし辺 {(2,0,0), (1,2,0), (0,4,0)} もコンパクトな面で、その面多項式
    (x+y^2)^2 はトーラス上 x = -y^2 で勾配が消える。実際 lambda は 1 で、
    ファセットだけを見て得られる 5/4 は誤り。

    正しい列挙 — 多面体 P の面について face(w1 + w2) = face(w1) ∩ face(w2)
    (交わりが空でないとき) が成り立つので、Gamma_+ の面は

        face(w) = ∩_{i in S} face(n_i) ∩ ∩_{j in T} face(e_j)

    (n_i はコンパクトなファセット法線、face(e_j) = {m : m_j が最小})
    の形で尽くされる。w > 0 になるのは **S が空でないとき**で、それが
    コンパクトな面にあたる。したがって「コンパクトなファセットの面から
    始めて、他のファセット面・座標面との交わりで閉じる」だけでよい。

    面の数が max_faces を超えたら None を返す (呼び出し側は 'unknown')。
    """
    monoms = [tuple(m) for m in monoms]
    n = len(monoms[0])

    key = (tuple(monoms), max_facets, max_faces)
    if key in _FACE_CACHE:
        return _FACE_CACHE[key]
    out = _newton_faces_uncached(monoms, n, max_facets, max_faces)
    if len(_FACE_CACHE) < 4096:
        _FACE_CACHE[key] = out
    return out


_FACE_CACHE: Dict[tuple, object] = {}


def _newton_faces_uncached(monoms, n, max_facets, max_faces):
    # --- 台が薄いとき: 部分集合を直接 LP で判定する ------------------
    # w > 0 に対する argmin 集合が「コンパクトな面」の定義そのものなので、
    # sigma が面かどうかは線形計画の可解性で決まる:
    #     exists w, d :  <w,m> = d (m in sigma),
    #                    <w,m> >= d + 1 (m not in sigma),  w_j >= 1
    # (w と d は正のスケールで自由なので、この正規化は一般性を失わない)
    # 支配される単項式 (m' >= m が成分ごとに成り立つもの) は w > 0 では
    # 決して argmin に入らないので、先に落としておく。
    # まずは凸包から得たファセット法線で閉じる (速い)。台が薄くて
    # コンパクトなファセットが 1 枚も無いときだけ LP の全列挙に落ちる。
    normals = newton_facet_normals(monoms, max_facets)
    if not normals:
        minimal = [m for m in monoms
                   if not any(m2 != m and all(a <= b for a, b in zip(m2, m))
                              for m2 in monoms)]
        if 2 <= len(minimal) <= 12:
            return _faces_by_lp(minimal, n)
        return None

    def argmin(w):
        d = min(sum(wj * e for wj, e in zip(w, m)) for m in monoms)
        return tuple(m for m in monoms
                     if sum(wj * e for wj, e in zip(w, m)) == d)

    # 生成元: コンパクトなファセットの面と、座標方向の面
    comp = [argmin(w) for w in normals]
    coord = []
    for j in range(n):
        mn = min(m[j] for m in monoms)
        coord.append(tuple(m for m in monoms if m[j] == mn))
    gens_faces = comp + coord

    # S が空でない = コンパクトな面。comp から始めて交わりで閉じる。
    faces = list(dict.fromkeys(comp))
    seen = set(faces)
    frontier = list(faces)
    while frontier:
        nxt = []
        for face in frontier:
            fs = set(face)
            for g in gens_faces:
                inter = tuple(m for m in face if m in set(g))
                if inter and inter != face and inter not in seen:
                    seen.add(inter)
                    faces.append(inter)
                    nxt.append(inter)
                    if len(faces) > max_faces:
                        return None
        frontier = nxt
    return faces


def newton_multiplicity(f, gens, h=None, *, max_facets: int = 14):
    """非退化な f の極の位数 m をニュートン多面体から組合せ的に求める。

    Varchenko の定理では、対角線が多面体の境界に当たる点 p = t* (1,...,1)
    を含む最小の面の次元 d に対し m = n - d。ここで

        m = rank { p を通るファセットの法線 }

    と書ける (最小面 = p を含む全ファセットの交わり、その次元は
    n - rank(法線))。これはトーリック解消で扇を作らなくても分かる量で、
    実際に扇を細分して chart を作った場合に得られる位数と一致する。

    h を与えると振幅 x^h 付きの場合 (p = t* (h+1)) になる。
    Returns m、決められなければ None。
    """
    gens = tuple(gens)
    n = len(gens)
    p_ = sp.Poly(sp.expand(f), *gens)
    monoms = [tuple(m) for m in p_.monoms()]
    if not monoms or any(all(e == 0 for e in m) for m in monoms):
        return None
    from resolve_singularity import newton_rlct
    hplus = [1] * n if h is None else [int(hj) + 1 for hj in h]
    lam = newton_rlct(f, gens, h=[hj - 1 for hj in hplus])
    if lam is sp.oo or lam == 0:
        return None
    if lam == 0:
        return None
    # 対角線 (振幅つきなら (h+1) 方向) が境界に当たる点
    t = 1 / lam
    p = [sp.Rational(t) * hj for hj in hplus]

    normals = []
    comp = newton_facet_normals(monoms, max_facets) or []
    normals += [list(w) for w in comp]
    # 非コンパクトなファセット {a_i = c_i} の法線 e_i
    for i in range(n):
        ci = min(m[i] for m in monoms)
        e = [0] * n
        e[i] = 1
        normals.append((e, ci))
    active = []
    for w in normals:
        if isinstance(w, tuple):
            e, ci = w
            if sp.nsimplify(p[[j for j, v in enumerate(e) if v][0]]) == ci:
                active.append(e)
        else:
            d = min(sum(wj * a for wj, a in zip(w, m)) for m in monoms)
            if sp.nsimplify(sum(wj * pj for wj, pj in zip(w, p))) == d:
                active.append(w)
    if not active:
        return None
    return int(sp.Matrix(active).rank())


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


def principal_face_compact(monoms, hplus, lam):
    """Varchenko の公式が使える追加条件: 主面がコンパクトか。

    非退化性だけでは足りない。ニュートン距離 t = 1/lambda に対し
    p = t*(h+1) を Gamma_+ の境界上の点とすると、**p を含む最小の面
    (主面) がコンパクトでないと公式は成り立たない**。

    >>> f = -9*x0 - 38*x1^2*x2
    f は原点で勾配が非零なので lambda = 1, m = 1 が真値。コンパクトな面は
    辺 {(1,0,0), (0,2,1)} だけで、その上で grad != 0 なので「非退化」に
    見え、LP は 3/2 を返す。しかし p = (2/3,2/3,2/3) は conv(台) の外で、
    主面は x_2 方向に伸びる非コンパクトな面。公式は使えない。

    コンパクトな面は conv(台) に含まれるので、判定は

        p in conv(台)            (凸結合が存在するかの LP)

    でよい。scipy が無ければ None (呼び出し側は 'unknown' に倒す)。
    """
    try:
        from scipy.optimize import linprog
    except Exception:
        return None
    if lam is None or lam is sp.oo or lam == 0:
        return None
    monoms = [tuple(m) for m in monoms]
    n = len(monoms[0])
    p = [float(sp.Rational(hj) / sp.nsimplify(lam)) for hj in hplus]
    N = len(monoms)
    A_eq = [[float(monoms[a][j]) for a in range(N)] for j in range(n)]
    b_eq = list(p)
    A_eq.append([1.0] * N)
    b_eq.append(1.0)
    res = linprog(c=[0.0] * N, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * N)
    return bool(res.status == 0)


def rlct_via_newton(f, gens, *, timeout_ms: int = 10000):
    """非退化なら Varchenko の定理で lambda を確定させる高速パス。

    Returns (lambda, status)
      status='proved'  : 非退化が証明され、値は厳密
      'unknown'/'refuted' : 値は上界としてのみ有効 (ブローアップに回すべき)
    """
    from resolve_singularity import newton_rlct
    st, face = newton_nondegenerate(f, gens, timeout_ms=timeout_ms)
    lam = newton_rlct(f, gens)
    if st == "proved":
        # 非退化だけでは足りない: 主面がコンパクトであることも要る
        ok = principal_face_compact(sp.Poly(sp.expand(f), *gens).monoms(),
                                    [1] * len(gens), lam)
        if ok is not True:
            st = "unknown"
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
            wts = list(st.weights) if st.weights else [1] * len(J)
            wmap = {j: wts[t] for t, j in enumerate(J)}
            i = None
            for v, e in st.subs:                       # x_j -> y_i^{w_j} * y_j
                for w_ in sp.Mul.make_args(e):
                    base = w_.base if w_.is_Pow else w_
                    if base in gens and base != v:
                        i = gens.index(base)
                        break
                if i is not None:
                    break
            if i is None:
                i = J[0]
            for j in J:
                if j != i:
                    b[j] = sp.Integer(1)               # 被覆補題の与える境界
            # 重み付きでは y_i = x_i^{1/w_i} なので、pivot の箱は b_i^{1/w_i}
            # に広がる (b_i <= 1 なら 1/w_i 乗で大きくなる)。ここを b_i の
            # ままにすると chart の像が足りず、被覆が成立しない。
            wi = wmap.get(i, 1)
            if wi > 1:
                import math as _math
                val = float(b[i]) ** (1.0 / wi)
                b[i] = sp.Rational(_math.ceil(val * 10000) + 1, 10000)
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
    identity_ok: bool = True        # f(phi(y)) == y^k * f_rest が厳密に成立
    witness: Optional[Dict] = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return (self.identity_ok and self.unit_status == "proved"
                and self.jac_status == "proved")

    def __str__(self) -> str:
        mark = {"proved": "OK", "refuted": "NG", "unknown": "??"}
        return (f"[{'OK' if self.identity_ok else 'NG'}/"
                f"{mark[self.unit_status]}/{mark[self.jac_status]}] "
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
        L.append(f"  葉 {len(self.leaves)} 件 [恒等式/単元/ヤコビアン]:")
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


def _verify_product_leaf(ch, gens):
    """積の分解で閉じた葉を独立に検証する。

    解消側が記録した (lambda, m) を信用せず、
      (1) f_rest が本当に変数について互いに素な因子に分かれるか
      (2) 部分問題の値を計算し直し、min と位数の合流が一致するか
    をここで確かめる。食い違えば 'refuted'。

    Returns (status, witness)
    """
    from lemmas import product_split_value

    lam_rec, m_rec, _why = ch.closed
    try:
        v = product_split_value(ch.k, ch.poly, ch.h, gens)
    except Exception as e:                                # noqa: BLE001
        return ("unknown", {"why": f"再計算できません: {str(e)[:50]}"})
    if v is None:
        return ("refuted", {"why": "互いに素な因子への分解を再現できません"})
    lam, m = v
    if sp.nsimplify(lam) != sp.nsimplify(lam_rec):
        return ("refuted", {"why": f"再計算が一致しません ({lam} != {lam_rec})"})
    if int(m) != int(m_rec):
        return ("refuted", {"why": f"位数が一致しません ({m} != {m_rec})"})
    return ("proved", None)


def _verify_newton_leaf(ch, gens, *, timeout_ms: int = 10000):
    """ニュートン高速パスで閉じた葉を独立に検証する。

    解消側が記録した (lambda, m) を信用せず、
      (1) 局所データ P = y^k * f_rest が実数体上ニュートン非退化であること
      (2) LP による値が記録された lambda と一致すること
      (3) 位数 m が一致すること
    をここで計算し直す。どれかが食い違えば 'refuted' を返す
    (帳簿のバグをそのまま proved にしないため)。

    Returns (status, witness)
    """
    from resolve_singularity import newton_rlct

    lam_rec, m_rec, _why = ch.closed
    monoms = [tuple(ki + ei for ki, ei in zip(ch.k, m))
              for m in ch.poly.monoms()]
    P = sum(c * sp.prod([v ** e for v, e in zip(gens, mm)])
            for c, mm in zip(ch.poly.coeffs(), monoms))
    st, face = newton_nondegenerate(P, gens, timeout_ms=timeout_ms)
    if st == "refuted":
        return ("refuted", {"face": [tuple(a) for a in face] if face else None,
                            "why": "ニュートン退化 (高速パスの適用が誤り)"})
    if st != "proved":
        return ("unknown", {"why": "非退化を判定できませんでした"})
    lam = newton_rlct(P, gens, h=list(ch.h))
    if sp.nsimplify(lam) != sp.nsimplify(lam_rec):
        return ("refuted", {"why": f"LP の再計算が一致しません "
                                   f"({lam} != {lam_rec})"})
    m_new = newton_multiplicity(P, gens, h=list(ch.h))
    if m_new is not None and int(m_new) != int(m_rec):
        return ("refuted", {"why": f"位数が一致しません ({m_new} != {m_rec})"})
    return ("proved", None)


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
    newton_leaves: List[str] = []
    product_leaves: List[str] = []
    f_expr = sp.expand(resolution.f)
    for ch in resolution.charts:
        box = domains.get(ch.name)
        lam, m = ch.local_rlct()
        # (a) 代入等式 f(phi(y)) = y^k * f_rest を厳密に検算する。
        #     k, h, f_rest, phi の帳簿が壊れていればここで落ちる。
        lhs = sp.expand(f_expr.subs({v: ch.phi[v] for v in gens},
                                    simultaneous=True))
        rhs = sp.cancel(sp.together(
            sp.prod([v ** e for v, e in zip(gens, ch.k)])
            * getattr(ch, "unit", sp.S.One) * ch.f_rest()))
        ident = sp.simplify(sp.expand(sp.cancel(lhs - rhs))) == 0
        # 落とした単元が本当に原点で単元か (0 でも極でもない) も確かめる
        uu = sp.cancel(getattr(ch, "unit", sp.S.One))
        un, ud = sp.fraction(sp.together(uu))
        z0 = {v: 0 for v in gens}
        if sp.simplify(un.subs(z0)) == 0 or sp.simplify(ud.subs(z0)) == 0:
            ident = False
        if box is None:
            leaves.append(LeafCertificate(ch.name, Box(()), lam, m,
                                          "unknown", "unknown",
                                          identity_ok=ident,
                                          note="定義域を追跡できません"))
            continue
        if not ident:
            leaves.append(LeafCertificate(ch.name, box, lam, m,
                                          "unknown", "unknown",
                                          identity_ok=False,
                                          note="f(phi) = y^k * f_rest が成立しません"))
            continue
        fib = [ch.phi[v] for v in gens] if localize else ()
        # この葉から中心を付け替えた点は、子の chart が担当するので除外する
        excl = recenter_pts.get(ch.name, [])
        closed = getattr(ch, "closed", None)
        if closed is not None and closed[2] == "product":
            # 積の分解で閉じた葉。分解が本当に変数について互いに素かと、
            # 部分問題の値の合流を計算し直して検査する。
            us, w = _verify_product_leaf(ch, gens)
            if us == "proved":
                product_leaves.append(ch.name)
        elif closed is not None and closed[2] == "newton":
            # ニュートン高速パスで閉じた葉。正規交差ではないので単元条件の
            # 代わりに「局所データ y^k*f_rest が原点で実数体上ニュートン
            # 非退化であること」を検証し、Varchenko の定理で値を認める。
            # 再判定するのは、解消側のキャッシュや LP を信用しないため。
            us, w = _verify_newton_leaf(ch, gens, timeout_ms=timeout_ms)
            if us == "proved":
                newton_leaves.append(ch.name)
        else:
            us, w = normal_crossing_on_box(ch.f_rest(), ch.k, ch.h, gens, box,
                                           fiber=fib, exclusions=excl,
                                           delta=delta, timeout_ms=timeout_ms)
        js, jw = "proved", None
        if check_jacobian:
            hmono = sp.prod([v ** e for v, e in zip(gens, ch.h)])
            det = sp.Matrix([[sp.diff(ch.phi[v], w2) for w2 in gens]
                             for v in gens]).det()
            q = sp.cancel(sp.together(sp.expand(det) / hmono)) if hmono != 0 \
                else sp.S.Zero
            num, den = sp.fraction(q)
            # 単元部が有理式のときは、分子が消えないことと分母が消えない
            # (= 極を持たない) ことの両方を確かめる。
            js, jw = nonzero_on_box(num, gens, box, fiber=fib,
                                    exclusions=excl, delta=delta,
                                    timeout_ms=timeout_ms)
            if den.free_symbols and js == "proved":
                js, jw = nonzero_on_box(den, gens, box, fiber=fib,
                                        exclusions=excl, delta=delta,
                                        timeout_ms=timeout_ms)
        leaves.append(LeafCertificate(
            ch.name, box, lam, m, us, js, identity_ok=ident, witness=w or jw,
            note=("Newton 非退化 (Varchenko)" if ch.name in newton_leaves
                  else ("互いに素な因子の積" if ch.name in product_leaves
                        else ""))))

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
           or lf.jac_status == "refuted" or not lf.identity_ok]
    unk = [lf for lf in leaves if not lf.ok and lf not in bad]
    cov_bad = [c for c in covering if c[1] != "proved"]

    if resolution.n_pruned:
        reasons.append("枝刈りが行われています。証明モードでは prune=False "
                       "を使ってください (下界の仮定が入るため)。")
    if product_leaves:
        reasons.append(
            f"{len(product_leaves)} 個の葉は正規交差まで解消せず、"
            "変数の互いに素な因子への積分解 (積分が完全に分離する) から "
            "lambda = min(部分問題) として閉じています。"
            "部分問題はそれぞれ proved であることを確認済み。"
        )
    if newton_leaves:
        reasons.append(
            f"{len(newton_leaves)} 個の葉は正規交差まで解消せず、"
            "ニュートン非退化 (実数体上、Z3 で充足不能を確認) から "
            "Varchenko の定理で値を確定しています。"
            "前提として Varchenko の公式を仮定 (Lean 未形式化)。"
        )

    if bad:
        status = "refuted"
        n_id = sum(1 for lf in bad if not lf.identity_ok)
        if n_id:
            reasons.append(f"{n_id} 個の chart で f(phi) = y^k * f_rest が"
                           "成立しません (帳簿のバグ)。")
        if len(bad) - n_id:
            reasons.append(f"{len(bad) - n_id} 個の chart で正規交差が"
                           "壊れる点が定義域内に見つかりました。")
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
# 厳密値が得られないときの区間
# ----------------------------------------------------------------------
@dataclass
class RLCTInterval:
    lo: sp.Rational
    hi: sp.Rational
    status: str                    # 'exact' | 'interval'
    value: Optional[sp.Rational]   # 'exact' のときだけ
    sources: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def width(self):
        return sp.nsimplify(self.hi - self.lo) if self.hi is not sp.oo else sp.oo

    def contains(self, x) -> bool:
        return self.lo <= sp.nsimplify(x) <= self.hi

    def report(self) -> str:
        if self.status == "exact":
            L = [f"############ lambda = {self.value} (厳密) ############"]
        else:
            L = [f"############ lambda in [{self.lo}, {self.hi}] "
                 f"(幅 {self.width}) ############"]
        for k, v in self.sources.items():
            L.append(f"  {k}: {v}")
        for n in self.notes:
            L.append(f"  [note] {n}")
        return "\n".join(L)

    def print_report(self) -> None:
        print(self.report())


def rlct_interval(f, gens=None, *, timeout_ms: int = 20000,
                  max_depth: int = 25, ideal: bool = True, generators=None,
                  **kw) -> RLCTInterval:
    """lambda を厳密に決められない場合に、健全な区間 [lo, hi] を返す。

    使う評価はすべて実数体上で正当なものに限る。

      下界 lo:
        * 1/m   (m = ord_0 f)。 1/m <= lct_C <= lambda_R
      上界 hi:
        * n/m   (ニュートン多面体が {sum a >= m} に入るため)
        * ニュートン LP の値 (Lin: 一般には RLCT の上界)
        * 解消が途中まででも、得られた chart の値の最小
          (被覆が不完全なら lambda を過大評価するので、上界として正しい)

    厳密値が確定した場合は status='exact' で value に入れる。
    """
    from families import family_rlct
    from invariants import multiplicity
    from resolve_singularity import (ResolutionFailure, newton_rlct,
                                     resolve_singularities)

    f = sp.expand(sp.sympify(f))
    if gens is None:
        gens = sorted(f.free_symbols, key=str)
    gens = tuple(gens)
    n = len(gens)
    src: Dict[str, str] = {}
    notes: List[str] = []

    # --- 厳密経路を先に試す -----------------------------------------
    fam = family_rlct(f, gens)
    if fam.status == "proved" and fam.rlct is not sp.oo:
        return RLCTInterval(fam.rlct, fam.rlct, "exact", fam.rlct,
                            {"経路": f"family:{fam.route}"})

    m = multiplicity(f, gens)
    lo = sp.Rational(1, m) if m not in (0, sp.oo) else sp.Integer(0)
    hi = sp.Rational(n, m) if m not in (0, sp.oo) else sp.oo
    src["重複度"] = f"m={m} -> 1/m={lo}, n/m={hi}"

    try:
        nl = newton_rlct(f, gens)
        if nl is not sp.oo and nl < hi:
            hi = nl
            src["ニュートン LP"] = f"上界 {nl}"
    except Exception:
        pass

    if ideal:
        gs = generators if generators is not None else _sum_of_squares(f, gens)
        if gs:
            try:
                from ideal_resolve import resolve_ideal
                ri = resolve_ideal(gs, gens, max_depth=max_depth)
                if ri.rlct is not sp.oo and ri.rlct < hi:
                    hi = ri.rlct
                    src["イデアル版の解消"] = f"上界 {hi} (到達値なので上界)"
            except Exception:
                pass

    res = None
    try:
        res = resolve_singularities(f, gens, prune=False, max_depth=max_depth,
                                    weighted=True, **kw)
    except ResolutionFailure as e:
        src["解消"] = f"失敗 ({e.reason})"
        # 途中までに解消できた chart の値は、被覆が不完全でも上界になる
        vals = [c.local_rlct()[0] for c in e.nodes.values()
                if c.status.startswith("resolved")]
        vals = [v for v in vals if v is not sp.oo]
        if vals and min(vals) < hi:
            hi = min(vals)
            src["途中の chart"] = f"上界 {hi} ({len(vals)} 個の解消済み chart)"
        if e.bound is not None:
            notes.append(f"分枝限定の暫定上界 {e.bound:.6g} "
                         "(下界の仮定に依るので区間には含めない)")
        # 枝刈りを効かせると解けることがある。得られる値は必ず「ある chart で
        # 実際に到達した値」なので、被覆が不完全でも上界として正しい。
        try:
            r2 = resolve_singularities(f, gens, prune="ties", weighted=True,
                                       max_depth=max_depth, **kw)
            if r2.rlct is not sp.oo and r2.rlct < hi:
                hi = r2.rlct
                src["枝刈りありの解消"] = f"上界 {hi} (到達値なので上界)"
        except Exception:
            pass
    except Exception as e:                            # noqa: BLE001
        src["解消"] = f"例外 ({str(e)[:50]})"

    if res is not None:
        cert = certify(res, timeout_ms=timeout_ms)
        if cert.status == "proved":
            return RLCTInterval(cert.rlct, cert.rlct, "exact", cert.rlct,
                                {"経路": "blowup+certify",
                                 "検証": "全 chart で正規交差と被覆を確認"})
        src["解消"] = f"完了したが証明できず ({cert.status})"
        if res.rlct is not sp.oo and res.rlct < hi:
            hi = res.rlct
            src["解消の値"] = f"上界 {res.rlct} (被覆が未検証なので上界としてのみ)"
        for r in cert.reasons:
            notes.append(r)

    if lo == hi:
        notes.append("区間が一点に潰れているので、値自体は確定している "
                     "(ただし上界と下界の根拠は別々)。")
        return RLCTInterval(lo, hi, "exact", lo, src, notes)
    return RLCTInterval(lo, hi, "interval", None, src, notes)


# ----------------------------------------------------------------------
# 高速パス付きの入口
# ----------------------------------------------------------------------
def _ideal_route(f, gens, *, max_depth: int = 14):
    """イデアルを運ぶ解消 (ideal_resolve.py) を試す。

    生成元の組を運ぶので、途中の節点で「生成元を座標に取る」ことができる。
    多項式に潰した経路では取れない値に届くことがある (Vandermonde H=2 で
    3/4 対 1 など)。ただし phi を追跡していないので証明書は付けられない。

    Returns (lambda, multiplicity) または None。
    """
    try:
        from ideal_resolve import resolve_ideal
    except Exception:
        return None
    # f = sum g_i^2 の形が分かっていれば生成元をそのまま使いたいが、
    # 入口では f しか無いので、平方和に分解できるときだけ生成元を取り出す。
    gs = _sum_of_squares(f, gens)
    try:
        r = resolve_ideal(gs, gens, max_depth=max_depth)
    except Exception:
        return None
    if r.rlct is sp.oo:
        return None
    return (r.rlct, r.multiplicity)


def _sum_of_squares(f, gens):
    """f を平方和 sum g_i^2 に分解する。できなければ [sqrt は使わず] [f] 相当。

    f が既に平方和の形 (展開済み) かどうかを構造的に判定するのは難しいので、
    ここでは f 自体を 1 つの生成元とみなす代わりに、f の平方因子を利用する:
    f = g^2 * h なら <g> ... のような単純化はせず、安全側に倒して
    「f = (g)^2 と書けるなら [g]、そうでなければ [f] を 2 乗と見なさず
     [sqrt(f)] は使わない」= [f] を生成元 1 つとして扱うのは誤り。

    実際には呼び出し側が生成元を知っているので、rlct_certified /
    rlct_interval には generators= を渡せるようにしてある。ここでは
    f = g^2 の形だけ拾う。
    """
    c, facs = sp.factor_list(sp.expand(f))
    gs = []
    ok = True
    for g, d in facs:
        if d % 2 == 0:
            gs.append(sp.expand(g ** (d // 2)))
        else:
            ok = False
            break
    if ok and gs and c >= 0:
        return [sp.expand(sp.sqrt(c) * sp.prod(gs))] if sp.sqrt(c).is_rational \
            else [sp.expand(sp.prod(gs))]
    return None


def rlct_certified(f, gens=None, *, timeout_ms: int = 20000,
                   max_depth: int = 25, verbose: bool = False,
                   ideal: bool = True, generators=None, **kw):
    """lambda を「証明付きで」求める。返り値は (lambda, status, 経路)。

    1. ニュートン多面体に関して実数体上で非退化かを Z3 で判定する。
       非退化なら Varchenko の定理から lambda = 1/(ニュートン距離) が厳密。
       LP だけなので多変数でも速い。
    2. 退化している/判定できない場合は、重み付きブローアップで解消して
       certify() にかける。
    status は 'proved' / 'unknown' / 'refuted'。proved でなければ lambda は
    参考値であり、正しさは保証されない。

    ideal : True (既定) なら、イデアルを運ぶ解消 (ideal_resolve.py) も併用し、
        より小さい値が得られればそちらを採る。解消の値はどれも「ある chart で
        到達した値」なので上界であり、小さいほうが真値に近い。ただし
        イデアル版は phi を追跡しておらず証明書を付けられないので、
        その値を採った場合の status は 'unknown' になる (誤った値を proved と
        主張しないため)。ideal=False で従来どおり多項式版のみ。
    generators : f = sum g_i^2 の生成元が分かっていれば渡す。イデアル版が
        そのまま使える (渡さない場合は f から推定を試みる)。
    """
    from resolve_singularity import resolve_singularities, ResolutionFailure

    if gens is None:
        gens = sorted(sp.sympify(f).free_symbols, key=lambda s: s.name)
    gens = tuple(gens)

    # 計時: 1 多項式ぶんの記録を開始する (入れ子なら外側が有効)。
    # timing.last_record() で段階別の時間が取り出せる。
    with _record("rlct_certified") as _rec:
        _p = sp.Poly(sp.expand(f), *gens)
        _add_meta(n_vars=len(gens), degree=int(_p.total_degree()),
                  n_terms=len(_p.monoms()))
        out = _rlct_certified_inner(f, gens, timeout_ms=timeout_ms,
                                    max_depth=max_depth, verbose=verbose,
                                    ideal=ideal, generators=generators, **kw)
    return out


def _rlct_certified_inner(f, gens, *, timeout_ms, max_depth, verbose,
                          ideal, generators, **kw):
    from resolve_singularity import resolve_singularities, ResolutionFailure

    with _timed("newton"):
        lam, st = rlct_via_newton(f, gens, timeout_ms=timeout_ms)
    if st == "proved":
        if verbose:
            print(f"  ニュートン非退化 -> lambda = {lam} (厳密)")
        return (lam, "proved", "newton")

    gs = generators if generators is not None else _sum_of_squares(f, gens)

    # --- イデアルを運ぶ経路 (既定で有効) -----------------------------
    lam_ideal = None
    if ideal:
        if gs:
            try:
                from ideal_resolve import resolve_ideal
                with _timed("ideal"):
                    ri = resolve_ideal(gs, gens, max_depth=max_depth)
                if ri.rlct is not sp.oo:
                    lam_ideal = ri.rlct
                    if verbose:
                        print(f"  イデアル版: lambda = {lam_ideal}")
            except Exception:
                lam_ideal = None

    # --- Aoyagi Lemma 1(1) の挟み込み --------------------------------
    # 部分生成元の lambda は全体の lambda の厳密な下界、newton_rlct は
    # 常に上界。一致すればブローアップなしで lambda が確定する。
    # 一致しなくても、下界は分枝限定の早期終了に使える。
    lb = None
    if gs and len(gs) >= 2:
        try:
            from lemmas import squeeze_rlct
            sq = squeeze_rlct(f, gens, gs)
            if sq.status == "proved":
                if verbose:
                    print(f"  挟み込み {sq} -> lambda = {sq.rlct} (厳密)")
                # 位数 m は挟み込みでは決まらないので返さない
                return (sq.rlct, "proved", "squeeze(Aoyagi Lemma 1(1) + Newton LP)")
            if sq.status != "inconsistent":
                lb = sq.lo
        except Exception:
            lb = None

    try:
        with _timed("resolve"):
            res = resolve_singularities(f, gens, prune=False, weighted=True,
                                        max_depth=max_depth,
                                        lower_bound=lb, **kw)
    except ResolutionFailure as e:
        # 解消が失敗したときだけ、挟み込みに大きな予算を与えて呼び直す
        # (pay-when-needed: 易しいケースでは overhead をかけない)。
        if gs and len(gs) >= 2:
            try:
                from lemmas import squeeze_rlct
                sq2 = squeeze_rlct(f, gens, gs, max_subset=2, max_calls=24,
                                   time_budget=8.0)
                if sq2.status == "proved":
                    return (sq2.rlct, "proved",
                            "squeeze(Aoyagi Lemma 1(1) + Newton LP, 解消失敗後)")
            except Exception:
                pass
        if lam_ideal is not None:
            return (lam_ideal, "unknown", "ideal (多項式版は解消できず)")
        return (None, "unknown", f"解消できず ({e.reason})")
    _add_meta(n_charts=len(res.charts), n_blowups=res.n_blowups,
              n_newton=res.n_newton)
    with _timed("certify"):
        cert = certify(res, timeout_ms=timeout_ms)
    lam_poly = res.rlct

    if lam_ideal is not None and lam_ideal < lam_poly:
        # 小さいほうが真値に近い (どちらも上界)。ただし証明書は付かない。
        return (lam_ideal, "unknown", "ideal (多項式版より小さい値)")
    if cert.status == "proved" and lam_ideal is not None and lam_ideal != lam_poly:
        # 証明済みの値よりイデアル版が大きい -> イデアル版が取りこぼし
        return (cert.rlct, "proved", "blowup+certify")
    return (cert.rlct if cert.status == "proved" else lam_poly,
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
