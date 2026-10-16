r"""
newton.py — **ニュートン多面体の層** (多面体・面の列挙・LP・非退化判定)

`certify.py` が判定器とニュートン多面体の道具を同居させていたので、下の層を
ここに分離した。**この分離だけで核心の循環依存が 4 辺消える**
(`resolve_singularity -> certify`、`families -> certify`、
`ideal_resolve -> certify`、`rlct_cache -> resolve_singularity`)。
詳細は `REORG_PLAN.ja.md` §2。

このモジュールはプロジェクト内の他モジュールに依存しない (`timing` の任意の
計時を除く) **葉**であり、依存の向きは常に「上の層がここを使う」だけ。

------------------------------------------------------------------
中身
------------------------------------------------------------------
* `Box`                     原点中心の箱 (定義域)
* `_to_z3`                  sympy 式 -> z3 の実数式
* `nonzero_on_box`          箱の上で消えないことの判定
* `normal_crossing_on_box`  例外因子との正規交叉の判定
* `newton_facet_normals`    Gamma_+ のコンパクトなファセットの外向き法線
* `newton_faces`            **全ての**コンパクトな面 (ファセットだけでは不足)
* `newton_multiplicity`     対角線が当たる面の次元から位数 m
* `newton_nondegenerate`    実数体上の非退化性 (Groebner でふるい、z3 で決める)
* `principal_face_compact`  Varchenko の公式に必要な追加条件
* `rlct_via_newton`         非退化なら lambda を確定させる高速パス
* `newton_rlct`             ニュートン多面体による lambda (LP)。常に上界
* `_lp_newton_value`        枝刈り用の LP (上界)
* `_ilp_min_hitting_set`    ブローアップの中心 (最小ヒッティング集合)

健全性の向きについての注意は各関数の docstring に書いてある。とくに
`_groebner_no_torus_zero` は **proved 側にしか健全でない** (複素トーラス零点の
非存在しか示さない) ので、`False` を返さない設計になっている。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

try:
    import z3 as _z3
    _HAS_Z3 = True
except Exception:                                     # pragma: no cover
    _z3 = None
    _HAS_Z3 = False

try:
    import pulp as _pulp
    _HAS_PULP = True
except Exception:                                     # pragma: no cover
    _pulp = None
    _HAS_PULP = False

try:
    from scipy.optimize import linprog as _linprog
    _HAS_SCIPY = True
except Exception:                                     # pragma: no cover
    _linprog = None
    _HAS_SCIPY = False

try:
    from timing import timed as _timed
except Exception:      # pragma: no cover
    from contextlib import contextmanager as _cm

    @_cm
    def _timed(_name):           # 計時モジュールが無くても動く
        yield


# ----------------------------------------------------------------------
# 箱 (定義域)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    """原点中心の箱 {|y_j| <= bound_j}。"""

    bounds: Tuple[sp.Rational, ...]

    def __str__(self) -> str:
        return "{" + ", ".join(f"|y{j}|<={b}" for j, b in enumerate(self.bounds)) + "}"


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


_GROEBNER_MAX_VARS = 8
_GROEBNER_MAX_TERMS = 200
_GROEBNER_MAX_DEG = 12


def _groebner_no_torus_zero(grads, gens):
    """grad f_sigma = 0 が **複素**トーラス (全ての x_i != 0) 上に解を持たない
    ことを Groebner 基底で示す。示せたら True、それ以外は None。

    Rabinowitsch のトリック: 新変数 t を足して

        I = <grad f_sigma> + <1 - t * prod x_i>   in  Q[x, t]

    を作る。トーラス上の零点は I の零点と 1 対 1 に対応するので、
    I = (1) (Groebner 基底が [1]) ならトーラス上に複素零点はない。

    **健全性の向きが片方だけ**であることが肝心。

      * GB = [1] なら複素零点がない。実零点は複素零点なので実トーラス上にも
        零点はなく、この面は (実数体上) 非退化 -> 'proved' に使える。
      * GB != [1] は「複素零点がある」だけで、それが実とは限らない。
        RLCT に必要なのは実数体上の非退化性なので、この向きからは何も
        言えない。よって False は返さず None を返して z3 に渡す。

    この関数が 'refuted' を出すことは構造上ありえない。
    """
    gens = tuple(gens)
    # まず単項式チェック: 勾配のどれかが単項式ならトーラス上で消えない。
    # Groebner を組むより桁違いに安く、面の多くはこれで片付く。
    for g in grads:
        if g == 0:
            continue
        try:
            pg = sp.Poly(g, *gens)
        except Exception:
            continue
        if len(pg.monoms()) == 1 and pg.coeffs()[0] != 0:
            return True
    if len(gens) > _GROEBNER_MAX_VARS:
        return None
    polys = []
    total = 0
    for g in grads:
        if g == 0:
            continue
        try:
            pg = sp.Poly(g, *gens)
        except Exception:
            return None
        if pg.total_degree() > _GROEBNER_MAX_DEG:
            return None
        total += len(pg.monoms())
        polys.append(g)
    if not polys or total > _GROEBNER_MAX_TERMS:
        return None
    t = sp.Dummy("t_rab")
    try:
        gb = sp.groebner(polys + [1 - t * sp.prod(gens)], *gens, t,
                         order="grevlex")
    except Exception:
        return None
    exprs = list(gb.exprs)
    if len(exprs) == 1 and exprs[0].is_number and exprs[0] != 0:
        return True
    return None


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
    coeff = {tuple(m): c for m, c in zip(p.monoms(), p.coeffs())}
    # z3 の Solver は面ごとに作り直さず、push/pop で使い回す (起動コスト)。
    s = _z3.Solver() if _HAS_Z3 else None
    if s is not None:
        s.set("timeout", timeout_ms)
        zvars = {v: _z3.Real(f"t{i}") for i, v in enumerate(gens)}
    for face in faces:
        if len(face) < 2:
            continue                       # 単項式の面は自動的に非退化
        fs = sum(coeff[m] * sp.prod([v ** e for v, e in zip(gens, m)])
                 for m in face)
        grads = [sp.expand(sp.diff(fs, v)) for v in gens]
        # 先に Gröbner で「複素トーラス上に零点なし」を試す。通れば
        # 実トーラス上にも零点はないのでこの面は非退化 (z3 を呼ばない)。
        if _groebner_no_torus_zero(grads, gens) is True:
            continue
        if s is None:
            return ("unknown", face)
        s.push()
        try:
            for v in gens:
                s.add(zvars[v] != 0)
            for g in grads:
                s.add(_to_z3(g, zvars) == 0)
        except ValueError:
            s.pop()
            return ("unknown", face)
        r = s.check()
        s.pop()
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
    st, face = newton_nondegenerate(f, gens, timeout_ms=timeout_ms)
    lam = newton_rlct(f, gens)
    if st == "proved":
        # 非退化だけでは足りない: 主面がコンパクトであることも要る
        ok = principal_face_compact(sp.Poly(sp.expand(f), *gens).monoms(),
                                    [1] * len(gens), lam)
        if ok is not True:
            st = "unknown"
    return (lam, st)


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
    # in-process の scipy(HiGHS) を常に優先する。CBC は解くたびに
    # サブプロセスを起動するので、初回 585ms / 以降 5.6ms に対し
    # scipy は 1.7ms (実測)。LP は整数制約がないので HiGHS で十分。
    del hot                              # 既存の呼び出し互換のため残す
    return "scipy" if _HAS_SCIPY else ("pulp" if _HAS_PULP else "none")


def _lp_var(name: str, *, low=None, cat=None):
    """pulp の LpVariable を版差を吸収して作る。

    pulp 4.0 で lowBound= キーワードが通らない環境が報告されているので、
    lowBound -> low_bound -> 位置引数 の順に試す。
    """
    kw = {}
    if cat is not None:
        kw["cat"] = cat
    if low is None:
        return _pulp.LpVariable(name, **kw)
    for key in ("lowBound", "low_bound"):
        try:
            return _pulp.LpVariable(name, **{key: low}, **kw)
        except TypeError:
            continue
    return _pulp.LpVariable(name, low, **kw)


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
        nu = [_lp_var(f"nu{a}", low=0) for a in range(N)]
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


def _ilp_min_hitting_set_scipy(supports: Sequence[frozenset],
                               n: int) -> Optional[List[int]]:
    """最小ヒッティング集合を scipy.optimize.milp (HiGHS) で解く。

    CBC と違ってプロセス起動がないので、初回 585ms の立ち上がりが消える
    (実測 2.7ms/回)。整数最適解なので結果は CBC と同じ。
    """
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception:
        return None
    A = np.zeros((len(supports), n))
    for i, S in enumerate(supports):
        for j in S:
            if 0 <= j < n:
                A[i, j] = 1.0
    if not A.any(axis=1).all():
        return None                  # 空の台 -> 被覆不能。呼び出し側に委ねる
    try:
        res = milp(c=np.ones(n),
                   constraints=LinearConstraint(A, lb=1.0, ub=np.inf),
                   integrality=np.ones(n), bounds=Bounds(0, 1))
    except Exception:
        return None
    if not res.success or res.x is None:
        return None
    return sorted(j for j in range(n) if res.x[j] > 0.5)


def _ilp_min_hitting_set(supports: Sequence[frozenset], n: int,
                         backend: str = "auto") -> Optional[List[int]]:
    """最小ヒッティング集合 (集合被覆) を 0-1 整数計画として解く。

        minimize  sum_j z_j
        s.t.      sum_{j in S} z_j >= 1  (各単項式の台 S),  z_j in {0,1}

    in-process の scipy.optimize.milp (HiGHS) を優先し、無ければ pulp
    (CBC) に投げる。解けなければ None を返し、呼び出し側の分枝限定に
    フォールバックする。
    """
    if not supports:
        return []
    if backend in ("auto", "scipy") and _HAS_SCIPY:
        r = _ilp_min_hitting_set_scipy(supports, n)
        if r is not None or backend == "scipy":
            return r
    if not _HAS_PULP:
        return None
    prob = _pulp.LpProblem("min_center", _pulp.LpMinimize)
    z = {j: _lp_var(f"z{j}", cat="Binary") for j in range(n)}
    prob += _pulp.lpSum(z.values())
    for S in supports:
        prob += _pulp.lpSum(z[j] for j in S) >= 1
    status = prob.solve(_pulp.PULP_CBC_CMD(msg=0))
    if _pulp.LpStatus[status] != "Optimal":
        return None
    return sorted(j for j in range(n) if z[j].value() is not None and z[j].value() > 0.5)


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


__all__ = [
    "Box",
    "nonzero_on_box",
    "normal_crossing_on_box",
    "newton_facet_normals",
    "newton_faces",
    "newton_multiplicity",
    "newton_nondegenerate",
    "principal_face_compact",
    "rlct_via_newton",
    "newton_rlct",
]
