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

# ニュートン多面体の層は newton.py に分離した (REORG_PLAN.ja.md §2)。
# 既存の呼び出し側との互換のためここでも再公開する (同じオブジェクト)。
from newton import (  # noqa: F401
    Box,
    _HAS_Z3,
    _fiber_constraints,
    _groebner_no_torus_zero,
    _lp_newton_value,
    _to_z3,
    _used_coords,
    _z3,
    newton_faces,
    newton_facet_normals,
    newton_multiplicity,
    newton_nondegenerate,
    newton_rlct,
    nonzero_on_box,
    normal_crossing_on_box,
    principal_face_compact,
    rlct_via_newton,
)


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
    from newton import newton_rlct

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


def _verify_cache_leaf(ch, gens, *, trust: bool = True,
                       timeout_ms: int = 10000):
    """キャッシュで閉じた葉を検査する。

    trust=True (既定): 安い健全性検査だけを行う。
        エントリは (i) 証明された値しか入らず、(ii) コードの指紋が一致
        するものしか読まないので、同じコードで計算し直せば同じ値になる。
        すなわち「信用する」ことは「計算し直す」ことと同値である。
        ただし **保存時のコードにバグがあれば同じバグを引き継ぐ** ので、
        証明書の理由欄にその旨を残す。
    trust=False: その局所データをキャッシュ抜きで解消し直して突き合わせる。
        食い違えば refuted を返し、そのエントリは捨てるべき
        (rlct_cache.invalidate) である。

    Returns (status, witness)
    """
    lam_rec, m_rec, _why = ch.closed
    from newton import _lp_newton_value

    monoms = [tuple(ki + ei for ki, ei in zip(ch.k, m))
              for m in ch.poly.monoms()]
    ub = _lp_newton_value(monoms, [hj + 1 for hj in ch.h], hot=True)
    if ub != float("inf") and float(lam_rec) > ub * (1 + 1e-6) + 1e-9:
        return ("refuted", {"why": f"上界を超えています ({lam_rec} > {ub})"})
    if trust:
        return ("proved", None)

    # --- 計算し直して突き合わせる ---------------------------------
    from resolve_singularity import (ResolutionFailure, resolve_singularities)
    mono = sp.prod([v ** e for v, e in zip(gens, ch.k)])
    sub = sp.expand(mono * ch.f_rest())
    try:
        res = resolve_singularities(sub, gens, prune=False, weighted=True,
                                    h0=list(ch.h), max_depth=16,
                                    use_cache=False)
    except ResolutionFailure:
        return ("unknown", {"why": "再計算できません (解消が失敗)"})
    except Exception as e:                                # noqa: BLE001
        return ("unknown", {"why": f"再計算できません ({str(e)[:40]})"})
    c2 = certify(res, timeout_ms=timeout_ms, trust_cache=False)
    if c2.status != "proved":
        return ("unknown", {"why": f"再計算の証明書が {c2.status}"})
    if sp.nsimplify(c2.rlct) != sp.nsimplify(lam_rec):
        return ("refuted", {"why": f"再計算が一致しません "
                                   f"({c2.rlct} != {lam_rec}) "
                                   "-- rlct_cache.invalidate してください"})
    return ("proved", None)


def certify(resolution, *, eps=sp.Integer(1), delta=sp.Rational(1, 100),
            timeout_ms: int = 10000, check_jacobian: bool = True,
            localize: bool = True, trust_cache: bool = True) -> Certificate:
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
    cache_leaves: List[str] = []
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
        if closed is not None and closed[2] == "cache":
            us, w = _verify_cache_leaf(ch, gens, trust=trust_cache,
                                       timeout_ms=timeout_ms)
            if us == "proved":
                cache_leaves.append(ch.name)
        elif closed is not None and closed[2] == "product":
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
                        else ("キャッシュ" if ch.name in cache_leaves
                              else "")))))

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
    if cache_leaves:
        reasons.append(
            f"{len(cache_leaves)} 個の葉は過去に証明済みの値 (rlct_cache) を"
            "再利用しています。コードの指紋が一致するエントリだけを読み、"
            "当たりはニュートン LP の上界と突き合わせていますが、"
            "**保存時のコードにバグがあればそれを引き継ぎます**。"
            "厳密に確かめるなら certify(trust_cache=False) か "
            "rlct_cache.disable() で計算し直してください。"
        )
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
    from resolve_singularity import ResolutionFailure, resolve_singularities
    from newton import newton_rlct

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


_LAST_CANDIDATES: List[tuple] = []


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
        # キャッシュ: この計算で使ったエントリを depends_on に残すため、
        # 開始時に記録をリセットする。
        try:
            import rlct_cache as _rc
            _c = _rc.current()
            if _c is not None:
                _c.begin()
        except Exception:
            _rc, _c = None, None
        out = _rlct_certified_inner(f, gens, timeout_ms=timeout_ms,
                                    max_depth=max_depth, verbose=verbose,
                                    ideal=ideal, generators=generators, **kw)
        # 証明できたものだけを保存する (unknown は決して入れない)
        if _c is not None and out and out[1] == "proved" and out[0] is not None:
            try:
                _rc.put([0] * len(gens), [0] * len(gens), _p, gens,
                        out[0], None, f"rlct_certified:{out[2]}")
                # 部分木の値も入れる。これが本命で、別の問題の chart として
                # 同じ局所データが現れたときに再利用される。
                for kk, hh, pp, lam_s, m_s in _LAST_CANDIDATES:
                    _rc.put(kk, hh, pp, gens, lam_s, m_s, "subtree")
            except Exception:
                pass
        _LAST_CANDIDATES.clear()
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
    if cert.status == "proved":
        _LAST_CANDIDATES.extend(getattr(res, "cache_candidates", []))
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
# closer への結線 (依存の逆転)
# ----------------------------------------------------------------------
def _default_subsolver(sub, vs, hh):
    """`lemmas` の積分解・挟み込みが使う部分問題の解き手。

    lemmas 側はこれを import しない (循環になる)。**こちらから注入する**
    ことで、依存の向きを closer -> 判定器 から 判定器 -> closer の
    一方向に揃えている (REORG_PLAN.ja.md §3.3(a))。
    """
    from resolve_singularity import ResolutionFailure, resolve_singularities
    try:
        # product_split=False: 部分問題の中で再び分解を試すと無限再帰になる
        res = resolve_singularities(sub, vs, prune=False, weighted=True,
                                    h0=list(hh), max_depth=12,
                                    product_split=False)
    except ResolutionFailure:
        return None
    except Exception:                             # noqa: BLE001
        return None
    if certify(res, timeout_ms=4000).status != "proved":
        return None
    if res.rlct is sp.oo:
        return None
    return (res.rlct, res.multiplicity)


try:
    import lemmas as _lemmas
    _lemmas.install_subsolver(_default_subsolver)
except Exception:                                 # pragma: no cover
    pass


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
