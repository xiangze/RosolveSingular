r"""
families.py
===========

族ごとの専用の証明経路。汎用のブローアップ + SMT 検証に回す前に、
構造が分かっている多項式については閉じた式で lambda と位数 m を確定させる。

実装している経路 (上から順に試す):

  1. monomial      f = c * x^k
                   lambda = min_{k_j>0} 1/k_j,  m = 最小を達成する j の個数
  2. product       f = f_1(u) * f_2(v) * ... (変数が互いに素)
                   zeta が積に分かれるので lambda = min lambda_i,
                   m = 最小を達成する成分の m_i の和。符号の条件は不要
                   (|f_1 f_2| = |f_1| |f_2|)。
  3. sum           f = f_1(u) + f_2(v) + ... (変数が互いに素、各項が非負)
                   lambda = sum lambda_i,  m = sum m_i - (成分数 - 1)
                   非負性が要る (|f_1 + f_2| ~ |f_1| + |f_2| を使うため)。
  4. power_form    f = g^p で g が 1 次形式または 2 次形式
                   g が 1 次: lambda = 1/p, m = 1
                   g が階数 r の 2 次形式で定値: lambda = r/(2p), m = 1
                   不定値: lambda = min(1/p, r/(2p))
  5. newton        ニュートン多面体に関して実数体上非退化 (Z3 で判定)
                   Varchenko の定理から lambda = 1/(ニュートン距離)

いずれも「証明経路」であると同時に「オラクル」でもある。bench では
汎用の経路が出した値をこれらと突き合わせて、proved の中身を検査する。

注意: 位数 m の合成則は、成分ごとの zeta の極が互いに打ち消さないことを
使っている。数値的な偶然の相殺は扱っていない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


import sympy as sp

__all__ = ["FamilyResult", "family_rlct", "is_nonneg_poly",
           "disjoint_factors", "disjoint_summands"]


@dataclass
class FamilyResult:
    rlct: Optional[sp.Rational]
    multiplicity: Optional[int]
    status: str            # 'proved' | 'unknown'
    route: str             # 経路の名前 (入れ子なら親子を '/' でつなぐ)

    def __str__(self) -> str:
        return (f"lambda={self.rlct}, m={self.multiplicity} "
                f"[{self.status} via {self.route}]")


_UNKNOWN = FamilyResult(None, None, "unknown", "-")


# ----------------------------------------------------------------------
# 補助
# ----------------------------------------------------------------------
def is_nonneg_poly(f, gens) -> bool:
    """全ての単項式の指数が偶数で係数が正 <=> 明らかに非負。"""
    p = sp.Poly(sp.expand(f), *gens)
    for mon, c in zip(p.monoms(), p.coeffs()):
        if any(e % 2 for e in mon):
            return False
        if c <= 0:
            return False
    return True


def _components(items, key_vars):
    """変数を共有するかどうかで items を連結成分に分ける。"""
    groups: List[Tuple[set, list]] = []
    for it in items:
        vs = set(key_vars(it))
        merged = [vs], [it]
        hit = []
        for idx, (gv, gi) in enumerate(groups):
            if gv & vs:
                hit.append(idx)
        newv = set(vs)
        newi = [it]
        for idx in hit:
            newv |= groups[idx][0]
            newi += groups[idx][1]
        groups = [g for idx, g in enumerate(groups) if idx not in hit]
        groups.append((newv, newi))
    return groups


def disjoint_factors(f, gens):
    """f を変数が互いに素な因子の積に分ける。分けられなければ None。"""
    c, facs = sp.factor_list(sp.expand(f))
    parts = []
    for g, d in facs:
        parts.append((g ** d, tuple(sorted(g.free_symbols, key=str))))
    if len(parts) <= 1:
        return None
    groups = _components(parts, lambda t: t[1])
    if len(groups) <= 1:
        return None
    out = []
    for _, items in groups:
        out.append(sp.expand(c * sp.prod([t[0] for t in items]))
                   if not out else sp.expand(sp.prod([t[0] for t in items])))
    return out


def disjoint_summands(f, gens):
    """f を変数が互いに素な和に分ける。分けられなければ None。"""
    terms = sp.Add.make_args(sp.expand(f))
    if len(terms) <= 1:
        return None
    items = [(t, tuple(sorted(t.free_symbols & set(gens), key=str)))
             for t in terms]
    groups = _components(items, lambda t: t[1])
    if len(groups) <= 1:
        return None
    return [sp.expand(sum(t[0] for t in items_)) for _, items_ in groups]


def _used(f, gens):
    return [v for v in gens if v in f.free_symbols]


# ----------------------------------------------------------------------
# 各経路
# ----------------------------------------------------------------------
def _route_monomial(f, gens) -> Optional[FamilyResult]:
    p = sp.Poly(sp.expand(f), *gens)
    if len(p.monoms()) != 1:
        return None
    k = p.monoms()[0]
    cand = [sp.Rational(1, e) for e in k if e > 0]
    if not cand:
        return FamilyResult(sp.oo, 0, "proved", "monomial(unit)")
    lam = min(cand)
    m = sum(1 for e in k if e > 0 and sp.Rational(1, e) == lam)
    return FamilyResult(lam, m, "proved", "monomial")


def _route_product(f, gens, depth) -> Optional[FamilyResult]:
    parts = disjoint_factors(f, gens)
    if not parts:
        return None
    subs = [family_rlct(g, _used(g, gens), _depth=depth + 1) for g in parts]
    if any(s.status != "proved" for s in subs):
        return None
    lam = min(s.rlct for s in subs)
    if lam is sp.oo:
        return FamilyResult(sp.oo, 0, "proved", "product")
    m = sum(s.multiplicity for s in subs if s.rlct == lam)
    route = "product(" + ", ".join(s.route for s in subs) + ")"
    return FamilyResult(lam, m, "proved", route)


def _route_sum(f, gens, depth) -> Optional[FamilyResult]:
    parts = disjoint_summands(f, gens)
    if not parts:
        return None
    if not all(is_nonneg_poly(g, _used(g, gens)) for g in parts):
        return None                      # 非負でないと和の規則が使えない
    subs = [family_rlct(g, _used(g, gens), _depth=depth + 1) for g in parts]
    if any(s.status != "proved" or s.rlct is sp.oo for s in subs):
        return None
    lam = sum(s.rlct for s in subs)
    m = sum(s.multiplicity for s in subs) - (len(subs) - 1)
    route = "sum(" + ", ".join(s.route for s in subs) + ")"
    return FamilyResult(lam, max(m, 1), "proved", route)


def _quadratic_rank_signature(g, gens):
    """g が斉次 2 次形式なら (階数, 定値か) を返す。"""
    p = sp.Poly(sp.expand(g), *gens)
    if p.total_degree() != 2 or any(sum(m) != 2 for m in p.monoms()):
        return None
    n = len(gens)
    M = sp.zeros(n, n)
    for mon, c in zip(p.monoms(), p.coeffs()):
        idx = [i for i, e in enumerate(mon) for _ in range(e)]
        if len(idx) == 2:
            i, j = idx
            if i == j:
                M[i, i] += c
            else:
                M[i, j] += sp.Rational(c, 2)
                M[j, i] += sp.Rational(c, 2)
    ev = M.eigenvals()
    pos = sum(mult for val, mult in ev.items() if val.is_positive)
    neg = sum(mult for val, mult in ev.items() if val.is_negative)
    zero = sum(mult for val, mult in ev.items() if val.is_zero)
    if pos + neg + zero != n:
        return None                      # 符号が判定できない
    r = pos + neg
    if r == 0:
        return None
    return (r, neg == 0 or pos == 0)


def _route_power_form(f, gens) -> Optional[FamilyResult]:
    c, facs = sp.factor_list(sp.expand(f))
    nonunit = [(g, d) for g, d in facs if g.subs({v: 0 for v in gens}) == 0]
    if len(nonunit) != 1:
        return None
    g, p = nonunit[0]
    gg = sp.expand(g)
    used = _used(gg, gens)
    pg = sp.Poly(gg, *used)
    if pg.total_degree() == 1 and all(sum(m) == 1 for m in pg.monoms()):
        return FamilyResult(sp.Rational(1, p), 1, "proved",
                            f"power_form(linear^{p})")
    q = _quadratic_rank_signature(gg, used)
    if q is None:
        return None
    r, definite = q
    if definite:
        lam = sp.Rational(r, 2 * p)
    else:
        lam = min(sp.Rational(1, p), sp.Rational(r, 2 * p))
    kind = "definite" if definite else "indefinite"
    return FamilyResult(lam, 1, "proved", f"power_form(quadratic {kind} rank {r}^{p})")


def _route_newton(f, gens) -> Optional[FamilyResult]:
    try:
        from certify import newton_nondegenerate
        from resolve_singularity import newton_rlct
    except Exception:
        return None
    from certify import principal_face_compact
    st, _ = newton_nondegenerate(f, gens, timeout_ms=8000)
    if st != "proved":
        return None
    lam = newton_rlct(f, gens)
    if lam is sp.oo:
        return FamilyResult(sp.oo, 0, "proved", "newton")
    # 非退化だけでは足りない: 主面がコンパクトでないと Varchenko の公式は
    # 成り立たない (certify.principal_face_compact の例を参照)
    if principal_face_compact(sp.Poly(sp.expand(f), *gens).monoms(),
                              [1] * len(gens), lam) is not True:
        return None
    # 位数は「対角線が当たる点を通るファセットの法線の階数」で決まる
    # (m = n - dim(最小面))。トーリック解消で扇を細分しても同じ値になる。
    try:
        from certify import newton_multiplicity
        m = newton_multiplicity(f, gens)
    except Exception:
        m = None
    return FamilyResult(lam, m, "proved", "newton")


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def family_rlct(f, gens=None, *, _depth: int = 0) -> FamilyResult:
    """族ごとの専用経路で lambda を確定させる。だめなら status='unknown'。"""
    f = sp.expand(sp.sympify(f))
    if gens is None:
        gens = sorted(f.free_symbols, key=str)
    gens = tuple(gens)
    if _depth > 8:
        return _UNKNOWN
    if not gens or f.is_number:
        return FamilyResult(sp.oo, 0, "proved", "constant") if f != 0 else _UNKNOWN

    routes = (("monomial", lambda: _route_monomial(f, gens)),
              ("product", lambda: _route_product(f, gens, _depth)),
              ("sum", lambda: _route_sum(f, gens, _depth)),
              ("power_form", lambda: _route_power_form(f, gens)),
              ("newton", lambda: _route_newton(f, gens)))
    for nm, fn in routes:
        # ニュートン経路だけは Z3 を呼ぶので別の phase で計時する
        with _timed("newton" if nm == "newton" else "family"):
            try:
                r = fn()
            except Exception:
                r = None
        if r is not None:
            return r
    return _UNKNOWN


def _demo():
    x, y, z, w = sp.symbols("x y z w", real=True)
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)
    cases = [
        ("x^2*y^3", x**2 * y**3),
        ("x^2+y^2", x**2 + y**2),
        ("x^2+y^3", x**2 + y**3),
        ("(x^2+y^2)(z^2+w^2)", (x**2 + y**2) * (z**2 + w**2)),
        ("(x-y)^2", (x - y) ** 2),
        ("(xy+z^2)^2", (x * y + z**2) ** 2),
        ("(a1b1+a2b2)^2", (a1 * b1 + a2 * b2) ** 2),
        ("x^2*y^2*z^2", x**2 * y**2 * z**2),
        ("sum of 5 squares", sum(v**2 for v in sp.symbols("u0:5", real=True))),
        ("x^2y^2+y^2z^2+z^2x^2", x**2*y**2 + y**2*z**2 + z**2*x**2),
        ("x^3+y^4+z^5", x**3 + y**4 + z**5),
    ]
    for label, f in cases:
        print(f"  {label:<24} {family_rlct(f)}")


if __name__ == "__main__":
    _demo()
