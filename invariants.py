r"""
invariants.py
=============

特異点の局所不変量。RLCT のような積分の漸近ではなく、有限次元の商環の
次元を数えるだけなので、ブローアップは要らない。

  multiplicity(f)        m   = ord_0 f (テイラー展開の最低次数)
  quasihomogeneous_weights(f)  重み w > 0 と次数 d (擬斉次でなければ None)
  milnor_number(f)       mu  = dim C[[x]] / <df/dx_1, ..., df/dx_n>
  tjurina_number(f)      tau = dim C[[x]] / <f, df/dx_1, ..., df/dx_n>

backend='auto' (既定) は Singular があればそれを使い、無ければ sympy に
落ちる。'singular' / 'sympy' で明示的に選べる。

------------------------------------------------------------------
計算の方針
------------------------------------------------------------------
* 擬斉次なら公式で即答する。f が重み (w_1,...,w_n)、次数 d に関して擬斉次で
  孤立特異点なら
        mu = prod_i (d / w_i - 1),   tau = mu
  (K. Saito の定理より mu = tau <=> 擬斉次)。
* そうでなければグレブナー基底で商環の次元を数える。ゼロ次元イデアルなら
  主導単項式イデアルの「階段」の外にある単項式の個数がそのまま次元になる。
  各変数 i について x_i の純冪が主導単項式に現れることが、ゼロ次元性
  (= 特異点が孤立) の判定条件。
* Singular があれば局所順序 (ds) で milnor/tjurina を呼ぶ (backend='auto'
  の既定)。これなら原点だけの局所不変量が厳密に得られる。
* Singular が無い場合の代替として、sympy の groebner を使う。ただし
  sympy は大域的な順序しか持たないので、得られるのは
  C[x]/I の大域的な次元である。原点以外に特異点があるとそれも数えてしまう
  ので、Z3 で「原点以外の特異点が存在しない」ことを確かめてから採用する
  (isolated='proved' のときだけ mu を返し、それ以外は locality を注記する)。

------------------------------------------------------------------
RLCT との関係
------------------------------------------------------------------
別の不変量だが互いを束縛する。

  1/m <= lambda            (m = 重複度)
  lambda <= n/m            のような上からの評価も得られる
  mu = tau  <=>  擬斉次     -> 重み付きブローアップ 1 回で単項式化できる

最後の同値は実用的で、resolve_singularity の「重みを使うかどうか」の
判定則そのものである。quasihomogeneous_weights() はその判定を、ニュートン
多面体の LP ではなく連立一次方程式で厳密に行う。

注意: mu と tau は代数閉体上の不変量である (商環の次元は係数体を
拡大しても変わらないので実係数で計算してよいが、意味づけは複素の話)。
実 RLCT の検算に使うなら、複素 lct 側の不等式を経由することになる。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

__all__ = ["multiplicity", "quasihomogeneous_weights", "milnor_number",
           "tjurina_number", "singularity_report", "LocalInvariants",
           "singular_available"]


# ----------------------------------------------------------------------
# Singular バックエンド (任意) — 局所順序が使えるので「原点での」mu が出る
# ----------------------------------------------------------------------
import shutil
import subprocess


def singular_available() -> bool:
    """Singular の実行ファイルがあるか。"""
    return shutil.which("Singular") is not None


def _singular_poly(f, gens) -> str:
    """sympy の多項式を Singular の構文に直す。"""
    p = sp.Poly(sp.expand(f), *gens)
    terms = []
    for mon, c in zip(p.monoms(), p.coeffs()):
        c = sp.Rational(c)
        parts = [f"({c.p}/{c.q})" if c.q != 1 else f"({c.p})"]
        for v, e in zip(gens, mon):
            if e:
                parts.append(f"{v}^{e}" if e > 1 else str(v))
        terms.append("*".join(parts))
    return " + ".join(terms) if terms else "0"


def _singular_invariants(f, gens, timeout: int = 60):
    """Singular で局所 (ds 順序) の mu, tau を計算する。

    ds は原点での局所順序なので、sympy の大域グレブナー基底と違って
    **原点だけ** の不変量が得られる。非孤立特異点では Singular は -1 を
    返すので、それを sp.oo に直す。

    Returns (mu, tau) または None (失敗時)。
    """
    if not singular_available():
        return None
    names = ",".join(str(v) for v in gens)
    script = (
        'LIB "sing.lib";\n'
        f"ring r=0,({names}),ds;\n"
        f"poly f={_singular_poly(f, gens)};\n"
        'printf("MU %s",milnor(f));\n'
        'printf("TAU %s",tjurina(f));\n'
        "quit;\n"
    )
    try:
        pr = subprocess.run(["Singular", "-q"], input=script, text=True,
                            capture_output=True, timeout=timeout)
    except Exception:
        return None
    mu = tau = None
    for line in pr.stdout.splitlines():
        line = line.strip()
        if line.startswith("MU "):
            mu = line[3:].strip()
        elif line.startswith("TAU "):
            tau = line[4:].strip()
    if mu is None or tau is None:
        return None

    def conv(t):
        try:
            v = int(t)
        except ValueError:
            return None
        return sp.oo if v < 0 else v

    return (conv(mu), conv(tau))


# ----------------------------------------------------------------------
# 重複度
# ----------------------------------------------------------------------
def multiplicity(f, gens=None) -> int:
    """原点における重複度 ord_0 f。"""
    f = sp.expand(sp.sympify(f))
    if gens is None:
        gens = sorted(f.free_symbols, key=str)
    p = sp.Poly(f, *gens)
    if p.is_zero:
        return sp.oo
    return min(sum(m) for m in p.monoms())


# ----------------------------------------------------------------------
# 擬斉次性
# ----------------------------------------------------------------------
def quasihomogeneous_weights(f, gens=None):
    """f が擬斉次なら (重み w (正整数), 次数 d) を返す。でなければ None。

    全ての単項式 a について <w, a> = d となる w > 0 を連立一次方程式で解く。
    d = 1 と正規化して解き、最後に整数に直す。
    """
    f = sp.expand(sp.sympify(f))
    if gens is None:
        gens = sorted(f.free_symbols, key=str)
    gens = tuple(gens)
    p = sp.Poly(f, *gens)
    monoms = [tuple(m) for m in p.monoms()]
    if not monoms or any(all(e == 0 for e in m) for m in monoms):
        return None
    n = len(gens)
    A = sp.Matrix([[sp.Integer(e) for e in m] for m in monoms])
    b = sp.ones(len(monoms), 1)
    sol = None
    try:
        # 最小二乗ではなく厳密解 (存在しなければ None)
        aug = A.row_join(b)
        if A.rank() != aug.rank():
            return None
        ws = sp.symbols(f"_w0:{n}")
        eqs = [sp.Eq(sum(A[i, j] * ws[j] for j in range(n)), 1)
               for i in range(len(monoms))]
        sol_set = sp.linsolve(eqs, ws)
        if not sol_set:
            return None
        sol = list(sol_set)[0]
    except Exception:
        return None
    # 自由変数が残る場合は、正の解になるように 1 を代入してみる
    free = sorted({s for v in sol for s in sp.sympify(v).free_symbols}, key=str)
    if free:
        sol = [sp.nsimplify(sp.sympify(v).subs({s: sp.Rational(1, 1)
                                                for s in free})) for v in sol]
    wvals = [sp.nsimplify(v) for v in sol]
    if any((not v.is_Rational) or v <= 0 for v in wvals):
        return None
    # d = 1 の解を整数化する
    dens = [sp.Rational(v).q for v in wvals]
    L = dens[0]
    for q in dens[1:]:
        L = sp.ilcm(L, q)
    wi = [int(sp.Rational(v) * L) for v in wvals]
    g = 0
    for v in wi:
        g = sp.igcd(g, v)
    if g > 1:
        wi = [v // g for v in wi]
        L = L // g
    d = int(L)
    # 検算
    degs = {sum(wi[j] * m[j] for j in range(n)) for m in monoms}
    if len(degs) != 1 or degs.pop() != d or any(v <= 0 for v in wi):
        return None
    return (wi, d)


# ----------------------------------------------------------------------
# ゼロ次元イデアルの次元 (階段の外の単項式を数える)
# ----------------------------------------------------------------------
def _quotient_dimension(gens_ideal, gens) -> Optional[int]:
    """dim_k k[x]/I を数える。ゼロ次元でなければ None (無限次元)。"""
    polys = [sp.expand(g) for g in gens_ideal if sp.expand(g) != 0]
    if not polys:
        return None
    try:
        G = sp.groebner(polys, *gens, order="grevlex")
    except Exception:
        return None
    if 1 in [sp.expand(g) for g in G.exprs]:
        return 0
    lms = []
    for g in G.exprs:
        pg = sp.Poly(g, *gens)
        lms.append(tuple(pg.monoms(order="grevlex")[0]))
    n = len(gens)
    # 各変数について純冪が主導単項式に現れるか (ゼロ次元性)
    bounds = []
    for i in range(n):
        pure = [lm[i] for lm in lms
                if all(e == 0 for j, e in enumerate(lm) if j != i)]
        if not pure:
            return None
        bounds.append(min(pure))
    if any(b == 0 for b in bounds):
        return 0
    count = 0
    for exps in itertools.product(*[range(b) for b in bounds]):
        if not any(all(e >= l for e, l in zip(exps, lm)) for lm in lms):
            count += 1
    return count


def _isolated_at_origin(f, gens, timeout_ms: int = 8000) -> str:
    """原点以外に特異点が無いか (grad f = 0 かつ x != 0 が充足不能か)。"""
    try:
        from certify import _HAS_Z3, _to_z3
        import z3
    except Exception:
        return "unknown"
    if not _HAS_Z3:
        return "unknown"
    zvars = {v: z3.Real(f"s{i}") for i, v in enumerate(gens)}
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    try:
        for v in gens:
            s.add(_to_z3(sp.expand(sp.diff(f, v)), zvars) == 0)
        s.add(z3.Or(*[zvars[v] != 0 for v in gens]))
    except Exception:
        return "unknown"
    r = s.check()
    if r == z3.unsat:
        return "proved"
    if r == z3.sat:
        return "refuted"
    return "unknown"


# ----------------------------------------------------------------------
# Milnor / Tjurina
# ----------------------------------------------------------------------
def milnor_number(f, gens=None, *, use_formula: bool = True,
                  backend: str = "auto"):
    """Milnor 数 mu。孤立特異点でなければ sp.oo。

    backend : 'auto'     -- Singular があれば使い、無ければ sympy
              'singular' -- Singular を使う (無ければ例外)
              'sympy'    -- sympy のグレブナー基底のみ (大域的な値)

    Singular は局所順序 (ds) を使えるので、原点だけの mu が得られる。
    sympy 経路は大域的な sum_p mu_p になることに注意。

    use_formula は擬斉次の公式との突き合わせに使う。公式は複素数体上で
    孤立している場合にのみ有効で、ゼロ次元性の判定 (グレブナー基底の
    主導単項式に各変数の純冪が現れるか) が先に立つ。

    Returns (mu, route)
    """
    f = sp.expand(sp.sympify(f))
    if gens is None:
        gens = sorted(f.free_symbols, key=str)
    gens = tuple(gens)
    if backend in ("auto", "singular"):
        got = _singular_invariants(f, gens)
        if got is not None:
            return (got[0], "Singular (局所順序 ds)")
        if backend == "singular":
            raise RuntimeError("Singular を呼び出せませんでした "
                               "(apt install singular などで導入してください)")

    jac = [sp.diff(f, v) for v in gens]
    dim = _quotient_dimension(jac, gens)
    qh = quasihomogeneous_weights(f, gens) if use_formula else None
    if dim is None:
        # ヤコビアンイデアルがゼロ次元でない = 複素数体上で孤立していない。
        # 擬斉次の公式はこの場合には使えない ((x^2+y^2)^2 が実例: 実点では
        # 原点しか特異でないが、複素では x = +-i y に沿って特異なので
        # mu = oo が正しい)。実数上の孤立性判定では不十分。
        return (sp.oo, "ヤコビアンイデアルがゼロ次元でない "
                       "(複素数体上で孤立特異点でない)")
    if qh:
        w, d = qh
        mu = sp.prod([sp.Rational(d, wi) - 1 for wi in w])
        if mu.is_Integer and int(mu) != dim:
            return (dim, f"グレブナー基底 (大域) / 擬斉次の公式は {mu} で不一致")
        if mu.is_Integer:
            return (dim, f"グレブナー基底 (大域) = 擬斉次の公式 (w={w}, d={d})")
    return (dim, "グレブナー基底 (大域)")


def tjurina_number(f, gens=None, *, backend: str = "auto"):
    """Tjurina 数 tau。Returns (tau, route)

    backend は milnor_number と同じ。
    """
    f = sp.expand(sp.sympify(f))
    if gens is None:
        gens = sorted(f.free_symbols, key=str)
    gens = tuple(gens)
    if backend in ("auto", "singular"):
        got = _singular_invariants(f, gens)
        if got is not None:
            return (got[1], "Singular (局所順序 ds)")
        if backend == "singular":
            raise RuntimeError("Singular を呼び出せませんでした")
    dim = _quotient_dimension([f] + [sp.diff(f, v) for v in gens], gens)
    if dim is None:
        return (sp.oo, "<f, grad f> がゼロ次元でない")
    return (dim, "グレブナー基底 (大域)")


# ----------------------------------------------------------------------
# まとめ
# ----------------------------------------------------------------------
@dataclass
class LocalInvariants:
    f: sp.Expr
    gens: Tuple[sp.Symbol, ...]
    multiplicity: int
    quasihomogeneous: Optional[Tuple[List[int], int]]
    milnor: object
    milnor_route: str
    tjurina: object
    tjurina_route: str
    isolated: str
    notes: List[str]

    def report(self) -> str:
        L = [f"############ 局所不変量: f = {self.f} ############",
             f"  変数 n = {len(self.gens)}",
             f"  重複度 m = {self.multiplicity}"]
        if self.quasihomogeneous:
            w, d = self.quasihomogeneous
            L.append(f"  擬斉次: はい (重み {w}, 次数 {d})"
                     " -> 重み付きブローアップ 1 回で単項式化できる"
                     + ("" if self.isolated == "proved"
                        else " (ただし孤立特異点ではないので mu の公式は使えない)"))
        else:
            L.append("  擬斉次: いいえ")
        L.append(f"  原点以外の特異点: {self.isolated}"
                 + {"proved": " (無い = 孤立)", "refuted": " (ある)",
                    "unknown": " (判定できず)"}[self.isolated])
        L.append(f"  Milnor 数  mu  = {self.milnor}   [{self.milnor_route}]")
        L.append(f"  Tjurina 数 tau = {self.tjurina}   [{self.tjurina_route}]")
        if (self.milnor is not sp.oo and self.tjurina is not sp.oo
                and isinstance(self.milnor, int) and isinstance(self.tjurina, int)):
            if self.milnor == self.tjurina:
                L.append("  mu = tau -> 擬斉次 (K. Saito)")
            else:
                L.append(f"  tau < mu (差 {self.milnor - self.tjurina})"
                         " -> 擬斉次でない")
        for nt in self.notes:
            L.append(f"  [note] {nt}")
        return "\n".join(L)

    def print_report(self) -> None:
        print(self.report())


def singularity_report(f, gens=None, *, timeout_ms: int = 8000,
                       backend: str = "auto") -> LocalInvariants:
    """重複度・擬斉次性・Milnor 数・Tjurina 数をまとめて計算する。"""
    f = sp.expand(sp.sympify(f))
    if gens is None:
        gens = sorted(f.free_symbols, key=str)
    gens = tuple(gens)
    notes: List[str] = []
    iso = _isolated_at_origin(f, gens, timeout_ms)
    mu, mr = milnor_number(f, gens, backend=backend)
    tau, tr = tjurina_number(f, gens, backend=backend)
    if "Singular" in mr:
        notes.append("Singular の局所順序で計算しているので、原点だけの "
                     "mu, tau です。")
    elif iso == "refuted" and mu is not sp.oo:
        notes.append("原点以外にも特異点があるため、大域的な次元は "
                     "sum_p mu_p であり、原点での局所的な mu より大きい "
                     "可能性があります (局所計算には Mora の標準基底が必要)。")
    if iso == "unknown":
        notes.append("孤立性が判定できていないので、mu, tau は大域的な値です。")
    return LocalInvariants(f=f, gens=gens, multiplicity=multiplicity(f, gens),
                           quasihomogeneous=quasihomogeneous_weights(f, gens),
                           milnor=mu, milnor_route=mr,
                           tjurina=tau, tjurina_route=tr,
                           isolated=iso, notes=notes)


# ----------------------------------------------------------------------
# デモ
# ----------------------------------------------------------------------
def _demo():
    x, y, z = sp.symbols("x y z", real=True)
    known = [
        ("A1: x^2+y^2", x**2 + y**2, 1, 1),
        ("A2: x^2+y^3", x**2 + y**3, 2, 2),
        ("A4: x^2+y^5", x**2 + y**5, 4, 4),
        ("E8: x^3+y^5", x**3 + y**5, 8, 8),
        ("D4: x^3-x*y^2", x**3 - x * y**2, 4, 4),
        ("x^3+y^4+z^5", x**3 + y**4 + z**5, 24, 24),
        ("x^2+y^2+z^2", x**2 + y**2 + z**2, 1, 1),
        ("非擬斉次 x^5+y^5+x^2y^2", x**5 + y**5 + x**2 * y**2, None, None),
        ("非孤立 (x-y)^2", (x - y) ** 2, sp.oo, sp.oo),
        ("非孤立 x^2*y^2", x**2 * y**2, sp.oo, sp.oo),
    ]
    print("=" * 70)
    print("既知の値との比較 (mu, tau)")
    print("=" * 70)
    for label, f, mu0, tau0 in known:
        mu, _ = milnor_number(f)
        tau, _ = tjurina_number(f)
        mk = ""
        if mu0 is not None:
            mk = " OK" if (mu == mu0 and tau == tau0) else f" !! 期待 {mu0},{tau0}"
        print(f"  {label:<26} m={multiplicity(f)}  mu={mu}  tau={tau}{mk}")

    print("\n" + "=" * 70)
    print(f"バックエンドの比較 (Singular available: {singular_available()})")
    print("=" * 70)
    print("  sympy 経路は大域グレブナー基底なので sum_p mu_p、")
    print("  Singular 経路は局所順序 ds なので原点だけの mu を返す。")
    for label, f in [("x^5+y^5+x^2y^2", x**5 + y**5 + x**2 * y**2),
                     ("x^2+y^3", x**2 + y**3),
                     ("x^3+y^4+z^5", x**3 + y**4 + z**5)]:
        row = [label]
        for be in ("sympy", "singular"):
            try:
                mu, _ = milnor_number(f, backend=be)
                tau, _ = tjurina_number(f, backend=be)
                row.append(f"{be}: mu={mu} tau={tau}")
            except Exception as e:                       # noqa: BLE001
                row.append(f"{be}: 使えません ({str(e)[:40]})")
        print(f"  {row[0]:<18} " + " | ".join(row[1:]))

    print("\n" + "=" * 70)
    print("詳しい報告")
    print("=" * 70)
    for f in (x**2 + y**3, x**5 + y**5 + x**2 * y**2, (x - y) ** 2,
              x**3 + y**4 + z**5):
        singularity_report(f).print_report()
        print()

    print("=" * 70)
    print("RLCT との関係 (1/m <= lambda <= n/m の確認)")
    print("=" * 70)
    try:
        from certify import rlct_certified
        for label, f, g in [("x^2+y^3", x**2 + y**3, (x, y)),
                            ("x^3+y^4+z^5", x**3 + y**4 + z**5, (x, y, z)),
                            ("x^2+y^2+z^2", x**2 + y**2 + z**2, (x, y, z))]:
            m = multiplicity(f, g)
            lam, st, how = rlct_certified(f, g, timeout_ms=8000)
            n = len(g)
            ok = (sp.Rational(1, m) <= lam <= sp.Rational(n, m))
            print(f"  {label:<16} m={m} n={n} lambda={lam} ({st}) "
                  f"1/m={sp.Rational(1,m)} n/m={sp.Rational(n,m)} "
                  f"-> {'OK' if ok else '!! 不等式が破れている'}")
    except Exception as e:                        # noqa: BLE001
        print(f"  (certify を読み込めません: {e})")


if __name__ == "__main__":
    _demo()
