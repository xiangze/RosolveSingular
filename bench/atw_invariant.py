r"""
bench/atw_invariant.py — **ATW の不変量 inv_p の試作と検算**

D. Abramovich, M. Temkin, J. Włodarczyk, *Functorial embedded resolution via
weighted blowings up*, arXiv:1906.07106 = Algebra & Number Theory 18-8 (2024).
関連: J. Włodarczyk, *Functorial resolution by torus actions*, arXiv:2203.03090.

**これは調査用の試作であって、解消器には組み込んでいない。**
組み込まない理由は `SPEEDUP_SURVEY.ja.md` §C1-0 に実測つきで書いた
(要点: 最大接触元のための線形座標変換で台が約 6 倍密になり、
コンパクト面が約 9 倍に増えて、Newton 高速路を失う)。

定義 (§5):
  b_1 = ord_p(f) = a_1
  x_1 = 最大接触元 (ord_p(d^{b_1-1} f / dx_1^{b_1-1}) = 1 となる座標)
  C(I,b) = < (D^a g)^{b!/(b-|a|)} : g in I の生成元, |a| < b >
  b_{i+1} = ord_p( C(I,b_i)|_{V(x_i)} )
  a_{i+1} = b_{i+1} / (b_i - 1)!
  中心の重み  w ∝ 1/a    (中心は (x_1^{1/w_1}, ..., x_k^{1/w_k}))

**階乗は不変量の計算には入らない。**order だけが必要で
  ord((D^a f)^e) = e * ord(D^a f),   ord(イデアル) = min(生成元の ord)
なので冪を展開しない。爆発するのは係数イデアルを実際にブローアップする
場合 (Villamayor 流) であって、ここではない。

検算 (`self_check()`): 論文の既知値 2 件を再現する。
  x^5+x^3y^3+y^8 -> inv_p = (5, 15/2), b_2 = 180, 中心 (x^{1/3}, y^{1/2})
  x^5+x^3y^3+y^7 -> inv_p = (5, 7),                中心 (x^{1/7}, y^{1/5})
"""

import itertools
import math
import sys
import time
import sympy as sp


def ordo(expr, gens):
    expr = sp.expand(expr)
    if expr == 0:
        return None                     # ord = infinity
    return min(sum(m) for m in sp.Poly(expr, *gens).monoms())


def derivs(f, gens, upto):
    """|a| <= upto の D^a f を返す {|a|: [式]}。"""
    out = {0: [sp.expand(f)]}
    for k in range(1, upto + 1):
        cur = []
        for a in itertools.combinations_with_replacement(range(len(gens)), k):
            d = f
            for i in a:
                d = sp.diff(d, gens[i])
            d = sp.expand(d)
            if d != 0:
                cur.append(d)
        out[k] = cur
    return out


def inv_p(f, gens, depth=4, verbose=True):
    """inv_p(f) の最初の depth 成分と、各段の最大接触元。"""
    gens = list(gens)
    a_list, contacts = [], []
    # 段 1 は超曲面 f そのもの
    items = [(sp.expand(f), 1)]            # (多項式, 冪)
    cur_gens = list(gens)
    prev_b = None
    for step in range(depth):
        # イデアルの order = 生成元の order の min (冪は e*ord)
        os = [(e * ordo(g, cur_gens)) for g, e in items
              if ordo(g, cur_gens) is not None]
        if not os:
            if verbose: print(f"      段 {step+1}: order = ∞ (消えた)")
            break
        b = min(os)
        a = sp.Integer(b) if step == 0 else sp.Rational(b, math.factorial(prev_b - 1))
        a_list.append(a)
        if verbose:
            print(f"      段 {step+1}: 変数 {len(cur_gens)} 本  生成元 {len(items)} 本  "
                  f"ord = {b}  -> a = {a}")
        if step == depth - 1 or b <= 1:
            break
        # 最大接触元: ord(d^{b-1} g / dx^{b-1}) = 1 となる座標
        i_mc = None
        for i, v in enumerate(cur_gens):
            for g, e in items:
                if e != 1:
                    continue
                d = sp.expand(sp.diff(g, v, b - 1))
                if ordo(d, cur_gens) == 1:
                    i_mc = i; break
            if i_mc is not None: break
        if i_mc is None:
            if verbose:
                print("        最大接触元が座標の中に無い -> 座標変換が必要 (ここで打ち切り)")
            break
        xi = cur_gens[i_mc]
        contacts.append(xi)
        rest = [v for v in cur_gens if v is not xi]
        # C(I,b) の生成元を (多項式, 冪) のまま作り、V(xi) に制限する
        fact = math.factorial(b)
        new_items = []
        for g, e in items:
            dd = derivs(g, cur_gens, b - 1)
            for k, lst in dd.items():
                if k >= b: continue
                ee = e * (fact // (b - k))
                for d in lst:
                    dr = sp.expand(d.subs({xi: 0}))
                    if dr != 0:
                        new_items.append((dr, ee))
        if verbose:
            print(f"        最大接触 {xi} -> C(I,{b}) 制限後 {len(new_items)} 本 "
                  f"(冪は展開しない。最大冪 {max((e for _, e in new_items), default=0)})")
        items = new_items
        cur_gens = rest
        prev_b = b
        if not cur_gens:
            break
    return a_list, contacts


def weights(a_list):
    inv = [sp.Rational(1) / sp.Rational(a) for a in a_list]
    L = sp.ilcm(*[sp.Rational(v).q for v in inv]) if len(inv) > 1 else sp.Rational(inv[0]).q
    w = [int(sp.Rational(v) * L) for v in inv]
    g = sp.igcd(*w) if len(w) > 1 else w[0]
    return [v // max(g, 1) for v in w]



def self_check(verbose: bool = True):
    """論文の既知値を再現するか。戻り値は NG のリスト。"""
    x, y = sp.symbols("x y", real=True)
    bad = []
    for nm, f, truth_a, truth_w in [
            ("x^5+x^3y^3+y^8", x**5 + x**3*y**3 + y**8,
             [sp.Integer(5), sp.Rational(15, 2)], [3, 2]),
            ("x^5+x^3y^3+y^7", x**5 + x**3*y**3 + y**7,
             [sp.Integer(5), sp.Integer(7)], [7, 5])]:
        a, _c = inv_p(f, (x, y), depth=2, verbose=False)
        w = weights(a) if a else None
        ok_a = [sp.Rational(v) for v in a] == [sp.Rational(v) for v in truth_a]
        ok_w = (w == truth_w)
        if not ok_a:
            bad.append(f"{nm}: inv_p {a} != 論文 {truth_a}")
        if not ok_w:
            bad.append(f"{nm}: 重み {w} != 論文 {truth_w}")
        if verbose:
            print(f"  {nm:<16} inv_p = {a} (論文 {truth_a}) "
                  f"{'OK' if ok_a else '**NG**'}   "
                  f"重み {w} (論文 {truth_w}) {'OK' if ok_w else '**NG**'}")
    if verbose:
        print(f"atw_invariant.self_check: "
              f"{'全て OK' if not bad else f'{len(bad)} 件 NG'}")
    return bad


if __name__ == "__main__":        # pragma: no cover
    sys.exit(1 if self_check() else 0)
