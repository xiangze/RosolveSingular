r"""
lemmas.py — Aoyagi の補題を「計算の短縮」に使う経路。

`bench/aoyagi_lemmas.py` は同じ補題を *検査* (真値を使わない整合性チェック)
に使っている。ここでは同じ関係式を *計算* に使う。使えるものと使えない
ものの区別が肝心なので、根拠と反例をすべて明記する。

------------------------------------------------------------------
(1) 使える: Lemma 1(1) 単調性 -> 下界、Newton LP -> 上界、挟み込み
------------------------------------------------------------------
S ⊂ {1..m} に対し、各点で

    sum_{i in S} g_i^2  <=  sum_{i=1}^m g_i^2

なので Lemma 1(1) から

    lambda(部分生成元 S)  <=  lambda(全体)

すなわち **部分生成元の lambda は全体の lambda の厳密な下界**である。
一方 newton_rlct(f) は非退化かどうかに関係なく常に上界 (Varchenko / Lin)。
したがって

    lambda_lo := max_S lambda(S)   <=   lambda   <=   lambda_hi := newton_rlct(f)

で、両端が一致すれば **ブローアップを一切せずに lambda が確定する**。
部分生成元は変数も次数も小さいので、族ごとの閉じた式 (families.py) だけで
厳密に出ることが多い。

*注意*: 挟み込みで決まるのは極の位置 lambda だけで、**位数 m は決まらない**。
この経路を採った場合 m は下限として返す (prune='ties' と同じ扱い)。

さらに lambda_lo は分枝限定の大域下界としても使える。到達値 (incumbent) が
lambda_lo に達したら、どの枝もそれより小さくできないので探索を打ち切れる。

------------------------------------------------------------------
(2) 使える: 積の分解 (任意の chart で)
------------------------------------------------------------------
chart の局所データ (y^k * f_rest, 振幅 y^h) で、f_rest が
**変数が互いに素な因子**の積 f_rest = A(u) * B(v) に分かれるなら、
k, h も u, v に分かれるので積分が完全に分離する:

    zeta(z) = ∫|u^{k_u} A|^z u^{h_u} du * ∫|v^{k_v} B|^z v^{h_v} dv

よって lambda = min(lambda_A, lambda_B)、位数は最小を与える側の和。
**k, h が何であっても成り立つ**ので、木のどのノードでも使える。

------------------------------------------------------------------
(3) 使えない: Lemma 2 (和の分離) を一般の chart で
------------------------------------------------------------------
f_rest = A(u) + B(v) と変数分離できても、局所データは

    P = y^k (A(u) + B(v)) = u^{k_u} v^{k_v} A + u^{k_u} v^{k_v} B

で、単項式因子が**両方の項に共通に掛かる**ため P 自体は変数分離しない。

反例: x(x^2 + y^2) = x^3 + x y^2 は (ニュートン非退化で) lambda = 2/3。
一方 lambda(x^3) + lambda(y^2) = 1/3 + 1/2 = 5/6 で一致しない。

したがって和の分離が使えるのは k = 0 のノードだけ (根・座標変換直後)。
それは families.py の和の経路が既に担当している。

------------------------------------------------------------------
(4) 使えない: Theorem 2 を「付け替えを省く」根拠に
------------------------------------------------------------------
「斉次なら最深点は原点だから中心の付け替えは要らない」は**誤り**。

反例: x^2 + y^2 - z^2 は斉次だが、原点での lambda は 1、錐上の原点以外の
点は滑らかなので lambda = 1/2 で、原点は最深ではない。

Theorem 2 は「f が変数群 w_1..w_j について斉次なら、その群を 0 に落としても
lambda は増えない」という**別々の変数群**についての主張であり、1 つの chart の
中で原点と例外因子上の点を比べる根拠にはならない。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import sympy as sp

__all__ = ["Squeeze", "monotone_lower_bound", "squeeze_rlct",
           "product_split_value"]

try:
    from timing import timed as _timed
except Exception:      # pragma: no cover
    from contextlib import contextmanager as _cm

    @_cm
    def _timed(_name):
        yield


# ----------------------------------------------------------------------
# (1) 単調性による下界と挟み込み
# ----------------------------------------------------------------------
@dataclass
class Squeeze:
    """挟み込みの結果。"""

    lo: Optional[sp.Rational] = None      # 部分生成元からの下界 (厳密)
    hi: Optional[sp.Rational] = None      # Newton LP による上界 (厳密)
    rlct: Optional[sp.Rational] = None    # lo == hi のときだけ値が入る
    status: str = "unknown"               # 'proved' | 'unknown'
    witness: str = ""                     # 下界を与えた部分生成元
    n_sub: int = 0                        # 試した部分集合の数

    def __str__(self) -> str:
        return (f"[{self.status}] lambda ∈ [{self.lo}, {self.hi}]"
                + (f" -> {self.rlct}" if self.rlct is not None else "")
                + (f"  (下界: {self.witness})" if self.witness else ""))


def monotone_lower_bound(generators: Sequence[sp.Expr], gens,
                         *, max_subset: int = 1, max_calls: int = 3,
                         time_budget: float = 0.5, max_terms: int = 40,
                         verbose: bool = False
                         ) -> Tuple[Optional[sp.Rational], str, int]:
    """Lemma 1(1) から lambda の厳密な下界を作る。

    生成元の部分集合 S について lambda(sum_{i in S} g_i^2) を族ごとの
    閉じた式で求め、**厳密に出たものだけ**を採用して最大値を返す。
    どれも出なければ (None, "", 0)。

    族経路は「証明された値」しか proved を返さないので、ここで採る値は
    常に真の下界であり、誤って大きすぎる下界を作ることはない。

    **コストの上限**: 族経路は内部でニュートン非退化判定 (凸包 + Z3) を
    呼ぶので、大きな生成元に当てると探索本体より高くつく。実測 (tanh 1-2-1,
    7 変数) では上限なしだと 7 秒 -> 146 秒になった。そこで
      * 部分集合の大きさは既定で 1 (単独の生成元だけ)
      * 呼び出し回数 max_calls、項数 max_terms、壁時計 time_budget 秒
    で打ち切る。下界が出なくても正しさには影響しない (単に使わないだけ)。

    既定は「安い 1 回目」の設定 (0.5 秒)。呼び出し側は解消が失敗したときだけ
    予算を増やして呼び直す (payer-when-needed): 易しいケースでは overhead を
    ほぼ 0 にし、難しいケースでだけ本気で探す。

    Returns (lower_bound, witness, n_calls)
    """
    import time as _time

    from families import family_rlct

    gens = tuple(gens)
    G = [sp.expand(g) for g in generators if sp.expand(g) != 0]
    if len(G) < 2:
        return (None, "", 0)

    best: Optional[sp.Rational] = None
    witness = ""
    calls = 0
    t0 = _time.time()
    # 生成元ごとの項数を先に数えておく。sum g_i^2 を展開してから大きさを見ると、
    # その展開自体が支配的コストになる (tanh のコアでは 20 項の生成元が
    # 2 乗で 235 項になり、捨てる集合の展開だけで数秒かかった)。
    sizes = []
    for g in G:
        try:
            sizes.append(len(sp.Poly(g, *gens).monoms()))
        except Exception:
            sizes.append(10 ** 6)

    # 小さい部分集合から。変数が少ないほど族経路で決まりやすい。
    for k in range(1, min(max_subset, len(G) - 1) + 1):
        for S in itertools.combinations(range(len(G)), k):
            if calls >= max_calls or _time.time() - t0 > time_budget:
                break
            # 展開前の見積もり: sum_i g_i^2 の項数は高々 sum_i n_i(n_i+1)/2
            if sum(sizes[i] * (sizes[i] + 1) // 2 for i in S) > max_terms:
                continue
            sub = sp.expand(sum(G[i] ** 2 for i in S))
            if sub == 0:
                continue
            used = tuple(v for v in gens if v in sub.free_symbols)
            if not used:
                continue
            calls += 1
            try:
                r = family_rlct(sub, used)
            except Exception:
                continue
            if r.status != "proved" or r.rlct is sp.oo:
                continue
            # 使われていない変数は積分の体積因子にしかならないので
            # lambda は変わらない (自由方向)。そのまま下界として使える。
            if best is None or r.rlct > best:
                best, witness = r.rlct, f"生成元 {list(S)} -> {r.route}"
                if verbose:
                    print(f"    下界 {best} ({witness})")
        if calls >= max_calls or _time.time() - t0 > time_budget:
            break
    return (best, witness, calls)


def squeeze_rlct(f, gens, generators: Optional[Sequence[sp.Expr]] = None,
                 *, max_subset: int = 1, max_calls: int = 3,
                 time_budget: float = 0.5, verbose: bool = False) -> Squeeze:
    """lambda_lo <= lambda <= lambda_hi を作り、一致すれば値を確定する。

    generators を渡せば Lemma 1(1) の下界が使える。渡さない場合でも
    上界 (Newton LP) は計算するので、分枝限定の incumbent には使える。

    位数 m は決まらない点に注意 (Squeeze は lambda だけを返す)。
    """
    from newton import newton_rlct

    gens = tuple(gens)
    out = Squeeze()
    with _timed("squeeze"):
        try:
            out.hi = newton_rlct(f, gens)
        except Exception:
            out.hi = None
        if generators:
            lo, w, n = monotone_lower_bound(generators, gens,
                                            max_subset=max_subset,
                                            max_calls=max_calls,
                                            time_budget=time_budget,
                                            verbose=verbose)
            out.lo, out.witness, out.n_sub = lo, w, n
        if (out.lo is not None and out.hi is not None
                and out.hi is not sp.oo):
            if out.lo > out.hi:
                # 下界 > 上界 はどちらかの実装のバグ。値は返さない。
                out.status = "inconsistent"
                return out
            if out.lo == out.hi:
                out.rlct = out.lo
                out.status = "proved"
    return out


# ----------------------------------------------------------------------
# (2) 積の分解 (chart ノードで使える)
# ----------------------------------------------------------------------
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _may_split(poly: sp.Poly) -> bool:
    """factor_list を呼ぶ前の、**健全で O(1) に近い**ふるい。

    f_rest は正規化で単項式因子を括り出した後なので、互いに素な変数群への
    分解 f_rest = A(u) * B(v) があれば A も B も 2 項以上をもつ。
    このとき指数ベクトルの台は直積 S = S_u × S_v になるので

        |S| = |S_u| * |S_v|,   |S_u|, |S_v| >= 2

    すなわち **単項式の個数は 4 以上の合成数**でなければならない。
    素数や 3 以下なら分解はありえないので factor_list を呼ばずに済む。
    (実測: tanh 1-2-1 のコアで product_split の時間が 44.8 秒 -> ほぼ 0 に)
    """
    N = len(poly.monoms())
    return N >= 4 and not _is_prime(N)


def product_split_value(k: Sequence[int], poly: sp.Poly, h: Sequence[int],
                        gens: Sequence[sp.Symbol], *, resolve_fn=None,
                        max_vars: int = 24, max_terms: int = 40,
                        max_degree: int = 24, cache: Optional[dict] = None):
    """f_rest が変数の互いに素な因子に分かれるとき (lambda, m) を返す。

    局所データ (y^k * f_rest, 振幅 y^h) は、f_rest = A(u) * B(v) で u, v が
    互いに素なら完全に分離するので

        lambda = min(lambda_A, lambda_B),
        m      = 最小を与える側の m の和

    A に対応する部分問題は「変数 u だけ、k|_u, h|_u を初期値とする解消」。
    u に現れない変数は体積因子にしかならないので落としてよい。

    resolve_fn(f, vars, h0=..., k0=...) は再帰用。None なら
    resolve_singularities を使う。分かれなければ None。
    """
    from families import disjoint_factors

    gens = tuple(gens)
    if len(gens) > max_vars:
        return None
    # factor_list は大きな多項式で非常に高くつく。健全なふるいで先に落とす。
    if (len(poly.monoms()) > max_terms
            or poly.total_degree() > max_degree
            or not _may_split(poly)):
        return None
    expr = poly.as_expr()
    if not expr.free_symbols:
        return None
    key = None
    if cache is not None:
        key = (tuple(k), tuple(h), tuple(poly.monoms()),
               tuple(map(str, poly.coeffs())))
        if key in cache:
            return cache[key]

    parts = disjoint_factors(expr, gens)
    if not parts or len(parts) < 2:
        if cache is not None:
            cache[key] = None
        return None

    # 各因子が使う変数。どの因子にも現れない変数 y_j は f_rest に現れないので
    # その方向の局所データは単項式 y_j^{k_j} (振幅 y_j^{h_j}) だけ。これも
    # 互いに素な「因子」なので、1 変数の部分問題として同じ規則で合流させる。
    supports = [tuple(v for v in gens if v in p.free_symbols) for p in parts]
    seen = set()
    for s in supports:
        seen |= set(s)

    result = None
    try:
        vals: List[Tuple[sp.Rational, int]] = []
        for p, sup in zip(parts, supports):
            if not sup:
                continue
            v = _sub_local_rlct(p, sup, k, h, gens)
            if v is None:
                vals = None
                break
            vals.append(v)
        if vals is not None:
            for j, v in enumerate(gens):
                if v in seen:
                    continue
                if k[j] > 0:                       # 単項式方向
                    vals.append((sp.Rational(h[j] + 1, k[j]), 1))
            if vals:
                lam = min(v[0] for v in vals)
                result = (lam, sum(m for l, m in vals if l == lam))
    except Exception:
        result = None
    if cache is not None:
        cache[key] = result
    return result


def _sub_local_rlct(p, sup, k, h, gens):
    """部分問題 (y^{k|sup} * p, 振幅 y^{h|sup}) の (lambda, m) を厳密に求める。

    まずニュートン非退化を試し、だめならブローアップして certify にかける。
    証明できなければ None (呼び出し側は分解をあきらめる)。
    """
    from certify import certify
    from newton import newton_multiplicity, newton_nondegenerate
    from resolve_singularity import ResolutionFailure, resolve_singularities
    from newton import newton_rlct

    vs = tuple(sup)
    kk = [k[gens.index(v)] for v in vs]
    hh = [h[gens.index(v)] for v in vs]
    mono = sp.prod([v ** e for v, e in zip(vs, kk)])
    sub = sp.expand(mono * p)

    st, _ = newton_nondegenerate(sub, vs, timeout_ms=4000)
    if st == "proved":
        lam = newton_rlct(sub, vs, h=hh)
        if lam is not sp.oo:
            m = newton_multiplicity(sub, vs, h=hh)
            return (sp.nsimplify(lam), int(m) if m else 1)
    try:
        # product_split=False: 部分問題の中で再び分解を試すと無限再帰になる
        res = resolve_singularities(sub, vs, prune=False, weighted=True,
                                    h0=hh, max_depth=12, product_split=False)
    except ResolutionFailure:
        return None
    except Exception:
        return None
    if certify(res, timeout_ms=4000).status != "proved":
        return None
    if res.rlct is sp.oo:
        return None
    return (res.rlct, res.multiplicity)
