r"""
bench/nn_cases.py — ニューラルネットの格子ベンチ。

目的は「どこまでなら proved が出るか」を**実測する**こと。予測ではなく
測定値として、次の関係を出す:

    パラメータ数 -> (自由方向 / ゲージ固定 / 正則方向) -> コア変数数 -> proved 率

torch は使わない。純 sympy で MLP を組み、活性化関数ごとに

  relu : theta* での前活性化の符号でセルを固定 (区分線形 -> 多項式)
  tanh : theta* の前活性化まわりのテイラー展開 (taylor_order 次で打ち切り)

として fiber ideal を作り、nn_rlct.local_rlct_from_ideal に渡す。

theta* は「退化した解」を意図的に置く。学習後の重みをそのまま使うと
ほぼ必ず正則点になり (lambda = 実効パラメータ数 / 2)、特異性の情報が
出ないため。退化の型は:

  redundant : 出力重みが 0 のユニットを混ぜる (冗長ユニット)
  duplicate : 同一の隠れユニットを複数置く (置換対称性の固定点)
  dead      : 全データで不活性なユニット (relu のみ)
  zero      : ユニットのパラメータを全て 0 にする (最も深い点)

------------------------------------------------------------------
注意: テイラー打ち切りについて
------------------------------------------------------------------
tanh を k 次で打ち切った多項式の RLCT が元の解析関数の RLCT と一致する
のは、fiber ideal が k 次で有限決定的なときだけである。深いネットでは
その次数が上がるので、打ち切り次数を変えて値が動かないことを確認する
必要がある (taylor_order を振って記録する)。
"""

from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp


# ----------------------------------------------------------------------
# 記号 MLP
# ----------------------------------------------------------------------
def _params(widths: Sequence[int]):
    """各層の重みとバイアスの記号。widths = [n_in, h1, ..., n_out]"""
    W, b = [], []
    for l in range(len(widths) - 1):
        W.append([[sp.Symbol(f"W{l}_{i}_{j}", real=True)
                   for j in range(widths[l])] for i in range(widths[l + 1])])
        b.append([sp.Symbol(f"b{l}_{i}", real=True)
                  for i in range(widths[l + 1])])
    return W, b


def _tanh_taylor(z0, order, max_den: int = 10 ** 4):
    """tanh を z0 のまわりで order 次まで展開した係数 (有理数化)。

    tanh(5/4) のような記号定数を残すと、sympy の多項式が有理数体 QQ では
    なく EX 領域になり、演算が桁で遅くなる (tanh の格子が 7 変数で
    タイムアウトしていた主因)。重みを有理数化するのと同じ扱いで、
    係数も分母を制限した有理数に丸める。
    """
    t = sp.Symbol("_t")
    f = sp.tanh(t)
    out = []
    der = f
    for k in range(order + 1):
        c = der.subs(t, sp.nsimplify(z0)) / sp.factorial(k)
        c = sp.Rational(float(c)).limit_denominator(max_den)
        out.append(c)
        der = sp.diff(der, t)
    return out


@dataclass
class NNCase:
    name: str
    generators: List[sp.Expr]
    variables: Tuple[sp.Symbol, ...]
    symmetries: List[Dict]
    activation: str
    widths: Tuple[int, ...]
    degeneracy: str
    taylor_order: Optional[int]
    n_data: int


def mlp_case(widths, activation="relu", degeneracy="redundant", n_data=6,
             taylor_order=3, seed=0) -> Optional[NNCase]:
    """記号 MLP の fiber ideal を作る。theta* は degeneracy で指定した退化点。"""
    rng = random.Random(seed)
    W, b = _params(widths)
    L = len(widths) - 1

    # --- theta* の数値 -------------------------------------------------
    Ws = [[[sp.Rational(rng.randint(-3, 3), 2) for _ in row] for row in layer]
          for layer in W]
    bs = [[sp.Rational(rng.randint(-2, 2), 2) for _ in layer] for layer in b]
    H = widths[1]
    if degeneracy == "redundant" and H >= 2:
        for i in range(H // 2, H):                     # 後半の出力重みを 0 に
            for r in range(len(Ws[1])):
                Ws[1][r][i] = sp.Integer(0)
    elif degeneracy == "duplicate" and H >= 2:
        for j in range(widths[0]):
            Ws[0][1][j] = Ws[0][0][j]
        bs[0][1] = bs[0][0]
    elif degeneracy == "dead" and H >= 2:
        for j in range(widths[0]):
            Ws[0][H - 1][j] = sp.Integer(0)
        bs[0][H - 1] = sp.Integer(-10)                 # 全データで不活性
    elif degeneracy == "duplicate3" and H >= 3:
        # 同一のユニットを 3 つ: 置換対称性が S_3 になり特異性が深くなる
        for i in (1, 2):
            for j in range(widths[0]):
                Ws[0][i][j] = Ws[0][0][j]
            bs[0][i] = bs[0][0]
    elif degeneracy == "pair" and H >= 4:
        # 2 組の重複ユニット
        for j in range(widths[0]):
            Ws[0][1][j] = Ws[0][0][j]
            Ws[0][3][j] = Ws[0][2][j]
        bs[0][1] = bs[0][0]
        bs[0][3] = bs[0][2]
    elif degeneracy == "opposite" and H >= 2:
        # ReLU(z) - ReLU(-z) = z。符号を反転した 2 ユニットは 1 次関数に潰れる
        for j in range(widths[0]):
            Ws[0][1][j] = -Ws[0][0][j]
        bs[0][1] = -bs[0][0]
        for r in range(len(Ws[1])):
            Ws[1][r][1] = -Ws[1][r][0]
    elif degeneracy == "scaled" and H >= 2:
        # 正斉次性の軌道上に置く: (a,b,c) -> (2a,2b,c/2)
        for j in range(widths[0]):
            Ws[0][1][j] = 2 * Ws[0][0][j]
        bs[0][1] = 2 * bs[0][0]
        for r in range(len(Ws[1])):
            Ws[1][r][1] = Ws[1][r][0] / 2
    elif degeneracy == "zero" and H >= 1:
        for j in range(widths[0]):
            Ws[0][H - 1][j] = sp.Integer(0)
        bs[0][H - 1] = sp.Integer(0)
        for r in range(len(Ws[1])):
            Ws[1][r][H - 1] = sp.Integer(0)

    X = [[sp.Rational(rng.randint(-4, 4), 2) for _ in range(widths[0])]
         for _ in range(n_data)]

    # --- 前向き計算 (記号と数値を並行) ---------------------------------
    def forward(x, symbolic: bool):
        cur = [sp.nsimplify(v) for v in x]
        num = [sp.nsimplify(v) for v in x]
        for l in range(L):
            new_s, new_n = [], []
            for i in range(widths[l + 1]):
                zs = sum((W[l][i][j] if symbolic else Ws[l][i][j]) * cur[j]
                         for j in range(widths[l])) \
                    + (b[l][i] if symbolic else bs[l][i])
                zn = sum(Ws[l][i][j] * num[j] for j in range(widths[l])) \
                    + bs[l][i]
                new_s.append(sp.expand(zs))
                new_n.append(sp.nsimplify(zn))
            if l < L - 1:                              # 中間層のみ活性化
                out_s, out_n = [], []
                for zs, zn in zip(new_s, new_n):
                    if activation == "relu":
                        if zn > 0:
                            out_s.append(zs)
                            out_n.append(zn)
                        elif zn < 0:
                            out_s.append(sp.Integer(0))
                            out_n.append(sp.Integer(0))
                        else:
                            # 前活性化がちょうど 0 = セル境界。不活性側の
                            # セルを取る (真の lambda はこの値以上)。
                            out_s.append(sp.Integer(0))
                            out_n.append(sp.Integer(0))
                    elif activation == "tanh":
                        cs = _tanh_taylor(zn, taylor_order)
                        d = sp.expand(zs - zn)
                        out_s.append(sp.expand(sum(cs[k] * d ** k
                                                   for k in range(len(cs)))))
                        out_n.append(sp.Rational(float(sp.tanh(zn))
                                                 ).limit_denominator(10 ** 4))
                    else:
                        raise ValueError(activation)
                cur, num = out_s, out_n
            else:
                cur, num = new_s, new_n
        return cur, num

    gens: List[sp.Expr] = []
    for x in X:
        r = forward(x, True)
        if r is None:
            return None
        sym, _ = r
        num = forward(x, False)[1]
        for s_, n_ in zip(sym, num):
            gens.append(sp.expand(s_ - n_))

    # --- theta = theta* + u へのシフト ----------------------------------
    U, shift = [], {}
    for l in range(L):
        for i in range(widths[l + 1]):
            for j in range(widths[l]):
                u = sp.Symbol(f"u{l}_{i}_{j}", real=True)
                U.append(u)
                shift[W[l][i][j]] = Ws[l][i][j] + u
            u = sp.Symbol(f"v{l}_{i}", real=True)
            U.append(u)
            shift[b[l][i]] = bs[l][i] + u
    gens = [sp.expand(g.subs(shift, simultaneous=True)) for g in gens]
    gens = [g for g in gens if g != 0]

    # --- relu のスケール対称性 ------------------------------------------
    syms: List[Dict] = []
    if activation == "relu" and L == 2:
        for i in range(widths[1]):
            vec = {}
            for j in range(widths[0]):
                if Ws[0][i][j] != 0:
                    vec[sp.Symbol(f"u0_{i}_{j}", real=True)] = Ws[0][i][j]
            if bs[0][i] != 0:
                vec[sp.Symbol(f"v0_{i}", real=True)] = bs[0][i]
            for r in range(widths[2]):
                if Ws[1][r][i] != 0:
                    vec[sp.Symbol(f"u1_{r}_{i}", real=True)] = -Ws[1][r][i]
            if vec:
                syms.append(vec)

    name = (f"{activation} {'-'.join(map(str, widths))} {degeneracy} "
            f"n={n_data}" + (f" T{taylor_order}" if activation == "tanh" else ""))
    return NNCase(name, gens, tuple(U), syms, activation, tuple(widths),
                  degeneracy, taylor_order if activation == "tanh" else None,
                  n_data)


# ----------------------------------------------------------------------
# 格子
# ----------------------------------------------------------------------
def mlp_case_retry(widths, activation, degeneracy, n_data, taylor_order,
                   tries: int = 6):
    """セル境界に当たったら種を変えて作り直す。"""
    for sd in range(tries):
        c = mlp_case(list(widths), activation, degeneracy, n_data=n_data,
                     taylor_order=taylor_order, seed=sd)
        if c is not None and c.generators:
            return c
    return None


#: ReLU 用の中小規模アーキテクチャ
RELU_ARCHS = [
    (1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1),
    (2, 2, 1), (2, 3, 1), (2, 4, 1),
    (3, 2, 1), (3, 3, 1),
    (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2),
    (1, 2, 1, 1), (1, 3, 2, 1), (2, 2, 2, 1), (1, 4, 2, 1),
]

#: 退化の型。後ろの 4 つは今回追加したもの
RELU_DEGS = ("redundant", "duplicate", "dead", "zero",
             "duplicate3", "pair", "opposite", "scaled")


def relu_specs(*, max_params: int = 48,
               n_data: Sequence[int] = (8, 10, 12, 16),
               degs: Sequence[str] = RELU_DEGS,
               seeds: Sequence[int] = (0, 1, 2, 3),
               archs: Sequence[tuple] = tuple(RELU_ARCHS)) -> List[tuple]:
    """(widths, degeneracy, n_data, seed) の一覧。**式は作らない**。

    格子は数千件になるので、全部の記号 MLP を先に組み立てると構築だけで
    分単位かかる。仕様だけ並べて、必要な分を case_from_spec() で作る。
    """
    out: List[tuple] = []
    for widths in archs:
        npar = sum(widths[l + 1] * widths[l] + widths[l + 1]
                   for l in range(len(widths) - 1))
        if npar > max_params:
            continue
        H = widths[1]
        for deg in degs:
            if deg == "duplicate3" and H < 3:
                continue
            if deg == "pair" and H < 4:
                continue
            if deg in ("duplicate", "opposite", "scaled", "dead") and H < 2:
                continue
            for nd in n_data:
                for sd in seeds:
                    out.append((tuple(widths), deg, int(nd), int(sd)))
    return out


def case_from_spec(spec) -> Optional[NNCase]:
    """relu_specs() の 1 要素から NNCase を作る。"""
    widths, deg, nd, sd = spec
    c = mlp_case(list(widths), "relu", deg, n_data=nd, seed=sd)
    if c is None or not c.generators:
        return None
    c.name = f"{c.name}/nd{nd}/s{sd}"
    return c


def relu_grid(*, max_params: int = 48,
              n_data: Sequence[int] = (8, 10, 12, 16),
              degs: Sequence[str] = RELU_DEGS,
              seeds: Sequence[int] = (0, 1, 2, 3),
              archs: Sequence[tuple] = tuple(RELU_ARCHS)) -> List[NNCase]:
    """ReLU ネットの中小規模な格子。キャッシュを貯めるのに使う。

    **データ点が多いほうがコアが残る**。少ないと fiber ideal の生成元が
    すべて正則方向として消えてしまい (core = 0)、解消が走らないので
    キャッシュに入れる値が生まれない。実測 (2-4-1 redundant) では
    n_data = 2..6 で core = 0、n_data = 10 で core = 5 だった。
    そのため既定の n_data は 8 以上にしてある。

    実測 (n_data = 2,3,4,6,10 / seed 0 のみ / 258 件) のコア変数の分布:
        core=0: 219, 1: 10, 2: 6, 3: 18, 4: 4, 5: 1   -> 非空は 39 件 (15%)
    非空になるのはほぼ n_data = 10 のケースだったので、データ点を増やし
    種も振ることで「貯める価値のあるケース」の収率を上げている。
    """
    out: List[NNCase] = []
    for spec in relu_specs(max_params=max_params, n_data=n_data, degs=degs,
                           seeds=seeds, archs=archs):
        c = case_from_spec(spec)
        if c is not None:
            out.append(c)
    return out


# ----------------------------------------------------------------------
# Transformer 1 層
# ----------------------------------------------------------------------
#: (T, d_model, d_k, d_ff, n_heads) の組
TRANSFORMER_SHAPES = [
    (2, 2, 2, 2, 1), (3, 2, 2, 2, 1), (2, 2, 1, 2, 1),
    (2, 2, 2, 3, 1), (3, 2, 1, 2, 1), (2, 4, 2, 2, 2),
]
TRANSFORMER_DEGS = ("none", "zero_ffn", "dead_ffn", "dup_ffn",
                    "uniform_att", "rank1_v", "dup_head")


def transformer_specs(*, shapes=tuple(TRANSFORMER_SHAPES),
                      degs: Sequence[str] = TRANSFORMER_DEGS,
                      taylor: Sequence[int] = (1, 2),
                      seeds: Sequence[int] = (0, 1)) -> List[tuple]:
    """Transformer 1 層の仕様の一覧 (式は作らない)。"""
    out: List[tuple] = []
    for (T, d, dk, dff, H) in shapes:
        for deg in degs:
            if deg == "dup_head" and H < 2:
                continue
            if deg == "rank1_v" and dk < 2:
                continue
            for od in taylor:
                for sd in seeds:
                    out.append((T, d, dk, dff, H, deg, int(od), int(sd)))
    return out


def transformer_case_from_spec(spec) -> Optional[NNCase]:
    """Transformer 1 層の仕様から NNCase を作る (run_case にそのまま渡せる)。"""
    from rlct.models.transformer_rlct import random_layer, transformer_layer_ideal

    T, d, dk, dff, H, deg, order, sd = spec
    rng = random.Random(1000 + sd)
    X = [[sp.Rational(rng.randint(-2, 2), 2) for _ in range(d)]
         for _ in range(T)]
    star = random_layer(d_model=d, d_k=dk, d_ff=dff, n_heads=H,
                        seed=sd, degeneracy=deg)
    ideal = transformer_layer_ideal(X, star, taylor_order=order)
    if not ideal.generators:
        return None
    return NNCase(
        name=f"transformer T{T} d{d} k{dk} ff{dff} H{H} {deg}/o{order}/s{sd}",
        generators=ideal.generators,
        variables=tuple(ideal.variables),
        symmetries=ideal.symmetry_vectors,
        activation="transformer",
        widths=(T, d, dk, dff, H),
        degeneracy=deg,
        taylor_order=order,
        n_data=T,
    )


def grid(max_params: int = 40) -> List[NNCase]:
    out = []
    archs = [(1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1), (2, 2, 1), (2, 3, 1),
             (1, 2, 2), (2, 2, 2), (1, 2, 1, 1), (1, 3, 2, 1)]
    for widths in archs:
        n_par = sum(widths[l + 1] * widths[l] + widths[l + 1]
                    for l in range(len(widths) - 1))
        if n_par > max_params:
            continue
        for act in ("relu", "tanh"):
            for deg in ("redundant", "duplicate", "dead", "zero"):
                if deg == "dead" and act != "relu":
                    continue
                orders = (2, 3) if act == "tanh" else (None,)
                for od in orders:
                    c = mlp_case_retry(widths, act, deg, 2 * widths[0] + 4,
                                       od or 3)
                    if c is not None and c.generators:
                        out.append(c)
    return out


# ----------------------------------------------------------------------
# 実行
# ----------------------------------------------------------------------
def _cache_store(loc, rec) -> None:
    """証明できたコアの値を永続キャッシュに入れる。

    ニューラルネットではアーキテクチャが違っても同じコアイデアルが何度も
    現れるので、根の値 (K, core_vars) を入れておくのが効く。部分木の値も
    あれば一緒に入れる。**証明できたときだけ**呼ぶこと。
    """
    try:
        import rlct.core.rlct_cache as _rc
    except Exception:
        return
    if _rc.current() is None:
        return
    n = 0
    try:
        if loc.core_gens and loc.core_vars:
            K = sp.expand(sum(g ** 2 for g in loc.core_gens))
            if K != 0:
                _rc.put([0] * len(loc.core_vars), [0] * len(loc.core_vars),
                        sp.Poly(K, *loc.core_vars), tuple(loc.core_vars),
                        loc.lam_core, loc.m_core,
                        f"nn-core:{getattr(loc, 'core_route', '')}")
                n += 1
        if loc.resolution is not None:
            for kk, hh, pp, lam_s, m_s in getattr(
                    loc.resolution, "cache_candidates", []):
                _rc.put(kk, hh, pp, tuple(loc.core_vars), lam_s, m_s,
                        "nn-subtree")
                n += 1
    except Exception:
        pass
    rec["n_cache_put"] = n


def run_case(c: NNCase, *, timeout_ms: int = 4000, max_depth: int = 8,
             timeout_s: int = 40) -> Dict:
    import signal

    from rlct.models.nn_rlct import local_rlct_from_ideal

    class _TO(Exception):
        pass

    def _h(*_):
        raise _TO()

    signal.signal(signal.SIGALRM, _h)
    rec: Dict = {"name": c.name, "activation": c.activation,
                 "arch": "-".join(map(str, c.widths)),
                 "degeneracy": c.degeneracy, "n_params": len(c.variables),
                 "taylor": c.taylor_order, "n_gens": len(c.generators)}
    from rlct.core.timing import add_meta, last_record, record, timed

    t0 = time.time()
    signal.alarm(timeout_s)
    _ctx = record(c.name, activation=c.activation, arch=rec["arch"],
                  n_params=len(c.variables), n_gens=len(c.generators),
                  taylor=c.taylor_order)
    _ctx.__enter__()
    _closed = False
    try:
        with timed("nn_reduce+resolve"):
            loc = local_rlct_from_ideal(c.generators, c.variables,
                                        symmetry_vectors=c.symmetries,
                                        max_depth=max_depth)
    except _TO:
        _ctx.__exit__(None, None, None); _closed = True
        rec["timing"] = last_record().as_dict() if last_record() else None
        rec.update(status="timeout", time=round(time.time() - t0, 1))
        return rec
    except Exception as e:                            # noqa: BLE001
        if not _closed:
            _ctx.__exit__(None, None, None); _closed = True
        rec["timing"] = last_record().as_dict() if last_record() else None
        rec.update(status="error", note=str(e)[:60],
                   time=round(time.time() - t0, 1))
        return rec
    finally:
        signal.alarm(0)
    rec.update(free=len(loc.free_vars), gauge=len(loc.gauge_fixed),
               regular=len(loc.regular_vars), core=len(loc.core_vars),
               rlct=str(loc.rlct), mult=loc.multiplicity)
    add_meta(core=len(loc.core_vars), regular=len(loc.regular_vars))
    if loc.resolution is not None:
        add_meta(n_charts=len(loc.resolution.charts),
                 n_blowups=loc.resolution.n_blowups,
                 n_newton=loc.resolution.n_newton)
        rec["n_charts"] = len(loc.resolution.charts)
        rec["n_newton"] = loc.resolution.n_newton
    # 証明書
    rec["core_route"] = getattr(loc, "core_route", "")
    if loc.resolution is None:
        rec["status"] = "proved"
        # コアが空 (正則方向だけ) か、コア全体がニュートン非退化で厳密か
        rec["cert"] = ("newton" if rec["core_route"] == "newton"
                       else "core-empty")
        _cache_store(loc, rec)
    else:
        try:
            from rlct.core.certify import certify
            signal.alarm(timeout_s)
            with timed("certify"):
                cert = certify(loc.resolution, timeout_ms=timeout_ms)
            signal.alarm(0)
            rec["status"] = cert.status
            rec["cert"] = "certify"
            if cert.status == "proved":
                _cache_store(loc, rec)
        except Exception as e:                        # noqa: BLE001
            signal.alarm(0)
            rec["status"] = "unknown"
            rec["cert"] = f"error:{str(e)[:30]}"
    if not _closed:
        _ctx.__exit__(None, None, None)
    rec["timing"] = last_record().as_dict() if last_record() else None
    rec["time"] = round(time.time() - t0, 1)
    return rec


def summarize(recs: List[Dict]) -> str:
    import collections
    L = [f"# ケース {len(recs)}"]
    cnt = collections.Counter(r.get("status", "?") for r in recs)
    L.append("# status: " + ", ".join(f"{k}={v}" for k, v in cnt.most_common()))

    L.append("# コア変数数ごとの proved 率  <- 本命の測定値")
    by = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        if "core" not in r:
            continue
        s = by[r["core"]]
        s[1] += 1
        s[0] += r.get("status") == "proved"
    for k in sorted(by):
        a, b = by[k]
        L.append(f"    コア {k} 変数: {a}/{b} proved ({100*a/b:5.1f}%)")

    L.append("# パラメータ数 -> コア変数数 (次元削減の効き方)")
    by2 = collections.defaultdict(list)
    for r in recs:
        if "core" in r:
            by2[r["n_params"]].append(r["core"])
    for k in sorted(by2):
        v = by2[k]
        L.append(f"    {k:>3} パラメータ -> コア {min(v)}..{max(v)} "
                 f"(中央 {sorted(v)[len(v)//2]})")

    for key in ("activation", "arch", "degeneracy"):
        L.append(f"# {key} 別")
        by3 = collections.defaultdict(lambda: [0, 0])
        for r in recs:
            s = by3[r.get(key)]
            s[1] += 1
            s[0] += r.get("status") == "proved"
        for k in sorted(by3, key=str):
            a, b = by3[k]
            L.append(f"    {str(k):<12} {a}/{b} proved ({100*a/b:5.1f}%)")

    L.append("# 実行時間 (秒)")
    try:
        from rlct.core.timing import timing_summary
        L.append(timing_summary(recs, by=("activation",)))
        L.append(timing_summary(recs, by=("arch",)))
        L.append(timing_summary(recs, by=("core",)))
        L.append(timing_summary(recs, by=("n_params",)))
    except Exception as e:                            # noqa: BLE001
        L.append(f"    集計に失敗: {e}")

    L.append("# tanh のテイラー打ち切り次数で lambda が動くか")
    seen = collections.defaultdict(dict)
    for r in recs:
        if r.get("activation") == "tanh" and "rlct" in r:
            seen[(r["arch"], r["degeneracy"])][r["taylor"]] = r["rlct"]
    for k, v in sorted(seen.items(), key=str):
        vals = set(v.values())
        L.append(f"    {k}: {v} {'一致' if len(vals) == 1 else '<- 動く'}")
    return "\n".join(L)


def _demo():
    import json
    cases = grid()
    print(f"格子: {len(cases)} ケース")
    recs = []
    for i, c in enumerate(cases):
        r = run_case(c)
        recs.append(r)
        print(f"[{i+1}/{len(cases)}] {c.name:<34} "
              f"vars={r.get('n_params')} core={r.get('core')} "
              f"lambda={r.get('rlct')} {r.get('status')} ({r.get('time')}s)",
              flush=True)
    with open("nn_bench.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=1)
    print()
    print(summarize(recs))


if __name__ == "__main__":
    _demo()
