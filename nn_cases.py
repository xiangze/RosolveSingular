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
def run_case(c: NNCase, *, timeout_ms: int = 4000, max_depth: int = 8,
             timeout_s: int = 40) -> Dict:
    import signal

    from nn_rlct import local_rlct_from_ideal

    class _TO(Exception):
        pass

    def _h(*_):
        raise _TO()

    signal.signal(signal.SIGALRM, _h)
    rec: Dict = {"name": c.name, "activation": c.activation,
                 "arch": "-".join(map(str, c.widths)),
                 "degeneracy": c.degeneracy, "n_params": len(c.variables),
                 "taylor": c.taylor_order, "n_gens": len(c.generators)}
    t0 = time.time()
    signal.alarm(timeout_s)
    try:
        loc = local_rlct_from_ideal(c.generators, c.variables,
                                    symmetry_vectors=c.symmetries,
                                    max_depth=max_depth)
    except _TO:
        rec.update(status="timeout", time=round(time.time() - t0, 1))
        return rec
    except Exception as e:                            # noqa: BLE001
        rec.update(status="error", note=str(e)[:60],
                   time=round(time.time() - t0, 1))
        return rec
    finally:
        signal.alarm(0)
    rec.update(free=len(loc.free_vars), gauge=len(loc.gauge_fixed),
               regular=len(loc.regular_vars), core=len(loc.core_vars),
               rlct=str(loc.rlct), mult=loc.multiplicity)
    # 証明書
    if loc.resolution is None:
        rec["status"] = "proved"          # コアが空 = 正則方向だけで厳密
        rec["cert"] = "core-empty"
    else:
        try:
            from certify import certify
            signal.alarm(timeout_s)
            cert = certify(loc.resolution, timeout_ms=timeout_ms)
            signal.alarm(0)
            rec["status"] = cert.status
            rec["cert"] = "certify"
        except Exception as e:                        # noqa: BLE001
            signal.alarm(0)
            rec["status"] = "unknown"
            rec["cert"] = f"error:{str(e)[:30]}"
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
