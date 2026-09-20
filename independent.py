r"""
bench/independent.py — 真値を使わない、生成側と何も共有しない検査。

証明書の弱点は、生成側 (resolve_singularity) と検査側 (certify) が同じ
phi, k, h を共有していることだった。共通モード故障はそこを素通りする。
ここでは **解消の結果を一切使わない** 二つの検査を用意する。

  rlct_monte_carlo(f, gens)   体積の漸近から lambda を数値推定する
  covering_sample_check(res)  ランダムな点が本当にどれかの chart の像に
                              入るかを直接確かめる

どちらも誤差や見落としがあるので「proved を与える」ためには使えない。
**間違いを間違いと判定する**ためだけに使う。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp


# ----------------------------------------------------------------------
# 1. 体積の漸近による lambda の数値推定
# ----------------------------------------------------------------------
@dataclass
class MonteCarloRLCT:
    lam_hat: Optional[float]
    r2: Optional[float]
    n_samples: int
    points: List[Tuple[float, float]]      # (log t, log V(t))
    note: str = ""

    def __str__(self):
        if self.lam_hat is None:
            return f"lambda_hat=?  ({self.note})"
        return (f"lambda_hat={self.lam_hat:.4f}  R^2={self.r2:.4f}  "
                f"N={self.n_samples}{(' ' + self.note) if self.note else ''}")


def rlct_monte_carlo(f, gens, *, eps: float = 0.3, n_samples: int = 200000,
                     n_levels: int = 12, seed: int = 0) -> MonteCarloRLCT:
    """体積の漸近 V(t) = vol{|x| < eps, |f(x)| < t} ~ c t^lambda から推定する。

    RLCT の定義と同値な特徴づけ:

        V(t) ~ c * t^lambda * (log 1/t)^{m-1}   (t -> 0)

    なので log V を log t に回帰した傾きが lambda。位数 m による log 補正は
    傾きを少しだけ歪めるが、1/2 と 1 のような差は十分見分けられる。

    **解消の結果を一切使わない**ので、chart の見落としや被覆の不足に対する
    独立な検出器になる。誤差があるので proved の根拠にはならない。
    """
    gens = tuple(gens)
    n = len(gens)
    fn = sp.lambdify(gens, sp.expand(f), "math")
    rng = random.Random(seed)

    vals = []
    for _ in range(n_samples):
        pt = [rng.uniform(-eps, eps) for _ in range(n)]
        try:
            v = abs(float(fn(*pt)))
        except Exception:
            continue
        if v == v and v != float("inf"):
            vals.append(v)
    if not vals:
        return MonteCarloRLCT(None, None, 0, [], "評価できませんでした")
    vals.sort()
    N = len(vals)

    # t の水準は |f| の分位点から取る。こうすると lambda の大小によらず
    # どの水準にも十分な件数が入る (固定の t 範囲だと lambda >= 1 で
    # 件数が足りなくなる)。
    pts: List[Tuple[float, float]] = []
    q_hi, q_lo = 0.05, 0.0005
    for i in range(n_levels):
        q = q_hi * (q_lo / q_hi) ** (i / max(n_levels - 1, 1))
        cnt = max(int(q * N), 1)
        if cnt < 30 or cnt >= N:
            continue
        t = vals[cnt - 1]
        if t <= 0:
            continue
        pts.append((math.log(t), math.log(cnt / N)))
    if len(pts) < 4:
        return MonteCarloRLCT(None, None, N, pts,
                              "十分な水準が取れません (サンプルを増やすか "
                              "t_hi を上げてください)")
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    if sxx == 0:
        return MonteCarloRLCT(None, None, N, pts, "回帰できません")
    slope = sxy / sxx
    r2 = (sxy ** 2) / (sxx * syy) if syy > 0 else 1.0
    return MonteCarloRLCT(slope, r2, N, pts)


def monte_carlo_disagrees(f, gens, lam, *, tol: float = 0.18, **kw) -> Optional[str]:
    """数値推定が lam と食い違っていれば理由の文字列を返す。

    tol は緩めに取る。log 補正 (位数 m) と有限サンプルの誤差があるため、
    厳密な一致は期待できない。1/2 と 1 のような取り違えを捕まえるのが目的。
    """
    mc = rlct_monte_carlo(f, gens, **kw)
    if mc.lam_hat is None or lam is None:
        return None
    lam_f = float(lam)
    # 位数 m >= 2 の log 補正は傾きを下げる向きに効くので、下側を少し広く取る
    if mc.lam_hat > lam_f * (1 + tol) + 0.05:
        return (f"数値推定 {mc.lam_hat:.3f} が lambda={lam} より有意に大きい "
                f"(R^2={mc.r2:.3f})")
    if mc.lam_hat < lam_f * (1 - tol) - 0.15:
        return (f"数値推定 {mc.lam_hat:.3f} が lambda={lam} より有意に小さい "
                f"(R^2={mc.r2:.3f})")
    return None


# ----------------------------------------------------------------------
# 2. 被覆のランダム点検証
# ----------------------------------------------------------------------
def covering_sample_check(resolution, *, eps=sp.Rational(1, 2),
                          n_per_chart: int = 4000, grid: int = 4,
                          seed: int = 0) -> Dict:
    """chart の像が元の箱を覆うかを、前向きの標本で確かめる。

    phi(y) = x を解くのではなく、各 chart の定義域から y を大量に取って
    x = phi(y) を計算し、元の箱を格子に切ってどのセルが当たるかを見る。
    逆問題を解かないので数値的に頑健。

    一度も当たらないセルがあれば **被覆の不足の候補** になる (標本誤差が
    あるので確定ではない)。k, h, f_rest を一切使わないので、生成側と
    独立な検査になる。
    """
    import numpy as np

    gens = resolution.gens
    n = len(gens)
    if n > 4:
        return {"status": "unknown", "note": "変数が多すぎます (n <= 4)"}
    from certify import compute_domains
    domains = compute_domains(resolution, eps)
    charts = list(resolution.charts)
    if not charts:
        return {"status": "unknown", "note": "chart がありません"}

    e = float(eps)
    cells_total = grid ** n
    rng = np.random.default_rng(seed)

    def cell_of(x):
        idx = 0
        for v in x:
            k = int((v + e) / (2 * e) * grid)
            k = min(max(k, 0), grid - 1)
            idx = idx * grid + k
        return idx

    hit = set()
    used = 0
    for ch in charts:
        box = domains.get(ch.name)
        if box is None:
            continue
        try:
            fn = sp.lambdify(gens, [sp.expand(ch.phi[v]) for v in gens], "numpy")
        except Exception:
            continue
        b = np.array([float(v) for v in box.bounds])
        Y = rng.uniform(-b, b, size=(n_per_chart, n))
        try:
            X = np.array(fn(*[Y[:, j] for j in range(n)]), dtype=float).T
        except Exception:
            continue
        used += 1
        inside = np.all(np.abs(X) <= e + 1e-12, axis=1)
        for x in X[inside]:
            hit.add(cell_of(x))
    miss = cells_total - len(hit)
    status = ("consistent" if miss == 0 else
              ("unknown" if used == 0 else "gap"))
    return {"status": status, "cells": cells_total, "hit": len(hit),
            "miss": miss, "charts_used": used,
            "note": ("全セルが覆われました" if miss == 0 else
                     f"{miss} セルが一度も覆われませんでした (被覆の不足の候補)")}


# ----------------------------------------------------------------------
# デモ
# ----------------------------------------------------------------------
def _demo():
    import sympy as sp
    from resolve_singularity import resolve_singularities

    x, y, z = sp.symbols("x y z", real=True)

    print("=" * 72)
    print("1. 体積の漸近による lambda の数値推定 (解消の結果を使わない)")
    print("=" * 72)
    known = [("x^2", x**2, (x,), sp.Rational(1, 2)),
             ("x^2+y^2", x**2 + y**2, (x, y), sp.Integer(1)),
             ("x^2*y^2", x**2 * y**2, (x, y), sp.Rational(1, 2)),
             ("(x-y)^2", (x - y) ** 2, (x, y), sp.Rational(1, 2)),
             ("x^2+y^3", x**2 + y**3, (x, y), sp.Rational(5, 6)),
             ("x^2+y^2+z^2", x**2 + y**2 + z**2, (x, y, z), sp.Rational(3, 2))]
    for label, f, g, lam in known:
        mc = rlct_monte_carlo(f, g, n_samples=120000)
        bad = monte_carlo_disagrees(f, g, lam, n_samples=120000)
        print(f"  {label:<14} 真値 {str(lam):<5} {mc}"
              + ("   ** " + bad if bad else "   一致"))

    print("\\n  誤った値を与えると検出できるか:")
    for label, f, g, wrong in [("(x-y)^2 に lambda=1", (x - y) ** 2, (x, y), 1),
                               ("x^2+y^2 に lambda=1/2", x**2 + y**2, (x, y),
                                sp.Rational(1, 2))]:
        bad = monte_carlo_disagrees(f, g, wrong, n_samples=120000)
        print(f"    {label:<24} -> {bad or '検出できず'}")

    print("\n" + "=" * 72)
    print("2. 被覆の前向き標本検証 (k, h, f_rest を使わない)")
    print("=" * 72)
    import copy
    for label, f, g in [("x^2+y^2", x**2 + y**2, (x, y)),
                        ("x^2+y^3", x**2 + y**3, (x, y)),
                        ("x^2+y^2+z^2", x**2 + y**2 + z**2, (x, y, z))]:
        res = resolve_singularities(f, g, prune=False, weighted=True)
        r = covering_sample_check(res)
        print(f"  {label:<14} {r['status']:<11} セル {r['hit']}/{r['cells']} "
              f"(chart {len(res.charts)})")
        broken = copy.deepcopy(res)
        if len(broken.charts) >= 2:
            broken.charts = broken.charts[:-1]
            rb = covering_sample_check(broken)
            print(f"    chart を 1 個落とすと -> {rb['status']:<11} "
                  f"セル {rb['hit']}/{rb['cells']}  "
                  f"{'検出' if rb['miss'] > 0 else '検出できず'}")


if __name__ == "__main__":
    _demo()
