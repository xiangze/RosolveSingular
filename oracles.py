r"""
bench/oracles.py — proved の中身を検査するための独立な値。

証明書が proved を返しても、証明していない前提 (Watanabe の定理など) が
残っているので「値が正しい」とは限らない。独立な経路の値と突き合わせる。

  family_oracle : families.py の閉じた式 (単項式/積/和/形式のべき/非退化)
  cross_oracles : 同じ f を設定違いで解いて一致を見る
"""
from __future__ import annotations

from typing import Dict, List, Optional

import sympy as sp

from families import family_rlct
from resolve_singularity import (ResolutionFailure, newton_rlct,
                                 resolve_singularities)


def family_oracle(f, gens):
    r = family_rlct(f, gens)
    return (r.rlct, r.multiplicity, r.route) if r.status == "proved" else (None, None, "-")


def cross_oracles(f, gens, max_depth: int = 20, timeout: float = 30.0):
    """設定違いで解いた lambda を集める。全部一致すべき。"""
    out: Dict[str, object] = {}
    settings = [
        ("weighted", dict(weighted=True, prune=False)),
        ("plain", dict(weighted=False, prune=False)),
        ("ties", dict(weighted=True, prune="ties")),
        ("support", dict(weighted=True, prune=False, center="support")),
    ]
    for name, kw in settings:
        try:
            r = resolve_singularities(f, gens, max_depth=max_depth, **kw)
            out[name] = (r.rlct, r.multiplicity)
        except ResolutionFailure as e:
            out[name] = ("FAIL", e.reason)
        except Exception as e:                      # noqa: BLE001
            out[name] = ("ERROR", str(e)[:60])
    try:
        out["newton_lp"] = newton_rlct(f, gens)     # 上界 (非退化なら一致)
    except Exception:
        out["newton_lp"] = None
    return out


def multiplicity_bounds(f, gens):
    """重複度から出る lambda の評価 1/m <= lambda <= n/m を返す。

    * 下界: 1/m <= lct_C <= lambda_R (実 RLCT は複素 lct 以上)
    * 上界: f の全単項式は全次数 m 以上なので、ニュートン多面体は
      {sum a >= m} に入る。よって対角線が到達する距離 t* >= m/n、
      lambda <= lambda_Newton = 1/t* <= n/m。

    どちらも実数体上で正当な不等式なので、値の検算に使える。
    (mu, tau は代数閉体上の量なので、値の検算には使わない。)
    """
    from invariants import multiplicity
    m = multiplicity(f, gens)
    if m is sp.oo or m == 0:
        return (None, None, m)
    n = len(gens)
    return (sp.Rational(1, m), sp.Rational(n, m), m)


def complexity_hints(f, gens, *, backend: str = "auto"):
    """計算の難しさを予測する指標。値の判定には一切使わない。

    * mu = oo は「複素数体上で特異軌跡が正の次元」を意味する。会話中に
      証明が通らなかった例 ((x-y)^2, x^2y^2 など) は全てこれだった。
    * 擬斉次なら重み付きブローアップ 1 回で単項式化できる。

    これらは bench の共変量として記録し、失敗クラスとの相関を見るために
    だけ使う。lambda の値にも proved/unknown の判定にも影響させない。
    """
    from invariants import milnor_number, quasihomogeneous_weights
    try:
        mu, route = milnor_number(f, gens, backend=backend)
    except Exception:
        mu, route = None, "-"
    qh = quasihomogeneous_weights(f, gens)
    return {"mu": (None if mu is None else str(mu)),
            "mu_infinite": (mu is sp.oo),
            "mu_route": route,
            "quasihomogeneous": bool(qh)}


def check_consistency(f, gens, lam) -> List[str]:
    """lam が既知の情報と矛盾していないか調べ、矛盾を文字列で返す。"""
    msgs: List[str] = []
    lo, hi, m = multiplicity_bounds(f, gens)
    if lam is not None and lo is not None and lam is not sp.oo:
        if lam < lo:
            msgs.append(f"重複度の下界 1/m = {lo} を下回っている (lambda={lam}, m={m})")
        if lam > hi:
            msgs.append(f"重複度の上界 n/m = {hi} を超えている (lambda={lam}, m={m})")
    fam_lam, fam_m, route = family_oracle(f, gens)
    if fam_lam is not None and lam is not None and fam_lam != lam:
        msgs.append(f"族の閉じた式 ({route}) は {fam_lam} だが {lam} が返された")
    try:
        nl = newton_rlct(f, gens)
        if lam is not None and nl is not None and nl is not sp.oo and lam > nl:
            msgs.append(f"ニュートン上界 {nl} を超えている ({lam})")
    except Exception:
        pass
    return msgs
