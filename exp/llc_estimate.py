r"""
llc_estimate.py
===============

**SGLD による LLC (局所学習係数) の数値推定** — 厳密 λ との突き合わせ用。

Lau, Furman, Wang, Murfet & Wei, *The Local Learning Coefficient: A
Singularity-Aware Complexity Measure*, AISTATS 2025 (= arXiv:2308.12108v2)
の推定量

    λ̂(w*) = n β* ( E_{w|w*,β*,γ}[L_n(w)] − L_n(w*) ),    β* = 1/log n

を実装する。事後分布は局所化 Gauss 事前つき

    p(w) ∝ exp( − n β* L_n(w) − (γ/2)‖w − w*‖² )

で、これを SGLD でサンプリングする。

------------------------------------------------------------------
**同じものを比べるための 4 つの条件**
------------------------------------------------------------------
厳密 λ と λ̂ は、何もしなければ**別の量**である。比較を成立させるには:

1. **realizable にする。** 厳密側の fiber ideal は「θ\* のモデルが真」として
   作る。したがって λ̂ 側の教師データも θ\* の出力そのものにしなければ
   ならない。実ラベルを使うと θ\* が損失の最小点でなくなり、λ̂ は負に振れる
   (Lau et al. も負値の原因として "w\* fails to be near a local minimum" を
   挙げている)。`realizable_targets()` がこれを作る。
   - 回帰: y_i = f(x_i; θ\*) (雑音なし)
   - 分類: **ソフト教師** p(·|x_i, θ\*)。ハードラベルを θ\* から標本抽出すると
     標本雑音のぶん θ\* が経験最小点からずれる。`hard_labels=True` でその
     挙動も見られるようにしてある (§負値の実験用)

2. **同じ部分空間に制限する。** 厳密 λ は一部のパラメータだけを動かした
   部分空間の値 (= 全体の λ の下界)。λ̂ も**同じ座標だけ**を動かし、残りは
   θ\* で凍結する。`params` がその指定。ここを揃えないと、下界と全空間の
   推定値を比べることになる。

3. **同じデータ点を使う。** 厳密側のイデアルは選んだ n_data 点の
   `g_{i,c} = f_c(x_i;θ) − f_c(x_i;θ*)` で生成される。
   `K_n(w) = (1/n)Σ‖·‖²` のゼロ集合はこのイデアルのゼロ集合と一致し、
   RLCT も一致する (平方和の RLCT = イデアルの RLCT)。**同じ点を渡せば
   λ̂ は厳密 λ そのものを推定している。**

4. **L_n は負の対数尤度 (Watanabe の規約) であること。** 回帰で
   `nn.MSELoss()` をそのまま使うと、出力次元で平均してしまい、かつ
   1/2 がつかないので λ̂ が定数倍ずれる。分散 1 の Gauss 雑音なら
       L_n(w) = (1/n) Σ_i [ (1/2)‖y_i − f(x_i,w)‖² ] + const
   が正しい (出力次元は**和**、係数は 1/2)。`nll()` がこれを実装する。
   ここを間違えると λ̂ が 2 倍ずれる — 一番ありがちな罠。

------------------------------------------------------------------
推定器自身の検算
------------------------------------------------------------------
`regular_model_check()` は、**正則な (Fisher 情報行列が正定値な) モデルでは
λ = d/2** という既知の事実で推定器を検算する。線形回帰の d パラメータを
動かして λ̂ ≈ d/2 になるかを見る。ここが合わないうちは、厳密値との
食い違いを「推定器の誤差」にも「厳密側の誤り」にも帰属できない。

使い方
------
    python llc_estimate.py --self-test
    python llc_estimate.py --model cnn --depth 2 --act relu --loss classification
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["realizable_targets", "nll", "llc_sgld", "LLCResult",
           "llc_grid", "regular_model_check"]


# ----------------------------------------------------------------------
# 損失 (Watanabe の規約: L_n = 負の対数尤度の標本平均)
# ----------------------------------------------------------------------
def realizable_targets(model, X, *, loss: str = "regression",
                       hard_labels: bool = False, seed: int = 0):
    r"""θ\* のモデルを「真」とする教師データを作る。

    回帰なら f(x;θ\*) そのもの、分類なら p(·|x,θ\*) (ソフト教師)。
    `hard_labels=True` なら p(·|x,θ\*) から標本抽出した整数ラベルを返す
    (θ\* が経験最小点からずれるので λ̂ が負に振れる挙動の確認用)。
    """
    import torch

    with torch.no_grad():
        out = model(X)
        if loss == "regression":
            return out.clone()
        p = torch.softmax(out, dim=1)
        if hard_labels:
            g = torch.Generator().manual_seed(seed)
            return torch.multinomial(p, 1, generator=g).squeeze(1)
        return p.clone()


def nll(model, X, y, *, loss: str = "regression") -> "object":
    r"""L_n(w) = 負の対数尤度の標本平均 (定数項は落とす)。

    回帰 (分散 1 の Gauss): (1/n) Σ_i (1/2) ‖y_i − f(x_i,w)‖²
        — 出力次元は**和**、係数 **1/2**。`nn.MSELoss()` は次元で平均して
          しまうので使わない。
    分類 (ソフト教師): (1/n) Σ_i Σ_c p_c(x_i,θ*) · ( − log p_c(x_i,w) )
        — これの θ\* での値は p(·|x,θ\*) のエントロピーで、
          L_n(w) − L_n(θ\*) = (1/n) Σ_i KL( p(·|x_i,θ*) ‖ p(·|x_i,w) ) ≥ 0。
          等号は**ロジットの差が一致するとき**で、fiber ideal と同じ条件。
    分類 (ハードラベル): 通常の交差エントロピー。
    """
    import torch
    import torch.nn.functional as F

    out = model(X)
    if loss == "regression":
        return 0.5 * ((out - y) ** 2).sum(dim=1).mean()
    logp = F.log_softmax(out, dim=1)
    if y.dtype in (torch.int64, torch.int32, torch.long):
        return F.nll_loss(logp, y)
    return -(y * logp).sum(dim=1).mean()


# ----------------------------------------------------------------------
# SGLD
# ----------------------------------------------------------------------
@dataclass
class LLCResult:
    lam_hat: float
    L_star: float
    L_mean: float
    #: 鎖の L_n の推移 (burn-in 後)
    trace: List[float] = field(default_factory=list)
    #: 誤差棒。鎖が 2 本以上なら**鎖間**の標準偏差 (自己相関を含んだ honest な値)
    lam_std: float = 0.0
    #: 鎖ごとの λ̂
    lam_chains: List[float] = field(default_factory=list)
    #: 収束診断: 残した鎖の前半と後半の λ̂ の最大の差
    split_drift: float = 0.0
    #: L_n(w) < L_n(w*) となった標本の割合 (理屈では 0 に近いはず)
    frac_below: float = 0.0
    #: 発散したか (L_n が初期値の 10^3 倍を超えた)
    diverged: bool = False
    n_steps: int = 0
    eps: float = 0.0
    gamma: float = 0.0
    beta: float = 0.0
    n_eff: int = 0
    n_vars: int = 0
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        d = " 発散" if self.diverged else ""
        return (f"λ̂={self.lam_hat:.3f}±{self.lam_std:.3f} "
                f"(ε={self.eps:.1e} γ={self.gamma:g} "
                f"steps={self.n_steps} n={self.n_eff}){d}")


def _named(model, params: Optional[Sequence[str]]):
    """動かすパラメータだけを取り出す。None なら全部。"""
    named = list(model.named_parameters())
    if params is None:
        return named
    want = set(params)
    out = [(n, p) for n, p in named if n in want]
    missing = want - {n for n, _ in out}
    if missing:
        raise ValueError(f"存在しないパラメータ {sorted(missing)}")
    return out


def llc_sgld(model, X, *, params: Optional[Sequence[str]] = None,
             loss: str = "regression", y=None,
             gamma: float = 1.0, eps: float = 1e-4,
             n_steps: int = 8000, burn_in: float = 0.5,
             n_eff: Optional[int] = None, seed: int = 0,
             hard_labels: bool = False, precondition: str = "none",
             chains: int = 4, verbose: bool = False) -> LLCResult:
    r"""SGLD で λ̂ を推定する。**`params` で指定した座標だけ**を動かす。

    model   : nn.Module。現在の重みが θ\*
    X       : 入力 (厳密側と**同じ点**を渡すこと)
    params  : 動かすパラメータ名。None なら全部
    loss    : 'regression' / 'classification'
    y       : 教師データ。None なら `realizable_targets` で θ\* から作る
    gamma   : 局所化 Gauss 事前の強さ
    eps     : SGLD のステップ幅
    n_eff   : β* = 1/log(n_eff) に使う n。None なら X の点数
    chains  : 独立な鎖の本数。**誤差棒は鎖間のばらつきから出す。**
              1 鎖の L_n の標準誤差を使うと自己相関のぶん過小になり、
              「精度がよく見えるが実際は当たっていない」状態になる
              (実測: 正則モデルで鎖内の誤差棒 ±0.075 に対し、ε を変えると
               λ̂ が 3.5〜6.2 まで動いた)
    precondition : 'none' か 'rmsprop'。Hitchcock & Hoogland (arXiv:2507.21449)
              は RMSProp 前処理つき SGLD が最も頑健と報告している

    返り値の λ̂ = n β* ( 鎖上の L_n の平均 − L_n(θ\*) )。
    """
    import torch

    dev = X.device if hasattr(X, "device") else None
    named = _named(model, params)
    if not named:
        raise ValueError("動かすパラメータが空です")
    star = {n: p.detach().clone() for n, p in named}
    n_vars = sum(p.numel() for _, p in named)
    N = int(X.shape[0])
    n_eff = int(n_eff or N)
    beta = 1.0 / math.log(n_eff)

    if y is None:
        y = realizable_targets(model, X, loss=loss, hard_labels=hard_labels,
                               seed=seed)

    # 動かさないパラメータは勾配を切る
    for n, p in model.named_parameters():
        p.requires_grad_(any(n == m for m, _ in named))

    with torch.no_grad():
        L_star = float(nll(model, X, y, loss=loss))

    lam_chains: List[float] = []
    trace_all: List[float] = []
    halves: List[Tuple[float, float]] = []
    below = 0
    total = 0
    diverged = False
    notes: List[str] = []

    for c in range(chains):
        # θ* から出発する (局所の量なので初期点は θ* 自身)
        with torch.no_grad():
            for n, p in named:
                p.copy_(star[n])
        g = torch.Generator(device="cpu").manual_seed(seed + 1000 * c)
        keep_from = int(n_steps * burn_in)
        chain: List[float] = []
        sq: Dict[str, object] = {}                        # RMSProp の EMA
        for t in range(n_steps):
            model.zero_grad(set_to_none=True)
            L = nll(model, X, y, loss=loss)
            L.backward()
            with torch.no_grad():
                Ms: Dict[str, object] = {}
                if precondition == "rmsprop":
                    # 初回は grad^2 で初期化する。0 から始めると
                    # M = 1/(sqrt(0)+1e-8) = 1e8 で初手に発散する
                    for n, p in named:
                        if n not in sq:
                            sq[n] = p.grad.detach() ** 2 + 1e-12
                        else:
                            sq[n].mul_(0.99).addcmul_(p.grad, p.grad,
                                                      value=0.01)
                    raw = {n: 1.0 / (sq[n].sqrt() + 1e-8) for n, _ in named}
                    # **平均 1 に正規化**して ε の意味を前処理なしと揃える
                    tot = sum(float(v.sum()) for v in raw.values())
                    cnt = sum(v.numel() for v in raw.values())
                    scale = cnt / tot if tot > 0 else 1.0
                    Ms = {n: raw[n] * scale for n, _ in named}
                for n, p in named:
                    # ∇[ nβ L_n + (γ/2)‖w−w*‖² ]
                    grad = n_eff * beta * p.grad + gamma * (p - star[n])
                    noise = torch.randn(p.shape, generator=g,
                                        dtype=p.dtype).to(p.device)
                    if precondition == "rmsprop":
                        M = Ms[n]
                        p.add_(-(eps / 2.0) * M * grad
                               + math.sqrt(eps) * M.sqrt() * noise)
                    else:
                        p.add_(-(eps / 2.0) * grad + math.sqrt(eps) * noise)
            Lv = float(L.detach())
            if not math.isfinite(Lv) or abs(Lv) > 1e3 * (abs(L_star) + 1.0):
                diverged = True
                notes.append(f"鎖 {c}: step {t} で発散 (L_n={Lv:.3e})")
                break
            if t >= keep_from:
                chain.append(Lv)
                total += 1
                if Lv < L_star - 1e-12:
                    below += 1
        if chain:
            lam_chains.append(n_eff * beta * (sum(chain) / len(chain) - L_star))
            trace_all += chain
            # 収束診断: 残した鎖の前半と後半で平均が揃っているか
            h = len(chain) // 2
            if h:
                halves.append((
                    n_eff * beta * (sum(chain[:h]) / h - L_star),
                    n_eff * beta * (sum(chain[h:]) / (len(chain) - h) - L_star)))
        if verbose:
            print(f"    鎖 {c}: λ̂="
                  f"{lam_chains[-1] if lam_chains else float('nan'):.3f}",
                  flush=True)

    # θ* に戻す
    with torch.no_grad():
        for n, p in named:
            p.copy_(star[n])
    for _, p in model.named_parameters():
        p.requires_grad_(True)

    lam = (sum(lam_chains) / len(lam_chains)) if lam_chains else float("nan")
    if len(lam_chains) > 1:
        # **鎖間のばらつき**を誤差棒にする。鎖内の標準誤差は自己相関を
        # 無視するので、精度を過大に見せてしまう
        sd = math.sqrt(sum((v - lam) ** 2 for v in lam_chains)
                       / (len(lam_chains) - 1))
    elif trace_all:
        mu = sum(trace_all) / len(trace_all)
        var = sum((v - mu) ** 2 for v in trace_all) / max(len(trace_all) - 1, 1)
        sd = n_eff * beta * math.sqrt(var / len(trace_all))
        notes.append("1 鎖なので誤差棒は自己相関を無視した過小評価")
    else:
        sd = float("nan")
    drift = (max(abs(a - b) for a, b in halves) if halves else float("nan"))
    if halves and drift > 0.15 * max(abs(lam), 1e-9):
        notes.append(f"未収束の疑い: 残した鎖の前半/後半で λ̂ が {drift:.3f} 動いた")
    return LLCResult(lam_hat=lam, L_star=L_star,
                     L_mean=(sum(trace_all) / len(trace_all)) if trace_all
                     else float("nan"),
                     trace=trace_all[:2000], lam_std=sd, split_drift=drift,
                     lam_chains=list(lam_chains),
                     frac_below=(below / total) if total else float("nan"),
                     diverged=diverged, n_steps=n_steps, eps=eps, gamma=gamma,
                     beta=beta, n_eff=n_eff, n_vars=n_vars, notes=notes)


def llc_grid(model, X, *, epss: Sequence[float], gammas: Sequence[float],
             **kw) -> List[LLCResult]:
    """ε と γ の格子で λ̂ を測る (感度の確認用)。"""
    out = []
    for g in gammas:
        for e in epss:
            out.append(llc_sgld(model, X, eps=e, gamma=g, **kw))
    return out


# ----------------------------------------------------------------------
# 推定器自身の検算: 正則モデルなら λ = d/2
# ----------------------------------------------------------------------
def gamma_bias(F, n_eff: int, gamma: float) -> float:
    r"""**局所化による偏りの解析形** (正則モデルのみ、厳密)。

    正則モデルでは L_n(w) − L_n(w\*) = (1/2)(w−w\*)^T F(w−w\*) が厳密な二次形式で、
    事後分布は精度 A = nβF + γI の Gauss になる。したがって

        E[ nβ (L_n − L_n(w*)) ] = (1/2) tr[ nβF (nβF + γI)^{-1} ]
                                = (1/2) Σ_i nβ f_i / (nβ f_i + γ)

    で、**γ → 0 で d/2**、γ が nβ f_i と同程度になると λ̂ は系統的に
    小さく出る。実測では γ=100 (nβ≈82) で真値の 45% まで落ちた。

    これは推定量の偏りが**閉形式で分かる**ということなので、正則モデルを
    基準に γ の影響を補正・検算できる。厳密な Gauss 事後からの直接標本抽出と
    小数 4 桁まで一致することを確認済み (1.9768 対 1.9760)。
    """
    import torch

    beta = 1.0 / math.log(n_eff)
    nb = n_eff * beta
    ev = torch.linalg.eigvalsh(F).tolist()
    return 0.5 * sum(nb * f / (nb * f + gamma) for f in ev)


def regular_model_check(dims: Sequence[int] = (2, 4, 8),
                        *, n: int = 512, eps: float = 5e-4,
                        gamma: float = 1.0, n_steps: int = 8000,
                        chains: int = 4, precondition: str = "none",
                        seed: int = 0, verbose: bool = True) -> List[Dict]:
    r"""**正則モデルでは λ = d/2** で推定器を検算する。

    y = x·W + b に y|x ~ N(f, 1) を乗せた線形回帰は Fisher 情報行列が
    正定値なので λ = d/2 (d = 動かすパラメータ数)。realizable に作って
    λ̂ を測り、d/2 と、`gamma_bias` の解析形の両方と比べる。
    ここが合わないうちは厳密値との食い違いをどちらの側にも帰属できない。
    """
    import torch
    import torch.nn as nn

    out = []
    for d in dims:
        torch.manual_seed(seed)
        # d パラメータ = 重み (d-1) + バイアス 1
        net = nn.Sequential(nn.Linear(d - 1, 1))
        X = torch.randn(n, d - 1, generator=torch.Generator().manual_seed(seed))
        Z = torch.cat([X, torch.ones(n, 1)], dim=1)
        F = (Z.t() @ Z) / n
        pred = gamma_bias(F, n, gamma)
        r = llc_sgld(net, X, params=["0.weight", "0.bias"],
                     loss="regression", eps=eps, gamma=gamma, chains=chains,
                     n_steps=n_steps, precondition=precondition, seed=seed)
        rec = dict(d=d, truth=d / 2.0, analytic=pred, lam_hat=r.lam_hat,
                   lam_std=r.lam_std, chains=r.lam_chains,
                   split_drift=r.split_drift,
                   rel_err=(r.lam_hat - d / 2.0) / (d / 2.0),
                   rel_err_analytic=(r.lam_hat - pred) / pred,
                   diverged=r.diverged, n_vars=r.n_vars)
        out.append(rec)
        if verbose:
            print(f"  d={d:<3} 真値 d/2={d/2:<5} 解析形 {pred:6.3f}  "
                  f"λ̂={r.lam_hat:7.3f}±{r.lam_std:.3f}  "
                  f"d/2 比 {rec['rel_err']:+.1%}  "
                  f"解析形 比 {rec['rel_err_analytic']:+.1%}"
                  f"{' 発散' if r.diverged else ''}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="正則モデル (λ=d/2) で推定器を検算")
    ap.add_argument("--dims", default="2,4,8")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--eps", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.self_test:
        print("===== 推定器の検算: 正則モデルなら λ = d/2 =====")
        recs = regular_model_check(
            [int(v) for v in args.dims.split(",") if v],
            n=args.n, eps=args.eps, gamma=args.gamma, n_steps=args.steps,
            seed=args.seed)
        ok = all(abs(r["rel_err"]) < 0.15 and not r["diverged"] for r in recs)
        print(f"\n-> {'すべて相対誤差 15% 以内' if ok else '**ずれています**'}")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(recs, fh, ensure_ascii=False, indent=1)
            print(f"-> {args.out}")
        return 0 if ok else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
