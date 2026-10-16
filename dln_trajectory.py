r"""
dln_trajectory.py
=================

**学習軌道に沿った厳密な λ — 深層線形網 (DLN) の鞍点から鞍点への遷移**

Wang, Hoogland, van Wingerden, Furman & Murfet (arXiv:2410.02984) は、学習中の
各チェックポイントで (refined) LLC を SGLD で推定し、その曲線から発達段階を
読んでいる。しかし推定量の正当化は「w\* が局所極小である」ことを前提にしており
(同論文 §3.3)、学習途中の点ではこれが成り立たない。

このモジュールは、同じ図を**真値つき**で描くためのもの。小さな初期値から
学習した DLN は、積 W_L⋯W_1 の階数を 0→1→2→… と 1 つずつ上げながら
鞍点を渡る (Saxe et al. 2014; Zhang, Saxe & Latham, arXiv:2512.20607)。
各チェックポイントで次を並べる。

1. **厳密な λ (自己教師)**: θ_t を階数 k の点 θ̄_t に「丸め」、
   K_{θ̄_t}(θ) = Σ_i ‖f_θ(x_i) − f_{θ̄_t}(x_i)‖² の局所 RLCT を
   Aoyagi の閉形式 `dln.lambda_dln(H, k)` から出す。位数 m も出る。
2. **自己教師データでの λ̂**: θ̄_t で、教師を θ̄_t 自身の出力にして SGLD。
   θ̄_t は損失 0 の大域解なので推定量の前提が**厳密に**満たされる。
   1 との差が「推定器そのものの誤差」。
3. **真のデータでの λ̂**: 生の θ_t で、本当の教師データのまま SGLD。
   これが Wang et al. の量にあたる。2 との差が「極小でないこと
   (勾配と非 realizable 性)」の寄与。
4. **二次近似による予測**: 3 の期待値の二次近似
       E[λ̂] ≈ ½ Σ_i nβh_i/(nβh_i+γ)  −  Σ_i (nβg_i)²(nβh_i+2γ) / (2(nβh_i+γ)²)
   (h_i, g_i は Hessian の固有値と、その固有基底での勾配成分)。
   第 1 項は `llc_estimate.gamma_bias` と同じもので、第 2 項が勾配による
   負の偏り。ただし式が意味を持つのは事後分布が正規化できるとき
   (全方向で nβh_i + γ > 0) だけ。

**実測で分かったこと** (既定設定、H=(6,6,6)、教師の特異値 8,4,2,1、γ=1):
学習が終わる前のプラトーはすべて鞍点で、真のデータの Hessian の最小固有値は
およそ −s_{k+1} (−8, −4, −2, −1)。nβ ≈ 15 なので nβh_min + γ < 0 となり、
事後分布が正規化できない。SGLD は鞍点から谷へ滑り落ち、λ̂_true は
k=0,1,2 で −634, −139, −16 と大きく負になる (鎖上で L_n < L_n(θ_t) と
なる割合 100%)。k=3 (h_min ≈ −1) では 3000 ステップの間に滑り落ちきらず
10.5 で、自己教師の 15.4 より小さい。
**負の λ̂ の主因は勾配ではなく負曲率**である。正規化には γ > nβ|h_min|
(≈ 120, 60, 30, 15) が必要で、そこまで γ を上げると今度は γ の尺度しか
見えなくなる。自己教師の θ̄_t は大域極小なので Hessian は半正定値で、
この問題は起きない。

層ごと (weight-refined) の厳密値も出す。第 l 層だけを動かし残りを θ̄_t に
凍結すると、出力は動かす重み V について線形なので K は V の二次形式になり、

    λ_l = rank(W_after) · rank(W_before X) / 2          (厳密、正則)

である (W_after = 第 l 層より出力側の積、W_before = 入力側の積)。

------------------------------------------------------------------
**階数の丸め (snap) が要る理由** — 実測
------------------------------------------------------------------
学習途中の θ_t の積は厳密には full rank (初期値由来の小さな特異値が残る)。
そのまま自己教師にすると正則点になり、λ は最大値 (= 階数 min(H) の値) に
張り付く。そこで

* 階数 k を、教師の特異値 s_i に対する比 u_i = σ_i(P)/s_i が閾値を
  超えたモードの数として決める。閾値 0.1 / 0.5 / 0.9 で k が食い違えば
  「遷移中」として `ambiguous=True` にする (プラトー内では一致する)。
* 各層の重みを SVD で階数 k に切り詰めて θ̄_t を作る。

**厳密値は Aoyagi の閉形式から取り、解消経路は使わない。** 理由は実測で、
`torch_rlct.torch_local_rlct` の (λ, m) は隠れ層の座標の取り方に依存する。
2-2-2 の DLN で積の階数が 1、各因子も階数 1 の点 (対称性 W1→GW1, W2→W2G⁻¹ で
すべて同じ軌道 = 真の (λ, m) は (2, 2)) を解消すると:

    隠れ層を座標軸に揃えた点 (`align_hidden`)  →  (2, 2)    正しい
    学習軌道の丸め点                            →  (2, 1)    m を過小評価
    一般の向きの点                              →  (5/2, 2)  λ を過大評価

`LocalRLCT` には証明書の status が無いので、この誤りは呼び出し側から
見えない。ガード `dln_trajectory_tests` は揃えた座標で解消経路と
Aoyagi が (λ, m) まで一致することを検査し、一般の座標での食い違いは
`known_issues` に未修正として出る。

なお位数 m が Aoyagi の θ と一致するのは「各因子が最小階数 k」の点で、
隠れ側の因子の階数が k より大きい点では λ は同じで m が下がる
(揃えた座標の解消経路で 2 → 1)。丸めを各層で行うのはこのため。

------------------------------------------------------------------
設定の目安
------------------------------------------------------------------
* 教師の特異値を幾何的に離す (既定 8, 4, 2, 1)。2 層 DLN では階数 i の
  遷移時刻はおよそ ln(s_i/u₀)/(lr·s_i) なので、比が 2 なら段がきれいに分かれる。
* 入力は白色化 (XᵀX/n = I)。Aoyagi の公式は入力の共分散が正定値で
  あることだけを要求する。
* λ は階数に対して単調増加だが、上端で飽和する (例 H=(6,6,6):
  27/2, 15, 16, 17, 35/2, 18, 18)。教師の階数は min(H) より小さくとる。

使い方
------
    python dln_trajectory.py                       # 厳密値だけ (数秒)
    python dln_trajectory.py --sgld --sgld-every 10  # λ̂ も重ねる
    python dln_trajectory.py --dims 6 6 6 6 --steps 6000 --lr 0.01
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["DLNConfig", "make_data", "make_dln", "aoyagi_H", "product",
           "detect_rank", "snap_to_rank", "align_hidden", "exact_full", "exact_layers",
           "quadratic_prediction", "train_checkpoints", "analyze", "plot"]


# ----------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------
@dataclass
class DLNConfig:
    """実験の設定。dims は入力側から (d_in, h_1, …, d_out)。"""
    dims: Tuple[int, ...] = (6, 6, 6)
    teacher_sv: Tuple[float, ...] = (8.0, 4.0, 2.0, 1.0)
    n_data: int = 64
    init_scale: float = 1e-3
    lr: float = 0.02
    n_steps: int = 600
    every: int = 3                  # 厳密値を計算する間隔 (安いので細かくてよい)
    noise: float = 0.0              # 教師の雑音。0 なら学習の終点は realizable
    seed: int = 0
    thresholds: Tuple[float, float, float] = (0.1, 0.5, 0.9)

    def validate(self) -> None:
        if len(self.dims) < 3:
            raise ValueError("dims は (d_in, 隠れ…, d_out) の 3 つ以上が必要です")
        R = len([s for s in self.teacher_sv if s > 0])
        if R > min(self.dims):
            raise ValueError(f"教師の階数 {R} が min(dims)={min(self.dims)} を超えます")
        if self.n_data < self.dims[0]:
            raise ValueError("白色化には n_data >= d_in が必要です")
        s = [x for x in self.teacher_sv if x > 0]
        if s != sorted(s, reverse=True):
            raise ValueError("teacher_sv は降順に並べてください")


# ----------------------------------------------------------------------
# データとモデル
# ----------------------------------------------------------------------
def make_data(cfg: DLNConfig):
    r"""白色化した入力 X (n×d_in, XᵀX/n = I) と教師 A = U diag(s) Vᵀ、y = XAᵀ。"""
    import torch

    g = torch.Generator().manual_seed(cfg.seed)
    d_in, d_out = cfg.dims[0], cfg.dims[-1]
    Z = torch.randn(cfg.n_data, d_in, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(Z)                       # 列が正規直交
    X = Q * math.sqrt(cfg.n_data)                   # XᵀX = n I
    U, _ = torch.linalg.qr(torch.randn(d_out, d_out, generator=g,
                                       dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(d_in, d_in, generator=g,
                                       dtype=torch.float64))
    s = torch.zeros(min(d_in, d_out), dtype=torch.float64)
    sv = [x for x in cfg.teacher_sv if x > 0]
    s[:len(sv)] = torch.tensor(sv, dtype=torch.float64)
    A = U[:, :len(s)] @ torch.diag(s) @ V[:, :len(s)].T
    y = X @ A.T
    if cfg.noise > 0:
        y = y + cfg.noise * torch.randn(y.shape, generator=g,
                                        dtype=torch.float64)
    return X, y, A


def make_dln(dims: Sequence[int], init_scale: float, seed: int = 0):
    """バイアスなしの線形層の列。重みは N(0, init_scale²)。float64。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed + 1)
    layers = [nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:])]
    net = nn.Sequential(*layers).double()
    with torch.no_grad():
        for m in net:
            m.weight.normal_(0.0, init_scale)
    return net


def aoyagi_H(dims: Sequence[int]) -> Tuple[int, ...]:
    """入力側からの dims を Aoyagi の H = (H^(1)=出力, …, H^(L+1)=入力) に直す。"""
    return tuple(int(d) for d in reversed(dims))


def product(net):
    """積 P = W_L ⋯ W_1 (d_out × d_in)。"""
    P = None
    for m in net:
        P = m.weight if P is None else m.weight @ P
    return P.detach()


# ----------------------------------------------------------------------
# 階数の判定と丸め
# ----------------------------------------------------------------------
@dataclass
class RankInfo:
    k: int                      # 中央の閾値での階数
    ks: Tuple[int, ...]         # 閾値ごとの階数
    ratios: List[float]         # u_i = σ_i(P) / s_i
    ambiguous: bool             # 閾値で k が食い違う = 遷移中


def detect_rank(net, teacher_sv: Sequence[float],
                thresholds: Sequence[float] = (0.1, 0.5, 0.9)) -> RankInfo:
    r"""教師の特異値に対する比で、学習済みのモードの数を数える。

    u_i = σ_i(P)/s_i が閾値を超えた i の個数を階数とする。σ_i(P) と s_i は
    どちらも降順なので、Saxe 型の学習ではモードが大きい順に 0→1 へ立ち上がる。
    """
    import torch

    P = product(net)
    sig = torch.linalg.svdvals(P).tolist()
    s = [x for x in teacher_sv if x > 0]
    ratios = [sig[i] / s[i] for i in range(len(s))]
    ks = tuple(sum(1 for u in ratios if u > t) for t in thresholds)
    return RankInfo(k=ks[len(ks) // 2], ks=ks, ratios=ratios,
                    ambiguous=len(set(ks)) > 1)


def snap_to_rank(net, k: int):
    r"""各層を SVD で階数 k に切り詰めた複製 θ̄ を返す。

    返り値は (net_bar, ok, rel_dist)。ok は「各因子の階数も積の階数も
    ちょうど k」— Aoyagi の θ が実現される最小階数の層に入ったか。
    rel_dist = ‖θ − θ̄‖ / ‖θ‖。
    """
    import torch

    bar = copy.deepcopy(net)
    num = 0.0
    den = 0.0
    with torch.no_grad():
        for m, mb in zip(net, bar):
            W = m.weight
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            S2 = S.clone()
            S2[k:] = 0.0
            mb.weight.copy_(U @ torch.diag(S2) @ Vh)
            num += float(((W - mb.weight) ** 2).sum())
            den += float((W ** 2).sum())
    ranks = [_rank(mb.weight) for mb in bar] + [_rank(product(bar))]
    ok = all(r == k for r in ranks)
    return bar, ok, math.sqrt(num / den) if den > 0 else 0.0


def align_hidden(net_bar, k: int):
    r"""対称性 W_l → G W_l, W_{l+1} → W_{l+1} G⁻¹ で隠れ層を座標軸に揃えた複製。

    各隠れ層で「入ってくる像 Im(W_l)」を e_1..e_k に、「次の層の核
    ker(W_{l+1})」を e_{k+1}.. に送る G を取る (最小階数の点では両者は
    直和になる)。揃えた後は、各層で使われない行・列がちょうど 0 になる。

    局所 RLCT とその位数はこの変換で不変なので、厳密値は変わらない。
    **しかし `torch_rlct` の解消経路が返す位数 m は座標に依存する**
    (実測: 2-2-2、階数 1 の点で、一般の座標 m=1、揃えた座標 m=2 = Aoyagi の θ)。
    解消経路で m を比べるときはこの関数を通すこと。
    """
    import torch

    ali = copy.deepcopy(net_bar)
    with torch.no_grad():
        for l in range(len(ali) - 1):
            W, Wn = ali[l].weight, ali[l + 1].weight
            h = W.shape[0]
            U, S, _ = torch.linalg.svd(W)
            img = U[:, :k]
            _, Sn, Vhn = torch.linalg.svd(Wn)
            ker = Vhn[k:].T if Vhn.shape[0] > k else Vhn[:0].T
            if ker.shape[1] < h - k:          # 次の層が横長で核が足りない場合
                return None
            B = torch.cat([img, ker[:, :h - k]], dim=1)
            if _rank(B) < h:
                return None                   # 最小階数の層にいない
            G = torch.linalg.inv(B)
            ali[l].weight.copy_(G @ W)
            ali[l + 1].weight.copy_(Wn @ B)
    return ali


def _rank(M, rtol: float = 1e-9) -> int:
    import torch

    s = torch.linalg.svdvals(M)
    if s.numel() == 0 or s[0].item() == 0.0:
        return 0
    return int((s > rtol * s[0]).sum())


# ----------------------------------------------------------------------
# 厳密値
# ----------------------------------------------------------------------
def exact_full(dims: Sequence[int], k: int):
    """全パラメータを動かしたときの (λ, m)。Aoyagi の閉形式。"""
    from dln import lambda_dln

    lam, th = lambda_dln(aoyagi_H(dims), k)
    return lam, int(th)


def exact_layers(net_bar, X) -> List:
    r"""層ごと (weight-refined) の厳密な λ_l = rank(W_after)·rank(W_before X)/2。

    第 l 層 V だけを動かすと f(x) = W_after V W_before x は V について線形で、
    K(V) = Σ_i ‖W_after (V−V̄) W_before x_i‖² は V の二次形式。その階数は
    Kronecker 積 (W_before X)ᵀ ⊗ W_after の階数 = 両者の階数の積。
    二次形式の RLCT は階数/2、位数 1。
    """
    import sympy as sp
    import torch

    Ws = [m.weight.detach() for m in net_bar]
    out = []
    for l in range(len(Ws)):
        before = X.T                                    # d_in × n
        for W in Ws[:l]:
            before = W @ before
        after = torch.eye(Ws[l].shape[0], dtype=Ws[l].dtype)
        for W in Ws[l + 1:]:
            after = W @ after
        out.append(sp.Rational(_rank(after) * _rank(before), 2))
    return out


# ----------------------------------------------------------------------
# 二次近似による λ̂ の予測 (勾配の項つき)
# ----------------------------------------------------------------------
def _flat_loss(net, X, y, names):
    import torch
    from torch.func import functional_call

    from llc_estimate import nll

    base = {n: p.detach() for n, p in net.named_parameters()}
    shapes = [base[n].shape for n in names]
    sizes = [base[n].numel() for n in names]

    def f(theta):
        parts = torch.split(theta, sizes)
        params = dict(base)
        for n, part, shp in zip(names, parts, shapes):
            params[n] = part.reshape(shp)

        class _M(torch.nn.Module):
            def forward(self, inp):
                return functional_call(net, params, (inp,))
        return nll(_M(), X, y)

    theta0 = torch.cat([base[n].reshape(-1) for n in names])
    return f, theta0


def quadratic_prediction(net, X, y, *, params: Optional[Sequence[str]] = None,
                         gamma: float = 1.0, n_eff: Optional[int] = None
                         ) -> Dict[str, float]:
    r"""E[λ̂] の二次近似。L_n を θ の周りで 2 次まで展開し、局所化 Gauss 事後
    (精度 A = nβH + γI) の下で nβ(E[L_n] − L_n(θ)) を厳密に計算する。

    固有基底 (H = Σ h_i e_i e_iᵀ, g_i = ⟨∇L_n, e_i⟩) で

        regular  = ½ Σ nβh_i / (nβh_i + γ)
        gradient = Σ (nβg_i)² (nβh_i + 2γ) / (2 (nβh_i + γ)²)
        total    = regular − gradient

    g = 0 なら `llc_estimate.gamma_bias` と一致する。nβh_i + γ ≤ 0 の方向が
    あれば事後分布が正規化できない (SGLD は逃げる) ので proper=False。
    """
    import torch

    names = list(params) if params is not None else [n for n, _ in
                                                      net.named_parameters()]
    f, th = _flat_loss(net, X, y, names)
    th = th.clone().requires_grad_(True)
    g = torch.autograd.grad(f(th), th)[0].detach()
    Hm = torch.autograd.functional.hessian(f, th.detach())
    Hm = 0.5 * (Hm + Hm.T)
    h, E = torch.linalg.eigh(Hm)
    gi = E.T @ g
    n = int(n_eff or X.shape[0])
    nb = n / math.log(n)
    A = nb * h + gamma
    proper = bool((A > 0).all())
    reg = float(0.5 * (nb * h / A).sum())
    grad = float((((nb * gi) ** 2) * (nb * h + 2 * gamma) / (2 * A ** 2)).sum())
    # 事後分布が正規化できない方向 (鞍点の負曲率)。この場合 SGLD は谷を
    # 下って L_n < L_n(θ) に滑り、λ̂ は大きく負になる。式の値は意味を持たない。
    n_unstable = int((A <= 0).sum())
    gamma_needed = float(max(0.0, -nb * float(h.min())))
    return dict(regular=reg, gradient=grad,
                total=(reg - grad) if proper else float("nan"),
                proper=proper, n_unstable=n_unstable,
                gamma_needed=gamma_needed,
                grad_norm=float(g.norm()), min_curv=float(h.min()))


# ----------------------------------------------------------------------
# 学習
# ----------------------------------------------------------------------
def train_checkpoints(cfg: DLNConfig, X, y) -> Tuple[List[int], List[dict],
                                                     List[float]]:
    """フルバッチ GD (損失は `llc_estimate.nll` と同じ ½‖·‖² の平均)。

    `cfg.every` ステップごとに state_dict を保存する。
    """
    import torch

    from llc_estimate import nll

    net = make_dln(cfg.dims, cfg.init_scale, cfg.seed)
    opt = torch.optim.SGD(net.parameters(), lr=cfg.lr)
    steps, states, losses = [], [], []
    for t in range(cfg.n_steps + 1):
        if t % cfg.every == 0:
            steps.append(t)
            states.append({k: v.detach().clone()
                           for k, v in net.state_dict().items()})
        opt.zero_grad()
        L = nll(net, X, y)
        losses.append(L.item())
        if t < cfg.n_steps:
            L.backward()
            opt.step()
    return steps, states, losses


# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------
def _sgld(net, X, *, y=None, params=None, gamma, eps, n_steps, chains, seed):
    from llc_estimate import llc_sgld

    r = llc_sgld(net, X, y=y, params=params, gamma=gamma, eps=eps,
                 n_steps=n_steps, chains=chains, seed=seed)
    return dict(lam=r.lam_hat, std=r.lam_std, frac_below=r.frac_below,
                diverged=r.diverged, notes=list(r.notes))


def analyze(cfg: DLNConfig, *, sgld: bool = False, sgld_every: int = 10,
            sgld_layers: bool = False, gamma: float = 1.0, eps: float = 1e-4,
            sgld_steps: int = 3000, chains: int = 2,
            verbose: bool = True) -> Dict:
    """学習してチェックポイントごとの記録を返す。

    返り値: {"config", "records", "losses"}。records の各要素は 1 チェックポイント。
    厳密値は全チェックポイントで、λ̂ は `sgld_every` 個おきにだけ計算する
    (SGLD は 1 点で数十秒かかる)。遷移中 (ambiguous) の点では λ̂ を計算しない。
    """
    import torch

    cfg.validate()
    X, y, A = make_data(cfg)
    steps, states, losses = train_checkpoints(cfg, X, y)
    net = make_dln(cfg.dims, cfg.init_scale, cfg.seed)
    names = [n for n, _ in net.named_parameters()]
    recs: List[Dict] = []
    for j, (t, sd) in enumerate(zip(steps, states)):
        net.load_state_dict(sd)
        ri = detect_rank(net, cfg.teacher_sv, cfg.thresholds)
        bar, ok, dist = snap_to_rank(net, ri.k)
        lam, m = exact_full(cfg.dims, ri.k)
        lays = exact_layers(bar, X)
        net.zero_grad()
        from llc_estimate import nll
        L = nll(net, X, y)
        L.backward()
        gnorm = math.sqrt(sum(float((p.grad ** 2).sum())
                              for p in net.parameters()))
        rec = dict(step=t, loss=losses[t], grad_norm=gnorm, k=ri.k,
                   k_by_threshold=list(ri.ks), ratios=ri.ratios,
                   ambiguous=ri.ambiguous, snap_ok=ok, snap_dist=dist,
                   lam_exact=float(lam), lam_exact_str=str(lam), m_exact=m,
                   lam_layers=[float(v) for v in lays],
                   lam_layers_str=[str(v) for v in lays])
        do_sgld = sgld and (j % sgld_every == 0) and not ri.ambiguous
        if do_sgld:
            # 2. 自己教師 (θ̄ で教師 = θ̄ 自身の出力): 推定量の前提が厳密に成立
            rec["self"] = _sgld(bar, X, gamma=gamma, eps=eps,
                                n_steps=sgld_steps, chains=chains, seed=cfg.seed)
            # 3. 真のデータ (生の θ_t): Wang et al. の量
            rec["true"] = _sgld(net, X, y=y, gamma=gamma, eps=eps,
                                n_steps=sgld_steps, chains=chains, seed=cfg.seed)
            # 4. 二次近似による 3 の予測
            rec["quad_true"] = quadratic_prediction(net, X, y, gamma=gamma)
            if sgld_layers:
                rec["self_layers"] = [
                    _sgld(bar, X, params=[nm], gamma=gamma, eps=eps,
                          n_steps=sgld_steps, chains=chains, seed=cfg.seed)
                    for nm in names]
                rec["true_layers"] = [
                    _sgld(net, X, y=y, params=[nm], gamma=gamma, eps=eps,
                          n_steps=sgld_steps, chains=chains, seed=cfg.seed)
                    for nm in names]
        recs.append(rec)
        if verbose and (do_sgld or j % max(1, len(steps) // 20) == 0):
            line = (f"step {t:6d}  loss {rec['loss']:.3e}  k={ri.k}"
                    f"{'?' if ri.ambiguous else ' '}  λ={rec['lam_exact_str']}"
                    f" m={m}  layers={rec['lam_layers_str']}")
            if do_sgld:
                q = rec["quad_true"]
                line += (f"  λ̂_self={rec['self']['lam']:.2f}"
                         f"  λ̂_true={rec['true']['lam']:.2f}"
                         + (f"  quad={q['total']:.2f}" if q["proper"] else
                            f"  quad=improper ({q['n_unstable']} dir,"
                            f" γ>{q['gamma_needed']:.1f} needed)"))
            print(line, flush=True)
    return dict(config=asdict(cfg), records=recs, losses=losses,
                sgld=dict(enabled=sgld, gamma=gamma, eps=eps,
                          steps=sgld_steps, chains=chains))


# ----------------------------------------------------------------------
# 出力
# ----------------------------------------------------------------------
# 既定の分類パレット (dataviz/palette.md の slot 1〜3、検証済み)。
# 色は「量の種類」に固定し、層の区別は線種とマーカーで行う。
_C_EXACT, _C_SELF, _C_TRUE = "#2a78d6", "#eb6834", "#1baf7a"
_C_QUAD = "#6b6a63"
_LAYER_STYLES = ["-", "--", ":", "-."]
_LAYER_MARKERS = ["o", "s", "^", "D"]


def write_csv(result: Dict, path: str) -> None:
    """表で見られるように 1 チェックポイント 1 行で書き出す。"""
    recs = result["records"]
    nl = len(recs[0]["lam_layers"]) if recs else 0
    cols = (["step", "loss", "grad_norm", "k", "ambiguous", "snap_ok",
             "snap_dist", "lam_exact", "m_exact"]
            + [f"lam_layer{l+1}" for l in range(nl)]
            + ["lam_hat_self", "lam_hat_self_std", "lam_hat_true",
               "lam_hat_true_std", "quad_regular", "quad_gradient",
               "quad_total"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in recs:
            q = r.get("quad_true", {})
            w.writerow([r["step"], r["loss"], r["grad_norm"], r["k"],
                        int(r["ambiguous"]), int(r["snap_ok"]), r["snap_dist"],
                        r["lam_exact_str"], r["m_exact"]]
                       + r["lam_layers_str"]
                       + [r.get("self", {}).get("lam", ""),
                          r.get("self", {}).get("std", ""),
                          r.get("true", {}).get("lam", ""),
                          r.get("true", {}).get("std", ""),
                          q.get("regular", ""), q.get("gradient", ""),
                          q.get("total", "")])


def plot(result: Dict, path: str, *, logx: bool = False) -> None:
    """損失・全体の λ・層ごとの λ を、x 軸を共有した 3 段で描く。

    y 軸は各段 1 本だけ (2 軸にしない)。遷移中 (ambiguous) の区間は灰色の帯。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = result["records"]
    steps = [r["step"] for r in recs]
    xs = [max(s, 1) for s in steps] if logx else steps
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1.4, 1.4]))
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(True, color="#e4e3dc", linewidth=0.6)
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
        # 遷移中の区間
        for r0, r1 in zip(recs, recs[1:]):
            if r0["ambiguous"]:
                x0 = max(r0["step"], 1) if logx else r0["step"]
                x1 = max(r1["step"], 1) if logx else r1["step"]
                a.axvspan(x0, x1, color="#d9d8d1", alpha=0.5, linewidth=0)
        if logx:
            a.set_xscale("log")

    # (a) 損失
    losses = result["losses"]
    lx = [max(t, 1) for t in range(len(losses))] if logx else range(len(losses))
    ax[0].plot(lx, losses, color="#3d3c37", linewidth=1.5)
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"training loss $L_n$")
    ax[0].set_title(r"Deep linear network: loss, exact $\lambda$, and SGLD estimates",
                    loc="left", fontsize=11)

    # (b) 全体の λ
    exact_vals = [r["lam_exact"] for r in recs]
    ax[1].step(xs, exact_vals, where="post",
               color=_C_EXACT, linewidth=2, label=r"exact $\lambda$ (self-teacher, Aoyagi)")
    # 縦軸は厳密値の範囲 (+ 余白) に固定する。範囲外の λ̂ (鞍点で事後分布が
    # 正規化できず大きく負になる点など) は端に ▼/▲ で置き、値を書き添える。
    lo, hi = min(exact_vals), max(exact_vals)
    pad = max(1.0, 0.35 * (hi - lo))
    ylo, yhi = min(0.0, lo - pad), hi + pad
    span = (max(xs) - min(xs)) or 1.0
    for key, col, mk, lab, dx in (
            ("self", _C_SELF, "o", r"$\hat\lambda$ self-teacher at $\bar\theta_t$",
             -0.006),
            ("true", _C_TRUE, "s", r"$\hat\lambda$ true data at $\theta_t$",
             +0.006)):
        # 同じ点の 2 系列が重ならないよう、x を少しだけずらす (対数軸では比で)
        def _sh(x, dx=dx):
            return x * (1 + 5 * dx) if logx else x + dx * span
        pts = [(_sh(x), r[key]["lam"], r[key]["std"]) for x, r in zip(xs, recs)
               if key in r]
        inr = [p for p in pts if ylo <= p[1] <= yhi]
        out = [p for p in pts if not ylo <= p[1] <= yhi]
        if inr:
            ax[1].errorbar([p[0] for p in inr], [p[1] for p in inr],
                           yerr=[p[2] for p in inr], color=col, marker=mk,
                           markersize=6, linewidth=0, elinewidth=1.2,
                           capsize=2, label=lab)
        for x, v, _ in out:
            yy = ylo if v < ylo else yhi
            ax[1].plot([x], [yy], marker="v" if v < ylo else "^", color=col,
                       markersize=8, linewidth=0, clip_on=False,
                       label=None if inr else lab)
            ax[1].annotate(f"{v:.0f}", (x, yy), textcoords="offset points",
                           xytext=(5, 6 if v < ylo else -12), fontsize=7,
                           color="#3d3c37")
            inr = inr or [None]          # 凡例は 1 回だけ
    ax[1].set_ylim(ylo, yhi)
    q = [(x, r["quad_true"]["total"]) for x, r in zip(xs, recs)
         if "quad_true" in r and r["quad_true"]["proper"]]
    if q:
        ax[1].plot([p[0] for p in q], [p[1] for p in q], color=_C_QUAD,
                   marker="x", markersize=7, linewidth=0,
                   label=r"quadratic prediction of $\hat\lambda$ (true data)")
    # 段の値を直接ラベル
    last = None
    for x, r in zip(xs, recs):
        if not r["ambiguous"] and r["lam_exact_str"] != last:
            ax[1].annotate(f"k={r['k']}: {r['lam_exact_str']}",
                           (x, r["lam_exact"]), textcoords="offset points",
                           xytext=(4, 4), fontsize=8, color="#3d3c37")
            last = r["lam_exact_str"]
    ax[1].set_ylabel(r"$\lambda$ (all layers)")
    ax[1].legend(fontsize=8, loc="lower right", frameon=False)

    # (c) 層ごとの λ
    nl = len(recs[0]["lam_layers"])
    for l in range(nl):
        ax[2].step(xs, [r["lam_layers"][l] for r in recs], where="post",
                   color=_C_EXACT, linestyle=_LAYER_STYLES[l % 4],
                   linewidth=1.8, label=rf"exact $\lambda$, layer {l+1}")
        for key, col in (("self_layers", _C_SELF), ("true_layers", _C_TRUE)):
            pts = [(x, r[key][l]["lam"]) for x, r in zip(xs, recs) if key in r]
            if pts:
                ax[2].plot([p[0] for p in pts], [p[1] for p in pts],
                           color=col, marker=_LAYER_MARKERS[l % 4],
                           linewidth=0, markersize=6,
                           label=rf"$\hat\lambda$ {key.split('_')[0]}, layer {l+1}")
    ax[2].set_ylabel(r"$\lambda$ (one layer, others frozen)")
    ax[2].set_xlabel("training step" + (" (log)" if logx else ""))
    ax[2].legend(fontsize=8, loc="lower right", frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--dims", type=int, nargs="+", default=[6, 6, 6],
                    help="入力側から d_in h_1 … d_out")
    ap.add_argument("--teacher-sv", type=float, nargs="+",
                    default=[8.0, 4.0, 2.0, 1.0])
    ap.add_argument("--n-data", type=int, default=64)
    ap.add_argument("--init", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sgld", action="store_true")
    ap.add_argument("--sgld-every", type=int, default=10)
    ap.add_argument("--sgld-layers", action="store_true")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--sgld-steps", type=int, default=3000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--logx", action="store_true")
    ap.add_argument("--out", default="result/dln_trajectory")
    a = ap.parse_args()

    import os

    cfg = DLNConfig(dims=tuple(a.dims), teacher_sv=tuple(a.teacher_sv),
                    n_data=a.n_data, init_scale=a.init, lr=a.lr,
                    n_steps=a.steps, every=a.every, noise=a.noise, seed=a.seed)
    res = analyze(cfg, sgld=a.sgld, sgld_every=a.sgld_every,
                  sgld_layers=a.sgld_layers, gamma=a.gamma, eps=a.eps,
                  sgld_steps=a.sgld_steps, chains=a.chains)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    write_csv(res, a.out + ".csv")
    plot(res, a.out + ".png", logx=a.logx)
    print(f"wrote {a.out}.json / .csv / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
