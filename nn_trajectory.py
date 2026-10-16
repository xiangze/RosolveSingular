r"""
nn_trajectory.py
================

**学習軌道に沿った λ — ReLU / tanh MLP・CNN・線形 attention への拡張**

`dln_trajectory.py` (深層線形網) と同じ図を、閉形式の λ が無いアーキテクチャで
描く。小さな初期値から学習すると、これらも「成分」を 1 つずつ獲得しながら
鞍点を渡る (Zhang, Saxe & Latham, arXiv:2512.20607):

    relu   1 次元入力の 2 層 ReLU MLP     成分 = 隠れユニット (折れ点)
    tanh   2 層 tanh MLP (バイアスなし)   成分 = 隠れユニット
    cnn    1 次元畳み込み + ReLU + 平均    成分 = 畳み込みカーネル (チャネル)
    attn   多頭の線形 attention (d_k=1)   成分 = ヘッド
    linear 2 層の深層線形網               成分 = 積の特異モード (検算用)

各チェックポイント θ_t で:

1. 成分ごとの強さ c_j を測り (ユニットなら |出力重み|·‖入力重み‖)、
   教師の最小の成分の強さ s_min に対する閾値 τ·s_min (τ = 0.1/0.5/0.9) で
   「獲得済み」の成分を数える。閾値で数が食い違えば遷移中 (ambiguous)。
2. 獲得していない成分のパラメータを 0 にした θ̄_t を作る (DLN の階数の
   切り詰めに相当)。自己教師 K_{θ̄_t}(θ) = Σ_i ‖f_θ(x_i) − f_{θ̄_t}(x_i)‖² を考える。
3. その局所 RLCT λ(θ̄_t) を次の 3 つで挟む / 求める:

   **下界 (健全)**  rank J(θ̄_t) / 2。J は全データ点の出力のパラメータ微分。
       Aoyagi Lemma 1(1) (部分生成元の単調性) と同じ論法。
   **上界 (数値)**  codim(F)/2。F は θ̄_t を通るファイバー {θ : f_θ = f_{θ̄_t}}
       の成分。θ̄_t + 小さな乱数から Gauss–Newton でファイバーに射影し、
       着地点 p での rank J(p) を codim とみなす (いくつかの出発点の最小)。
       根拠: 生成元は Lipschitz なので K ≤ L²·dist(θ, F)²、したがって
       vol{K < t} ≳ t^{codim F / 2} で λ ≤ codim F / 2 (`lc_filter` の U1 を
       数値で行うもの)。**仮定**: (a) 着地した成分の閉包が θ̄_t を含む
       (半径を小さく取る)、(b) その成分上で K が横断方向に 2 次で消える
       (被約)、(c) 数値階数が正しい。どれかが破れると上界にならない。
   **厳密 (証明つき)**  `torch_rlct.torch_local_rlct` (attn は自前の
       イデアル) で解消し、`certify` が proved を返したときだけ厳密値として
       描く。解消経路は座標に依存して誤ることがある (dln_trajectory の
       known issue) ので、proved でない値は描かない。各プラトーの代表点だけで
       計算し、子プロセスで時間制限をかける。

linear では厳密値を Aoyagi の閉形式から取り、上の括弧がそれを挟むことを
ガード `nn_trajectory_tests` で検査する。

------------------------------------------------------------------
読み方の注意
------------------------------------------------------------------
* 括弧が潰れる (下界 = 上界) 点では λ がほぼ確定する (上界の仮定つき)。
* 獲得していない成分が 0 の点は、ReLU ではセル境界 (前活性化が恒等的に 0)
  にある。torch の微分はそこで 0 を返すので下界は保守側に倒れる。厳密経路は
  境界のセルを全部調べきれないことが多い (status が proved にならない)。
* tanh の厳密経路は θ̄ のまわりの Taylor 打ち切りモデル (`taylor_order` 次)
  の値で、tanh そのものの値ではない。
* 「獲得済み」の判定は閾値に依存する。プラトーでは閾値によらず一致し、
  遷移中だけ食い違う (図の灰色の帯)。

使い方
------
    python nn_trajectory.py --arch relu                 # 1 つ
    python nn_trajectory.py --arch all --compare        # 全部 + 比較図
    python nn_trajectory.py --arch attn --sgld          # λ̂ も重ねる
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["TrajConfig", "DEFAULTS", "make_task", "train_checkpoints",
           "component_strengths", "snap", "jacobian", "lower_bound",
           "upper_bound", "exact_lambda", "analyze", "plot", "plot_compare"]

ARCHS = ("linear", "relu", "tanh", "cnn", "attn")
#: 成分を「積 W2·W1 の特異モード」で数えるアーキテクチャ。tanh は小さな初期値
#: では線形網と同じく特異モードを順に学習し、ユニットどうしは同じ向きに
#: 揃わない (実測: 3 ユニットの向きの余弦は ±0.9 程度で、1 つに縮退しない)。
MODE_ARCHS = ("linear", "tanh")
_W2 = {"linear": "1.weight", "tanh": "2.weight"}


# ----------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------
@dataclass
class TrajConfig:
    arch: str = "relu"
    n_data: int = 24
    width: int = 3                     # 隠れユニット / チャネル / ヘッドの数
    teacher_strengths: Tuple[float, ...] = (4.0, 2.0, 1.0)
    init_scale: float = 1e-3
    lr: float = 0.05
    n_steps: int = 3000
    every: int = 10
    seed: int = 0
    #: 獲得済みの判定 τ·s_min。成分は指数的に育つので対数的に離して取る
    thresholds: Tuple[float, ...] = (0.02, 0.05, 0.15)
    # アーキテクチャ固有
    d_in: int = 2                      # tanh / attn / linear の入力次元
    d_out: int = 1                     # tanh / linear の出力次元
    seq_len: int = 5                   # cnn の入力長 / attn のトークン数
    kernel: int = 2                    # cnn のカーネル幅


#: アーキテクチャごとの既定 (段がきれいに見えるように調整済み)
DEFAULTS: Dict[str, Dict] = {
    "linear": dict(width=3, d_in=3, d_out=3, n_data=16,
                   teacher_strengths=(4.0, 2.0, 1.0), lr=0.05, n_steps=600,
                   every=3, init_scale=1e-3, thresholds=(0.1, 0.5, 0.9)),
    "relu": dict(width=6, n_data=24, teacher_strengths=(4.0, 2.0, 1.0),
                 lr=0.05, n_steps=3000, every=10, init_scale=1e-3),
    "tanh": dict(width=3, d_in=3, d_out=3, n_data=16,
                 teacher_strengths=(8.0, 2.0, 0.5), lr=0.05, n_steps=1500,
                 every=5, init_scale=1e-3),
    "cnn": dict(width=3, n_data=16, seq_len=5, kernel=2,
                teacher_strengths=(8.0, 1.0), lr=0.05, n_steps=3000,
                every=10, init_scale=1e-3),
    "attn": dict(width=3, d_in=2, seq_len=3, n_data=16,
                 teacher_strengths=(4.0, 2.0, 1.0), lr=0.005, n_steps=3000,
                 every=10, init_scale=1e-2),
}


def config_for(arch: str, **over) -> TrajConfig:
    kw = dict(DEFAULTS.get(arch, {}))
    kw.update({k: v for k, v in over.items() if v is not None})
    return TrajConfig(arch=arch, **kw)


# ----------------------------------------------------------------------
# モデル
# ----------------------------------------------------------------------
def _nn():
    import torch.nn as nn
    return nn


class LinearAttention(_nn().Module):
    r"""多頭の線形 attention (softmax なし、d_k = d_v = 1)。

        f(X)_t = (1/T) Σ_h Σ_s (q_h·x_t)(k_h·x_s)(v_h·x_s)

    X は (n, T, d)、出力は (n, T)。各ヘッドは (q_h, k_h, v_h) ∈ R^{3d} で、
    スケール q→aq, k→bk, v→v/(ab) が 2 次元の対称性。パラメータは
    q, k, v の 3 つの (H, d) 行列。
    """

    def __init__(self, d: int, heads: int):
        import torch
        super().__init__()
        self.q = torch.nn.Parameter(torch.zeros(heads, d))
        self.k = torch.nn.Parameter(torch.zeros(heads, d))
        self.v = torch.nn.Parameter(torch.zeros(heads, d))

    def forward(self, X):
        Q = X @ self.q.T                                  # (n, T, H)
        K = X @ self.k.T
        V = X @ self.v.T
        KV = (K * V).sum(dim=1, keepdim=True)             # (n, 1, H)
        return (Q * KV).sum(dim=2) / X.shape[1]           # (n, T)


def build_model(cfg: TrajConfig, width: Optional[int] = None):
    """アーキテクチャの nn.Module (float64)。width を変えれば教師にも使う。"""
    nn = _nn()
    H = cfg.width if width is None else width
    a = cfg.arch
    if a == "linear":
        m = nn.Sequential(nn.Linear(cfg.d_in, H, bias=False),
                          nn.Linear(H, cfg.d_out, bias=False))
    elif a == "relu":
        m = nn.Sequential(nn.Linear(1, H), nn.ReLU(),
                          nn.Linear(H, 1, bias=False))
    elif a == "tanh":
        m = nn.Sequential(nn.Linear(cfg.d_in, H, bias=False), nn.Tanh(),
                          nn.Linear(H, cfg.d_out, bias=False))
    elif a == "cnn":
        L = cfg.seq_len - cfg.kernel + 1
        m = nn.Sequential(nn.Conv2d(1, H, (1, cfg.kernel), bias=False),
                          nn.ReLU(), nn.AvgPool2d((1, L)), nn.Flatten(),
                          nn.Linear(H, 1, bias=False))
    elif a == "attn":
        m = LinearAttention(cfg.d_in, H)
    else:
        raise ValueError(f"未知の arch {a}")
    return m.double()


# パラメータを「成分」ごとに束ねる: 各要素は (パラメータ名, index の組) の列。
def component_slices(model, cfg: TrajConfig) -> List[List[Tuple[str, tuple]]]:
    a = cfg.arch
    H = cfg.width
    if a == "relu":
        return [[("0.weight", (j,)), ("0.bias", (j,)), ("2.weight", (0, j))]
                for j in range(H)]
    if a in ("tanh", "linear"):
        return [[("0.weight", (j,)), ("1.weight" if a == "linear"
                                      else "2.weight", (slice(None), j))]
                for j in range(H)]
    if a == "cnn":
        return [[("0.weight", (j,)), ("4.weight", (0, j))] for j in range(H)]
    if a == "attn":
        return [[("q", (h,)), ("k", (h,)), ("v", (h,))] for h in range(H)]
    raise ValueError(a)


def component_strengths(model, cfg: TrajConfig) -> List[float]:
    r"""成分の強さ。積の形のパラメータなので「各因子のノルムの積」。

    relu: |a_j|·‖(w_j, b_j)‖、tanh/linear: ‖W2[:, j]‖·‖W1[j]‖、
    cnn: |r_j|·‖kernel_j‖、attn: ‖q_h‖‖k_h‖‖v_h‖。
    linear と tanh は成分 = 隠れユニットではなく積 W2·W1 の特異値
    (`MODE_ARCHS` の説明を参照)。
    """
    import torch

    P = dict(model.named_parameters())
    if cfg.arch in MODE_ARCHS:
        prod = P[_W2[cfg.arch]] @ P["0.weight"]
        return torch.linalg.svdvals(prod.detach()).tolist()
    out = []
    for comp in component_slices(model, cfg):
        s = 1.0
        for name, idx in comp:
            s *= float(P[name].detach()[idx].norm())
        out.append(s)
    return out


# ----------------------------------------------------------------------
# 教師とデータ
# ----------------------------------------------------------------------
def _set_teacher(teacher, cfg: TrajConfig, g) -> None:
    """教師の成分を、強さが teacher_strengths になるように置く。"""
    import torch

    S = [s for s in cfg.teacher_strengths if s > 0]
    a = cfg.arch
    P = dict(teacher.named_parameters())
    with torch.no_grad():
        for p in teacher.parameters():
            p.zero_()
        if a == "relu":
            # 折れ点を [-0.6, 0.6] に等間隔、傾きの変化の符号を交互に
            kinks = torch.linspace(-0.6, 0.6, len(S), dtype=torch.float64)
            for j, (s, c) in enumerate(zip(S, kinks)):
                sgn = 1.0 if j % 2 == 0 else -1.0
                w = sgn * 1.0
                P["0.weight"][j, 0] = w
                P["0.bias"][j] = -w * c
                P["2.weight"][0, j] = s / math.sqrt(1 + float(c) ** 2)
        elif a in ("tanh", "linear"):
            W1 = "0.weight"
            W2 = "1.weight" if a == "linear" else "2.weight"
            U, _ = torch.linalg.qr(torch.randn(cfg.d_out, cfg.d_out,
                                               generator=g, dtype=torch.float64))
            V, _ = torch.linalg.qr(torch.randn(cfg.d_in, cfg.d_in,
                                               generator=g, dtype=torch.float64))
            for j, s in enumerate(S):
                # 入力側は単位ベクトル × sqrt(s)、出力側も sqrt(s) (釣り合い)
                P[W1][j] = math.sqrt(s) * V[:, j % cfg.d_in]
                P[W2][:, j] = math.sqrt(s) * U[:, j % cfg.d_out]
        elif a == "cnn":
            for j, s in enumerate(S):
                ker = torch.randn(cfg.kernel, generator=g, dtype=torch.float64)
                ker = ker / ker.norm()
                P["0.weight"][j, 0, 0] = ker
                P["4.weight"][0, j] = s
        elif a == "attn":
            for h, s in enumerate(S):
                r = s ** (1.0 / 3.0)
                for nm in ("q", "k", "v"):
                    u = torch.randn(cfg.d_in, generator=g, dtype=torch.float64)
                    P[nm][h] = r * u / u.norm()


def make_inputs(cfg: TrajConfig, g):
    import torch

    a = cfg.arch
    if a == "relu":
        # 折れ点 (±0.6, 0) にちょうど乗らないようにずらした格子
        return (torch.linspace(-1, 1, cfg.n_data, dtype=torch.float64)
                .reshape(-1, 1) + 0.013)
    if a in ("tanh", "linear"):
        Z = torch.randn(cfg.n_data, cfg.d_in, generator=g, dtype=torch.float64)
        if a == "linear":
            Q, _ = torch.linalg.qr(Z)
            return Q * math.sqrt(cfg.n_data)           # 白色化
        return Z * 0.7
    if a == "cnn":
        return torch.randn(cfg.n_data, 1, 1, cfg.seq_len, generator=g,
                           dtype=torch.float64)
    if a == "attn":
        return torch.randn(cfg.n_data, cfg.seq_len, cfg.d_in, generator=g,
                           dtype=torch.float64)
    raise ValueError(a)


def make_task(cfg: TrajConfig):
    """(X, y, teacher)。教師は学生と同じ形で、成分数 = len(teacher_strengths)。"""
    import torch

    g = torch.Generator().manual_seed(cfg.seed)
    X = make_inputs(cfg, g)
    teacher = build_model(cfg, width=max(cfg.width,
                                         len(cfg.teacher_strengths)))
    _set_teacher(teacher, cfg, g)
    with torch.no_grad():
        y = teacher(X).reshape(X.shape[0], -1).clone()
    return X, y, teacher


def make_student(cfg: TrajConfig):
    """小さな初期値の学生。呼び出し側の既定 dtype に依らず同じ初期値にする。

    `nn.Linear` の生成は既定 dtype で乱数を消費する量が変わるので、ここで
    float64 に固定してから作る (固定しないと float32 既定の CLI と float64
    既定のスクリプトで軌道が変わった: ReLU で段が 1 つ消えた)。
    """
    import torch

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(cfg.seed + 1)
        m = build_model(cfg)
        with torch.no_grad():
            for p in m.parameters():
                p.normal_(0.0, cfg.init_scale)
    finally:
        torch.set_default_dtype(prev)
    return m


def _flat_out(model, X):
    out = model(X)
    return out.reshape(out.shape[0], -1)


def loss_fn(model, X, y):
    """L_n = (1/n) Σ ½‖y − f‖² (`llc_estimate.nll` の回帰と同じ規約)。"""
    return 0.5 * ((_flat_out(model, X) - y) ** 2).sum(dim=1).mean()


# ----------------------------------------------------------------------
# 学習
# ----------------------------------------------------------------------
def train_checkpoints(cfg: TrajConfig, X, y):
    import torch

    net = make_student(cfg)
    opt = torch.optim.SGD(net.parameters(), lr=cfg.lr)
    steps, states, losses = [], [], []
    for t in range(cfg.n_steps + 1):
        if t % cfg.every == 0:
            steps.append(t)
            states.append({k: v.detach().clone()
                           for k, v in net.state_dict().items()})
        opt.zero_grad()
        L = loss_fn(net, X, y)
        losses.append(L.item())
        if t < cfg.n_steps:
            L.backward()
            opt.step()
    return net, steps, states, losses


# ----------------------------------------------------------------------
# 成分の判定と丸め
# ----------------------------------------------------------------------
@dataclass
class CompInfo:
    k: int                      # 獲得済みの「機能的な」成分の数 (クラスタ数)
    ks: Tuple[int, ...]         # 閾値ごとの k
    strengths: List[float]      # 成分 (ユニット等) ごとの強さ
    active: List[int]           # 中央の閾値で獲得済みの成分
    clusters: List[List[int]]   # active を同じ向きでまとめたもの
    ambiguous: bool


#: 同じクラスタとみなす向きの余弦の下限
CLUSTER_COS = 0.995


def _signatures(model, cfg: TrajConfig) -> List[List]:
    r"""成分ごとの「入力側の向き」。同じ向きの成分は関数として 1 つに縮退する。

    relu / cnn: 正の斉次性 a·relu(c·u) = ac·relu(u) (c > 0) があるので、入力側
      (w_j, b_j) / カーネルの**向き**だけで決まる。
    attn: ヘッドは (q·x_t)(k·x_s)(v·x_s) なので q, k, v それぞれの向き (符号込みで
      ±1 倍は出力の符号に吸収されるため、向きの比較は |cos| で行う)。
    """
    import torch

    P = dict(model.named_parameters())
    out = []
    for comp in component_slices(model, cfg):
        if cfg.arch == "relu":
            name_w, name_b = comp[0][0], comp[1][0]
            j = comp[0][1][0]
            out.append([torch.cat([P[name_w].detach()[j],
                                   P[name_b].detach()[j:j + 1]])])
        elif cfg.arch == "cnn":
            j = comp[0][1][0]
            out.append([P["0.weight"].detach()[j].reshape(-1)])
        elif cfg.arch == "attn":
            h = comp[0][1][0]
            out.append([P[n].detach()[h] for n in ("q", "k", "v")])
        else:
            out.append([])
    return out


def _same(a: List, b: List, arch: str) -> bool:
    for u, v in zip(a, b):
        nu, nv = float(u.norm()), float(v.norm())
        if nu == 0 or nv == 0:
            return False
        c = float(u @ v) / (nu * nv)
        if arch == "attn":
            c = abs(c)
        if c < CLUSTER_COS:
            return False
    return True


def _cluster(active: List[int], sigs, arch: str) -> List[List[int]]:
    cl: List[List[int]] = []
    for j in active:
        for c in cl:
            if _same(sigs[c[0]], sigs[j], arch):
                c.append(j)
                break
        else:
            cl.append([j])
    return cl


def classify(model, cfg: TrajConfig) -> CompInfo:
    r"""獲得済みの成分を数える。

    強さ c_j が τ·s_min を超えた成分を active とし、active を同じ向きで
    クラスタにまとめた数を k とする (linear は積の特異値の数)。
    閾値 τ ごとに k を出し、食い違えば遷移中。
    """
    c = component_strengths(model, cfg)
    smin = min(s for s in cfg.teacher_strengths if s > 0)
    sigs = None if cfg.arch in MODE_ARCHS else _signatures(model, cfg)
    ks = []
    act_mid, cl_mid = [], []
    mid = cfg.thresholds[len(cfg.thresholds) // 2]
    for t in cfg.thresholds:
        act = [j for j, v in enumerate(c) if v > t * smin]
        cl = [[j] for j in act] if sigs is None else _cluster(act, sigs,
                                                              cfg.arch)
        ks.append(len(cl))
        if t == mid:
            act_mid, cl_mid = act, cl
    ks_t = tuple(ks)
    return CompInfo(k=ks_t[len(ks_t) // 2], ks=ks_t, strengths=c,
                    active=act_mid, clusters=cl_mid,
                    ambiguous=len(set(ks_t)) > 1)


def snap(model, cfg: TrajConfig, info: CompInfo):
    r"""θ̄: 獲得していない成分を 0 にし、同じクラスタの成分の向きを揃えた複製。

    「成分が 0」「同じクラスタの成分が同じ向き」はどちらも勾配流の不変多様体
    (置換対称性・小さな初期値の極限) で、学習軌道はその近くを通る。
    向きを揃えるときは各成分の大きさを保ち、向きだけクラスタ平均に置き換える
    linear / tanh は各層の重みを SVD で階数 k に切り詰める
    (`dln_trajectory.snap_to_rank` と同じ操作)。
    """
    import torch

    if cfg.arch in MODE_ARCHS:
        return _snap_rank(model, info.k)
    bar = copy.deepcopy(model)
    P = dict(bar.named_parameters())
    comps = component_slices(bar, cfg)
    with torch.no_grad():
        for j, comp in enumerate(comps):
            if j in info.active:
                continue
            for name, idx in comp:
                P[name][idx] = 0.0
        for cl in info.clusters:
            if len(cl) < 2:
                continue
            if cfg.arch == "relu":
                vecs = [torch.cat([P["0.weight"][j], P["0.bias"][j:j + 1]])
                        for j in cl]
                mean = sum(v / v.norm() for v in vecs)
                mean = mean / mean.norm()
                for j, v in zip(cl, vecs):
                    nv = v.norm() * mean
                    P["0.weight"][j] = nv[:-1]
                    P["0.bias"][j] = nv[-1]
            elif cfg.arch == "cnn":
                vecs = [P["0.weight"][j].clone() for j in cl]
                mean = sum(v / v.norm() for v in vecs)
                mean = mean / mean.norm()
                for j, v in zip(cl, vecs):
                    P["0.weight"][j] = v.norm() * mean
            elif cfg.arch == "attn":
                for n in ("q", "k", "v"):
                    ref = P[n][cl[0]].clone()
                    vecs = [P[n][h].clone() for h in cl]
                    sg = [1.0 if float(v @ ref) >= 0 else -1.0 for v in vecs]
                    mean = sum(s * v / v.norm() for s, v in zip(sg, vecs))
                    mean = mean / mean.norm()
                    for h, s, v in zip(cl, sg, vecs):
                        P[n][h] = s * v.norm() * mean
    return bar


def _snap_rank(model, k: int):
    """重みを持つ層をすべて SVD で階数 k に切り詰めた複製。"""
    import torch

    bar = copy.deepcopy(model)
    with torch.no_grad():
        for m in bar:
            if not hasattr(m, "weight") or m.weight is None:
                continue
            U, S, Vh = torch.linalg.svd(m.weight, full_matrices=False)
            S[k:] = 0.0
            m.weight.copy_(U @ torch.diag(S) @ Vh)
    return bar


# ----------------------------------------------------------------------
# 下界・上界
# ----------------------------------------------------------------------
def _param_vector(model):
    import torch
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _functional(model, X):
    """θ (平坦なベクトル) → 全データ点の出力 (平坦)。"""
    import torch
    from torch.func import functional_call

    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    sizes = [p.numel() for _, p in model.named_parameters()]

    def f(theta):
        parts = torch.split(theta, sizes)
        params = {n: t.reshape(s) for n, t, s in zip(names, parts, shapes)}
        return functional_call(model, params, (X,)).reshape(-1)
    return f


def jacobian(model, X, theta=None):
    import torch

    f = _functional(model, X)
    th = _param_vector(model) if theta is None else theta
    return torch.autograd.functional.jacobian(f, th)


#: 数値階数の相対閾値。下界と上界で**同じ値**を使うこと (別の値だと、θ̄ で
#: 1e-7 と 1e-6 の間に本物の特異値があった tanh で 下界 > 上界 になった)。
#: J は float64 で 1e-15 程度まで正確なので、1e-9 より上の特異値は本物とみなす。
RANK_RTOL = 1e-9


def _num_rank(J, rtol: float = RANK_RTOL) -> int:
    import torch

    s = torch.linalg.svdvals(J)
    if s.numel() == 0 or s[0].item() == 0.0:
        return 0
    return int((s > rtol * s[0]).sum())


def lower_bound(model_bar, X, rtol: float = RANK_RTOL) -> float:
    """rank J(θ̄)/2 (健全な下界)。"""
    return _num_rank(jacobian(model_bar, X), rtol) / 2.0


@dataclass
class UpperInfo:
    value: float                 # min codim / 2 (着地できなければ d/2)
    codims: List[int]            # 出発点ごとの codim (着地できたもの)
    dists: List[float]           # 着地点と θ̄ の距離
    n_fail: int
    trivial: bool = False        # True なら value = d/2 (自明な上界)


def upper_bound(model_bar, X, *, n_starts: int = 10,
                radii: Sequence[float] = (3e-3, 1e-2, 3e-2),
                rtol: float = RANK_RTOL,
                tol: float = 1e-11, max_iter: int = 40,
                seed: int = 0) -> UpperInfo:
    r"""ファイバー F = {f_θ = f_θ̄} の成分の余次元の最小 / 2 (数値)。

    θ̄ + ρu から Levenberg–Marquardt で ‖f_θ − f_θ̄‖ → 0 に射影し、着地点 p
    での rank J(p) を codim とみなす。p が θ̄ から離れすぎた (>10ρ) 出発点は捨てる。

    `max_iter` 回以内に `tol` まで下がらない出発点も捨てる。被約な成分の
    滑らかな点には LM が 2 次収束で着地するが、特異点・被約でない点には
    劣線形にしか近づかない (実測: tanh の階数 1 の点で 200 回後も 1e-9)。
    そこでは rank J が codim を過小評価しうるので、上界として使わない。
    """
    import torch

    f = _functional(model_bar, X)
    th0 = _param_vector(model_bar)
    y0 = f(th0).detach()
    # θ̄ を通る成分の接錐は ker J(θ̄) に入るので codim ≥ rank J(θ̄)。
    # これより小さい codim の着地点は θ̄ を通らない成分 (か数値誤差) なので捨てる
    # (実測: tanh の階数 1 の点で codim 16 < rank 17 の着地があった)。
    r0 = _num_rank(torch.autograd.functional.jacobian(f, th0), rtol)
    scale = max(float(th0.norm()), 1.0)
    g = torch.Generator().manual_seed(seed)
    codims, dists, fails = [], [], 0
    for rho in radii:
        for _ in range(n_starts):
            u = torch.randn(th0.shape, generator=g, dtype=th0.dtype)
            th = th0 + rho * scale * u / u.norm()
            mu = 1e-3
            ok = False
            for _ in range(max_iter):
                r = f(th) - y0
                nr = float(r.norm())
                if nr < tol:
                    ok = True
                    break
                J = torch.autograd.functional.jacobian(f, th)
                A = J.T @ J
                step = torch.linalg.solve(
                    A + mu * torch.eye(A.shape[0], dtype=A.dtype), J.T @ r)
                th_new = th - step
                if float((f(th_new) - y0).norm()) < nr:
                    th, mu = th_new, max(mu / 3, 1e-12)
                else:
                    mu *= 4
            d = float((th - th0).norm())
            if not ok or d > 10 * rho * scale:
                fails += 1
                continue
            J = torch.autograd.functional.jacobian(f, th)
            cd = _num_rank(J, rtol)
            if cd < r0:
                fails += 1
                continue
            codims.append(cd)
            dists.append(d)
    # λ ≤ d/2 は常に成り立つ (K ≥ 0 が θ̄ で消えるだけで十分)。射影が 2 次収束で
    # 着地しない (= 着地点が特異で被約でない疑い; rank J は codim を過小評価
    # しうる) ときはこの自明な上界に退く。
    d = th0.numel()
    val = min(min(codims), d) / 2.0 if codims else d / 2.0
    return UpperInfo(value=val, codims=codims, dists=dists, n_fail=fails,
                     trivial=not codims)


# ----------------------------------------------------------------------
# θ̄ を「有理数で書ける」点に置き直す
# ----------------------------------------------------------------------
def _rat(x: float, den: int = 16):
    from fractions import Fraction
    return Fraction(float(x)).limit_denominator(den)


def rationalize(bar, cfg: TrajConfig, info: CompInfo, den: int = 16):
    r"""θ̄ の構造 (0 の成分・平行な成分・階数) を**有理数のまま**保つ値に置き直す。

    厳密経路 (`torch_rlct`) は重みを成分ごとに有理数化するので、そのままだと
    平行だった成分が平行でなくなり、階数 k の行列が full rank に戻る
    (= 別の層の点を解くことになる)。そこで
      relu / cnn / attn : 成分 = (有理数の大きさ) × (クラスタで共通の有理数の向き)
      linear / tanh     : 各層 W = F·G (F: H×k, G: k×d を有理数化) で階数 k を保つ
    と書き直す。分母は den 以下の積・和なので、`torch_rlct` の
    max_denominator (既定 10⁶) で正確に復元される。下界・上界・厳密値は
    すべてこの点で計算する。
    """
    import torch

    def R(t):
        return torch.tensor([float(_rat(v, den)) for v in t.reshape(-1).tolist()],
                            dtype=torch.float64).reshape(t.shape)

    out = copy.deepcopy(bar)
    with torch.no_grad():
        if cfg.arch in MODE_ARCHS:
            for m in out:
                if not hasattr(m, "weight") or m.weight is None:
                    continue
                W = m.weight
                if info.k == 0:
                    W.zero_()
                    continue
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                F = R(U[:, :info.k] * S[:info.k])
                G = R(Vh[:info.k])
                W.copy_(F @ G)
            return out
        P = dict(out.named_parameters())
        if cfg.arch == "relu":
            for cl in info.clusters:
                d = R(torch.cat([P["0.weight"][cl[0]], P["0.bias"][cl[0]:cl[0] + 1]])
                      / torch.cat([P["0.weight"][cl[0]],
                                   P["0.bias"][cl[0]:cl[0] + 1]]).norm())
                for j in cl:
                    n = float(_rat(torch.cat([P["0.weight"][j],
                                              P["0.bias"][j:j + 1]]).norm(), den))
                    P["0.weight"][j] = n * d[:-1]
                    P["0.bias"][j] = n * d[-1]
            P["2.weight"].copy_(R(P["2.weight"]))
        elif cfg.arch == "cnn":
            for cl in info.clusters:
                v0 = P["0.weight"][cl[0]]
                d = R(v0 / v0.norm())
                for j in cl:
                    n = float(_rat(P["0.weight"][j].norm(), den))
                    P["0.weight"][j] = n * d
            P["4.weight"].copy_(R(P["4.weight"]))
        elif cfg.arch == "attn":
            for cl in info.clusters:
                for nm in ("q", "k", "v"):
                    v0 = P[nm][cl[0]]
                    d = R(v0 / v0.norm())
                    for h in cl:
                        sg = 1.0 if float(P[nm][h] @ v0) >= 0 else -1.0
                        n = float(_rat(P[nm][h].norm(), den))
                        P[nm][h] = sg * n * d
    return out


# ----------------------------------------------------------------------
# 厳密値 (証明つき)
# ----------------------------------------------------------------------
def attention_ideal(bar, X, max_den: int = 10 ** 6):
    r"""線形 attention の fiber ideal を sympy で作る。

    変数は θ − θ̄ (q, k, v の全成分)。生成元は全サンプル・全トークンの
    f_θ(X) − f_θ̄(X) (θ̄ で恒等的に 0)。0 でないヘッドのスケール対称性
    (q→(1+ε)q, v→(1−ε)v) と (k→(1+ε)k, v→(1−ε)v) を接ベクトルとして渡す
    (0 のヘッドでは作用が自由でないので渡さない)。
    """
    import sympy as sp
    from fractions import Fraction

    def Q(x):
        f = Fraction(float(x)).limit_denominator(max_den)
        return sp.Rational(f.numerator, f.denominator)

    P = {n: p.detach() for n, p in bar.named_parameters()}
    H, d = P["q"].shape
    sym = {}
    val = {}
    for nm in ("q", "k", "v"):
        for h in range(H):
            for i in range(d):
                s_ = sp.Symbol(f"d{nm}_{h}_{i}")
                sym[(nm, h, i)] = s_
                val[(nm, h, i)] = Q(P[nm][h, i])
    variables = list(sym.values())
    Xr = [[[Q(X[n, t, i]) for i in range(d)] for t in range(X.shape[1])]
          for n in range(X.shape[0])]
    T = X.shape[1]

    def dot(nm, h, x, shift):
        return sum((val[(nm, h, i)] + (sym[(nm, h, i)] if shift else 0)) * x[i]
                   for i in range(d))

    gens = []
    for n in range(X.shape[0]):
        for t in range(T):
            def f(shift):
                tot = 0
                for h in range(H):
                    kv = sum(dot("k", h, Xr[n][s_], shift)
                             * dot("v", h, Xr[n][s_], shift) for s_ in range(T))
                    tot += dot("q", h, Xr[n][t], shift) * kv
                return tot / T
            g = sp.expand(f(True) - f(False))
            if g != 0:
                gens.append(g)
    symv = []
    for h in range(H):
        if all(val[(nm, h, i)] == 0 for nm in ("q", "k", "v") for i in range(d)):
            continue
        for a in ("q", "k"):
            vec = {}
            for i in range(d):
                if val[(a, h, i)] != 0:
                    vec[sym[(a, h, i)]] = val[(a, h, i)]
                if val[("v", h, i)] != 0:
                    vec[sym[("v", h, i)]] = -val[("v", h, i)]
            if vec:
                symv.append(vec)
    return gens, variables, symv


def _exact_child(arch, bar, X, taylor_order, q):
    """子プロセスの中身: 解消して certify する。"""
    import sys
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)        # tanh の Taylor 係数が 4300 桁を超える
    try:
        from certify import certify
        if arch == "attn":
            from nn_rlct import local_rlct_from_ideal
            gens, variables, symv = attention_ideal(bar, X)
            loc = local_rlct_from_ideal(gens, variables, symmetry_vectors=symv)
            notes = list(loc.notes)
        else:
            from torch_rlct import torch_local_rlct
            kw = dict(loss="regression", max_vars=60, taylor_order=taylor_order)
            Xin = X
            if arch == "cnn":
                kw["input_shape"] = tuple(X.shape[1:])
                Xin = X.reshape(X.shape[0], -1)
            rep = torch_local_rlct(bar, Xin, **kw)
            loc = rep.local
            notes = list(loc.notes)
        if loc.resolution is None:
            status = "proved"
        else:
            status = certify(loc.resolution).status
        # 下界にしかならない計算 (セル境界を一部しか調べていない等) は proved にしない
        if any("下界" in n or "[warn]" in n for n in notes) and status == "proved":
            status = "lower-bound-only"
        q.put(dict(lam=str(loc.rlct), lam_f=float(loc.rlct),
                   m=int(loc.multiplicity), status=status,
                   core=len(loc.core_vars), notes=[n[:160] for n in notes]))
    except Exception as e:                                 # noqa: BLE001
        q.put(dict(status="error", note=f"{type(e).__name__}: {str(e)[:160]}"))


def exact_lambda(bar, X, cfg: TrajConfig, info: CompInfo, *,
                 timeout: float = 60.0, taylor_order: int = 3) -> Dict:
    r"""θ̄ での厳密な (λ, m) と証明書の状態。

    linear は Aoyagi の閉形式 (status='closed-form')。それ以外は子プロセスで
    解消し、`certify` の結果を status に入れる ('proved' / 'refuted' /
    'unknown' / 'timeout' / 'error' / 'lower-bound-only')。
    """
    import multiprocessing as mp

    t0 = time.time()
    if cfg.arch == "linear":
        from dln import lambda_dln
        H = (cfg.d_out, cfg.width, cfg.d_in)
        lam, th = lambda_dln(H, info.k)
        return dict(lam=str(lam), lam_f=float(lam), m=int(th),
                    status="closed-form", time=0.0)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_exact_child,
                    args=(cfg.arch, bar, X, taylor_order, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return dict(status="timeout", time=round(time.time() - t0, 1))
    try:
        out = q.get(timeout=5)
    except Exception:                                      # noqa: BLE001
        out = dict(status="crash")
    out["time"] = round(time.time() - t0, 1)
    if cfg.arch == "tanh":
        out["note_taylor"] = f"Taylor {taylor_order} 次の打ち切りモデルの値"
    return out


# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------
def representatives(recs: List[Dict], min_len: int = 3) -> List[int]:
    """各プラトー (同じ k が続く非 ambiguous の区間) の中央と、最後の点の index。"""
    reps, i, n = [], 0, len(recs)
    while i < n:
        if recs[i]["ambiguous"]:
            i += 1
            continue
        j = i
        while j + 1 < n and not recs[j + 1]["ambiguous"] \
                and recs[j + 1]["k"] == recs[i]["k"]:
            j += 1
        if j - i + 1 >= min_len:
            reps.append((i + j) // 2)
        i = j + 1
    if n and (n - 1) not in reps and not recs[-1]["ambiguous"]:
        reps.append(n - 1)
    return sorted(set(reps))


def analyze(cfg: TrajConfig, *, bound_every: int = 6, exact: bool = True,
            exact_timeout: float = 30.0, taylor_order: int = 3,
            sgld: bool = False, gamma: float = 1.0, eps: float = 1e-4,
            sgld_steps: int = 2000, chains: int = 2,
            verbose: bool = True) -> Dict:
    r"""学習して、チェックポイントごとの記録を返す。

    * 成分の判定は全チェックポイント。
    * 下界・上界は `bound_every` 個おき + 代表点 (遷移中の点では計算しない)。
    * 厳密値と SGLD は代表点 (各プラトーの中央と最後の点) だけ。
    """
    import torch

    X, y, teacher = make_task(cfg)
    net, steps, states, losses = train_checkpoints(cfg, X, y)
    m = make_student(cfg)
    recs: List[Dict] = []
    infos: List[CompInfo] = []
    for t, sd in zip(steps, states):
        m.load_state_dict(sd)
        ci = classify(m, cfg)
        infos.append(ci)
        recs.append(dict(step=t, loss=losses[t], k=ci.k, ks=list(ci.ks),
                         ambiguous=ci.ambiguous, strengths=ci.strengths,
                         clusters=ci.clusters))
    reps = representatives(recs)
    t0 = time.time()
    for j, (rec, sd, ci) in enumerate(zip(recs, states, infos)):
        is_rep = j in reps
        if rec["ambiguous"] or not (is_rep or j % bound_every == 0):
            continue
        m.load_state_dict(sd)
        bar = rationalize(snap(m, cfg, ci), cfg, ci)
        rec["lower"] = lower_bound(bar, X)
        up = upper_bound(bar, X, seed=0)
        rec["upper"] = up.value
        rec["upper_codims"] = sorted(set(up.codims))
        rec["upper_trivial"] = up.trivial
        if not is_rep:
            continue
        rec["rep"] = True
        if exact:
            rec["exact"] = exact_lambda(bar, X, cfg, ci, timeout=exact_timeout,
                                        taylor_order=taylor_order)
        if sgld:
            from llc_estimate import llc_sgld, realizable_targets
            yb = realizable_targets(bar, X)
            if yb.dim() > 2:
                yb = yb.reshape(yb.shape[0], -1)
            r1 = llc_sgld(bar, X, y=yb, gamma=gamma, eps=eps,
                          n_steps=sgld_steps, chains=chains, seed=cfg.seed)
            m.load_state_dict(sd)
            r2 = llc_sgld(m, X, y=y, gamma=gamma, eps=eps,
                          n_steps=sgld_steps, chains=chains, seed=cfg.seed)
            rec["self"] = dict(lam=r1.lam_hat, std=r1.lam_std)
            rec["true"] = dict(lam=r2.lam_hat, std=r2.lam_std)
        if verbose:
            ex = rec.get("exact", {})
            print(f"  [{cfg.arch}] step {rec['step']:6d}  k={rec['k']}  "
                  f"λ ∈ [{rec['lower']}, {rec['upper']}]  exact: "
                  f"{ex.get('lam', '-')} ({ex.get('status', '-')}, "
                  f"{ex.get('time', 0)}s)"
                  + (f"  λ̂ self {rec['self']['lam']:.2f} true "
                     f"{rec['true']['lam']:.2f}" if "self" in rec else ""),
                  flush=True)
    if verbose:
        print(f"  [{cfg.arch}] 解析 {time.time() - t0:.0f}s", flush=True)
    return dict(config=asdict(cfg), records=recs, losses=losses,
                representatives=reps)


# ----------------------------------------------------------------------
# 出力
# ----------------------------------------------------------------------
# dataviz の既定パレット (検証済み): 青 = 厳密/括弧、橙 = 自己教師 λ̂、緑 = 真のデータ λ̂
_C_EXACT, _C_SELF, _C_TRUE = "#2a78d6", "#eb6834", "#1baf7a"
_C_INK, _C_MUTED, _C_GRID, _C_SURF = "#3d3c37", "#8a8980", "#e4e3dc", "#fcfcfb"
_TITLES = {"linear": "deep linear (2 layers)", "relu": "ReLU MLP (1-D input)",
           "tanh": "tanh MLP", "cnn": "CNN (1-D conv + ReLU)",
           "attn": "linear attention (multi-head)"}


def _style(ax):
    ax.set_facecolor(_C_SURF)
    ax.grid(True, color=_C_GRID, linewidth=0.6)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)


def _shade(ax, recs):
    for r0, r1 in zip(recs, recs[1:]):
        if r0["ambiguous"]:
            ax.axvspan(r0["step"], r1["step"], color="#d9d8d1", alpha=0.5,
                       linewidth=0)


def _draw_lambda(ax, res: Dict, *, legend: bool = True, annotate: bool = True):
    recs = res["records"]
    b = [r for r in recs if "lower" in r]
    xs = [r["step"] for r in b]
    lo = [r["lower"] for r in b]
    hi = [r["upper"] for r in b]
    ax.fill_between(xs, lo, hi, step="post", color=_C_EXACT, alpha=0.18,
                    linewidth=0, label=r"$\lambda$ bracket")
    ax.step(xs, hi, where="post", color=_C_EXACT, linewidth=1.8,
            label=r"upper: codim/2 (numerical)")
    ax.step(xs, lo, where="post", color=_C_EXACT, linewidth=1.2,
            linestyle="--", label=r"lower: rank $J$/2")
    ok = [r for r in recs if r.get("exact", {}).get("status")
          in ("proved", "closed-form")]
    if ok:
        ax.plot([r["step"] for r in ok], [r["exact"]["lam_f"] for r in ok],
                linestyle="none", marker="D", markersize=7, color=_C_EXACT,
                markeredgecolor="white", markeredgewidth=1.2,
                label=r"exact $\lambda$ (proved / closed form)")
        if annotate:
            for r in ok:
                ax.annotate(f"{r['exact']['lam']} (m={r['exact']['m']})",
                            (r["step"], r["exact"]["lam_f"]),
                            textcoords="offset points", xytext=(5, 6),
                            fontsize=7.5, color=_C_INK)
    span = (recs[-1]["step"] - recs[0]["step"]) or 1
    for key, col, mk, lab, dx in (
            ("self", _C_SELF, "o", r"$\hat\lambda$ self-teacher at $\bar\theta_t$",
             -0.008),
            ("true", _C_TRUE, "s", r"$\hat\lambda$ true data at $\theta_t$",
             +0.008)):
        # 同じ点の 2 系列が重ならないよう x を少しずらす
        pts = [dict(r, step=r["step"] + dx * span) for r in recs if key in r]
        if not pts:
            continue
        ymin = min(lo) - 1 if lo else -1
        inr = [r for r in pts if r[key]["lam"] >= ymin]
        out = [r for r in pts if r[key]["lam"] < ymin]
        if inr:
            ax.errorbar([r["step"] for r in inr], [r[key]["lam"] for r in inr],
                        yerr=[r[key]["std"] for r in inr], color=col,
                        marker=mk, markersize=5.5, linewidth=0,
                        elinewidth=1.1, capsize=2, label=lab)
        for r in out:
            ax.plot([r["step"]], [ymin], marker="v", color=col, markersize=7,
                    linewidth=0, clip_on=False, label=None if inr else lab)
            ax.annotate(f"{r[key]['lam']:.0f}", (r["step"], ymin),
                        textcoords="offset points", xytext=(4, 5), fontsize=7,
                        color=_C_INK)
            inr = inr or [None]
    if legend:
        ax.legend(fontsize=7.5, loc="best", frameon=True, framealpha=0.85,
                  edgecolor="none")


def _exact_summary(res: Dict) -> str:
    from collections import Counter
    c = Counter(r["exact"]["status"] for r in res["records"] if "exact" in r)
    return ", ".join(f"{k} {v}" for k, v in c.most_common()) or "not run"


def plot(res: Dict, path: str) -> None:
    """損失・λ・成分の強さの 3 段 (x 軸共有、y 軸は各段 1 本)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = res["config"]
    recs = res["records"]
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1.4, 1.1]))
    for a in ax:
        _style(a)
        _shade(a, recs)
    L = res["losses"]
    ax[0].plot(range(len(L)), L, color=_C_INK, linewidth=1.5)
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"training loss $L_n$")
    ax[0].set_title(f"{_TITLES.get(cfg['arch'], cfg['arch'])}: loss, "
                    r"$\lambda$ at the snapped point $\bar\theta_t$, components",
                    loc="left", fontsize=11)
    _draw_lambda(ax[1], res)
    ax[1].set_ylabel(r"$\lambda(\bar\theta_t)$")
    ax[1].set_title("exact solver at plateau representatives: "
                    + _exact_summary(res), loc="left", fontsize=8,
                    color=_C_MUTED)
    S = list(zip(*[r["strengths"] for r in recs]))
    steps = [r["step"] for r in recs]
    for j, series in enumerate(S):
        ax[2].plot(steps, [max(v, 1e-12) for v in series], color=_C_INK,
                   linewidth=1.1, alpha=0.85,
                   label="component strength" if j == 0 else None)
    smin = min(s for s in cfg["teacher_strengths"] if s > 0)
    for t_ in cfg["thresholds"]:
        ax[2].axhline(t_ * smin, color=_C_MUTED, linewidth=0.8, linestyle=":")
    ax[2].set_yscale("log")
    ax[2].set_ylim(bottom=max(1e-6, 0.3 * min(
        (v for s in S for v in s if v > 0), default=1e-6)))
    unit = {"linear": "singular values of $W_2W_1$",
            "tanh": "singular values of $W_2W_1$",
            "relu": r"$|a_j|\,\|(w_j,b_j)\|$ per unit",
            "cnn": r"$|r_j|\,\|\kappa_j\|$ per kernel",
            "attn": r"$\|q_h\|\|k_h\|\|v_h\|$ per head"}[cfg["arch"]]
    ax[2].set_ylabel("component strength")
    ax[2].text(0.01, 0.04, unit + "   (dotted: thresholds)",
               transform=ax[2].transAxes, fontsize=7.5, color=_C_MUTED)
    ax[2].set_xlabel("training step")
    # 上の段に k を直接書く
    last, last_x = None, -1e18
    span = (recs[-1]["step"] - recs[0]["step"]) or 1
    for r in recs:
        if not r["ambiguous"] and r["k"] != last:
            if r["step"] - last_x > 0.04 * span:     # 近すぎるラベルは省く
                ax[0].annotate(f"k={r['k']}", (r["step"], r["loss"]),
                               textcoords="offset points", xytext=(3, 4),
                               fontsize=7.5, color=_C_INK)
                last_x = r["step"]
            last = r["k"]
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare(results: Dict[str, Dict], path: str) -> None:
    """アーキテクチャを列に並べた比較図 (行: 損失、λ、獲得済みの成分数 k)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = [a for a in ARCHS if a in results]
    n = len(archs)
    fig, ax = plt.subplots(3, n, figsize=(3.3 * n, 7.6), squeeze=False,
                           gridspec_kw=dict(height_ratios=[1, 1.4, 0.8]))
    for c, a in enumerate(archs):
        res = results[a]
        recs = res["records"]
        for r_ in range(3):
            _style(ax[r_][c])
            _shade(ax[r_][c], recs)
        L = res["losses"]
        ax[0][c].plot(range(len(L)), L, color=_C_INK, linewidth=1.3)
        ax[0][c].set_yscale("log")
        ax[0][c].set_title(_TITLES.get(a, a), fontsize=9.5, loc="left")
        _draw_lambda(ax[1][c], res, legend=False, annotate=False)
        ax[1][c].set_title("exact: " + _exact_summary(res), fontsize=7,
                           loc="left", color=_C_MUTED)
        steps = [r["step"] for r in recs]
        ks = [r["k"] if not r["ambiguous"] else float("nan") for r in recs]
        ax[2][c].step(steps, ks, where="post", color=_C_INK, linewidth=1.5)
        ax[2][c].set_ylim(-0.3, max(r["k"] for r in recs) + 0.5)
        ax[2][c].set_xlabel("training step", fontsize=8.5)
        for r_ in range(3):
            ax[r_][c].tick_params(labelsize=7.5)
    ax[0][0].set_ylabel(r"loss $L_n$")
    ax[1][0].set_ylabel(r"$\lambda(\bar\theta_t)$")
    ax[2][0].set_ylabel("components k")
    fig.suptitle(r"Saddle-to-saddle training: loss, $\lambda$ at the snapped "
                 r"point, and learned components", x=0.01, ha="left",
                 fontsize=11)
    # 凡例は図の下に 1 本だけ (データに重ねない)
    hs, ls = [], []
    for c in range(n):
        for h, l in zip(*ax[1][c].get_legend_handles_labels()):
            if l not in ls:
                hs.append(h)
                ls.append(l)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(hs, ls, loc="lower center", ncol=min(len(ls), 5), fontsize=8,
               frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(res: Dict, path: str) -> None:
    cols = ["step", "loss", "k", "ambiguous", "lower", "upper", "upper_codims",
            "exact_lambda", "exact_m", "exact_status", "exact_time",
            "lam_hat_self", "lam_hat_self_std", "lam_hat_true",
            "lam_hat_true_std", "strengths"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in res["records"]:
            ex = r.get("exact", {})
            w.writerow([r["step"], r["loss"], r["k"], int(r["ambiguous"]),
                        r.get("lower", ""), r.get("upper", ""),
                        " ".join(map(str, r.get("upper_codims", []))),
                        ex.get("lam", ""), ex.get("m", ""),
                        ex.get("status", ""), ex.get("time", ""),
                        r.get("self", {}).get("lam", ""),
                        r.get("self", {}).get("std", ""),
                        r.get("true", {}).get("lam", ""),
                        r.get("true", {}).get("std", ""),
                        " ".join(f"{v:.4g}" for v in r["strengths"])])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="学習軌道に沿った λ (ReLU/tanh/CNN/attention/linear)")
    ap.add_argument("--arch", default="relu",
                    help="relu / tanh / cnn / attn / linear / all")
    ap.add_argument("--compare", action="store_true",
                    help="アーキテクチャを並べた比較図も描く")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--init", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bound-every", type=int, default=6)
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--exact-timeout", type=float, default=30.0)
    ap.add_argument("--taylor-order", type=int, default=3)
    ap.add_argument("--sgld", action="store_true")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--sgld-steps", type=int, default=2000)
    ap.add_argument("--out", default="result/nn_trajectory")
    ap.add_argument("--replot", action="store_true",
                    help="学習せず --out の JSON から図だけ描き直す")
    a = ap.parse_args()

    if a.replot:
        results = {}
        for arch in ARCHS:
            fn = os.path.join(a.out, arch + ".json")
            if os.path.exists(fn):
                with open(fn) as fh:
                    results[arch] = json.load(fh)
                plot(results[arch], os.path.join(a.out, arch + ".png"))
        if len(results) > 1:
            plot_compare(results, os.path.join(a.out, "compare.png"))
        print(f"replotted {sorted(results)}")
        return 0

    archs = list(ARCHS) if a.arch == "all" else a.arch.split(",")
    os.makedirs(a.out, exist_ok=True)
    results = {}
    for arch in archs:
        cfg = config_for(arch, n_steps=a.steps, lr=a.lr, width=a.width,
                         init_scale=a.init, seed=a.seed)
        print(f"== {arch}: {asdict(cfg)}", flush=True)
        res = analyze(cfg, bound_every=a.bound_every, exact=not a.no_exact,
                      exact_timeout=a.exact_timeout,
                      taylor_order=a.taylor_order, sgld=a.sgld,
                      gamma=a.gamma, eps=a.eps, sgld_steps=a.sgld_steps)
        results[arch] = res
        base = os.path.join(a.out, arch)
        with open(base + ".json", "w") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
        write_csv(res, base + ".csv")
        plot(res, base + ".png")
        print(f"  wrote {base}.json / .csv / .png", flush=True)
    if a.compare and len(results) > 1:
        plot_compare(results, os.path.join(a.out, "compare.png"))
        print(f"  wrote {os.path.join(a.out, 'compare.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
