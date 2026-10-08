r"""
train_rlct.py
=============

分類データセットで MLP / CNN / ViT (Transformer) を**学習させてから**、
その学習済みパラメータ theta* における局所 RLCT を計算する。

xiangze/vision_transformer_demo の vision_demo.py を参考に組んだつもりだが、
このセッションからは当該 URL を読み取れなかった (権限要求が未応答)。
典型的な ViT の構成 (パッチ埋め込み + cls トークン + 位置埋め込み +
Transformer エンコーダ層 + 分類ヘッド) を仮定している。実際の構成が違う
場合はファイルを貼っていただければ合わせる。

------------------------------------------------------------------
分類の fiber ideal (ここが回帰と違う)
------------------------------------------------------------------
分類モデルは p(y|x,theta) = softmax(f(x;theta))。p が theta* と一致するのは
**ロジットの差が一致するとき**で、ロジット全体の平行移動 f -> f + c(x) は p を
変えない。したがって基準クラス C を 1 つ選び

    g_{i,c} = (f_c - f_C)(theta) - (f_c - f_C)(theta*),    c != C

を生成元に取る (torch_rlct の loss="classification")。平行移動の方向は
対称性としてゲージ固定する。二乗誤差の fiber ideal
(g = f_c(theta) - f_c(theta*)) を分類に使うと、識別できない方向まで
拘束してしまい lambda を過大評価する。

------------------------------------------------------------------
学習済みモデルの lambda について正直に言えること
------------------------------------------------------------------
本物の ViT は 10^5〜10^6 パラメータで、厳密な記号計算は不可能。できるのは
**一部のパラメータだけを動かし、残りを学習後の値で凍結する**こと。
部分空間に制限すると積分領域が狭くなるので

    lambda(制限) <= lambda(全体)

すなわち得られる値は**全体の lambda の下界**になる (例: K = x^2 + y^2 なら
lambda = 1 だが y = 0 に制限すると 1/2)。報告ではこの向きを必ず明記する。

------------------------------------------------------------------
データセット
------------------------------------------------------------------
このコンテナは外部へのダウンロードが遮断されているので、既定は
**合成データセット** (`synthetic_classification`)。torchvision が使えて
データが手元にある環境なら `--dataset mnist/cifar10` でそちらを使える。

------------------------------------------------------------------
使い方
------------------------------------------------------------------
    python train_rlct.py --model mlp --epochs 30 --rlct
    python train_rlct.py --model cnn --epochs 20 --rlct
    python train_rlct.py --model vit --epochs 20 --rlct --vit-layer 0
    python train_rlct.py --model vit --dataset mnist --data-root ./data
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "synthetic_classification", "synthetic_regression", "load_dataset",
    "MLPClassifier", "SmallCNN", "DeepCNN", "TinyViT", "ViTEncoderLayer",
    "train", "rlct_of_trained",
]


# ----------------------------------------------------------------------
# データセット
# ----------------------------------------------------------------------
@dataclass
class Dataset:
    X: object                     # (N, C, H, W) の tensor
    y: object                     # (N,) の long tensor
    n_classes: int
    shape: Tuple[int, int, int]   # (C, H, W)
    name: str = "synthetic"
    #: 'classification' なら y は (N,) の long、'regression' なら (N, n_out) の float
    task: str = "classification"


def synthetic_classification(n: int = 256, shape=(1, 8, 8), n_classes: int = 3,
                             seed: int = 0, noise: float = 0.15) -> Dataset:
    """ダウンロード不要の合成画像分類データ。

    クラスごとに固定のランダムな「テンプレート画像」を用意し、ガウス雑音を
    足したものをサンプルにする。線形分離できるほど簡単ではないが、小さな
    MLP / CNN / ViT でも 1 に近い精度まで学習できる程度の難しさにしてある。
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    C, Hh, Ww = shape
    templates = torch.randn(n_classes, C, Hh, Ww, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    X = templates[y] + noise * torch.randn(n, C, Hh, Ww, generator=g)
    return Dataset(X=X, y=y, n_classes=n_classes, shape=shape,
                   name=f"synthetic{shape}x{n_classes}")


def load_dataset(name: str = "synthetic", *, root: str = "./data",
                 n: int = 256, shape=(1, 8, 8), n_classes: int = 3,
                 seed: int = 0, download: bool = False) -> Dataset:
    """'synthetic' / 'mnist' / 'cifar10'。後者は torchvision が要る。"""
    if name == "synthetic":
        return synthetic_classification(n=n, shape=shape,
                                        n_classes=n_classes, seed=seed)
    import torch
    try:
        import torchvision
        import torchvision.transforms as T
    except Exception as e:                                # noqa: BLE001
        raise RuntimeError(
            f"torchvision が必要です ({e})。--dataset synthetic を使ってください。")
    if name == "mnist":
        tf = T.Compose([T.Resize((shape[1], shape[2])), T.ToTensor()])
        ds = torchvision.datasets.MNIST(root, train=True, download=download,
                                        transform=tf)
        C = 1
    elif name == "cifar10":
        tf = T.Compose([T.Resize((shape[1], shape[2])), T.ToTensor()])
        ds = torchvision.datasets.CIFAR10(root, train=True, download=download,
                                          transform=tf)
        C = 3
    else:
        raise ValueError(f"未知のデータセット {name}")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]
    X = torch.stack([ds[i][0] for i in idx])
    y = torch.tensor([ds[i][1] for i in idx], dtype=torch.long)
    keep = y < n_classes
    X, y = X[keep], y[keep]
    return Dataset(X=X, y=y, n_classes=int(y.max().item()) + 1,
                   shape=(C, shape[1], shape[2]), name=name)


# ----------------------------------------------------------------------
# モデル
# ----------------------------------------------------------------------
def MLPClassifier(shape, n_classes: int, hidden: Sequence[int] = (8,)):
    """平坦化 -> Linear -> ReLU -> ... -> Linear。nn.Sequential なので
    torch_local_rlct の記号フォワードがそのまま通る。"""
    import torch.nn as nn

    C, Hh, Ww = shape
    dims = [C * Hh * Ww] + list(hidden)
    mods: List = [nn.Flatten()]
    for a, b in zip(dims, dims[1:]):
        mods += [nn.Linear(a, b), nn.ReLU()]
    mods.append(nn.Linear(dims[-1], n_classes))
    return nn.Sequential(*mods)


def SmallCNN(shape, n_classes: int, channels: int = 4, hidden: int = 8,
             kernel: int = 3):
    """Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> ReLU -> Linear。

    torch_rlct は Conv2d / MaxPool2d / AvgPool2d を記号で通せる
    (MaxPool は theta* での argmax を固定する)。
    """
    import torch.nn as nn

    C, Hh, Ww = shape
    conv = nn.Conv2d(C, channels, kernel, padding=kernel // 2)
    oh, ow = Hh // 2, Ww // 2
    return nn.Sequential(
        conv, nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(channels * oh * ow, hidden), nn.ReLU(),
        nn.Linear(hidden, n_classes),
    )


def _act_module(act: str):
    """活性化の名前から nn モジュールを作る。"""
    import torch.nn as nn

    a = act.lower()
    if a == "relu":
        return nn.ReLU()
    if a == "tanh":
        return nn.Tanh()
    if a == "sigmoid":
        return nn.Sigmoid()
    if a in ("leaky", "leakyrelu"):
        return nn.LeakyReLU(0.1)
    raise ValueError(f"未知の活性化 {act}")


def DeepCNN(shape, n_out: int, *, depth: int = 2, channels: int = 2,
            hidden: int = 4, kernel: int = 3, act: str = "relu",
            pool: bool = True):
    r"""**層数を指定できる** CNN。RLCT を層数別に比べるために使う。

        [Conv2d -> act -> (MaxPool2d(2))] x depth -> Flatten
          -> Linear -> act -> Linear(n_out)

    - `act='relu'` なら区分線形なのでセルを固定すれば厳密に多項式
    - `act='tanh'` なら `torch_rlct` が θ\* の前活性化まわりで
      `taylor_order` 次に展開する (**次数を変えるとモデルが変わる**)
    - プーリングは一辺が偶数で 2 以上のあいだだけ入れる。奇数になった時点で
      やめるので、小さい画像でも depth を上げられる
    - `n_out` は分類ならクラス数、回帰なら出力次元

    チャンネル数は層ごとに増やさない (RLCT の変数が増えすぎるため)。
    層数だけを動かして比べるのが目的なので、むしろ揃えておくほうが素直。
    """
    import torch.nn as nn

    C, Hh, Ww = shape
    mods: List = []
    cin, h, w = C, Hh, Ww
    # 奇数カーネルなら padding で一辺を保てるが、偶数カーネルでは保てない。
    # 左右非対称な padding を入れるより、素直に縮ませて形を追うほうが安全。
    pad = kernel // 2 if kernel % 2 == 1 else 0
    for _ in range(depth):
        if h + 2 * pad < kernel or w + 2 * pad < kernel:
            raise ValueError(
                f"depth={depth} は画像 {Hh}x{Ww} に対して深すぎます "
                f"(kernel={kernel} の手前で {h}x{w} まで縮みました)")
        mods += [nn.Conv2d(cin, channels, kernel, padding=pad),
                 _act_module(act)]
        cin = channels
        h, w = h + 2 * pad - (kernel - 1), w + 2 * pad - (kernel - 1)
        if pool and h >= 2 and w >= 2 and h % 2 == 0 and w % 2 == 0:
            mods.append(nn.MaxPool2d(2))
            h, w = h // 2, w // 2
    mods += [nn.Flatten(), nn.Linear(cin * h * w, hidden), _act_module(act),
             nn.Linear(hidden, n_out)]
    return nn.Sequential(*mods)


def synthetic_regression(n: int = 128, shape=(1, 8, 8), n_out: int = 1,
                         seed: int = 0, noise: float = 0.05) -> Dataset:
    """回帰用の合成データ。`y` が (N, n_out) の float になる。

    教師は固定のランダムな線形写像 + 弱い非線形項で作る。RLCT を見るのは
    θ\* まわりなので教師そのものの難しさは本質ではないが、ネットワークが
    学習しきれる程度にしてある。
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    C, Hh, Ww = shape
    X = torch.randn(n, C, Hh, Ww, generator=g)
    W = torch.randn(C * Hh * Ww, n_out, generator=g) / math.sqrt(C * Hh * Ww)
    flat = X.reshape(n, -1)
    y = flat @ W + 0.3 * torch.tanh(flat @ W)
    y = y + noise * torch.randn(n, n_out, generator=g)
    return Dataset(X=X, y=y, n_classes=n_out, shape=shape,
                   name=f"synthreg{shape}x{n_out}", task="regression")


def _make_encoder_layer(d_model: int, n_heads: int, d_ff: int,
                        layernorm: bool):
    """nn.TransformerEncoderLayer 相当の層。layernorm=False も選べる。

    LayerNorm を外せる形にしてあるのは、LayerNorm 入りの層だと RLCT が
    「LayerNorm を外した別のモデル」の値になってしまうため。
    layernorm=False で学習すれば、計算した lambda がそのモデル自身の値になる。
    """
    import torch
    import torch.nn as nn

    class ViTEncoderLayerImpl(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(
                d_model, n_heads, batch_first=True, bias=True)
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.act = nn.ReLU()
            self.use_ln = layernorm
            if layernorm:
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            a, _ = self.self_attn(x, x, x, need_weights=False)
            x = x + a
            if self.use_ln:
                x = self.norm1(x)
            h = self.linear2(self.act(self.linear1(x)))
            x = x + h
            if self.use_ln:
                x = self.norm2(x)
            return x

    return ViTEncoderLayerImpl()


#: 外から参照できるように別名を用意する
def ViTEncoderLayer(d_model: int, n_heads: int = 1, d_ff: int = 4,
                    layernorm: bool = False):
    return _make_encoder_layer(d_model, n_heads, d_ff, layernorm)


def TinyViT(shape, n_classes: int, *, patch: int = 4, d_model: int = 8,
            n_heads: int = 1, d_ff: int = 8, depth: int = 1,
            layernorm: bool = False, cls_token: bool = True):
    """小さな Vision Transformer。

    パッチ埋め込み (Conv2d) -> [cls トークン] + 位置埋め込み ->
    エンコーダ層 x depth -> cls トークン (または平均) -> 分類ヘッド。

    既定は **LayerNorm なし**。LayerNorm は 1/sqrt が入って多項式にならず、
    RLCT を計算すると「LayerNorm を外した別モデル」の値になるため、
    学習の段階から外しておくほうが筋が通る。--layernorm で入れられる。
    """
    import torch
    import torch.nn as nn

    C, Hh, Ww = shape
    if Hh % patch or Ww % patch:
        raise ValueError(f"画像サイズ {Hh}x{Ww} はパッチ {patch} で割れません")
    n_patch = (Hh // patch) * (Ww // patch)

    class TinyViTImpl(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(C, d_model, patch, stride=patch)
            self.cls = (nn.Parameter(torch.zeros(1, 1, d_model))
                        if cls_token else None)
            n_tok = n_patch + (1 if cls_token else 0)
            self.pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
            nn.init.normal_(self.pos, std=0.02)
            if cls_token:
                nn.init.normal_(self.cls, std=0.02)
            self.layers = nn.ModuleList(
                [_make_encoder_layer(d_model, n_heads, d_ff, layernorm)
                 for _ in range(depth)])
            self.head = nn.Linear(d_model, n_classes)
            self.meta = dict(patch=patch, d_model=d_model, n_heads=n_heads,
                             d_ff=d_ff, depth=depth, n_tokens=n_tok,
                             layernorm=layernorm, cls_token=cls_token)

        def tokens(self, x):
            """エンコーダに入る直前のトークン列 (B, T, d_model)。"""
            z = self.patch_embed(x)                   # (B, d, h, w)
            z = z.flatten(2).transpose(1, 2)          # (B, n_patch, d)
            if self.cls is not None:
                z = torch.cat([self.cls.expand(z.shape[0], -1, -1), z], dim=1)
            return z + self.pos

        def forward_upto(self, x, layer: int):
            """エンコーダ層 `layer` の**入力**になるトークン列を返す。"""
            z = self.tokens(x)
            for l in self.layers[:layer]:
                z = l(z)
            return z

        def forward(self, x):
            z = self.tokens(x)
            for l in self.layers:
                z = l(z)
            pooled = z[:, 0] if self.cls is not None else z.mean(dim=1)
            return self.head(pooled)

    return TinyViTImpl()


# ----------------------------------------------------------------------
# 学習
# ----------------------------------------------------------------------
@dataclass
class TrainReport:
    model: object
    dataset: Dataset
    history: List[Dict[str, float]] = field(default_factory=list)
    final_loss: float = 0.0
    final_acc: float = 0.0

    def print_report(self) -> None:
        print(f"  データ {self.dataset.name}: "
              f"{tuple(self.dataset.X.shape)} / {self.dataset.n_classes} クラス")
        n = sum(p.numel() for p in self.model.parameters())
        print(f"  モデル {type(self.model).__name__}: パラメータ {n}")
        print(f"  学習後: loss={self.final_loss:.5f} acc={self.final_acc:.3f}")


def train(model, ds: Dataset, *, epochs: int = 50, lr: float = 0.05,
          batch: int = 0, weight_decay: float = 0.0, seed: int = 0,
          verbose: bool = False) -> TrainReport:
    """分類の交差エントロピー、または回帰の二乗誤差で学習する。

    どちらかは `ds.task` で決まる (`'classification'` / `'regression'`)。

    RLCT を見るのは「損失がほぼ 0 の局所解」なので、既定では**全データ
    バッチで十分な回数**回して過学習させる (realizable な点に落とす)。
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    reg = getattr(ds, "task", "classification") == "regression"
    lossf = nn.MSELoss() if reg else nn.CrossEntropyLoss()
    rep = TrainReport(model=model, dataset=ds)
    N = ds.X.shape[0]
    for ep in range(epochs):
        model.train()
        if batch and batch < N:
            perm = torch.randperm(N)
            tot = 0.0
            for s in range(0, N, batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                out = model(ds.X[idx])
                l = lossf(out, ds.y[idx])
                l.backward()
                opt.step()
                tot += float(l.detach()) * len(idx)
            cur = tot / N
        else:
            opt.zero_grad()
            out = model(ds.X)
            l = lossf(out, ds.y)
            l.backward()
            opt.step()
            cur = float(l.detach())
        with torch.no_grad():
            out = model(ds.X)
            if reg:
                # 回帰では「精度」の代わりに決定係数 R^2 を残す
                ss_res = float(((out - ds.y) ** 2).sum())
                ss_tot = float(((ds.y - ds.y.mean(dim=0)) ** 2).sum()) or 1.0
                acc = 1.0 - ss_res / ss_tot
            else:
                acc = float((out.argmax(dim=1) == ds.y).float().mean())
        rep.history.append(dict(epoch=ep, loss=cur, acc=acc))
        if verbose and (ep % max(epochs // 10, 1) == 0 or ep == epochs - 1):
            print(f"    ep{ep:<4} loss={cur:.5f} acc={acc:.3f}", flush=True)
    rep.final_loss, rep.final_acc = rep.history[-1]["loss"], rep.history[-1]["acc"]
    return rep


# ----------------------------------------------------------------------
# 学習済みモデルの RLCT
# ----------------------------------------------------------------------
def _stratified(ds: Dataset, n: int) -> List[int]:
    """クラスを均等にまたぐように n 点選ぶ。回帰では等間隔に取る。"""
    import torch

    if getattr(ds, "task", "classification") == "regression":
        N = int(ds.X.shape[0])
        if n >= N:
            return list(range(N))
        step = N / n
        return [int(i * step) for i in range(n)]
    by: Dict[int, List[int]] = {}
    for i, c in enumerate(ds.y.tolist()):
        by.setdefault(int(c), []).append(i)
    out: List[int] = []
    r = 0
    while len(out) < n and any(len(v) > r for v in by.values()):
        for c in sorted(by):
            if r < len(by[c]) and len(out) < n:
                out.append(by[c][r])
        r += 1
    return out[:n]


def _param_subset(model, kind: str, max_vars: int) -> List[str]:
    """max_vars に収まるように、**後ろの層から**パラメータを選ぶ。

    出力に近い層のほうが退化の構造 (ユニットの重複・死んだユニット) が
    直接 lambda に効くので、ヘッド側から取る。
    """
    names = [(n, p.numel()) for n, p in model.named_parameters()]
    out, tot = [], 0
    for n, k in reversed(names):
        if tot + k > max_vars:
            continue
        out.append(n)
        tot += k
    return list(reversed(out))


def rlct_of_trained(model, ds: Dataset, *, kind: str = "mlp",
                    n_data: int = 8, max_vars: int = 20,
                    params: Optional[Sequence[str]] = None,
                    max_denominator: int = 64,
                    vit_layer: int = 0, taylor_order: int = 0,
                    loss: Optional[str] = None,
                    verbose: bool = False, **resolve_kwargs):
    """学習済みモデルの theta* における局所 RLCT。

    kind='mlp' / 'cnn' : torch_local_rlct(loss='classification') をそのまま
        使う。`params` で動かすパラメータを絞る (指定しなければ出力側から
        max_vars 個ぶん)。**凍結した方向を除いた部分空間の値なので、
        全体の lambda の下界**になる。
    kind='vit' : 学習済み ViT のエンコーダ層 `vit_layer` を取り出し、その層の
        入力トークン列を学習後の活性で固定して、**その層だけの局所 RLCT** を
        計算する。ViT 全体の lambda ではない (別の局所モデルの値)。

    loss : None なら `ds.task` から決める ('classification' / 'regression')。
        分類モデルに回帰のイデアルを使うと lambda を過大評価するので、
        既定で揃えてある。
    taylor_order : tanh など解析的な活性化を θ\* まわりで何次まで展開するか。
        **次数を変えるとモデルが変わる**ので lambda も変わりうる。ReLU は
        区分線形なので無関係。
    """
    import torch

    from torch_rlct import torch_local_rlct, torch_transformer_rlct

    # データ点はクラスをまたいで選ぶ。先頭から取ると同じクラスばかりになり、
    # 隠れ層が全データで不活性 (= 残差が恒等的に 0, lambda = 0) という
    # 情報のない部分問題になりやすい。
    X = ds.X[_stratified(ds, n_data)]
    Xflat = X.reshape(X.shape[0], -1)
    if kind == "vit":
        with torch.no_grad():
            Z = model.forward_upto(X, vit_layer)[0]     # 1 サンプルのトークン列
        layer = model.layers[vit_layer]
        ln = "ignore" if getattr(layer, "use_ln", False) else "error"
        if verbose:
            print(f"  ViT 層 {vit_layer}: トークン列 {tuple(Z.shape)}, "
                  f"LayerNorm={'あり' if ln == 'ignore' else 'なし'}")
        return torch_transformer_rlct(layer, Z.tolist(), layernorm=ln,
                                      taylor_order=taylor_order,
                                      max_denominator=max_denominator,
                                      verbose=verbose, **resolve_kwargs)
    names = list(params) if params else _param_subset(model, kind, max_vars)
    if loss is None:
        loss = ("regression"
                if getattr(ds, "task", "classification") == "regression"
                else "classification")
    if verbose:
        print(f"  動かすパラメータ: {names}  (loss={loss}, "
              f"taylor_order={taylor_order})")
    return torch_local_rlct(model, Xflat, loss=loss, params=names,
                            input_shape=ds.shape, max_vars=max_vars + 4,
                            max_denominator=max_denominator,
                            taylor_order=taylor_order,
                            verbose=verbose, **resolve_kwargs)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp",
                    choices=("mlp", "cnn", "deepcnn", "vit"))
    ap.add_argument("--task", default="classification",
                    choices=("classification", "regression"))
    ap.add_argument("--act", default="relu",
                    choices=("relu", "tanh", "sigmoid", "leaky"))
    ap.add_argument("--out-dim", type=int, default=1,
                    help="回帰の出力次元")
    ap.add_argument("--dataset", default="synthetic",
                    choices=("synthetic", "mnist", "cifar10"))
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--n", type=int, default=128, help="データ点の数")
    ap.add_argument("--size", type=int, default=8, help="画像の一辺")
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # モデルの形
    ap.add_argument("--hidden", type=int, default=6)
    ap.add_argument("--conv-channels", type=int, default=2)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=4)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--d-ff", type=int, default=4)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--layernorm", action="store_true",
                    help="ViT に LayerNorm を入れる (RLCT は LN を外した"
                         "別モデルの値になる)")
    # RLCT
    ap.add_argument("--rlct", action="store_true", help="学習後に RLCT を計算")
    ap.add_argument("--rlct-data", type=int, default=6)
    ap.add_argument("--max-vars", type=int, default=16)
    ap.add_argument("--params", default="", help="動かすパラメータ名 (カンマ区切り)")
    ap.add_argument("--max-denominator", type=int, default=64)
    ap.add_argument("--vit-layer", type=int, default=0)
    ap.add_argument("--taylor-order", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--cache", default="off")
    args = ap.parse_args()

    import rlct_cache
    if str(args.cache).lower() in ("off", "none", ""):
        rlct_cache.disable()
    else:
        rlct_cache.enable(args.cache)

    shape = (args.channels, args.size, args.size)
    if args.task == "regression":
        if args.dataset != "synthetic":
            print("  [注意] 回帰は合成データのみです。--dataset synthetic に"
                  "読み替えます。")
        ds = synthetic_regression(n=args.n, shape=shape, n_out=args.out_dim,
                                  seed=args.seed)
    else:
        ds = load_dataset(args.dataset, root=args.data_root, n=args.n,
                          shape=shape, n_classes=args.classes, seed=args.seed,
                          download=args.download)
    n_out = ds.n_classes
    if args.model == "mlp":
        model = MLPClassifier(ds.shape, n_out, hidden=(args.hidden,))
    elif args.model == "cnn":
        model = SmallCNN(ds.shape, n_out, channels=args.conv_channels,
                         hidden=args.hidden)
    elif args.model == "deepcnn":
        model = DeepCNN(ds.shape, n_out, depth=args.depth,
                        channels=args.conv_channels, hidden=args.hidden,
                        act=args.act)
    else:
        model = TinyViT(ds.shape, n_out, patch=args.patch,
                        d_model=args.d_model, n_heads=args.heads,
                        d_ff=args.d_ff, depth=args.depth,
                        layernorm=args.layernorm)
    print(f"===== 学習 ({args.model}) =====")
    rep = train(model, ds, epochs=args.epochs, lr=args.lr, batch=args.batch,
                seed=args.seed, verbose=True)
    rep.print_report()

    if not args.rlct:
        return 0
    print(f"\n===== 学習済み theta* の局所 RLCT ({args.model}) =====")
    names = [v for v in args.params.split(",") if v] or None
    import time
    t = time.time()
    try:
        out = rlct_of_trained(model, ds,
                              kind="cnn" if args.model == "deepcnn"
                              else args.model,
                              n_data=args.rlct_data, max_vars=args.max_vars,
                              params=names,
                              max_denominator=args.max_denominator,
                              vit_layer=args.vit_layer,
                              taylor_order=args.taylor_order,
                              max_depth=args.max_depth, verbose=True)
    except Exception as e:                                # noqa: BLE001
        print(f"  計算できませんでした: {type(e).__name__}: {e}")
        return 1
    out.print_report()
    print(f"  ({time.time()-t:.1f}s)")
    if args.model == "vit":
        print("  [注意] これはエンコーダ層 1 枚の局所 RLCT で、"
              "ViT 全体の lambda ではありません。")
    else:
        print("  [注意] 凍結した方向を除いた部分空間の値なので、"
              "全体の lambda の**下界**です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
