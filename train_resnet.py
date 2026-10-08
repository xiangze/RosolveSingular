r"""
train_resnet.py
===============

**ResNet18 / ResNet50 を実データで学習し、学習済み θ\* の RLCT を計算する。**
GPU と実データがある環境で動かす想定 (このコンテナでは外部ダウンロードが
遮断されているので `--data random` でしか回せない)。

------------------------------------------------------------------
手順
------------------------------------------------------------------
    # 1. 学習 (GPU)
    python train_resnet.py --arch resnet18 --data cifar10 --download \
        --epochs 60 --batch 128 --lr 0.1 --out run/r18

    # 2. 動かせるパラメータの一覧 (融合後の名前)
    python resnet_rlct.py --arch resnet18 --ckpt run/r18/best.pt --list-params

    # 3. RLCT (部分空間に制限した下界)
    python train_resnet.py --arch resnet18 --ckpt run/r18/best.pt \
        --data cifar10 --rlct --params fc.bias --tail-from layer4 --n-data 16

------------------------------------------------------------------
θ\* が「適切な解」であるために
------------------------------------------------------------------
RLCT は**損失が (ほぼ) 0 の局所解**での量である。CIFAR-10 の ResNet18 は
weight decay とデータ拡張を入れると訓練損失が 0 にならない。RLCT を見る
θ\* としては、

- `--wd 0 --no-augment` で**訓練データに過学習させる** (realizable に近づく)
- もしくは `--subset 2000` のように小さくして完全に覚えさせる

のどちらかを選ぶ。既定は**過学習寄り** (wd=0, 拡張なし) にしてある。
素性の良い汎化モデルが欲しい場合は `--wd 5e-4 --augment` を付けるが、
その θ\* は損失 0 ではないので「realizable な設定での λ」ではなくなる。

> なお fiber ideal は**常に θ\* のモデルを真**として作る (`resnet_rlct`)。
> したがって λ の計算自体は θ\* が訓練損失 0 かどうかに依らない。
> 上の話は「その θ\* が興味のある点かどうか」という問題である。

------------------------------------------------------------------
BatchNorm について
------------------------------------------------------------------
BN は評価モードで畳み込みに融合される (`resnet_rlct.fuse_conv_bn`)。
融合後のネットワークは**元と数値的に同一**かつ多項式。ただし λ は
「BN の統計量を固定したモデル」の値で、統計量をパラメータとして動かす
モデルの λ ではない。

------------------------------------------------------------------
どのパラメータを動かせるか — 費用の目安
------------------------------------------------------------------
記号計算の費用は **変数の数 × その影響が広がる範囲**で決まる。
`resnet_rlct` は「ずれだけ」を疎に運ぶので、出力に近いほど安い。

| `--params` / `--tail-from` | 目安 |
|---|---|
| `fc.bias` / `layer4` | 変数 = クラス数。秒 |
| `fc.weight` / `layer4` | 変数 = 512×クラス数。絞らないと重い |
| `layer4.1.conv2.weight` / `layer4.1` | 512×512×9 なので**必ず絞る** |

`fc.weight` 全部のように大きすぎる場合は、`--params-slice` で先頭 n 成分だけに
絞れる。絞ると部分空間がさらに小さくなるので λ はより小さい下界になる。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["build_model", "load_dataset", "load_eval_batch", "train",
           "TrainState"]


# ----------------------------------------------------------------------
# モデル
# ----------------------------------------------------------------------
def build_model(arch: str = "resnet18", num_classes: int = 10,
                *, small_input: bool = True):
    r"""torchvision の ResNet。`small_input=True` で 32x32 向けに頭を直す。

    ImageNet 用の ResNet は 7x7 stride 2 の conv1 と maxpool で入力を
    1/4 にするので、32x32 では情報が落ちすぎる。CIFAR では 3x3 stride 1 の
    conv1 にして maxpool を外すのが標準。
    """
    import torch.nn as nn
    import torchvision.models as tvm

    model = getattr(tvm, arch)(weights=None, num_classes=num_classes)
    if small_input:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                bias=False)
        model.maxpool = nn.Identity()
    return model


# ----------------------------------------------------------------------
# データ
# ----------------------------------------------------------------------
_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)


def load_dataset(name: str = "cifar10", *, root: str = "./data",
                 download: bool = False, augment: bool = False,
                 size: int = 32, classes: int = 10, subset: int = 0):
    """'cifar10' / 'cifar100' / 'folder' / 'random'。"""
    import torch
    from torch.utils.data import Subset, TensorDataset

    if name == "random":
        g = torch.Generator().manual_seed(0)
        n = subset or 512
        X = torch.randn(n, 3, size, size, generator=g)
        y = torch.randint(0, classes, (n,), generator=g)
        return TensorDataset(X, y), TensorDataset(X, y), classes

    import torchvision
    import torchvision.transforms as T

    norm = [T.ToTensor(), T.Normalize(_MEAN, _STD)]
    train_tf = T.Compose(([T.RandomCrop(size, padding=4),
                           T.RandomHorizontalFlip()] if augment else [])
                         + ([T.Resize((size, size))] if size != 32 else [])
                         + norm)
    test_tf = T.Compose(([T.Resize((size, size))] if size != 32 else [])
                        + norm)
    if name == "cifar10":
        tr = torchvision.datasets.CIFAR10(root, train=True,
                                          download=download,
                                          transform=train_tf)
        te = torchvision.datasets.CIFAR10(root, train=False,
                                          download=download,
                                          transform=test_tf)
        nc = 10
    elif name == "cifar100":
        tr = torchvision.datasets.CIFAR100(root, train=True,
                                           download=download,
                                           transform=train_tf)
        te = torchvision.datasets.CIFAR100(root, train=False,
                                           download=download,
                                           transform=test_tf)
        nc = 100
    else:
        tr = torchvision.datasets.ImageFolder(os.path.join(root, "train"),
                                              transform=train_tf)
        te = torchvision.datasets.ImageFolder(os.path.join(root, "val"),
                                              transform=test_tf)
        nc = len(tr.classes)
    if subset:
        tr = Subset(tr, list(range(min(subset, len(tr)))))
    return tr, te, nc


def load_eval_batch(name: str = "cifar10", *, root: str = "./data",
                    n: int = 8, size: int = 32, classes: int = 10,
                    download: bool = False, stratified: bool = True):
    r"""RLCT を見るためのデータ点を取る。**クラスをまたいで層化抽出**する。

    先頭から取ると同じクラスばかりになり、隠れユニットが全データで不活性
    (残差が恒等的に 0) という情報のない部分問題になりやすい
    (小さい CNN で実測: 先頭 4 点だと λ=0、層化して 6 点で λ=2)。
    """
    import torch

    _, te, nc = load_dataset(name, root=root, download=download,
                             size=size, classes=classes)
    if not stratified:
        idx = list(range(n))
    else:
        by: Dict[int, List[int]] = {}
        for i in range(min(len(te), 20 * max(n, 1))):
            c = int(te[i][1])
            by.setdefault(c, []).append(i)
        idx, r = [], 0
        while len(idx) < n and any(len(v) > r for v in by.values()):
            for c in sorted(by):
                if r < len(by[c]) and len(idx) < n:
                    idx.append(by[c][r])
            r += 1
        idx = idx[:n]
    X = torch.stack([te[i][0] for i in idx])
    y = torch.tensor([int(te[i][1]) for i in idx])
    return X, y


# ----------------------------------------------------------------------
# 学習
# ----------------------------------------------------------------------
@dataclass
class TrainState:
    epoch: int = 0
    train_loss: float = float("nan")
    train_acc: float = float("nan")
    test_acc: float = float("nan")
    best_acc: float = 0.0
    history: List[Dict] = field(default_factory=list)


def train(model, train_set, test_set, *, epochs: int = 60, lr: float = 0.1,
          batch: int = 128, wd: float = 0.0, momentum: float = 0.9,
          device: str = "", workers: int = 4, out_dir: str = "",
          label_smoothing: float = 0.0, cosine: bool = True,
          verbose: bool = True) -> TrainState:
    r"""SGD + モーメンタム + cosine 減衰。既定は **wd=0** (過学習寄り)。

    `out_dir` を渡すと `best.pt` と `last.pt` と `log.json` を書く。
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    tl = DataLoader(train_set, batch_size=batch, shuffle=True,
                    num_workers=workers, drop_last=False)
    vl = DataLoader(test_set, batch_size=max(batch, 256), shuffle=False,
                    num_workers=workers)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=wd, nesterov=momentum > 0)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)
    lossf = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    st = TrainState()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for ep in range(epochs):
        model.train()
        tot = cor = 0
        s = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            L = lossf(out, yb)
            L.backward()
            opt.step()
            s += float(L.detach()) * yb.numel()
            cor += int((out.argmax(1) == yb).sum())
            tot += yb.numel()
        if sched:
            sched.step()
        st.epoch, st.train_loss, st.train_acc = ep, s / tot, cor / tot
        model.eval()
        tc = tt = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(dev), yb.to(dev)
                tc += int((model(xb).argmax(1) == yb).sum())
                tt += yb.numel()
        st.test_acc = tc / max(tt, 1)
        st.history.append(dict(epoch=ep, train_loss=st.train_loss,
                               train_acc=st.train_acc,
                               test_acc=st.test_acc))
        if out_dir:
            torch.save(dict(model=model.state_dict(), state=st.history[-1]),
                       os.path.join(out_dir, "last.pt"))
            if st.test_acc >= st.best_acc:
                st.best_acc = st.test_acc
                torch.save(dict(model=model.state_dict(),
                                state=st.history[-1]),
                           os.path.join(out_dir, "best.pt"))
            with open(os.path.join(out_dir, "log.json"), "w") as fh:
                json.dump(st.history, fh, indent=1)
        if verbose:
            print(f"  ep{ep:<4} train loss={st.train_loss:.4f} "
                  f"acc={st.train_acc:.4f}  test acc={st.test_acc:.4f}",
                  flush=True)
    return st


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18",
                    help="resnet18 / resnet34 / resnet50 / ...")
    ap.add_argument("--data", default="cifar10",
                    choices=("cifar10", "cifar100", "folder", "random"))
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--subset", type=int, default=0,
                    help="訓練データをこの数に絞る (完全に覚えさせたいとき)")
    ap.add_argument("--imagenet-stem", action="store_true",
                    help="conv1 を 7x7 stride2 のまま使う (224 入力向け)")
    # 学習
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--wd", type=float, default=0.0,
                    help="既定 0 (過学習寄り = realizable に近づける)")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default="run")
    ap.add_argument("--ckpt", default="", help="学習を飛ばして読み込む")
    # RLCT
    ap.add_argument("--rlct", action="store_true")
    ap.add_argument("--params", default="fc.bias")
    ap.add_argument("--params-slice", type=int, default=0,
                    help="各パラメータの先頭 n 成分だけを動かす (0 で全部)")
    ap.add_argument("--tail-from", default="layer4")
    ap.add_argument("--loss", default="classification",
                    choices=("classification", "regression"))
    ap.add_argument("--n-data", type=int, default=8)
    ap.add_argument("--max-vars", type=int, default=40)
    ap.add_argument("--max-denominator", type=int, default=10 ** 6)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--llc", action="store_true",
                    help="SGLD の λ̂ も測って厳密値と比べる")
    ap.add_argument("--llc-gammas", default="1e-3,1e-2,1e-1,1")
    ap.add_argument("--llc-eps", default="5e-4,1e-3")
    ap.add_argument("--llc-steps", type=int, default=8000)
    ap.add_argument("--rlct-out", default="")
    args = ap.parse_args()

    import torch

    model = build_model(args.arch, args.classes,
                        small_input=not args.imagenet_stem)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"===== {args.arch} ({n_par:,} パラメータ) =====")

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model", sd))
        print(f"  重みを {args.ckpt} から読み込みました")
    else:
        tr, te, nc = load_dataset(args.data, root=args.data_root,
                                  download=args.download,
                                  augment=args.augment, size=args.size,
                                  classes=args.classes, subset=args.subset)
        print(f"  データ {args.data}: 訓練 {len(tr)} / 評価 {len(te)}, "
              f"{nc} クラス, 拡張={'あり' if args.augment else 'なし'}, "
              f"wd={args.wd}")
        st = train(model, tr, te, epochs=args.epochs, lr=args.lr,
                   batch=args.batch, wd=args.wd, device=args.device,
                   workers=args.workers, out_dir=args.out,
                   label_smoothing=args.label_smoothing)
        print(f"  学習後: train acc={st.train_acc:.4f} "
              f"loss={st.train_loss:.4f} / test acc={st.test_acc:.4f} "
              f"(best {st.best_acc:.4f})")
        print(f"  -> {args.out}/best.pt, {args.out}/last.pt")

    if not args.rlct:
        return 0

    from resnet_rlct import fuse_conv_bn, resnet_local_rlct

    model = model.cpu().eval()
    X, _ = load_eval_batch(args.data, root=args.data_root, n=args.n_data,
                           size=args.size, classes=args.classes,
                           download=args.download)
    params = [v for v in args.params.split(",") if v]
    if args.params_slice:
        print(f"  [注意] 各パラメータの先頭 {args.params_slice} 成分だけを"
              "動かします。部分空間がさらに小さくなるので λ はより小さい下界です")

    print(f"\n===== 学習済み theta* における局所 RLCT =====")
    t = time.time()
    try:
        rep = resnet_local_rlct(model, X, params=params,
                                tail_from=args.tail_from, loss=args.loss,
                                max_denominator=args.max_denominator,
                                max_vars=args.max_vars,
                                params_slice=args.params_slice,
                                verbose=True, max_depth=args.max_depth)
    except Exception as e:                                # noqa: BLE001
        print(f"  計算できませんでした: {type(e).__name__}: {e}")
        return 1
    rep.print_report()
    print(f"  ({time.time()-t:.1f}s)")

    if args.llc:
        from llc_estimate import llc_sgld
        fused = fuse_conv_bn(model)
        print("\n===== SGLD による λ̂ (同じ部分空間に制限) =====")
        print("  realizable (教師は theta* の出力) ・同じデータ点・"
              "同じ座標だけを動かす")
        best = None
        for gamma in [float(v) for v in args.llc_gammas.split(",") if v]:
            for eps in [float(v) for v in args.llc_eps.split(",") if v]:
                r = llc_sgld(fused, X, params=params, loss=args.loss,
                             gamma=gamma, eps=eps, n_steps=args.llc_steps,
                             chains=4, seed=0)
                mark = ""
                if best is None or abs(r.lam_hat - float(rep.local.rlct)) < \
                        abs(best - float(rep.local.rlct)):
                    best = r.lam_hat
                    mark = "  <- 厳密値に最も近い"
                print(f"  γ={gamma:<7g} ε={eps:<8g} λ̂={r.lam_hat:8.3f}"
                      f"±{r.lam_std:.3f}  厳密 λ={rep.local.rlct} "
                      f"誤差 {(r.lam_hat - float(rep.local.rlct)) / float(rep.local.rlct):+7.1%}"
                      f"{' 発散' if r.diverged else ''}{mark}", flush=True)
        print("  [注意] 位数 m は λ̂ が原理的に返せない量です "
              f"(厳密計算では m={rep.local.multiplicity})")

    if args.rlct_out:
        with open(args.rlct_out, "w", encoding="utf-8") as fh:
            json.dump(dict(arch=args.arch, n_params=n_par, params=params,
                           tail_from=args.tail_from, loss=args.loss,
                           rlct=str(rep.local.rlct),
                           mult=rep.local.multiplicity,
                           n_vars=len(rep.variables), n_gens=rep.n_gens),
                      fh, ensure_ascii=False, indent=1)
        print(f"-> {args.rlct_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
