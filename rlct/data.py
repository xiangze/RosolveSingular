r"""
vision_data.py — 画像データの読み込み (学習と評価で共有する葉)

`train_resnet` に置いていたが、`resnet_rlct` の CLI も評価バッチを必要とし、
`train_resnet -> resnet_rlct` と合わせて循環になっていた
(REORG_PLAN.ja.md §2)。データ読み込みは**両者より下の層**なので分離した。

    from vision_data import load_dataset, load_eval_batch
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple


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
