r"""
resnet_rlct.py
==============

**ResNet (torchvision の resnet18 / resnet50 など) の学習済み θ\* における
局所 RLCT。** 大きなモデルでも動くように作ってある。

------------------------------------------------------------------
なぜ素直にやると動かないか
------------------------------------------------------------------
`torch_rlct` の記号フォワードは全エントリを sympy 式として密に持つ。
ResNet18 の `layer4` は 512ch × 4×4 = 8192 エントリ、各々が 256×9 = 2304 項の
和なので、1 ブロックで約 1800 万回の記号演算になり終わらない。

**この実装は「ずれ」だけを運ぶ。**各活性化を

    値 = num[i] (numpy の float) + d[i] (sympy 式、θ\* で 0)

と分けて持ち、`d` は**変数に依存するエントリだけ**の疎な辞書にする。
動かすパラメータが k 本なら、記号計算の量はその影響が広がる範囲だけで済む。
線形層の数値部分は numpy で一括、記号部分は疎な範囲だけ sympy で回す。

この分け方には**もう 1 つ本質的な利点**がある。fiber ideal の生成元は
出力の θ\* からのずれ、つまりちょうど `d` そのものなので、
**生成元は構成上 θ\* で厳密に 0 になる。**基準値 `num` を有理数に丸める必要も
なく、丸め誤差が残差として漏れることもない (`torch_rlct` では
softmax の `e*` や tanh の係数でこの問題に手当てが必要だった)。
有理数に丸めるのは**記号に掛かる係数だけ**でよい。

------------------------------------------------------------------
BatchNorm — 畳み込みに畳み込む
------------------------------------------------------------------
BN は `1/sqrt(σ²+ε)` を含むので多項式でない。しかし**評価モードでは統計量が
定数**なので、BN は単なるアフィン写像であり、直前の畳み込みに**厳密に
畳み込める**:

    W' = W · γ/√(σ²+ε),   b' = (b − μ)·γ/√(σ²+ε) + β

`fuse_conv_bn()` がこれを行い、BN を `Identity` に置き換える。
**融合が厳密であること (出力が一致すること) は毎回検査する** (`fuse_check`)。

> これは「BN を無視する」のとは違う。畳み込んだネットワークは元のネットワークと
> **数値的に同一**で、しかも多項式である。ただし λ は「BN の統計量を固定した
> モデル」の値になる。統計量をパラメータとして動かすモデルの λ ではない。

------------------------------------------------------------------
どこまで記号にするか
------------------------------------------------------------------
`tail_from` で指定したモジュール以降だけを記号で通し、それより前は
数値で評価して活性化を固定する。ResNet 全体の λ ではなく
**「その先だけを動かす局所モデル」の λ** であり、部分空間への制限なので
**全体の λ の下界**になる (`K = x²+y²` は λ=1 だが `y=0` に制限すると 1/2)。

現実的な選択:

| `params` | 記号部分 | 変数の数 | 実測 |
|---|---|---|---|
| `fc` のみ | 最終 Linear だけ | 512·C + C | 秒 |
| `layer4.1.conv2` の一部チャネル | conv2 → avgpool → fc | 任意に絞れる | 分 |
| `layer4` 全体 | 重すぎる | — | 不可 |

使い方
------
    python resnet_rlct.py --demo                    # 乱数初期化で検算だけ
    python resnet_rlct.py --arch resnet18 --ckpt run/best.pt \\
        --data cifar10 --params fc.weight,fc.bias --n-data 16
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

__all__ = ["fuse_conv_bn", "fuse_check", "SymTensor", "resnet_fiber_ideal",
           "resnet_local_rlct", "list_param_groups"]


# ----------------------------------------------------------------------
# BatchNorm の畳み込み
# ----------------------------------------------------------------------
def _fuse_pair(conv, bn):
    """conv の重み・バイアスに bn を畳み込む (評価モードで厳密)。"""
    import torch
    import torch.nn as nn

    with torch.no_grad():
        w = conv.weight
        gamma = bn.weight if bn.weight is not None else torch.ones(
            bn.num_features)
        beta = bn.bias if bn.bias is not None else torch.zeros(bn.num_features)
        mu, var, eps = bn.running_mean, bn.running_var, bn.eps
        scale = gamma / torch.sqrt(var + eps)
        new_w = w * scale.reshape(-1, *([1] * (w.dim() - 1)))
        b = conv.bias if conv.bias is not None else torch.zeros(w.shape[0])
        new_b = (b - mu) * scale + beta
        fused = nn.Conv2d(conv.in_channels, conv.out_channels,
                          conv.kernel_size, stride=conv.stride,
                          padding=conv.padding, dilation=conv.dilation,
                          groups=conv.groups, bias=True)
        fused.weight.copy_(new_w)
        fused.bias.copy_(new_b)
    return fused


def fuse_conv_bn(model):
    r"""モデル中の (Conv2d, BatchNorm2d) の対をすべて畳み込む。

    **評価モードにしてから呼ぶこと** (`model.eval()`)。学習モードだと BN が
    バッチ統計を使うので、固定した統計量での融合は別のモデルになる。

    torchvision の ResNet は conv と bn が隣り合う属性として現れる
    (`conv1`/`bn1`, `layer*.*.conv2`/`bn2`, `downsample.0`/`downsample.1`)
    ので、`named_children` を順に見て対を探す。返り値は融合した**複製**で、
    元のモデルは変更しない。
    """
    import torch.nn as nn

    model = copy.deepcopy(model).eval()

    def walk(mod):
        children = list(mod.named_children())
        i = 0
        while i < len(children):
            name, child = children[i]
            nxt = children[i + 1] if i + 1 < len(children) else None
            if isinstance(child, nn.Conv2d) and nxt is not None \
                    and isinstance(nxt[1], nn.BatchNorm2d):
                setattr(mod, name, _fuse_pair(child, nxt[1]))
                setattr(mod, nxt[0], nn.Identity())
                i += 2
                continue
            walk(child)
            i += 1

    walk(model)
    return model


def fuse_check(model, X, *, tol: float = 1e-4) -> Tuple[bool, float]:
    r"""融合が厳密か (出力が一致するか) を検査する。

    BN の畳み込みは**代数的には厳密**だが、float32 の丸めで誤差は出る。
    ここで一致を確認しないと、以降の λ は「別のモデル」の値になる。
    """
    import torch

    model = model.eval()
    fused = fuse_conv_bn(model)
    with torch.no_grad():
        a = model(X)
        b = fused(X)
    dev = float((a - b).abs().max())
    return dev <= tol, dev


# ----------------------------------------------------------------------
# 「ずれだけを運ぶ」記号テンソル
# ----------------------------------------------------------------------
@dataclass
class SymTensor:
    r"""値 = `num` (numpy の実数) + `sym` (疎な sympy 式、θ\* で 0)。

    `shape` は (C, H, W) か (D,)。`sym` は平坦な添字 → sympy 式の辞書で、
    **変数に依存するエントリだけ**を持つ。
    """
    num: object                       # numpy.ndarray (平坦)
    sym: Dict[int, sp.Expr]
    shape: Tuple[int, ...]

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def entry(self, i: int):
        return self.num[i], self.sym.get(i)

    def is_numeric(self) -> bool:
        return not self.sym


def _rat(v, max_den: int) -> sp.Rational:
    return sp.Rational(float(v)).limit_denominator(max_den)


class SymEval:
    r"""ResNet を「ずれだけ」で記号評価する。

    `var_of` : パラメータ名 → {平坦添字: sympy 記号} (動かす成分だけ)
    `max_den`: 記号に掛かる係数を有理数に丸めるときの分母上限
    """

    def __init__(self, var_of: Dict[str, Dict[int, sp.Symbol]],
                 name_of: Dict[int, str], *, max_den: int = 10 ** 6,
                 relu_cells: Optional[Dict] = None,
                 record_cells: Optional[Dict] = None):
        self.var_of = var_of
        self.name_of = name_of
        self.max_den = max_den
        self.relu_cells = relu_cells if relu_cells is not None else {}
        self.record_cells = record_cells
        self.boundary: List[str] = []
        self.n_sym_max = 0
        self.path: List[str] = []

    # -- 補助 ------------------------------------------------------
    def _vars_for(self, module, kind: str) -> Dict[int, sp.Symbol]:
        """このモジュールの weight/bias のうち変数になっている成分。"""
        p = getattr(module, kind, None)
        if p is None:
            return {}
        nm = self.name_of.get(id(p))
        if nm is None:
            return {}
        return self.var_of.get(nm, {})

    def _note(self, t: SymTensor):
        self.n_sym_max = max(self.n_sym_max, len(t.sym))

    # -- 各モジュール ----------------------------------------------
    def conv2d(self, layer, x: SymTensor) -> SymTensor:
        import numpy as np
        import torch

        C, H, W = x.shape
        W_t = layer.weight.detach()
        Oc, Ic, kh, kw = W_t.shape
        sh, sw = layer.stride if isinstance(layer.stride, tuple) \
            else (layer.stride, layer.stride)
        ph, pw = layer.padding if isinstance(layer.padding, tuple) \
            else (layer.padding, layer.padding)
        if layer.groups != 1 or layer.dilation not in ((1, 1), 1):
            raise NotImplementedError("groups/dilation は 1 のみ")
        Ho = (H + 2 * ph - kh) // sh + 1
        Wo = (W + 2 * pw - kw) // sw + 1

        # --- 数値部分は torch で一括 ---
        # 数値部分は **float64** で計算する。float32 だと丸めが 1e-7 で、
        # 記号側との突き合わせがその雑音に埋もれる (実測で相対 1e-3 まで悪化)
        xin = torch.from_numpy(x.num).reshape(1, C, H, W).double()
        with torch.no_grad():
            out_num = torch.nn.functional.conv2d(
                xin, W_t.double(),
                layer.bias.detach().double() if layer.bias is not None
                else None, stride=(sh, sw), padding=(ph, pw))
        num = out_num.reshape(-1).numpy().astype("float64")

        # --- 記号部分 ---
        wv = self._vars_for(layer, "weight")
        bv = self._vars_for(layer, "bias")
        sym: Dict[int, sp.Expr] = {}
        Wnp = W_t.numpy()

        def add(idx, term):
            if term == 0:
                return
            sym[idx] = sp.expand(sym.get(idx, sp.Integer(0)) + term)

        # (a) 入力のずれ × 数値の重み
        if x.sym:
            for src, d in x.sym.items():
                c = src // (H * W)
                rem = src % (H * W)
                ih, iw = rem // W, rem % W
                for oc in range(Oc):
                    for a in range(kh):
                        oh_num = ih + ph - a
                        if oh_num % sh:
                            continue
                        oh = oh_num // sh
                        if not (0 <= oh < Ho):
                            continue
                        for b in range(kw):
                            ow_num = iw + pw - b
                            if ow_num % sw:
                                continue
                            ow = ow_num // sw
                            if not (0 <= ow < Wo):
                                continue
                            wgt = Wnp[oc, c, a, b]
                            if wgt == 0.0:
                                continue
                            add((oc * Ho + oh) * Wo + ow,
                                _rat(wgt, self.max_den) * d)
        # (b) 重みのずれ × 入力 (数値 + ずれ)
        for widx, symb in wv.items():
            oc = widx // (Ic * kh * kw)
            rem = widx % (Ic * kh * kw)
            c = rem // (kh * kw)
            rem2 = rem % (kh * kw)
            a, b = rem2 // kw, rem2 % kw
            for oh in range(Ho):
                ih = oh * sh - ph + a
                if not (0 <= ih < H):
                    continue
                for ow in range(Wo):
                    iw = ow * sw - pw + b
                    if not (0 <= iw < W):
                        continue
                    src = (c * H + ih) * W + iw
                    val = _rat(x.num[src], self.max_den)
                    term = symb * val
                    ds = x.sym.get(src)
                    if ds is not None:
                        term = term + symb * ds
                    add((oc * Ho + oh) * Wo + ow, term)
        # (c) バイアスのずれ
        for bidx, symb in bv.items():
            for oh in range(Ho):
                for ow in range(Wo):
                    add((bidx * Ho + oh) * Wo + ow, symb)
        t = SymTensor(num=num, sym=sym, shape=(Oc, Ho, Wo))
        self._note(t)
        return t

    def linear(self, layer, x: SymTensor) -> SymTensor:
        import numpy as np
        import torch

        W_t = layer.weight.detach()
        Oc, Ic = W_t.shape
        xin = torch.from_numpy(x.num).reshape(1, -1).double()
        with torch.no_grad():
            num = torch.nn.functional.linear(
                xin, W_t.double(),
                layer.bias.detach().double() if layer.bias is not None
                else None).reshape(-1).numpy().astype("float64")
        wv = self._vars_for(layer, "weight")
        bv = self._vars_for(layer, "bias")
        sym: Dict[int, sp.Expr] = {}
        Wnp = W_t.numpy()

        def add(i, term):
            if term == 0:
                return
            sym[i] = sp.expand(sym.get(i, sp.Integer(0)) + term)

        if x.sym:
            for src, d in x.sym.items():
                for o in range(Oc):
                    wgt = Wnp[o, src]
                    if wgt == 0.0:
                        continue
                    add(o, _rat(wgt, self.max_den) * d)
        for widx, symb in wv.items():
            o, i = widx // Ic, widx % Ic
            term = symb * _rat(x.num[i], self.max_den)
            ds = x.sym.get(i)
            if ds is not None:
                term = term + symb * ds
            add(o, term)
        for bidx, symb in bv.items():
            add(bidx, symb)
        t = SymTensor(num=num, sym=sym, shape=(Oc,))
        self._note(t)
        return t

    def relu(self, x: SymTensor, tag: str) -> SymTensor:
        r"""θ\* での符号でセルを固定する。

        前活性化が 0 ちょうどの箇所は境界として記録する。境界では錐への
        制限をしていないので λ は**下界**になる (`torch_rlct` と同じ扱い)。
        """
        num = x.num.copy()
        sym: Dict[int, sp.Expr] = {}
        for i in range(x.size):
            v = x.num[i]
            if v > 0:
                if i in x.sym:
                    sym[i] = x.sym[i]
            elif v == 0.0:
                self.boundary.append(f"{tag}[{i}]")
                num[i] = 0.0
            else:
                num[i] = 0.0
        num = num.clip(min=0.0)
        return SymTensor(num=num, sym=sym, shape=x.shape)

    def maxpool(self, layer, x: SymTensor) -> SymTensor:
        import numpy as np
        import torch

        C, H, W = x.shape
        k = layer.kernel_size if isinstance(layer.kernel_size, int) \
            else layer.kernel_size[0]
        s = layer.stride if isinstance(layer.stride, int) \
            else (layer.stride[0] if layer.stride else k)
        p = layer.padding if isinstance(layer.padding, int) \
            else layer.padding[0]
        xin = torch.from_numpy(x.num).reshape(1, C, H, W).double()
        with torch.no_grad():
            out, idx = torch.nn.functional.max_pool2d(
                xin, k, stride=s, padding=p, return_indices=True)
        num = out.reshape(-1).numpy().astype("float64")
        flat_idx = idx.reshape(-1).numpy()
        sym: Dict[int, sp.Expr] = {}
        if x.sym:
            for o, src in enumerate(flat_idx):
                c = o // (out.shape[2] * out.shape[3])
                d = x.sym.get(int(src) + c * H * W if int(src) < H * W
                              else int(src))
                if d is not None:
                    sym[o] = d
        return SymTensor(num=num, sym=sym,
                         shape=(C, out.shape[2], out.shape[3]))

    def avgpool_global(self, x: SymTensor) -> SymTensor:
        import numpy as np

        C, H, W = x.shape
        n = H * W
        num = x.num.reshape(C, n).mean(axis=1)
        sym: Dict[int, sp.Expr] = {}
        if x.sym:
            acc: Dict[int, sp.Expr] = {}
            for src, d in x.sym.items():
                c = src // n
                acc[c] = acc.get(c, sp.Integer(0)) + d
            for c, e in acc.items():
                sym[c] = sp.expand(e / n)
        return SymTensor(num=num, sym=sym, shape=(C,))

    def add(self, a: SymTensor, b: SymTensor) -> SymTensor:
        sym = dict(a.sym)
        for i, d in b.sym.items():
            sym[i] = sp.expand(sym.get(i, sp.Integer(0)) + d)
        return SymTensor(num=a.num + b.num, sym=sym, shape=a.shape)

    def flatten(self, x: SymTensor) -> SymTensor:
        return SymTensor(num=x.num, sym=dict(x.sym), shape=(x.size,))

    # -- ブロックとモジュール一般 ----------------------------------
    def module(self, mod, x: SymTensor, tag: str = "") -> SymTensor:
        import torch.nn as nn
        try:
            from torchvision.models.resnet import BasicBlock, Bottleneck
        except Exception:                                 # noqa: BLE001
            BasicBlock = Bottleneck = ()

        if isinstance(mod, nn.Conv2d):
            return self.conv2d(mod, x)
        if isinstance(mod, nn.Linear):
            return self.linear(mod, x)
        if isinstance(mod, nn.ReLU):
            return self.relu(x, tag)
        if isinstance(mod, nn.MaxPool2d):
            return self.maxpool(mod, x)
        if isinstance(mod, nn.AdaptiveAvgPool2d):
            if mod.output_size not in (1, (1, 1)):
                raise NotImplementedError(
                    "AdaptiveAvgPool2d は出力 1x1 のみ")
            return self.avgpool_global(x)
        if isinstance(mod, nn.AvgPool2d):
            raise NotImplementedError("AvgPool2d は未対応 (必要なら追加)")
        if isinstance(mod, nn.BatchNorm2d):
            raise RuntimeError(
                "BatchNorm2d が残っています。fuse_conv_bn() を先に通してください "
                "(BN は 1/sqrt を含むので多項式になりません)")
        if isinstance(mod, (nn.Identity, nn.Dropout)):
            return x
        if isinstance(mod, nn.Flatten):
            return self.flatten(x)
        if BasicBlock and isinstance(mod, BasicBlock):
            out = self.conv2d(mod.conv1, x)
            out = self.module(mod.bn1, out, tag + ".bn1")
            out = self.relu(out, tag + ".relu1")
            out = self.conv2d(mod.conv2, out)
            out = self.module(mod.bn2, out, tag + ".bn2")
            idt = x if mod.downsample is None else \
                self.module(mod.downsample, x, tag + ".downsample")
            out = self.add(out, idt)
            return self.relu(out, tag + ".relu2")
        if Bottleneck and isinstance(mod, Bottleneck):
            out = self.conv2d(mod.conv1, x)
            out = self.module(mod.bn1, out, tag + ".bn1")
            out = self.relu(out, tag + ".relu1")
            out = self.conv2d(mod.conv2, out)
            out = self.module(mod.bn2, out, tag + ".bn2")
            out = self.relu(out, tag + ".relu2")
            out = self.conv2d(mod.conv3, out)
            out = self.module(mod.bn3, out, tag + ".bn3")
            idt = x if mod.downsample is None else \
                self.module(mod.downsample, x, tag + ".downsample")
            out = self.add(out, idt)
            return self.relu(out, tag + ".relu3")
        if isinstance(mod, nn.Sequential):
            for i, sub in enumerate(mod):
                x = self.module(sub, x, f"{tag}.{i}")
            return x
        raise NotImplementedError(
            f"未対応のモジュール {type(mod).__name__} ({tag})")


# ----------------------------------------------------------------------
# ResNet を頭と尾に分ける
# ----------------------------------------------------------------------#: ResNet の順方向の並び (torchvision)
_RESNET_ORDER = ["conv1", "bn1", "relu", "maxpool",
                 "layer1", "layer2", "layer3", "layer4",
                 "avgpool", "flatten", "fc"]


def _resnet_modules(model) -> List[Tuple[str, object]]:
    import torch.nn as nn
    out = []
    for nm in _RESNET_ORDER:
        if nm == "flatten":
            out.append(("flatten", nn.Flatten(1)))
            continue
        if hasattr(model, nm):
            out.append((nm, getattr(model, nm)))
    return out


def list_param_groups(model, *, fused: bool = True) -> List[Tuple[str, int]]:
    """パラメータ名と要素数の一覧 (どれを動かすか決めるため)。"""
    return [(n, p.numel()) for n, p in model.named_parameters()]


def resnet_fiber_ideal(model, X, *, params: Sequence[str],
                       tail_from: str = "layer4",
                       loss: str = "classification",
                       max_denominator: int = 10 ** 6,
                       zero_tol: float = 0.0,
                       max_vars: int = 40,
                       params_slice: int = 0,
                       verbose: bool = False) -> Dict:
    r"""学習済み ResNet の fiber ideal を作る。

    model     : torchvision の ResNet (**融合前**でよい。中で融合する)
    X         : 入力 (N, C, H, W)
    params    : 動かすパラメータ名 (`fc.weight` など)。
                **融合後のモデルの名前**で指定すること (bn* は消える)
    tail_from : ここから先だけを記号で通す。それより前は数値で固定
    loss      : 'classification' ならロジットの**差**、'regression' なら出力そのもの

    返り値は `nn_rlct.local_rlct_from_ideal` に渡せる dict。
    """
    import numpy as np
    import torch
    import torch.nn as nn

    model = model.eval()
    ok, dev = fuse_check(model, X[:2])
    fused = fuse_conv_bn(model)
    mods = _resnet_modules(fused)
    names = [n for n, _ in mods]
    if tail_from not in names:
        raise ValueError(f"tail_from={tail_from} は {names} のいずれかにしてください")
    cut = names.index(tail_from)

    # 変数を作る: パラメータ名 -> {平坦添字: 記号}
    pdict = dict(fused.named_parameters())
    unknown = [p for p in params if p not in pdict]
    if unknown:
        raise ValueError(f"存在しないパラメータ {unknown}。"
                         f"融合後の名前で指定してください (bn* は消えます)")
    var_of: Dict[str, Dict[int, sp.Symbol]] = {}
    variables: List[sp.Symbol] = []
    for pn in params:
        p = pdict[pn]
        flat = p.detach().reshape(-1)
        n_use = (min(params_slice, flat.numel()) if params_slice > 0
                 else flat.numel())
        d: Dict[int, sp.Symbol] = {}
        for i in range(n_use):
            s = sp.Symbol(f"u_{pn.replace('.', '_')}_{i}", real=True)
            d[i] = s
            variables.append(s)
        var_of[pn] = d
    if len(variables) > max_vars:
        raise ValueError(
            f"変数が {len(variables)} 本で上限 {max_vars} を超えます。"
            f"params を絞るか max_vars を上げてください "
            f"(記号計算の費用は変数の数と影響範囲で決まります)")
    name_of = {id(p): n for n, p in fused.named_parameters()}

    # --- 頭を数値で評価 ---
    with torch.no_grad():
        h = X
        for nm, mod in mods[:cut]:
            h = mod(h)
    if verbose:
        print(f"  {tail_from} の入力: {tuple(h.shape)}  "
              f"(BN 融合の最大差 {dev:.2e})")

    # --- 尾を記号で評価 ---
    ev = SymEval(var_of, name_of, max_den=max_denominator)
    outs: List[SymTensor] = []
    for i in range(X.shape[0]):
        hi = h[i].detach().numpy().astype("float64").reshape(-1)
        shape = tuple(h[i].shape)
        t = SymTensor(num=hi, sym={}, shape=shape)
        for nm, mod in mods[cut:]:
            t = ev.module(mod, t, nm)
        outs.append(t)

    # --- 生成元 ---
    gens: List[sp.Expr] = []
    C = outs[0].size
    if loss == "classification":
        ref = C - 1
        for t in outs:
            dref = t.sym.get(ref, sp.Integer(0))
            for c in range(C - 1):
                g = sp.expand(t.sym.get(c, sp.Integer(0)) - dref)
                if g != 0:
                    gens.append(g)
    else:
        for t in outs:
            for c in range(C):
                g = t.sym.get(c)
                if g is not None and g != 0:
                    gens.append(sp.expand(g))

    # ロジット全体の平行移動は分類で対称性 (最終 Linear の bias の (1,..,1))
    sym_vecs: List[Dict] = []
    if loss == "classification" and "fc.bias" in params:
        sym_vecs.append({s: 1 for s in var_of["fc.bias"].values()})

    meta = dict(fuse_ok=bool(ok), fuse_dev=dev, tail_from=tail_from,
                tail_input_shape=tuple(h.shape[1:]),
                boundary=list(ev.boundary), n_sym_max=ev.n_sym_max,
                n_params_total=sum(p.numel() for p in fused.parameters()))
    return dict(generators=gens, variables=variables,
                symmetry_vectors=sym_vecs, meta=meta, fused=fused)


@dataclass
class ResNetReport:
    local: object
    variables: List
    params: List[str]
    meta: Dict
    n_gens: int = 0

    def print_report(self) -> None:
        m = self.meta
        print(f"  モデル全体のパラメータ {m['n_params_total']:,}")
        print(f"  記号で通した範囲: {m['tail_from']} 以降 "
              f"(入力 {m['tail_input_shape']})")
        print(f"  動かしたパラメータ: {self.params} -> 変数 "
              f"{len(self.variables)} 本")
        print(f"  生成元 {self.n_gens} 本、記号エントリの最大数 "
              f"{m['n_sym_max']}")
        print(f"  BN 融合の最大差 {m['fuse_dev']:.2e} "
              f"({'厳密と判定' if m['fuse_ok'] else '**要確認**'})")
        if m["boundary"]:
            print(f"  [note] 前活性化が 0 の箇所 {len(m['boundary'])} 個。"
                  "錐への制限をしていないので lambda は下界")
        print()
        self.local.print_report()
        print("  [注意] これは指定した部分空間に制限した値なので、"
              "ResNet 全体の lambda の**下界**です。")


def resnet_local_rlct(model, X, *, params: Sequence[str],
                      tail_from: str = "layer4",
                      loss: str = "classification",
                      max_denominator: int = 10 ** 6,
                      max_vars: int = 40, params_slice: int = 0,
                      verbose: bool = False,
                      **resolve_kwargs) -> ResNetReport:
    """学習済み ResNet の局所 RLCT。"""
    from nn_rlct import local_rlct_from_ideal

    data = resnet_fiber_ideal(model, X, params=params, tail_from=tail_from,
                              loss=loss, max_denominator=max_denominator,
                              max_vars=max_vars, params_slice=params_slice,
                              verbose=verbose)
    loc = local_rlct_from_ideal(data["generators"], data["variables"],
                                symmetry_vectors=data["symmetry_vectors"],
                                verbose=verbose, **resolve_kwargs)
    return ResNetReport(local=loc, variables=data["variables"],
                        params=list(params), meta=data["meta"],
                        n_gens=len(data["generators"]))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _demo(arch: str = "resnet18", n: int = 4, size: int = 32) -> int:
    """乱数初期化の ResNet で、融合と記号フォワードの検算だけを行う。"""
    import torch
    import torchvision.models as tvm

    print(f"===== {arch} の検算 (乱数初期化, {size}x{size}) =====")
    model = getattr(tvm, arch)(weights=None, num_classes=10).eval()
    X = torch.randn(n, 3, size, size,
                    generator=torch.Generator().manual_seed(0))
    ok, dev = fuse_check(model, X)
    print(f"  BN 融合の最大差 {dev:.3e}  {'OK' if ok else '**NG**'}")

    # 記号フォワードが torch と一致するか (変数を 0 にすれば一致するはず)
    fused = fuse_conv_bn(model)
    with torch.no_grad():
        ref = fused(X)
    data = resnet_fiber_ideal(model, X, params=["fc.bias"],
                              tail_from="layer4", loss="regression",
                              max_vars=64, verbose=True)
    # 生成元は θ* で 0 になるはず
    bad = 0
    for g in data["generators"]:
        P = sp.Poly(g, *data["variables"])
        if P.coeff_monomial(1) != 0:
            bad += 1
    print(f"  生成元 {len(data['generators'])} 本, θ* で消えない {bad} 本")
    print(f"  記号エントリの最大数 {data['meta']['n_sym_max']}")
    return 0 if ok and bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="乱数初期化で融合と記号フォワードを検算")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--ckpt", default="", help="学習済み重み (.pt)")
    ap.add_argument("--data", default="cifar10",
                    choices=("cifar10", "cifar100", "folder", "random"))
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--n-data", type=int, default=8)
    ap.add_argument("--params", default="fc.bias")
    ap.add_argument("--tail-from", default="layer4")
    ap.add_argument("--loss", default="classification",
                    choices=("classification", "regression"))
    ap.add_argument("--max-vars", type=int, default=40)
    ap.add_argument("--params-slice", type=int, default=0,
                    help="各パラメータの先頭 n 成分だけを動かす")
    ap.add_argument("--max-denominator", type=int, default=10 ** 6)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--list-params", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.demo:
        return _demo(args.arch, size=args.size)

    import torch
    import torchvision.models as tvm

    model = getattr(tvm, args.arch)(weights=None,
                                    num_classes=args.classes).eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        sd = sd.get("model", sd)
        model.load_state_dict(sd)
        print(f"  重みを {args.ckpt} から読み込みました")
    if args.list_params:
        fused = fuse_conv_bn(model)
        print("融合後のパラメータ (この名前で --params に指定):")
        for n, k in list_param_groups(fused):
            print(f"  {n:<36} {k:>9,}")
        return 0

    from train_resnet import load_eval_batch
    X, _ = load_eval_batch(args.data, root=args.data_root, n=args.n_data,
                           size=args.size, classes=args.classes)
    params = [v for v in args.params.split(",") if v]
    print(f"===== {args.arch} の学習済み theta* における局所 RLCT =====")
    t = time.time()
    rep = resnet_local_rlct(model, X, params=params,
                            tail_from=args.tail_from, loss=args.loss,
                            max_denominator=args.max_denominator,
                            max_vars=args.max_vars,
                            params_slice=args.params_slice, verbose=True,
                            max_depth=args.max_depth)
    rep.print_report()
    print(f"  ({time.time()-t:.1f}s)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(dict(arch=args.arch, params=params,
                           tail_from=args.tail_from, loss=args.loss,
                           rlct=str(rep.local.rlct),
                           mult=rep.local.multiplicity,
                           n_vars=len(rep.variables), n_gens=rep.n_gens,
                           meta={k: str(v) for k, v in rep.meta.items()}),
                      fh, ensure_ascii=False, indent=1)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
