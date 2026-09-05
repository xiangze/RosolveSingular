r"""
torch_rlct.py
=============

PyTorch のモデルをそのまま読み込み、theta* (現在の重み) における局所 RLCT を
計算する。中身は nn_rlct.py と同じ 3 段階:

  (i)   theta* とデータで決まる活性化セルを固定し、そのセル上でネットワークを
        パラメータの多項式に書き換える (深いネットでも各層の前活性化の符号を
        theta* で固定すれば全体が多項式になる)
  (ii)  スケール対称性をゲージ固定し、自由方向と正則方向を落とす
  (iii) 残ったコアイデアルを resolve_singularity.py に渡す

対応レイヤ
----------
  nn.Linear                     : 記号 or 定数の重み
  nn.ReLU, nn.LeakyReLU         : theta* での符号でセルを固定 (区分線形)
  nn.Identity, nn.Flatten       : 素通り
  nn.Tanh, nn.Sigmoid, nn.SiLU,
  nn.Softplus, nn.GELU          : theta* の前活性化まわりのテイラー展開
                                  (taylor_order 次で打ち切り)
  それ以外                      : extra_handlers で追加するか例外

注意
----
* 変数の数がそのまま解消の難しさになるので、params= で「動かすパラメータ」を
  必ず絞ること (既定の上限 max_vars=40)。残りは theta* に凍結される。
* 学習後の重みは厳密に 0 にならないので、zero_tol 以下の成分は 0 に丸める。
  「近くにある退化した点」の RLCT を見るための操作であり、丸めないと
  たいてい正則点 (lambda = パラメータ数/2) になる。
* 重みは有理数化して厳密演算する (max_denominator)。
* ゲージ固定は、その対称性が動かす全パラメータが変数になっている場合のみ
  適用する。一部を凍結していると、それはもう対称性ではないため。

使い方
------
    import torch.nn as nn, torch
    from torch_rlct import torch_local_rlct

    net = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1))
    X = torch.tensor([[-2.], [-1.], [0.5], [1.], [2.]])
    rep = torch_local_rlct(net, X)          # realizable (真の関数 = 現在の net)
    rep.print_report()
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import sympy as sp

from nn_rlct import LocalRLCT, local_rlct_from_ideal

__all__ = [
    "TorchReport",
    "torch_local_rlct",
    "fiber_ideal_from_torch",
]


# ----------------------------------------------------------------------
# 数値 <-> 記号
# ----------------------------------------------------------------------
def _rat(v, zero_tol: float, max_den: int) -> sp.Rational:
    v = float(v)
    if abs(v) <= zero_tol:
        return sp.Integer(0)
    return sp.Rational(v).limit_denominator(max_den)


# ----------------------------------------------------------------------
# レイヤの取り出し
# ----------------------------------------------------------------------
def _flatten_layers(module) -> List:
    import torch.nn as nn

    if isinstance(module, nn.Sequential):
        out = []
        for m in module:
            out.extend(_flatten_layers(m))
        return out
    kids = list(module.children())
    if kids:
        out = []
        for m in kids:
            out.extend(_flatten_layers(m))
        return out
    return [module]


# ----------------------------------------------------------------------
# 活性化のハンドラ
# ----------------------------------------------------------------------
def _relu_like(slope):
    def handler(sym, num, sign, order):
        # sign > 0 なら恒等、sign < 0 なら slope 倍 (theta* の近傍で符号は不変)
        k = 1 if sign > 0 else slope
        return sp.expand(k * sym), k * num
    return handler


def _analytic(fexpr):
    """解析的な活性化: theta* の前活性化まわりでテイラー展開する。"""
    z = sp.Symbol("_z")
    f = fexpr(z)

    def handler(sym, num, sign, order):
        z0 = sp.nsimplify(num)
        delta = sp.expand(sym - z0)
        val = sp.Integer(0)
        der = f
        for k in range(order + 1):
            coeff = der.subs(z, z0)
            val += sp.nsimplify(coeff) / sp.factorial(k) * delta ** k
            der = sp.diff(der, z)
        return sp.expand(val), float(f.subs(z, z0))
    return handler


def _default_handlers():
    import torch.nn as nn

    h = {
        nn.ReLU: lambda m: _relu_like(0),
        nn.LeakyReLU: lambda m: _relu_like(sp.nsimplify(m.negative_slope)),
        nn.Tanh: lambda m: _analytic(sp.tanh),
        nn.Sigmoid: lambda m: _analytic(lambda z: 1 / (1 + sp.exp(-z))),
        nn.Softplus: lambda m: _analytic(lambda z: sp.log(1 + sp.exp(z))),
        nn.SiLU: lambda m: _analytic(lambda z: z / (1 + sp.exp(-z))),
        nn.GELU: lambda m: _analytic(
            lambda z: z * (1 + sp.erf(z / sp.sqrt(2))) / 2),
    }
    return h


_PIECEWISE = ("ReLU", "LeakyReLU")          # 正斉次 = スケール対称性を持つ


# ----------------------------------------------------------------------
# 記号フォワード
# ----------------------------------------------------------------------
@dataclass
class _Sym:
    layers: List
    var_names: List[str]
    zero_tol: float
    max_den: int
    order: int
    handlers: Dict

    def __post_init__(self):
        import torch.nn as nn
        self.nn = nn
        self.symbols: Dict[Tuple[str, Tuple[int, ...]], sp.Symbol] = {}
        self.star: Dict[Tuple[str, Tuple[int, ...]], sp.Rational] = {}
        self.u_vars: List[sp.Symbol] = []
        self.param_of: Dict[sp.Symbol, str] = {}

    # --- パラメータ 1 成分を「theta* + u」または定数にする ------------
    def entry(self, name: str, tensor, idx: Tuple[int, ...]):
        key = (name, idx)
        if key in self.star:
            star = self.star[key]
        else:
            star = _rat(tensor[idx].item(), self.zero_tol, self.max_den)
            self.star[key] = star
        if name not in self.var_names:
            return star, star
        if key not in self.symbols:
            u = sp.Symbol("u_" + name.replace(".", "_") + "_"
                          + "_".join(map(str, idx)), real=True)
            self.symbols[key] = u
            self.u_vars.append(u)
            self.param_of[u] = name
        return star + self.symbols[key], star

    # --- 1 サンプルの前向き計算 (記号 + 数値を並行して持つ) -----------
    def forward(self, x_row, name_of, signs, record_signs=None):
        sym = [sp.Rational(float(v)).limit_denominator(self.max_den) for v in x_row]
        num = [float(v) for v in x_row]
        for li, layer in enumerate(self.layers):
            if isinstance(layer, self.nn.Linear):
                W, b = layer.weight, layer.bias
                out_s, out_n = [], []
                wname = name_of.get(id(W), None)
                bname = name_of.get(id(b), None) if b is not None else None
                for j in range(W.shape[0]):
                    s = sp.Integer(0)
                    n = 0.0
                    for k in range(W.shape[1]):
                        ws, wn = self.entry(wname, W, (j, k))
                        s += ws * sym[k]
                        n += wn * num[k]
                    if b is not None:
                        bs, bn = self.entry(bname, b, (j,))
                        s += bs
                        n += float(bn)
                    out_s.append(sp.expand(s))
                    out_n.append(float(n))
                sym, num = out_s, out_n
            elif isinstance(layer, (self.nn.Identity, self.nn.Flatten)):
                continue
            else:
                key = type(layer)
                if key not in self.handlers:
                    raise NotImplementedError(
                        f"未対応のレイヤ {key.__name__}。extra_handlers で追加してください。")
                h = self.handlers[key](layer)
                piecewise = type(layer).__name__ in _PIECEWISE
                out_s, out_n = [], []
                for j in range(len(sym)):
                    if piecewise:
                        # 区分線形な活性化だけがセル (符号) の概念を持つ
                        sgn = signs.get((li, j), None)
                        if sgn is None:
                            sgn = 0 if abs(num[j]) <= self.zero_tol else (
                                1 if num[j] > 0 else -1)
                            if record_signs is not None:
                                record_signs[(li, j)] = sgn
                    else:
                        sgn = None
                    s, n = h(sym[j], num[j], -1 if not sgn else sgn, self.order)
                    out_s.append(s)
                    out_n.append(n)
                sym, num = out_s, out_n
        return sym, num


# ----------------------------------------------------------------------
# 本体
# ----------------------------------------------------------------------
@dataclass
class TorchReport:
    local: LocalRLCT
    variables: List[sp.Symbol]
    param_names: List[str]
    frozen: List[str]
    signs: Dict[Tuple[int, int, int], int]        # (sample, layer, unit) -> +/-/0
    boundary: List[Tuple[int, int, int]]
    dead_units: List[Tuple[int, int]]
    symmetry_units: List[Tuple[int, int]]
    skipped_symmetries: List[Tuple[int, int]]
    cells: List[Tuple[Tuple[int, ...], LocalRLCT]] = field(default_factory=list)

    def print_report(self, with_history: bool = False) -> None:
        print("############ PyTorch モデルの局所 RLCT ############")
        print(f"  変数にしたパラメータ: {self.param_names}")
        if self.frozen:
            print(f"  theta* に凍結: {self.frozen}")
        print(f"  変数の数: {len(self.variables)}")
        n_act = len({(l, u) for (_, l, u) in self.signs})
        print(f"  活性化ユニット数: {n_act},  前活性化が 0 の箇所: {len(self.boundary)}")
        if self.dead_units:
            print(f"  全データで不活性なユニット (層, 番号): {self.dead_units}")
        print(f"  スケール対称性を商にしたユニット: {self.symmetry_units}")
        if self.skipped_symmetries:
            print(f"  対称性を使えなかったユニット (一部を凍結しているため): "
                  f"{self.skipped_symmetries}")
        if self.boundary:
            print("  theta* はセル境界 -> 接する各錐に制限した RLCT の最小値が真の値。"
                  "以下は錐に制限していないので下界。")
            if self.cells:
                from collections import Counter
                cnt = Counter((str(l.rlct), l.multiplicity) for _, l in self.cells)
                print(f"    接するセル {len(self.cells)} 個: ((lambda, m) -> 個数)")
                for (lam, m), k in sorted(cnt.items(),
                                          key=lambda t: float(sp.Rational(t[0][0]))):
                    print(f"      lambda={lam}, m={m}  ({k} セル)")
        print()
        self.local.print_report(with_history=with_history)


def fiber_ideal_from_torch(
    model,
    X,
    *,
    targets=None,
    params: Optional[Sequence[str]] = None,
    zero_tol: float = 1e-8,
    max_denominator: int = 10 ** 6,
    taylor_order: int = 3,
    extra_handlers: Optional[Dict] = None,
    max_vars: int = 40,
    residual_tol: float = 1e-3,
):
    """torch モデルから fiber ideal の生成元・変数・対称ベクトルを作る。

    Returns
    -------
    dict(generators=..., variables=..., symmetry_vectors=..., meta=...)
    """
    import torch
    import torch.nn as nn

    layers = _flatten_layers(model)
    handlers = _default_handlers()
    if extra_handlers:
        handlers.update(extra_handlers)

    # パラメータ名 (id -> 名前)
    name_of = {id(p): n for n, p in model.named_parameters()}
    all_names = list(name_of.values())
    if params is None:
        var_names = list(all_names)
    else:
        var_names = [n for n in all_names if n in set(params)]
        unknown = set(params) - set(all_names)
        if unknown:
            raise ValueError(f"存在しないパラメータ名: {sorted(unknown)}"
                             f" / 使えるのは {all_names}")
    frozen = [n for n in all_names if n not in var_names]

    X = torch.as_tensor(X, dtype=torch.float64)
    if X.dim() == 1:
        X = X.unsqueeze(1)
    Xl = X.tolist()

    with torch.no_grad():
        S = _Sym(layers, var_names, zero_tol, max_denominator, taylor_order, handlers)

        # --- 1 回目: 活性化パターン (符号) を theta* で決める ----------
        signs_per_sample: List[Dict[Tuple[int, int], int]] = []
        for row in Xl:
            rec: Dict[Tuple[int, int], int] = {}
            S.forward(row, name_of, {}, record_signs=rec)
            signs_per_sample.append(rec)

        boundary = [(i, l, u) for i, rec in enumerate(signs_per_sample)
                    for (l, u), s in rec.items() if s == 0]

        # --- 2 回目: 符号を固定して記号フォワード -------------------
        residual_check: List[float] = []

        def build(assign: Tuple[int, ...]):
            sgn_maps = [dict(rec) for rec in signs_per_sample]
            for (i, l, u), s in zip(boundary, assign):
                sgn_maps[i][(l, u)] = s
            gens = []
            for i, row in enumerate(Xl):
                sym, num = S.forward(row, name_of, sgn_maps[i])
                zero = {u: 0 for u in S.u_vars}
                # fiber ideal は「theta* のモデルが真」としたときの残差で定義される。
                # targets が与えられた場合は theta* がその教師データのゼロ点に
                # なっているかの検査にのみ使う (下で residual_tol と比較)。
                y = [sp.expand(e).subs(zero) for e in sym]
                if targets is not None:
                    ti = targets[i]
                    ti = ti if isinstance(ti, (list, tuple)) else [ti]
                    for yv, tv in zip(y, ti):
                        residual_check.append(abs(float(yv) - float(tv)))
                for s_expr, y_val in zip(sym, y):
                    gens.append(sp.expand(s_expr - y_val))
            return gens

        gens0 = build(tuple(-1 for _ in boundary))
        # theta* が教師データのゼロ点かどうかは、丸めていない元の重みで判定する
        rounding_shift = max(residual_check) if residual_check else 0.0
        raw_res = 0.0
        if targets is not None:
            dt = next(model.parameters()).dtype
            pred = model(X.to(dt)).detach().reshape(len(Xl), -1).tolist()
            for pi, ti in zip(pred, targets):
                ti = ti if isinstance(ti, (list, tuple)) else [ti]
                for a_, b_ in zip(pi, ti):
                    raw_res = max(raw_res, abs(float(a_) - float(b_)))
            if raw_res > residual_tol:
                raise ValueError(
                    f"theta* が教師データのゼロ点になっていません (最大残差 "
                    f"{raw_res:.3g} > residual_tol={residual_tol})。RLCT は"
                    " realizable な点 (損失がほぼ 0 の局所解) でのみこの形で"
                    " 定義されます。targets=None にすると theta* のモデル自身を"
                    "真の関数として扱います。")

    if len(S.u_vars) > max_vars:
        raise ValueError(
            f"変数が {len(S.u_vars)} 個あります (上限 {max_vars})。"
            " params= で動かすパラメータを絞ってください。"
            f" 例: params={all_names[:2]}")

    # --- スケール対称性 (Linear -> 正斉次活性化 -> Linear) -------------
    sym_vecs, sym_units, skipped = [], [], []
    lin_idx = [i for i, l in enumerate(layers) if isinstance(l, nn.Linear)]
    for a, b in zip(lin_idx, lin_idx[1:]):
        between = [l for l in layers[a + 1:b]
                   if type(l).__name__ not in ("Identity", "Flatten")]
        if len(between) != 1 or type(between[0]).__name__ not in _PIECEWISE:
            continue
        L1, L2 = layers[a], layers[b]
        n1, n2 = name_of[id(L1.weight)], name_of[id(L2.weight)]
        nb1 = name_of[id(L1.bias)] if L1.bias is not None else None
        needed = {n1, n2} | ({nb1} if nb1 else set())
        for j in range(L1.weight.shape[0]):
            vec, complete = {}, True
            for k in range(L1.weight.shape[1]):
                key = (n1, (j, k))
                if S.star.get(key, 0) != 0:
                    if key in S.symbols:
                        vec[S.symbols[key]] = S.star[key]
                    else:
                        complete = False
            if nb1 is not None and S.star.get((nb1, (j,)), 0) != 0:
                if (nb1, (j,)) in S.symbols:
                    vec[S.symbols[(nb1, (j,))]] = S.star[(nb1, (j,))]
                else:
                    complete = False
            for r in range(L2.weight.shape[0]):
                key = (n2, (r, j))
                if S.star.get(key, 0) != 0:
                    if key in S.symbols:
                        vec[S.symbols[key]] = -S.star[key]
                    else:
                        complete = False
            if not vec:
                continue
            if complete:
                sym_vecs.append(vec)
                sym_units.append((a, j))
            else:
                skipped.append((a, j))

    # --- 死んだユニット ------------------------------------------------
    act_keys = set()
    for rec in signs_per_sample:
        act_keys |= set(rec.keys())
    dead = [(l, u) for (l, u) in sorted(act_keys)
            if all(rec.get((l, u), -1) <= 0 for rec in signs_per_sample)]

    meta = dict(
        raw_residual=raw_res, rounding_shift=rounding_shift,
        build=build, boundary=boundary, signs_per_sample=signs_per_sample,
        dead_units=dead, var_names=var_names, frozen=frozen,
        symmetry_units=sym_units, skipped=skipped, layers=layers,
    )
    return dict(generators=gens0, variables=list(S.u_vars),
                symmetry_vectors=sym_vecs, meta=meta)


def torch_local_rlct(
    model,
    X,
    *,
    targets=None,
    params: Optional[Sequence[str]] = None,
    zero_tol: float = 1e-8,
    max_denominator: int = 10 ** 6,
    taylor_order: int = 3,
    extra_handlers: Optional[Dict] = None,
    max_vars: int = 40,
    residual_tol: float = 1e-3,
    boundary: str = "both",
    max_cells: int = 32,
    verbose: bool = False,
    **resolve_kwargs,
) -> TorchReport:
    """PyTorch モデルの現在の重み theta* における局所 RLCT を計算する。

    model    : nn.Module (Sequential 相当の直列構造)
    X        : 入力データ (tensor / ndarray / list)
    targets  : 教師データ。fiber ideal 自体は常に「theta* のモデルが真の関数」
               として作られ、targets は theta* がその教師データのゼロ点に
               なっているか (残差 <= residual_tol) の検査にだけ使われる
    params   : 変数として動かすパラメータ名 (model.named_parameters() の名前)。
               None なら全部だが、max_vars を超えると例外。
    zero_tol : この絶対値以下の重みは 0 に丸める (退化点を見るため)
    boundary : 前活性化が 0 の箇所の扱い ('both' / 'active' / 'inactive')
    """
    data = fiber_ideal_from_torch(
        model, X, targets=targets, params=params, zero_tol=zero_tol,
        max_denominator=max_denominator, taylor_order=taylor_order,
        extra_handlers=extra_handlers, max_vars=max_vars,
        residual_tol=residual_tol)
    meta = data["meta"]
    build, bnd = meta["build"], meta["boundary"]

    if not bnd:
        assigns = [()]
    elif boundary == "active":
        assigns = [tuple(1 for _ in bnd)]
    elif boundary == "inactive":
        assigns = [tuple(-1 for _ in bnd)]
    else:
        assigns = list(itertools.product((1, -1), repeat=len(bnd)))[:max_cells]

    def run(assign):
        return local_rlct_from_ideal(
            build(assign), data["variables"],
            symmetry_vectors=data["symmetry_vectors"],
            verbose=verbose, **resolve_kwargs)

    cells = []
    if len(assigns) == 1:
        local = run(assigns[0])
    else:
        for a in assigns:
            cells.append((a, run(a)))
        local = min(cells, key=lambda t: (float(t[1].rlct), -t[1].multiplicity))[1]

    if targets is not None:
        local.notes.append(
            f"theta* の教師データに対する最大残差 {meta['raw_residual']:.3g}"
            f" / 有理数化と zero_tol による出力のずれ {meta['rounding_shift']:.3g}"
            " (後者は「近くの退化点を見る」ための意図的な丸め)。")
    if meta["dead_units"]:
        local.notes.append(
            f"全データで不活性なユニット {meta['dead_units']} のパラメータは "
            "K に現れない (自由方向)。")
    if bnd:
        local.notes.append(
            "theta* はセル境界。錐への制限をしていないので lambda は下界。")
    if meta["skipped"]:
        local.notes.append(
            f"ユニット {meta['skipped']} のスケール対称性は、関係するパラメータの"
            "一部を凍結しているため商にできませんでした (次元は落ちていない)。")

    signs = {(i, l, u): s for i, rec in enumerate(meta["signs_per_sample"])
             for (l, u), s in rec.items()}
    return TorchReport(
        local=local, variables=data["variables"], param_names=meta["var_names"],
        frozen=meta["frozen"], signs=signs, boundary=bnd,
        dead_units=meta["dead_units"], symmetry_units=meta["symmetry_units"],
        skipped_symmetries=meta["skipped"], cells=cells)


# ----------------------------------------------------------------------
# デモ
# ----------------------------------------------------------------------
def _demo():
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    X = torch.tensor([[-2.0], [-1.0], [0.3], [1.0], [2.0], [3.0]])

    def head(t):
        print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

    def setw(net, W1, b1, W2, b2):
        with torch.no_grad():
            net[0].weight.copy_(torch.tensor(W1, dtype=torch.float32))
            net[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
            net[2].weight.copy_(torch.tensor(W2, dtype=torch.float32))
            net[2].bias.copy_(torch.tensor(b2, dtype=torch.float32))
        return net

    head("例 1: 1-2-1 ReLU、全ユニット活性、正則点 -> lambda = 1")
    net = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
    setw(net, [[1.0]], [3.0], [[2.0]], [0.0])
    torch_local_rlct(net, X).print_report()

    head("例 2: 冗長ユニット (c*=0, a*=0, b*>0) -> nn_rlct の例 2 と一致するはず"
         "\n  期待: lambda = 1, m = 2")
    net = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
    setw(net, [[0.0]], [1.0], [[0.0]], [0.0])
    torch_local_rlct(net, X).print_report()

    head("例 3: 1-2-1、ユニット 1 が全データで不活性 (死んだ ReLU)")
    net = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1))
    setw(net, [[1.0], [1.0]], [3.0, -10.0], [[2.0, 5.0]], [0.0])
    torch_local_rlct(net, X).print_report()

    head("例 4: 学習後の重みを想定 — 小さい重みを zero_tol で 0 に丸める")
    net = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1))
    setw(net, [[1.0], [3e-3]], [0.5, 1.0], [[1.0, 2e-3]], [0.0])
    print("  zero_tol=0 (丸めない -> 正則点に見える):")
    torch_local_rlct(net, X, zero_tol=0.0).local.print_report()
    print("\n  zero_tol=1e-2 (2 番目のユニットの出力重みを 0 とみなす"
          " -> 近くの退化点を見る):")
    torch_local_rlct(net, X, zero_tol=1e-2).local.print_report()

    head("例 5: 深いネット 1-2-2-1、一部のパラメータだけ動かす")
    net = nn.Sequential(nn.Linear(1, 2), nn.ReLU(),
                        nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0], [-1.0]]))
        net[0].bias.copy_(torch.tensor([0.5, 0.5]))
        net[2].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        net[2].bias.copy_(torch.tensor([0.2, 0.1]))
        net[4].weight.copy_(torch.tensor([[1.0, -1.0]]))
        net[4].bias.copy_(torch.tensor([0.0]))
    print("  全パラメータ:")
    torch_local_rlct(net, X).print_report()
    print("\n  最終層だけを変数にする:")
    torch_local_rlct(net, X, params=["4.weight", "4.bias"]).print_report()

    head("例 7: 実際に学習させたネットの局所 RLCT (1-3-1 で 1-1-1 の関数を学習)")
    Xtr = torch.linspace(-2, 2, 24).unsqueeze(1)
    true = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
    with torch.no_grad():
        true[0].weight.copy_(torch.tensor([[1.0]]))
        true[0].bias.copy_(torch.tensor([0.3]))
        true[2].weight.copy_(torch.tensor([[2.0]]))
        true[2].bias.copy_(torch.tensor([-0.5]))
    Y = true(Xtr).detach()
    for seed in range(10):          # 損失がほぼ 0 の局所解が出るまで初期値を変える
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        for _ in range(4000):
            opt.zero_grad()
            loss = ((net(Xtr) - Y) ** 2).mean()
            loss.backward()
            opt.step()
        if float(loss) < 1e-9:
            break
    print(f"  学習後の損失: {float(loss):.3e}  (パラメータ 10 個、素朴には lambda=5)")
    torch_local_rlct(net, Xtr, targets=Y.squeeze(1).tolist(),
                     zero_tol=1e-2, max_denominator=1000).print_report()

    head("例 6: tanh (解析的活性化) — テイラー展開で多項式化")
    net = nn.Sequential(nn.Linear(1, 1), nn.Tanh(), nn.Linear(1, 1))
    setw(net, [[0.0]], [0.0], [[0.0]], [0.0])
    torch_local_rlct(net, X, taylor_order=3).print_report()


if __name__ == "__main__":
    _demo()
