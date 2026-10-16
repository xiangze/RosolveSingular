r"""
transformer_rlct.py
===================

典型的な Transformer 1 層分 (Self-Attention + 残差 + MLP(ReLU) + 残差) の
fiber ideal を作り、nn_rlct.local_rlct_from_ideal に渡せる形にする。

------------------------------------------------------------------
層の定義
------------------------------------------------------------------
入力 X ∈ R^{T x d} (T = 系列長, d = d_model) に対し、ヘッド h = 1..H で

    Q^h = X W_Q^h + b_Q^h,  K^h = X W_K^h + b_K^h,  V^h = X W_V^h + b_V^h
    S^h_{ij} = scale * (Q^h_i . K^h_j)
    A^h_{i.} = softmax_j(S^h_{i.})
    attn_i = concat_h ( sum_j A^h_{ij} V^h_j )          (R^{H d_k})
    Y_i    = X_i + attn_i W_O + b_O                     (残差 1)
    H_i    = ReLU(Y_i W_1 + b_1)
    Z_i    = Y_i + H_i W_2 + b_2                        (残差 2)

LayerNorm は既定では**入れない**。1/sqrt が入って多項式でなくなるうえ、
追加のスケール対称性が増えるため。torch から読むときに LayerNorm が
有効な層を渡した場合は、明示的に layernorm="ignore" を指定しない限り
エラーにする (黙って別のモデルを計算しないため)。

------------------------------------------------------------------
softmax をどう多項式にするか
------------------------------------------------------------------
softmax は多項式ではないが、**分母は theta* の近傍で単元**なので払える。
RLCT はイデアルの不変量で、単元倍で変わらないので、これは近似ではなく
厳密な変形である (指数の打ち切りだけが近似)。

    delta_{ij} = S_{ij}(theta) - S*_{ij}
    E_{ij}     = e*_{ij} * sum_{k<=n} delta_{ij}^k / k!      (多項式)
    D_i        = sum_j E_{ij}                                 (theta* で > 0)
    A_{ij}     = E_{ij} / D_i

ここで e*_{ij} は exp(S*_{ij}) を有理数に丸めたもの。**記号側と数値側で
同じ e*_{ij} を使う**ので、theta = theta* で生成元がぴったり 0 になる
(丸めが残差として漏れない)。

出力の残差 g_i = Z_i(theta) - Z_i(theta*) は 1/D_i を高々 1 次でしか含まない
ので、g~_i = D_i * g_i は多項式。D_i は単元なので <g~> = <g> (局所)。

ReLU はデータ点ごと・ユニットごとに theta* での前活性化の符号でセルを固定する
(nn_rlct.py 冒頭の議論と同じ)。

------------------------------------------------------------------
対称性 (ゲージ固定に使う接ベクトル)
------------------------------------------------------------------
Transformer が特異なのは、パラメータが積でしか現れない場所が多いため。
theta* での軌道の接ベクトルを渡すと local_rlct_from_ideal が商を取る。

  1. QK ゲージ  GL(d_k):  W_Q -> W_Q M, b_Q -> b_Q M,
                          W_K -> W_K M^{-T}, b_K -> b_K M^{-T}
     -> Q_i . K_j が不変。ヘッドごとに d_k^2 次元。
  2. VO ゲージ  GL(d_k):  W_V -> W_V N, b_V -> b_V N, W_O -> N^{-1} W_O
     -> attn W_O が不変。ヘッドごとに d_k^2 次元。
  3. softmax のシフト不変性: b_K -> b_K + c
     -> logit が j によらない量だけずれるので softmax は不変。
        ヘッドごとに d_k 次元。
  4. ReLU の正斉次性: (W_1[:,m], b_1[m]) -> t(...), W_2[m,:] -> W_2[m,:]/t
     -> FFN のユニットごとに 1 次元。

d=d_k=2, H=1, d_ff=2 なら 4 + 4 + 2 + 2 = 12 方向がゲージで落ちる。

------------------------------------------------------------------
使い方
------------------------------------------------------------------
    from transformer_rlct import transformer_local_rlct, random_layer
    star = random_layer(d_model=2, d_k=2, d_ff=2, seed=0)
    rep = transformer_local_rlct(X, star, taylor_order=2)
    rep.print_report()

torch からは torch_rlct.torch_transformer_rlct(layer, X) を使う。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

from rlct.models.nn_rlct import LocalRLCT, local_rlct_from_ideal

__all__ = [
    "TransformerStar", "TransformerIdeal", "TransformerReport",
    "transformer_symbols", "transformer_layer_ideal",
    "transformer_local_rlct", "random_layer",
]


# ----------------------------------------------------------------------
# theta* (数値) の入れ物
# ----------------------------------------------------------------------
@dataclass
class TransformerStar:
    """1 層分の theta* (すべて sympy の有理数)。"""

    d_model: int
    d_k: int
    d_ff: int
    n_heads: int = 1
    Wq: List = field(default_factory=list)   # [h][d_model][d_k]
    bq: List = field(default_factory=list)   # [h][d_k]
    Wk: List = field(default_factory=list)
    bk: List = field(default_factory=list)
    Wv: List = field(default_factory=list)
    bv: List = field(default_factory=list)
    Wo: List = field(default_factory=list)   # [H*d_k][d_model]
    bo: List = field(default_factory=list)   # [d_model]
    W1: List = field(default_factory=list)   # [d_model][d_ff]
    b1: List = field(default_factory=list)   # [d_ff]
    W2: List = field(default_factory=list)   # [d_ff][d_model]
    b2: List = field(default_factory=list)   # [d_model]

    @property
    def d_att(self) -> int:
        return self.n_heads * self.d_k


def random_layer(d_model: int = 2, d_k: int = 2, d_ff: int = 2,
                 n_heads: int = 1, seed: int = 0,
                 degeneracy: str = "none",
                 lo: int = -3, hi: int = 3, den: int = 2) -> TransformerStar:
    """有理数の theta* をランダムに作る。degeneracy で退化を仕込める。

    degeneracy:
      'none'        一般の点
      'dup_head'    2 つのヘッドを同一にする (H >= 2)
      'dup_ffn'     FFN のユニットを 2 つ同一にする (d_ff >= 2)
      'dead_ffn'    FFN のユニットを 1 つ全データで不活性にする
      'zero_ffn'    FFN のユニットを 1 つ完全に 0 にする
      'rank1_v'     W_V を階数 1 にする (VO ゲージが自由に動けない方向)
      'uniform_att' W_Q = 0 -> 注意が一様になる (softmax の logit が縮退)
    """
    rng = random.Random(seed)

    def r():
        return sp.Rational(rng.randint(lo, hi), den)

    st = TransformerStar(d_model=d_model, d_k=d_k, d_ff=d_ff, n_heads=n_heads)
    st.Wq = [[[r() for _ in range(d_k)] for _ in range(d_model)]
             for _ in range(n_heads)]
    st.bq = [[r() for _ in range(d_k)] for _ in range(n_heads)]
    st.Wk = [[[r() for _ in range(d_k)] for _ in range(d_model)]
             for _ in range(n_heads)]
    st.bk = [[r() for _ in range(d_k)] for _ in range(n_heads)]
    st.Wv = [[[r() for _ in range(d_k)] for _ in range(d_model)]
             for _ in range(n_heads)]
    st.bv = [[r() for _ in range(d_k)] for _ in range(n_heads)]
    st.Wo = [[r() for _ in range(d_model)] for _ in range(n_heads * d_k)]
    st.bo = [r() for _ in range(d_model)]
    st.W1 = [[r() for _ in range(d_ff)] for _ in range(d_model)]
    st.b1 = [r() for _ in range(d_ff)]
    st.W2 = [[r() for _ in range(d_model)] for _ in range(d_ff)]
    st.b2 = [r() for _ in range(d_model)]

    if degeneracy == "dup_head" and n_heads >= 2:
        st.Wq[1] = [row[:] for row in st.Wq[0]]
        st.bq[1] = st.bq[0][:]
        st.Wk[1] = [row[:] for row in st.Wk[0]]
        st.bk[1] = st.bk[0][:]
        st.Wv[1] = [row[:] for row in st.Wv[0]]
        st.bv[1] = st.bv[0][:]
    elif degeneracy == "dup_ffn" and d_ff >= 2:
        for i in range(d_model):
            st.W1[i][1] = st.W1[i][0]
        st.b1[1] = st.b1[0]
    elif degeneracy == "dead_ffn" and d_ff >= 1:
        for i in range(d_model):
            st.W1[i][d_ff - 1] = sp.Integer(0)
        st.b1[d_ff - 1] = sp.Integer(-10)       # 全データで不活性
    elif degeneracy == "zero_ffn" and d_ff >= 1:
        for i in range(d_model):
            st.W1[i][d_ff - 1] = sp.Integer(0)
        st.b1[d_ff - 1] = sp.Integer(0)
        for c in range(d_model):
            st.W2[d_ff - 1][c] = sp.Integer(0)
    elif degeneracy == "rank1_v" and d_k >= 2 and d_model >= 1:
        for h in range(n_heads):
            for i in range(d_model):
                st.Wv[h][i][1] = 2 * st.Wv[h][i][0]
            st.bv[h][1] = 2 * st.bv[h][0]
    elif degeneracy == "uniform_att":
        for h in range(n_heads):
            st.Wq[h] = [[sp.Integer(0)] * d_k for _ in range(d_model)]
            st.bq[h] = [sp.Integer(0)] * d_k
    return st


# ----------------------------------------------------------------------
# 記号
# ----------------------------------------------------------------------
def _S(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


def transformer_symbols(st: TransformerStar, prefix: str = "t"):
    """theta = theta* + u の u を作る。戻り値は (記号の入れ物, 変数の一覧)。"""
    H, d, dk, dff = st.n_heads, st.d_model, st.d_k, st.d_ff
    sym: Dict[str, object] = {}
    variables: List[sp.Symbol] = []

    def mat(name, rows, cols):
        M = [[_S(f"{prefix}{name}_{i}_{j}") for j in range(cols)]
             for i in range(rows)]
        for row in M:
            variables.extend(row)
        return M

    def vec(name, n):
        v = [_S(f"{prefix}{name}_{i}") for i in range(n)]
        variables.extend(v)
        return v

    sym["Wq"] = [mat(f"Wq{h}", d, dk) for h in range(H)]
    sym["bq"] = [vec(f"bq{h}", dk) for h in range(H)]
    sym["Wk"] = [mat(f"Wk{h}", d, dk) for h in range(H)]
    sym["bk"] = [vec(f"bk{h}", dk) for h in range(H)]
    sym["Wv"] = [mat(f"Wv{h}", d, dk) for h in range(H)]
    sym["bv"] = [vec(f"bv{h}", dk) for h in range(H)]
    sym["Wo"] = mat("Wo", H * dk, d)
    sym["bo"] = vec("bo", d)
    sym["W1"] = mat("W1", d, dff)
    sym["b1"] = vec("b1", dff)
    sym["W2"] = mat("W2", dff, d)
    sym["b2"] = vec("b2", d)
    return sym, variables


# ----------------------------------------------------------------------
# 前向き計算 (記号と数値を並行)
# ----------------------------------------------------------------------
@dataclass
class TransformerIdeal:
    generators: List[sp.Expr]
    variables: List[sp.Symbol]
    symmetry_vectors: List[Dict[sp.Symbol, sp.Rational]]
    attn_pattern: List[List[List[sp.Rational]]]   # [h][i][j] = A*_{ij}
    relu_pattern: List[List[int]]                 # [i][m] = sign
    interior: bool
    notes: List[str]
    meta: Dict[str, object]
    # 外からヘッド (分類器など) を載せられるように、分母を払った形の
    # 出力と分母をそのまま返す。Z_sym[i][c] / D_sym[i] が本当の Z_i,c。
    Z_sym: List[List[sp.Expr]] = field(default_factory=list)
    D_sym: List[sp.Expr] = field(default_factory=list)
    Z_star: List[List[sp.Rational]] = field(default_factory=list)
    D_star: List[sp.Rational] = field(default_factory=list)
    symbols: Dict[str, object] = field(default_factory=dict)


def _exp_rat(x, max_den: int) -> sp.Rational:
    """exp(x) を有理数に丸める (記号側・数値側で同じ値を使う)。"""
    return sp.Rational(math.exp(float(x))).limit_denominator(max_den)


def transformer_layer_ideal(
    X,
    star: TransformerStar,
    *,
    taylor_order: int = 1,
    scale: Optional[sp.Rational] = None,
    targets=None,
    max_denominator: int = 10 ** 4,
    prefix: str = "t",
) -> TransformerIdeal:
    """1 層分の fiber ideal (多項式) と対称性を作る。

    X       : 入力 (T x d_model のリスト)
    star    : theta*
    targets : 真の出力 (T x d_model)。None なら realizable (theta* の層)。
    scale   : softmax の logit スケール。None なら 1/sqrt(d_k) を有理数に丸める。

    taylor_order の意味と費用 (d_model=2, T=2, exp の分母上限 256 で実測):

      0 : **注意パターンを theta* で凍結**する。E_{ij} = e*_{ij} (定数) なので
          A = A* になり、W_Q, W_K, b_Q, b_K は生成元に現れなくなる
          (= 自由方向。lambda には効かない)。value/output 回路と FFN だけの
          モデルになるので **別のモデルの lambda** だが、格段に安い。
      1 : softmax を 1 次まで展開する。注意もパラメータとして効く。
      2 以上: より高次。項数が急激に増える。

      | order | dk=1 ff=1 | dk=1 ff=2 | dk=2 ff=2 |
      |-------|-----------|-----------|-----------|
      | 0     | 0.1s (9 項) | 1.2s (66 項) | 2.7s (207 項) |
      | 1     | 3.8s (101 項) | 119s | 10 分でも終わらず |

    律速は係数の大きさではなく **生成元の項数** (softmax の展開で密になる)。
    exp の分母上限を 16 まで下げても項数は減らず、時間もほぼ変わらなかった。
    """
    H, d, dk, dff = star.n_heads, star.d_model, star.d_k, star.d_ff
    T = len(X)
    Xr = [[sp.nsimplify(v) for v in row] for row in X]
    if scale is None:
        scale = sp.Rational(1 / math.sqrt(dk)).limit_denominator(max_denominator)
    sym, variables = transformer_symbols(star, prefix)
    notes: List[str] = []

    def P(sym_block, star_block):
        """theta* + u を組み立てる (行列)。"""
        return [[star_block[i][j] + sym_block[i][j]
                 for j in range(len(sym_block[0]))]
                for i in range(len(sym_block))]

    def Pv(sym_block, star_block):
        return [star_block[i] + sym_block[i] for i in range(len(sym_block))]

    # --- パラメータ (記号 = theta* + u / 数値 = theta*) ------------------
    pWq = [P(sym["Wq"][h], star.Wq[h]) for h in range(H)]
    pbq = [Pv(sym["bq"][h], star.bq[h]) for h in range(H)]
    pWk = [P(sym["Wk"][h], star.Wk[h]) for h in range(H)]
    pbk = [Pv(sym["bk"][h], star.bk[h]) for h in range(H)]
    pWv = [P(sym["Wv"][h], star.Wv[h]) for h in range(H)]
    pbv = [Pv(sym["bv"][h], star.bv[h]) for h in range(H)]
    pWo = P(sym["Wo"], star.Wo)
    pbo = Pv(sym["bo"], star.bo)
    pW1 = P(sym["W1"], star.W1)
    pb1 = Pv(sym["b1"], star.b1)
    pW2 = P(sym["W2"], star.W2)
    pb2 = Pv(sym["b2"], star.b2)

    def forward(Wq, bq, Wk, bk, Wv, bv, Wo, bo, W1, b1, W2, b2,
                *, numeric: bool, relu_mask=None, ehat=None):
        """1 層分の前向き。numeric=True なら theta* での値 (有理数) を返す。

        戻り値は (Z, 分母 D_i の列, e*_{ij}, ReLU のマスク, 前活性化)。
        """
        attn = [[[None] * dk for _ in range(H)] for _ in range(T)]
        Ds = []
        ehat_out = []
        for h in range(H):
            Qh = [[sum(Xr[i][a] * Wq[h][a][r] for a in range(d)) + bq[h][r]
                   for r in range(dk)] for i in range(T)]
            Kh = [[sum(Xr[j][a] * Wk[h][a][r] for a in range(d)) + bk[h][r]
                   for r in range(dk)] for j in range(T)]
            Vh = [[sum(Xr[j][a] * Wv[h][a][r] for a in range(d)) + bv[h][r]
                   for r in range(dk)] for j in range(T)]
            Sh = [[sp.expand(scale * sum(Qh[i][r] * Kh[j][r]
                                         for r in range(dk)))
                   for j in range(T)] for i in range(T)]
            if ehat is None:
                # theta* での exp を有理数に丸めて固定する
                eh = [[_exp_rat(Sh[i][j], max_denominator) for j in range(T)]
                      for i in range(T)]
            else:
                eh = ehat[h]
            ehat_out.append(eh)
            # E_{ij} = e*_{ij} * sum_k delta^k / k!
            Sstar = ehat_star_logits[h] if ehat is not None else Sh
            E = []
            for i in range(T):
                row = []
                for j in range(T):
                    delta = sp.expand(Sh[i][j] - Sstar[i][j])
                    poly = sum(delta ** k / sp.factorial(k)
                               for k in range(taylor_order + 1))
                    row.append(sp.expand(eh[i][j] * poly))
                E.append(row)
            Drow = [sp.expand(sum(E[i][j] for j in range(T))) for i in range(T)]
            Ds.append(Drow)
            for i in range(T):
                for r in range(dk):
                    # 分子だけを持ち、分母 D_h[i] は後でまとめて払う
                    attn[i][h][r] = sp.expand(
                        sum(E[i][j] * Vh[j][r] for j in range(T)))

        # Y_i = X_i + attn_i W_O + b_O   (分母は行ごとに共通の積)
        Y = []
        Dprod = []
        for i in range(T):
            den = sp.Integer(1)
            for h in range(H):
                den = sp.expand(den * Ds[h][i])
            Dprod.append(den)
            # ヘッド h の寄与は num/D_h なので、Dprod を掛けると
            # num * prod_{h' != h} D_{h'} になる。式の同一性 (is) ではなく
            # **添字で** 他のヘッドの分母を集める (同じ式になるヘッドがあっても
            # 正しく動くように)。
            others = []
            for h in range(H):
                o = sp.Integer(1)
                for h2 in range(H):
                    if h2 != h:
                        o = sp.expand(o * Ds[h2][i])
                others.append(o)
            row = []
            for c in range(d):
                acc = sp.Integer(0)
                for h in range(H):
                    for r in range(dk):
                        acc = sp.expand(acc + attn[i][h][r] * others[h]
                                        * Wo[h * dk + r][c])
                row.append(sp.expand(Xr[i][c] * den + acc + bo[c] * den))
            Y.append(row)          # Y_i * Dprod_i (分母を払った形)

        # H_i = ReLU(Y_i W_1 + b_1)  : 前活性化も分母を払った形
        pre = [[sp.expand(sum(Y[i][a] * W1[a][m] for a in range(d))
                          + b1[m] * Dprod[i])
                for m in range(dff)] for i in range(T)]
        mask = relu_mask
        if mask is None:
            mask = [[1 if pre[i][m] > 0 else 0 for m in range(dff)]
                    for i in range(T)]
        Hh = [[(pre[i][m] if mask[i][m] else sp.Integer(0))
               for m in range(dff)] for i in range(T)]

        # Z_i = Y_i + H_i W_2 + b_2 : Y と pre は Dprod 倍されているので、
        # H_i W_2 も Dprod 倍。全体を Dprod 倍した形で返す。
        Z = [[sp.expand(Y[i][c]
                        + sum(Hh[i][m] * W2[m][c] for m in range(dff))
                        + b2[c] * Dprod[i])
              for c in range(d)] for i in range(T)]
        return Z, Dprod, ehat_out, mask, pre

    # --- theta* での値 (有理数) -----------------------------------------
    ehat_star_logits = None
    Zs, Dstar, ehat, mask, pre_star = forward(
        star.Wq, star.bq, star.Wk, star.bk, star.Wv, star.bv,
        star.Wo, star.bo, star.W1, star.b1, star.W2, star.b2,
        numeric=True)
    # theta* での logit をそのまま保持しておく (記号側の delta の基準)
    ehat_star_logits = []
    for h in range(H):
        Qh = [[sum(Xr[i][a] * star.Wq[h][a][r] for a in range(d))
               + star.bq[h][r] for r in range(dk)] for i in range(T)]
        Kh = [[sum(Xr[j][a] * star.Wk[h][a][r] for a in range(d))
               + star.bk[h][r] for r in range(dk)] for j in range(T)]
        ehat_star_logits.append(
            [[sp.expand(scale * sum(Qh[i][r] * Kh[j][r] for r in range(dk)))
              for j in range(T)] for i in range(T)])

    if taylor_order == 0:
        notes.append("taylor_order=0: 注意パターンを theta* で凍結しました。"
                     "W_Q, W_K, b_Q, b_K は生成元に現れないので自由方向に"
                     "なります (value/output 回路と FFN だけのモデル)。")
    interior = all(pre_star[i][m] != 0 for i in range(T) for m in range(dff))
    if not interior:
        notes.append("theta* が ReLU セルの境界にあります "
                     "(前活性化がちょうど 0)。不活性側のセルを取ったので、"
                     "得られる lambda は真値の下界です。")

    # --- 記号側 ----------------------------------------------------------
    Zsym, Dsym, _, _, _ = forward(
        pWq, pbq, pWk, pbk, pWv, pbv, pWo, pbo, pW1, pb1, pW2, pb2,
        numeric=False, relu_mask=mask, ehat=ehat)

    # --- 生成元 g~ = D(theta) * Z(theta*) - ... --------------------------
    # Zsym[i][c] は「Dsym_i 倍した Z」、Zs[i][c] は「Dstar_i 倍した Z*」。
    # したがって Z(theta) - Z* = Zsym/Dsym - Zs/Dstar。両辺に Dsym*Dstar を
    # 掛ければ多項式 (Dstar は正の有理数、Dsym は theta* で単元)。
    tgt = None
    if targets is not None:
        tgt = [[sp.nsimplify(v) for v in row] for row in targets]
    gens: List[sp.Expr] = []
    for i in range(T):
        for c in range(d):
            zstar = sp.cancel(Zs[i][c] / Dstar[i]) if tgt is None else tgt[i][c]
            g = sp.expand(Zsym[i][c] - zstar * Dsym[i])
            if g != 0:
                gens.append(g)

    # --- 対称性 -----------------------------------------------------------
    symv: List[Dict[sp.Symbol, sp.Rational]] = []
    for h in range(H):
        # (1) QK ゲージ GL(d_k)
        for r in range(dk):
            for s in range(dk):
                vec: Dict[sp.Symbol, sp.Rational] = {}
                for i in range(d):
                    vec[sym["Wq"][h][i][s]] = star.Wq[h][i][r]
                    vec[sym["Wk"][h][i][r]] = -star.Wk[h][i][s]
                vec[sym["bq"][h][s]] = vec.get(sym["bq"][h][s], 0) + star.bq[h][r]
                vec[sym["bk"][h][r]] = vec.get(sym["bk"][h][r], 0) - star.bk[h][s]
                if any(v != 0 for v in vec.values()):
                    symv.append(vec)
        # (2) VO ゲージ GL(d_k)
        for r in range(dk):
            for s in range(dk):
                vec = {}
                for i in range(d):
                    vec[sym["Wv"][h][i][s]] = star.Wv[h][i][r]
                vec[sym["bv"][h][s]] = vec.get(sym["bv"][h][s], 0) + star.bv[h][r]
                for c in range(d):
                    vec[sym["Wo"][h * dk + r][c]] = -star.Wo[h * dk + s][c]
                if any(v != 0 for v in vec.values()):
                    symv.append(vec)
        # (3) softmax のシフト不変性: b_K -> b_K + c
        for r in range(dk):
            symv.append({sym["bk"][h][r]: sp.Integer(1)})
    # (4) ReLU の正斉次性 (FFN のユニットごと)
    for m in range(dff):
        vec = {}
        for i in range(d):
            vec[sym["W1"][i][m]] = star.W1[i][m]
        vec[sym["b1"][m]] = star.b1[m]
        for c in range(d):
            vec[sym["W2"][m][c]] = -star.W2[m][c]
        if any(v != 0 for v in vec.values()):
            symv.append(vec)

    Astar = []
    for h in range(H):
        row_h = []
        for i in range(T):
            den = sum(ehat[h][i][j] for j in range(T))
            row_h.append([sp.Rational(ehat[h][i][j], 1) / den
                          for j in range(T)])
        Astar.append(row_h)

    return TransformerIdeal(
        generators=gens,
        variables=variables,
        symmetry_vectors=symv,
        attn_pattern=Astar,
        relu_pattern=mask,
        interior=interior,
        notes=notes,
        meta=dict(T=T, d_model=d, d_k=dk, d_ff=dff, n_heads=H,
                  taylor_order=taylor_order, scale=scale,
                  n_params=len(variables), n_gens=len(gens),
                  n_symmetries=len(symv)),
        Z_sym=Zsym, D_sym=Dsym, Z_star=Zs, D_star=Dstar, symbols=sym,
    )


# ----------------------------------------------------------------------
# 報告
# ----------------------------------------------------------------------
@dataclass
class TransformerReport:
    local: LocalRLCT
    ideal: TransformerIdeal

    def print_report(self, with_history: bool = False) -> None:
        m = self.ideal.meta
        print("############ Transformer 1 層の局所 RLCT ############")
        print(f"  T={m['T']}, d_model={m['d_model']}, d_k={m['d_k']}, "
              f"heads={m['n_heads']}, d_ff={m['d_ff']}, "
              f"softmax の打ち切り次数={m['taylor_order']}")
        print(f"  パラメータ {m['n_params']}, 生成元 {m['n_gens']}, "
              f"対称性 {m['n_symmetries']} 方向")
        print("  theta* での注意重み A*:")
        for h, Ah in enumerate(self.ideal.attn_pattern):
            for i, row in enumerate(Ah):
                print(f"    head{h} row{i}: "
                      + " ".join(f"{float(v):.3f}" for v in row))
        print("  ReLU の活性化パターン (行=系列位置, 列=FFN ユニット):")
        for i, row in enumerate(self.ideal.relu_pattern):
            print(f"    pos{i}: " + " ".join("+" if v else "-" for v in row))
        for n in self.ideal.notes:
            print(f"  [note] {n}")
        print()
        self.local.print_report(with_history=with_history)


def transformer_local_rlct(X, star: TransformerStar, *, taylor_order: int = 1,
                           targets=None, scale=None, verbose: bool = False,
                           **resolve_kwargs) -> TransformerReport:
    """Transformer 1 層の theta* における局所 RLCT。"""
    ideal = transformer_layer_ideal(X, star, taylor_order=taylor_order,
                                    targets=targets, scale=scale)
    loc = local_rlct_from_ideal(ideal.generators, ideal.variables,
                                symmetry_vectors=ideal.symmetry_vectors,
                                verbose=verbose, **resolve_kwargs)
    for n in ideal.notes:
        loc.notes.append(n)
    return TransformerReport(local=loc, ideal=ideal)


# ----------------------------------------------------------------------
# デモ
# ----------------------------------------------------------------------
def _demo():
    import time

    X = [[sp.Rational(1), sp.Rational(0)],
         [sp.Rational(0), sp.Rational(1)],
         [sp.Rational(1, 2), sp.Rational(-1)]]
    for deg in ("none", "zero_ffn", "dead_ffn", "uniform_att"):
        star = random_layer(d_model=2, d_k=2, d_ff=2, n_heads=1, seed=0,
                            degeneracy=deg)
        t = time.time()
        ideal = transformer_layer_ideal(X, star, taylor_order=1)
        print(f"{deg:<12} パラメータ {ideal.meta['n_params']}, "
              f"生成元 {ideal.meta['n_gens']}, "
              f"対称性 {ideal.meta['n_symmetries']}, "
              f"構築 {time.time()-t:.2f}s")


if __name__ == "__main__":
    _demo()
