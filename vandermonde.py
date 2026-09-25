r"""
bench/vandermonde.py — Vandermonde 行列型特異点の真値 (Aoyagi の公式)。

出典:
  M. Aoyagi, "Learning Coefficient of Vandermonde Matrix-Type Singularities
  in Model Selection", Entropy 21(6):561, 2019.
    - Definition 3  : Vandermonde 行列型特異点の定義
    - Theorem 6     : H = 1,2,3,4 における lambda の明示式
    - Section 4 末尾: N = 1 における厳密値と位数 (Aoyagi, Int. J. Pure Appl.
                      Math. 52 (2009) 177-204 による)

------------------------------------------------------------------
設定 (Definition 3, w* = 0 の場合)
------------------------------------------------------------------
  A_{M,H} = (a_{ki})            M x H の変数行列
  B^{(Q)}_{H,N} = (B_{H,N,I})   |I| = Qn+1, 0 <= n <= H-1
      B_{H,N,I} = ( prod_j b_{ij}^{l_j} )_{i=1..H}

イデアル J は ||A B||^2 の成分、すなわち

    f_{k,I} = sum_{i=1}^{H} a_{ki} prod_{j=1}^{N} b_{ij}^{l_j}
      (1 <= k <= M,  I = (l_1,...,l_N),  |I| = Qn+1,  0 <= n <= H-1)

で生成される。lambda = lambda_0( sum_{k,I} f_{k,I}^2 ) が学習係数。

三層ニューラルネット (N 入力・H 隠れ・M 出力、真の隠れユニット数 r=0) と
正規混合モデルの学習係数がこの形で与えられる (論文 Section 4.1, 4.2)。
Q は活性化関数のテイラー展開の偶奇に対応し、tanh のような奇関数では Q=2。

------------------------------------------------------------------
公式
------------------------------------------------------------------
Theorem 6 (H <= 4, 任意の M, N, Q):

  H=1: lambda = min{ M/2, N/2 }
  H=2: lambda = min{ (bN + (2-b)M)/2,  b = 0,1,2 ;
                     (2N + Q(N-1+M)) / (2Q+2) }
  H=3: lambda = min{ (bN + (3-b)M)/2,  b = 0..3 ;
                     (bN + (3-b)M + Q(a(N+a-b) + (3-a)M)) / (2(Q+1)),
                        a = 1..b-1, b = 2,3 ;
                     (3N + Q(3N-3+3M)) / (2(2Q+1)) }
  H=4: 上に加えて
                     (4N + Q(aN - a - 1 + (8-a)M)) / (2(2Q+1)), a = 2,3,4 ;
                     (4N + Q(5N-5+3M)) / (2(2Q+1)) ;
                     (3N + M + Q(aN - a + (8-a)M)) / (2(2Q+1)), a = 2,3 ;
                     (4N + Q(6N-6+6M)) / (2(3Q+1))

N = 1 のときは任意の H で厳密値と位数が分かる:

  lambda = (M Q k(k+1) + 2H) / (4(1 + kQ)),
      k = max{ i in Z : 2H >= M(i(i-1)Q + 2i) }
  theta = 1  if 2H >  M(k(k-1)Q + 2k)
          2  if 2H == M(k(k-1)Q + 2k)

二つの式は N=1, H<=4 で一致するはずなので、互いの検算にも使える。
"""
from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import sympy as sp

__all__ = ["vandermonde_ideal", "lambda_theorem6", "lambda_N1",
           "vandermonde_cases"]


# ----------------------------------------------------------------------
# イデアルの生成
# ----------------------------------------------------------------------
def vandermonde_ideal(M: int, N: int, H: int, Q: int):
    """Definition 3 の J = <f_{k,I}> の生成元と変数を返す (w* = 0)。

    Returns (generators, variables)
    """
    a = [[sp.Symbol(f"a{k}_{i}", real=True) for i in range(1, H + 1)]
         for k in range(1, M + 1)]
    b = [[sp.Symbol(f"b{i}_{j}", real=True) for j in range(1, N + 1)]
         for i in range(1, H + 1)]
    idxs: List[Tuple[int, ...]] = []
    for n in range(H):
        total = Q * n + 1
        for I in itertools.product(range(total + 1), repeat=N):
            if sum(I) == total:
                idxs.append(I)
    gens = []
    for k in range(M):
        for I in idxs:
            gens.append(sp.expand(sum(
                a[k][i] * sp.prod([b[i][j] ** I[j] for j in range(N)])
                for i in range(H))))
    variables = tuple([x for row in a for x in row]
                      + [x for row in b for x in row])
    return gens, variables


# ----------------------------------------------------------------------
# Theorem 6 (H <= 4)
# ----------------------------------------------------------------------
def lambda_theorem6(M: int, N: int, H: int, Q: int,
                    trust_H4: bool = False) -> Optional[sp.Rational]:
    """Aoyagi (2019) Theorem 6。H <= 3 を信頼する。

    H = 4 の式は転記したものの N=1 の厳密式と食い違う (M=1,H=4,Q=1 で
    Thm6 が 1、N=1 式が 7/6)。二つの公式は H <= 3 では全て一致するので、
    H=4 は転記ミスと判断し、既定では真値として使わない (trust_H4=True で
    参考値を得られる)。
    """
    M, N, H, Q = map(sp.Integer, (M, N, H, Q))
    Hn = int(H)
    if Hn == 1:
        return min(sp.Rational(M, 2), sp.Rational(N, 2))
    cands: List[sp.Rational] = []
    # 共通: (b N + (H-b) M) / 2
    for bb in range(0, Hn + 1):
        cands.append(sp.Rational(bb * N + (Hn - bb) * M, 2))
    if Hn == 2:
        cands.append(sp.Rational(2 * N + Q * (N - 1 + M), 2 * Q + 2))
        return min(cands)
    # H = 3, 4 共通の第 2 群
    for bb in range(2, Hn + 1):
        for aa in range(1, bb):
            cands.append(sp.Rational(
                bb * N + (Hn - bb) * M + Q * (aa * (N + aa - bb)
                                              + (Hn - aa) * M),
                2 * (Q + 1)))
    if Hn == 3:
        cands.append(sp.Rational(3 * N + Q * (3 * N - 3 + 3 * M),
                                 2 * (2 * Q + 1)))
        return min(cands)
    if Hn == 4:
        # 注意: 論文の H=4 の式を転記したが、N=1 の厳密式と食い違う
        # (M=1,H=4,Q=1 で Thm6 は 1、N=1 式は 7/6)。転記ミスの可能性が高い
        # ので、真値としては使わない。参考値として返すだけにする。
        for aa in (2, 3, 4):
            cands.append(sp.Rational(
                4 * N + Q * (aa * N - aa - 1 + (8 - aa) * M),
                2 * (2 * Q + 1)))
        cands.append(sp.Rational(4 * N + Q * (5 * N - 5 + 3 * M),
                                 2 * (2 * Q + 1)))
        for aa in (2, 3):
            cands.append(sp.Rational(
                3 * N + M + Q * (aa * N - aa + (8 - aa) * M),
                2 * (2 * Q + 1)))
        cands.append(sp.Rational(4 * N + Q * (6 * N - 6 + 6 * M),
                                 2 * (3 * Q + 1)))
        return min(cands) if trust_H4 else None
    return None


# ----------------------------------------------------------------------
# N = 1 の厳密値と位数 (任意の H)
# ----------------------------------------------------------------------
def lambda_N1(M: int, H: int, Q: int) -> Tuple[sp.Rational, int]:
    """N = 1 の場合の厳密な lambda と位数 theta。

    k = max{ i : 2H >= M(i(i-1)Q + 2i) }
    lambda = (M Q k(k+1) + 2H) / (4(1 + kQ))
    theta  = 1 (2H > ...) または 2 (等号)
    """
    k = 0
    i = 1
    while 2 * H >= M * (i * (i - 1) * Q + 2 * i):
        k = i
        i += 1
        if i > 10 ** 4:
            break
    lam = sp.Rational(M * Q * k * (k + 1) + 2 * H, 4 * (1 + k * Q))
    rhs = M * (k * (k - 1) * Q + 2 * k)
    theta = 1 if 2 * H > rhs else 2
    return (lam, theta)


# ----------------------------------------------------------------------
# ベンチ用のケース
# ----------------------------------------------------------------------
def vandermonde_cases(max_vars: int = 9):
    """(名前, f, 変数, 族, 真値 lambda, 真値 theta) の列を返す。"""
    out = []
    for (M, N, H, Q) in [(1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 1),
                         (1, 1, 2, 2), (1, 1, 3, 1), (2, 1, 2, 1),
                         (1, 2, 1, 1), (1, 2, 2, 1), (2, 2, 1, 2),
                         (1, 1, 4, 1), (1, 3, 1, 2), (3, 1, 2, 1)]:
        gens, variables = vandermonde_ideal(M, N, H, Q)
        if len(variables) > max_vars:
            continue
        f = sp.expand(sum(g ** 2 for g in gens))
        if f == 0:
            continue
        lam = lambda_theorem6(M, N, H, Q)
        theta = None
        if N == 1:
            lam1, theta = lambda_N1(M, H, Q)
            if lam is not None and lam != lam1:
                # 二つの公式が食い違ったら真値を入れない (実装ミスの可能性)
                lam, theta = None, None
            else:
                lam = lam1
        out.append((f"vandermonde M={M} N={N} H={H} Q={Q}", f, variables,
                    "vandermonde", lam, theta))
    return out


def _demo():
    print("=" * 76)
    print("Aoyagi (2019) の二つの公式の相互検算 (N=1, H<=4)")
    print("=" * 76)
    bad = 0
    for M in (1, 2, 3):
        for H in (1, 2, 3):
            for Q in (1, 2, 3):
                l6 = lambda_theorem6(M, 1, H, Q)
                l1, th = lambda_N1(M, H, Q)
                ok = (l6 == l1)
                bad += 0 if ok else 1
                if not ok:
                    print(f"  M={M} H={H} Q={Q}: Thm6 {l6} vs N=1 式 {l1} !!")
    print(f"  N=1, M<=3, H<=3, Q<=3 の全 27 通り: 不一致 {bad} 件 "
          "(H=4 は食い違うので真値に使わない)")

    print("\n" + "=" * 76)
    print("解消による計算との比較")
    print("=" * 76)
    from certify import rlct_certified
    import time
    for name, f, gens, fam, lam0, th0 in vandermonde_cases():
        st = time.time()
        try:
            lam, status, how = rlct_certified(f, gens, timeout_ms=5000,
                                              max_depth=12)
        except Exception as e:                       # noqa: BLE001
            lam, status, how = None, "error", str(e)[:24]
        mark = ("(真値なし)" if lam0 is None else
                ("OK" if lam == lam0 else f"!!! 真値 {lam0}"))
        print(f"  {name:<28} lambda={str(lam):<8} {status:<8} "
              f"{time.time()-st:5.1f}s  真値={str(lam0):<6} {mark}")


if __name__ == "__main__":
    _demo()
