r"""
stage_dln.py — 学習の段階ごとの厳密 (λ, m) と数値 λ̂ の突き合わせ (段階 1: 深層線形網)

小さな初期値から学習させた深層線形網は、鞍点から鞍点へと階数を 1 つずつ
上げながら学習する (saddle-to-saddle)。各プラトーで止まっている点 θ_k は、
積 A_1⋯A_L の階数が k の「部分ネットワーク」(各層も階数 k) に乗っている。

------------------------------------------------------------------
段階の λ の定義
------------------------------------------------------------------
θ_k は真の損失の最小点ではない (鞍点) ので、通常の局所 λ は定義されない
(K − K(θ_k) が負になる方向がある)。そこで

    λ_k := θ_k のモデル自身を真とした (realizable に付け替えた) 問題の
           局所 RLCT,  K_k(w) = E_x ‖f(x;w) − f(x;θ_k)‖² / 2

と定義する。これは `llc_estimate.realizable_targets` がしていることと
同じで、数値側 (SGLD) もこの量を推定する。θ_k が各層の階数 k の層
(閉包順序で最小の軌道) に乗っていれば、この点の近傍はすべての層と交わるので
局所 RLCT = 大域 RLCT = Aoyagi の λ(H, k)。

------------------------------------------------------------------
3 つの経路
------------------------------------------------------------------
* 厳密 1: Aoyagi の閉形式 `dln.lambda_dln(H, k)` → (λ_k, m_k)
* 厳密 2: `strata_closer.lambda_dln_strata(H, k)` (対称性の層の公式、独立な実装)
* 数値  : `llc_estimate.llc_sgld` を θ_k の層への射影で実行。温度 t = nβ を
          複数取り、λ̂(t) = λ − Σ_{i<m} 1/(log t − r_i) の形から m も読む

データの階数 (L2): 入力が q 次元部分空間に乗っていれば、入力幅を q に
置き換えた λ(H_1,…,H_L, q; k) になるはず (第 1 層の V^⊥ 成分は平坦)。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["train_dln", "detect_stages", "project_to_stratum",
           "stage_exact", "stage_sgld", "fit_m"]


# ----------------------------------------------------------------------
# 学習
# ----------------------------------------------------------------------
def make_teacher(H_out: int, H_in: int, svals: Sequence[float], seed: int = 0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((H_out, H_out)))
    V, _ = np.linalg.qr(rng.standard_normal((H_in, H_in)))
    S = np.zeros((H_out, H_in))
    for i, s in enumerate(svals):
        S[i, i] = s
    return U @ S @ V.T


def make_data(n: int, H_in: int, q: Optional[int] = None, seed: int = 1):
    """n 点の入力。q を与えると q 次元のランダム部分空間に乗せる (データ階数 q)。"""
    rng = np.random.default_rng(seed)
    if q is None or q >= H_in:
        return rng.standard_normal((n, H_in))
    B, _ = np.linalg.qr(rng.standard_normal((H_in, q)))
    return rng.standard_normal((n, q)) @ B.T


def _prod(Ws):
    """Ws = [A_1 (出力側), …, A_L (入力側)]、積 A_1⋯A_L。"""
    P = Ws[0]
    for W in Ws[1:]:
        P = P @ W
    return P


def train_dln(H: Sequence[int], A_star, X, *, init: float = 1e-4,
              lr: float = 0.05, steps: int = 6000, record_every: int = 10,
              seed: int = 2):
    """全バッチ勾配降下。H = (H_1 出力, …, H_{L+1} 入力)。

    損失 L(W) = (1/2n) Σ ‖P x_i − A* x_i‖² (Watanabe の規約)。
    返り値: dict(t, loss, ckpt=[Ws], svals=[積の特異値])
    """
    rng = np.random.default_rng(seed)
    L = len(H) - 1
    Ws = [init * rng.standard_normal((H[s], H[s + 1])) for s in range(L)]
    n = X.shape[0]
    C = X.T @ X / n                                  # 入力の 2 次モーメント
    rec = dict(t=[], loss=[], ckpt=[], svals=[])
    for it in range(steps + 1):
        P = _prod(Ws)
        E = P - A_star
        if it % record_every == 0:
            rec["t"].append(it)
            rec["loss"].append(0.5 * float(np.trace(E @ C @ E.T)))
            rec["ckpt"].append([W.copy() for W in Ws])
            rec["svals"].append(np.linalg.svd(P, compute_uv=False))
        G = E @ C                                    # ∂L/∂P
        grads = []
        for s in range(L):
            left = _prod(Ws[:s]) if s > 0 else np.eye(H[0])
            right = _prod(Ws[s + 1:]) if s < L - 1 else np.eye(H[-1])
            grads.append(left.T @ G @ right.T)
        for s in range(L):
            Ws[s] -= lr * grads[s]
    return rec


# ----------------------------------------------------------------------
# 段階の検出と層への射影
# ----------------------------------------------------------------------
def detect_stages(rec, svals_star: Sequence[float], *, frac: float = 0.5,
                  min_len: int = 5):
    r"""プラトーを切り出す。

    各記録点で「学習済みの階数」k = #{σ_i(P) > frac·σ_i*} を数え、
    k が一定の区間をプラトーとする。区間の中央の記録点を代表にする。
    (frac=0.5 は各モードの S 字の中点。中点を跨いだ時刻が遷移)
    """
    s_star = np.array(sorted(svals_star, reverse=True))
    ks = []
    for sv in rec["svals"]:
        sv = np.sort(sv)[::-1][:len(s_star)]
        ks.append(int(np.sum(sv > frac * s_star[:len(sv)])))
    stages = []
    i = 0
    while i < len(ks):
        j = i
        while j + 1 < len(ks) and ks[j + 1] == ks[i]:
            j += 1
        if j - i + 1 >= min_len:
            # 区間内で損失の相対変化が最も遅い点 (= 鞍点に最も近い点) を代表にする
            lo = rec["loss"]
            speed = [abs(lo[min(q + 1, len(lo) - 1)] - lo[max(q - 1, 0)])
                     / (abs(lo[q]) + 1e-300) for q in range(i, j + 1)]
            mid = i + int(np.argmin(speed))
            stages.append(dict(k=ks[i], i0=i, i1=j, mid=mid,
                               t_mid=rec["t"][mid], loss=rec["loss"][mid]))
        i = j + 1
    return stages, ks


def project_to_stratum(Ws, k: int):
    r"""各層を階数 k に切り詰める (SVD)。最小の層 (各層・積とも階数 k) に乗せる。

    返り値: (射影後の Ws, 診断 dict)。診断には射影の大きさと、
    射影後の部分積の階数プロファイル (全部 k なら最小の層) を入れる。
    """
    out = []
    moved = 0.0
    for W in Ws:
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        S2 = S.copy()
        S2[k:] = 0.0
        Wp = (U * S2) @ Vt
        moved = max(moved, float(np.linalg.norm(W - Wp)))
        out.append(Wp)
    L = len(out)
    prof = {}
    for a in range(L):
        for b in range(a + 1, L + 1):
            M = _prod(out[a:b])
            sv = np.linalg.svd(M, compute_uv=False)
            prof[(a, b)] = int(np.sum(sv > 1e-8 * max(1.0, sv[0] if len(sv) else 1)))
    minimal = all(v == k for v in prof.values())
    return out, dict(moved=moved, rank_profile=prof, minimal=minimal)


# ----------------------------------------------------------------------
# 厳密
# ----------------------------------------------------------------------
def stage_exact(H: Sequence[int], k: int, *, q: Optional[int] = None,
                strata: bool = True):
    """(λ, m) を Aoyagi と層の公式の 2 経路で。q はデータの階数 (L2)。"""
    from bench.dln import lambda_dln
    Hq = list(H)
    if q is not None and q < Hq[-1]:
        Hq[-1] = q
    lam, th = lambda_dln(Hq, k)
    rec = dict(H_eff=tuple(Hq), lam=Fraction(str(lam)), m=int(th))
    if strata:
        from rlct.closers.strata_closer import lambda_dln_strata
        lam2, m2, info = lambda_dln_strata(Hq, k)
        rec.update(lam_strata=lam2, m_strata=int(m2), rho=info["rho"])
    return rec


# ----------------------------------------------------------------------
# 数値 (SGLD)
# ----------------------------------------------------------------------
def _torch_model(Ws):
    import torch
    import torch.nn as nn
    mods = []
    for W in reversed(Ws):                     # 入力側から
        lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(W, dtype=torch.float64))
        mods.append(lin)
    return nn.Sequential(*mods).double()


def stage_sgld(Ws, X, *, n_effs: Sequence[int] = (256, 1024, 4096),
               gamma: float = 1.0, c_eps: float = 0.01, n_steps: int = 6000,
               chains: int = 4, seed: int = 0, precondition: str = "none"):
    r"""温度 t = n_eff/log n_eff ごとに λ̂ を出す。ε = c_eps / (t · 曲率)。

    曲率は θ_k での K の Hessian の最大固有値を、積の作用素ノルムから
    粗く見積もる: ‖∂P/∂A_s‖² ≤ Π_{s'≠s} ‖A_{s'}‖² · ‖C‖。
    """
    import torch
    from exp.llc_estimate import llc_sgld

    Xt = torch.tensor(X, dtype=torch.float64)
    C = X.T @ X / X.shape[0]
    norms = [np.linalg.norm(W, 2) for W in Ws]
    curv = max(np.prod([norms[j] ** 2 for j in range(len(Ws)) if j != s])
               for s in range(len(Ws))) * np.linalg.norm(C, 2)
    curv = max(curv, 1.0)
    out = []
    for ne in n_effs:
        t = ne / math.log(ne)
        eps = c_eps / (t * curv + gamma)
        model = _torch_model(Ws)
        t0 = time.time()
        r = llc_sgld(model, Xt, loss="regression", gamma=gamma, eps=eps,
                     n_steps=n_steps, n_eff=ne, chains=chains, seed=seed,
                     precondition=precondition)
        out.append(dict(n_eff=ne, t=t, eps=eps, lam_hat=r.lam_hat,
                        lam_std=r.lam_std, drift=r.split_drift,
                        diverged=r.diverged, sec=round(time.time() - t0, 1)))
    return out


def _bprod(Ws):
    P = Ws[0]
    for W in Ws[1:]:
        P = P @ W
    return P


def stage_hmc(Ws_star, C, *, ts: Sequence[float] = (30, 100, 300, 1000, 3000),
              gamma: float = 1.0, n_chains: int = 64, n_iter: int = 1500,
              burn: int = 500, n_leap: int = 20, target_acc: float = 0.8,
              seed: int = 0):
    r"""Metropolis 補正つき HMC で t·E_t[K] を厳密な事後から測る (離散化の偏りなし)。

    事後 ∝ exp( −t K(w) − (γ/2)‖w − w*‖² ),  K(w) = ½ tr( E C Eᵀ ),
    E = A_1⋯A_L − P*。深層線形網ではデータは入力の 2 次モーメント C を通してしか
    入らないので、K は n 点の和を取らずに厳密に評価できる。
    n_chains 本の鎖を一括で回し、鎖ごとの平均のばらつきから誤差棒を出す。
    ステップ幅は burn-in 中に受理率 target_acc に合わせて調整する。
    """
    rng = np.random.default_rng(seed)
    L = len(Ws_star)
    Pstar = _prod(Ws_star)
    shapes = [W.shape for W in Ws_star]
    W0 = [np.broadcast_to(W, (n_chains,) + W.shape).copy() for W in Ws_star]

    def K_and_grad(Ws, t):
        P = _bprod(Ws)
        E = P - Pstar
        EC = E @ C
        K = 0.5 * np.einsum("bij,bij->b", EC, E)
        grads = []
        for s in range(L):
            G = EC
            if s > 0:
                left = _bprod(Ws[:s])
                G = np.swapaxes(left, 1, 2) @ G
            if s < L - 1:
                right = _bprod(Ws[s + 1:])
                G = G @ np.swapaxes(right, 1, 2)
            grads.append(t * G + gamma * (Ws[s] - W0[s]))
        U = t * K + 0.5 * gamma * sum(
            np.einsum("bij,bij->b", Ws[s] - W0[s], Ws[s] - W0[s]) for s in range(L))
        return K, U, grads

    out = []
    old = np.seterr(over="ignore", invalid="ignore")
    for t in ts:
        Ws = [w.copy() for w in W0]
        K, U, g = K_and_grad(Ws, t)
        # 初期ステップ幅: 最大曲率 ~ t·(積の他の層のノルム²)·‖C‖ + γ
        nrm = [np.linalg.norm(W, 2) for W in Ws_star]
        curv = max(np.prod([nrm[j] ** 2 for j in range(L) if j != s])
                   for s in range(L)) * np.linalg.norm(C, 2)
        h = 0.5 / math.sqrt(t * max(curv, 1e-2) + gamma)
        acc_hist = []
        sums = np.zeros(n_chains)
        cnt = 0
        for it in range(n_iter):
            p = [rng.standard_normal(w.shape) for w in Ws]
            H0 = U + 0.5 * sum(np.einsum("bij,bij->b", q, q) for q in p)
            nl = max(1, int(n_leap * rng.uniform(0.5, 1.5)))
            Wn = [w.copy() for w in Ws]
            pn = [q - 0.5 * h * gg for q, gg in zip(p, g)]
            for l in range(nl):
                Wn = [w + h * q for w, q in zip(Wn, pn)]
                Kn, Un, gn = K_and_grad(Wn, t)
                if l < nl - 1:
                    pn = [q - h * gg for q, gg in zip(pn, gn)]
            pn = [q - 0.5 * h * gg for q, gg in zip(pn, gn)]
            H1 = Un + 0.5 * sum(np.einsum("bij,bij->b", q, q) for q in pn)
            dH = H1 - H0
            dH = np.where(np.isfinite(dH), dH, np.inf)
            acc = np.log(rng.uniform(size=n_chains)) < -dH
            for s in range(L):
                Ws[s][acc] = Wn[s][acc]
                g[s][acc] = gn[s][acc]
            K = np.where(acc, Kn, K)
            U = np.where(acc, Un, U)
            a = float(acc.mean())
            if it < burn:
                h *= math.exp(0.5 * (a - target_acc))      # 受理率に合わせる
            else:
                acc_hist.append(a)
                sums += t * K
                cnt += 1
        per_chain = sums / max(cnt, 1)
        lam = float(per_chain.mean())
        se = float(per_chain.std(ddof=1) / math.sqrt(n_chains))
        out.append(dict(t=float(t), lam_hat=lam, lam_std=se, h=h,
                        acc=float(np.mean(acc_hist)) if acc_hist else float("nan")))
    np.seterr(**old)
    return out


def fit_m(lam: float, pts: Sequence[Dict]):
    r"""λ が既知のとき、D(t) = λ − λ̂(t) を (m−1)/(log t − r) で当てはめる。

    返り値 (m_fit, r_fit, 残差)。m−1 ≥ 0 に制約。点が 2 個未満なら None。
    D が負 (λ̂ > λ) の点は推定器の上振れ (局所化・離散化) なので、そのまま使う
    (当てはめが m−1 ≈ 0 を返す)。
    """
    pts = [p for p in pts if math.isfinite(p["lam_hat"])]
    if len(pts) < 2:
        return None
    Ls = np.array([math.log(p["t"]) for p in pts])
    D = np.array([lam - p["lam_hat"] for p in pts])
    best = None
    for r in np.linspace(-15, min(Ls) - 0.2, 400):
        x = 1.0 / (Ls - r)
        a = max(0.0, float(x @ D / (x @ x)))
        res = float(np.sum((D - a * x) ** 2))
        if best is None or res < best[2]:
            best = (1.0 + a, float(r), res)
    return best


# ----------------------------------------------------------------------
# 実行
# ----------------------------------------------------------------------
def run(H: Sequence[int], svals: Sequence[float], *, q: Optional[int] = None,
        n: int = 256, n_effs=(1024,), steps: int = 6000,
        lr: float = 0.05, init: float = 1e-4, n_steps: int = 6000,
        chains: int = 4, gamma: float = 1.0, sgld: bool = True,
        hmc: bool = True, ts=(30, 100, 300, 1000, 3000, 10000, 30000),
        hmc_iter: int = 3000, init_scale: Optional[float] = None,
        seed: int = 0, verbose: bool = True):
    H = list(H)
    A_star = make_teacher(H[0], H[-1], svals, seed=seed)
    X = make_data(n, H[-1], q=q, seed=seed + 1)
    rec = train_dln(H, A_star, X, init=init, lr=lr, steps=steps, seed=seed + 2)
    C = X.T @ X / X.shape[0]
    stages, ks = detect_stages(rec, svals)
    out = dict(H=H, svals=list(svals), q=q, n=n, stages=[],
               loss_curve=dict(t=rec["t"], loss=rec["loss"], k=ks))
    if verbose:
        print(f"== H={tuple(H)} 教師の特異値 {list(svals)}"
              + (f" データ階数 q={q}" if q else "") + f"  段階 {len(stages)}")
    for st in stages:
        k = st["k"]
        Ws = rec["ckpt"][st["mid"]]
        Wp, diag = project_to_stratum(Ws, k)
        ex = stage_exact(H, k, q=q)
        row = dict(k=k, t_mid=st["t_mid"], loss=st["loss"],
                   plateau=(rec["t"][st["i0"]], rec["t"][st["i1"]]),
                   moved=diag["moved"], minimal=diag["minimal"],
                   lam=str(ex["lam"]), m=ex["m"],
                   lam_strata=str(ex.get("lam_strata")),
                   m_strata=ex.get("m_strata"), rho=ex.get("rho"))
        if hmc:
            pts = stage_hmc(Wp, C, ts=ts, gamma=gamma, n_iter=hmc_iter,
                            burn=hmc_iter // 3, seed=seed)
            row["hmc"] = pts
            f = fit_m(float(ex["lam"]), pts)
            if f:
                row["m_fit"], row["r_fit"] = round(f[0], 2), round(f[1], 2)
        if sgld:
            row["sgld"] = stage_sgld(Wp, X, n_effs=n_effs, gamma=gamma,
                                     c_eps=0.3, n_steps=n_steps,
                                     chains=chains, seed=seed)
        out["stages"].append(row)
        if verbose:
            s = (f"  k={k} 区間 {row['plateau']} 損失 {row['loss']:.3g} "
                 f"射影 {row['moved']:.1e} 最小層 {row['minimal']} | "
                 f"厳密 λ={row['lam']} m={row['m']} (層の公式 "
                 f"{row['lam_strata']}, {row['m_strata']}; ρ/2={Fraction(row['rho'] or 0, 2)})")
            print(s, flush=True)
            for p in row.get("hmc", []):
                print(f"      HMC  t={p['t']:8.0f}  tE[K]={p['lam_hat']:.3f} ± "
                      f"{p['lam_std']:.3f}  受理 {p['acc']:.2f}")
            for p in row.get("sgld", []):
                print(f"      SGLD t={p['t']:8.1f}  λ̂={p['lam_hat']:.3f} ± "
                      f"{p['lam_std']:.3f}  drift {p['drift']:.3f}"
                      f"{'  発散' if p['diverged'] else ''}  ({p['sec']}s)")
            if "m_fit" in row:
                print(f"      当てはめ m≈{row['m_fit']} (r={row['r_fit']})  厳密 m={row['m']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", default="3,3,3")
    ap.add_argument("--svals", default="4,1,0.25")
    ap.add_argument("--q", type=int, default=None)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--n-effs", default="1024")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--sgld-steps", type=int, default=8000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--no-sgld", action="store_true")
    ap.add_argument("--no-hmc", action="store_true")
    ap.add_argument("--init", type=float, default=1e-4)
    ap.add_argument("--ts", default="30,100,300,1000,3000,10000,30000")
    ap.add_argument("--hmc-iter", type=int, default=3000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = run([int(x) for x in a.H.split(",")],
              [float(x) for x in a.svals.split(",")], q=a.q, n=a.n,
              n_effs=[int(x) for x in a.n_effs.split(",")], steps=a.steps,
              lr=a.lr, n_steps=a.sgld_steps, chains=a.chains, gamma=a.gamma,
              sgld=not a.no_sgld, hmc=not a.no_hmc, init=a.init,
              ts=[float(x) for x in a.ts.split(",")], hmc_iter=a.hmc_iter)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, default=str, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
