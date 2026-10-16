r"""
stage_mlp.py — 2 層 tanh 網の段階ごとの (λ, m): 厳密 / 記号 / HMC の突き合わせ

モデル (Aoyagi の Vandermonde 族と同じ N = M = 1、バイアスなし):

    f(x; a, b) = Σ_{i=1}^{H} a_i tanh(b_i x)

段階 k の点 θ_k: k 個のユニットが学習済み (a_j, b_j ≠ 0、b_j は相異なる)、
残り H − k 個は死んでいる (a_i = b_i = 0)。段階の λ は stage_dln と同じく
「θ_k 自身を真とした realizable 問題の局所 RLCT」と定義する。

------------------------------------------------------------------
段階の漸化式 (予想)
------------------------------------------------------------------
    λ_k = k + λ_V(H − k),     m_k = m_V(H − k)

λ_V(h), m_V(h) は h ユニット・原点の Vandermonde 特異点 (Aoyagi, N=1, Q=2):
k = 0 は Aoyagi の定理そのもの。k ≥ 1 は次の消去の議論による予想:
学習済み k ユニットは 2k 個の正則方向を持ち (寄与 k)、Taylor 係数の列の
2k 次元の部分空間を吸収する。死んだユニットの寄与は、その部分空間を法として
もとの Vandermonde イデアルと同じ形 (各生成元に単数が掛かるだけ) になる。

深層線形網の λ(H, r) = ρ/2 + λ(H − r, 0) と**同じ構造**:
段階のコア = 死んだ部分ネットワークの原点での特異点。

検証:
  (i)  記号: RosolveSingular の `local_rlct_from_ideal` (コアが ⟨ab⟩ 型のときだけ
       信用できる。Vandermonde の h ≥ 2 は解消経路が過大評価することが既知)
  (ii) 数値: Metropolis 補正つき HMC で t·E_t[K] を測り、λ と m を当てはめる
  (iii) 学習: 小さな初期値から学習させ、実際にどの層を通るかを調べる
        (死んだユニットが 0 に留まるか、b が揃う「凝縮」の層を通るか)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["stage_point", "predict", "symbolic", "hmc_mlp", "train_mlp"]


def predict(H: int, k: int):
    """予想 (λ_k, m_k)。k = 0 は Aoyagi の定理。"""
    from bench.vandermonde import lambda_N1
    h = H - k
    if h == 0:
        return Fraction(k), 1
    lam, th = lambda_N1(1, h, 2)
    return Fraction(k) + Fraction(str(lam)), int(th)


def stage_point(H: int, k: int, a_star: Sequence[float], b_star: Sequence[float]):
    a = np.zeros(H)
    b = np.zeros(H)
    a[:k] = a_star[:k]
    b[:k] = b_star[:k]
    return a, b


# ----------------------------------------------------------------------
# 記号
# ----------------------------------------------------------------------
def symbolic(H: int, k: int, a_star, b_star, *, K_terms: Optional[int] = None,
             max_depth: int = 14):
    r"""Taylor 係数のイデアル ⟨Σ_i a_i b_i^{2j+1} − Σ_{l<k} a*_l b*_l^{2j+1}⟩ の局所 RLCT。

    tanh(bx) の x^{2j+1} の係数は c_j b^{2j+1} (c_j ≠ 0 は定数) なので、
    データの支持が開集合を含めば fiber ideal はこれで生成される。
    係数は有理数に丸める (b* が有理数なら厳密)。
    """
    import sympy as sp
    from rlct.models.nn_rlct import local_rlct_from_ideal

    K_terms = K_terms or (2 * H + 2)
    us = sp.symbols(f"u0:{H}", real=True)
    vs = sp.symbols(f"v0:{H}", real=True)
    a0 = [sp.Rational(str(a_star[i])) if i < k else 0 for i in range(H)]
    b0 = [sp.Rational(str(b_star[i])) if i < k else 0 for i in range(H)]
    gens = []
    for j in range(K_terms):
        e = 2 * j + 1
        g = sum((a0[i] + us[i]) * (b0[i] + vs[i]) ** e for i in range(H)) \
            - sum(a0[i] * b0[i] ** e for i in range(H))
        gens.append(sp.expand(g))
    rep = local_rlct_from_ideal(gens, list(us) + list(vs), max_depth=max_depth)
    return rep


# ----------------------------------------------------------------------
# HMC (Metropolis 補正つき)
# ----------------------------------------------------------------------
def hmc_mlp(a0, b0, x, *, ts=(30, 100, 300, 1000, 3000, 10000, 30000),
            gamma: float = 1.0, n_chains: int = 64, n_iter: int = 3000,
            burn: Optional[int] = None, n_leap: int = 20,
            target_acc: float = 0.8, seed: int = 0):
    r"""t·E_t[K] を測る。K(θ) = (1/2n) Σ_j (f(x_j;θ) − f(x_j;θ_k))²。"""
    rng = np.random.default_rng(seed)
    burn = burn if burn is not None else n_iter // 3
    H = len(a0)
    A0 = np.broadcast_to(a0, (n_chains, H)).copy()
    B0 = np.broadcast_to(b0, (n_chains, H)).copy()
    fstar = np.tanh(np.outer(x, b0)) @ a0                    # (n,)

    def KUg(a, b, t):
        Z = b[:, None, :] * x[None, :, None]                 # (B,n,H)
        T = np.tanh(Z)
        f = np.einsum("bnh,bh->bn", T, a)
        E = f - fstar[None, :]
        K = 0.5 * np.mean(E * E, axis=1)
        ga = np.einsum("bn,bnh->bh", E, T) / len(x)
        gb = np.einsum("bn,bnh->bh", E, a[:, None, :] * x[None, :, None]
                       * (1 - T * T)) / len(x)
        da, db = a - A0, b - B0
        U = t * K + 0.5 * gamma * (np.sum(da * da, 1) + np.sum(db * db, 1))
        return K, U, t * ga + gamma * da, t * gb + gamma * db

    out = []
    old = np.seterr(over="ignore", invalid="ignore")
    for t in ts:
        a, b = A0.copy(), B0.copy()
        K, U, ga, gb = KUg(a, b, t)
        curv = max(1.0, float(np.max(np.abs(a0))) ** 2 * float(np.mean(x * x)))
        h = 0.5 / math.sqrt(t * curv + gamma)
        sums = np.zeros(n_chains)
        cnt = 0
        accs = []
        for it in range(n_iter):
            pa = rng.standard_normal(a.shape)
            pb = rng.standard_normal(b.shape)
            H0 = U + 0.5 * (np.sum(pa * pa, 1) + np.sum(pb * pb, 1))
            nl = max(1, int(n_leap * rng.uniform(0.5, 1.5)))
            an, bn = a.copy(), b.copy()
            qa, qb = pa - 0.5 * h * ga, pb - 0.5 * h * gb
            for l in range(nl):
                an, bn = an + h * qa, bn + h * qb
                Kn, Un, gan, gbn = KUg(an, bn, t)
                if l < nl - 1:
                    qa, qb = qa - h * gan, qb - h * gbn
            qa, qb = qa - 0.5 * h * gan, qb - 0.5 * h * gbn
            H1 = Un + 0.5 * (np.sum(qa * qa, 1) + np.sum(qb * qb, 1))
            dH = np.where(np.isfinite(H1 - H0), H1 - H0, np.inf)
            acc = np.log(rng.uniform(size=n_chains)) < -dH
            a[acc], b[acc] = an[acc], bn[acc]
            ga[acc], gb[acc] = gan[acc], gbn[acc]
            K = np.where(acc, Kn, K)
            U = np.where(acc, Un, U)
            if it < burn:
                h *= math.exp(0.5 * (float(acc.mean()) - target_acc))
            else:
                accs.append(float(acc.mean()))
                sums += t * K
                cnt += 1
        pc = sums / max(cnt, 1)
        out.append(dict(t=float(t), lam_hat=float(pc.mean()),
                        lam_std=float(pc.std(ddof=1) / math.sqrt(n_chains)),
                        acc=float(np.mean(accs))))
    np.seterr(**old)
    return out


# ----------------------------------------------------------------------
# 学習: 実際にどの層を通るか
# ----------------------------------------------------------------------
def train_mlp(H: int, a_star, b_star, x, *, init: float = 1e-3,
              lr: float = 0.05, steps: int = 20000, record_every: int = 20,
              seed: int = 0):
    rng = np.random.default_rng(seed)
    a = init * rng.standard_normal(H)
    b = init * rng.standard_normal(H)
    fstar = np.tanh(np.outer(x, b_star)) @ np.asarray(a_star)
    rec = dict(t=[], loss=[], a=[], b=[])
    for it in range(steps + 1):
        T = np.tanh(np.outer(x, b))
        E = T @ a - fstar
        if it % record_every == 0:
            rec["t"].append(it)
            rec["loss"].append(0.5 * float(np.mean(E * E)))
            rec["a"].append(a.copy())
            rec["b"].append(b.copy())
        ga = T.T @ E / len(x)
        gb = ((1 - T * T) * x[:, None]).T @ E * a / len(x)
        a -= lr * ga
        b -= lr * gb
    return rec


def describe_point(a, b, *, tol: float = 0.05):
    """点の層を記述: 活性ユニット (|a|,|b| > tol)、b の重なり (凝縮)。"""
    act = [i for i in range(len(a)) if abs(a[i]) > tol and abs(b[i]) > tol]
    groups: List[List[int]] = []
    for i in act:
        for g in groups:
            if abs(abs(b[g[0]]) - abs(b[i])) < tol * max(1.0, abs(b[i])):
                g.append(i)
                break
        else:
            groups.append([i])
    return dict(active=len(act), distinct=len(groups),
                groups=[[int(i) for i in g] for g in groups])


def hmc_moments(a0, b0, *, J: Optional[int] = None,
                ts=(100, 1000, 10000, 100000), gamma: float = 1.0,
                n_chains: int = 64, n_iter: int = 2000, n_leap: int = 20,
                seed: int = 0):
    r"""tanh 網と**同じ局所イデアル**を多項式で書いたモデルで t·E_t[K] を測る。

    K(θ) = ½ Σ_{j<J} ( m_j(θ) − m_j(θ_k) )²,   m_j = Σ_i a_i b_i^{2j+1}

    tanh(bx) の Taylor 係数は c_j b^{2j+1} なので、全モーメントの生成するイデアルが
    tanh 網のファイバーイデアルと一致する (定数倍は RLCT を変えない)。活性化の
    飽和や Fisher のゆるい方向 (sloppiness) が無いので、特異点の構造だけを測れる。
    """
    import torch
    H = len(a0)
    J = J or (2 * H + 2)
    dt = torch.float64
    a0t = torch.tensor(np.asarray(a0, float), dtype=dt)
    b0t = torch.tensor(np.asarray(b0, float), dtype=dt)
    w0 = torch.cat([a0t, b0t])
    m_star = torch.stack([(a0t * b0t ** (2 * j + 1)).sum() for j in range(J)])
    g = torch.Generator().manual_seed(seed)

    def UG(w, t):
        w = w.detach().requires_grad_(True)
        a, b = w[:, :H], w[:, H:]
        m = torch.stack([(a * b ** (2 * j + 1)).sum(1) for j in range(J)], 1)
        K = 0.5 * ((m - m_star) ** 2).sum(1)
        U = t * K + 0.5 * gamma * ((w - w0) ** 2).sum(1)
        gr, = torch.autograd.grad(U.sum(), w)
        return K.detach(), U.detach(), gr

    out = []
    for t in ts:
        w = w0.repeat(n_chains, 1)
        K, U, G = UG(w, t)
        h = 0.5 / math.sqrt(t * max(1.0, float((a0t ** 2).max() + (b0t ** 2).max())) + gamma)
        sums = torch.zeros(n_chains, dtype=dt)
        cnt, burn, accs = 0, n_iter // 3, []
        for it in range(n_iter):
            p = torch.randn(w.shape, generator=g, dtype=dt)
            H0 = U + 0.5 * (p * p).sum(1)
            nl = max(1, int(n_leap * (0.5 + float(torch.rand(1, generator=g)))))
            wn, q = w.clone(), p - 0.5 * h * G
            for l in range(nl):
                wn = wn + h * q
                Kn, Un, Gn = UG(wn, t)
                if l < nl - 1:
                    q = q - h * Gn
            q = q - 0.5 * h * Gn
            dH = torch.nan_to_num(Un + 0.5 * (q * q).sum(1) - H0, nan=1e30,
                                  posinf=1e30)
            acc = torch.log(torch.rand(n_chains, generator=g, dtype=dt)) < -dH
            w[acc], G[acc] = wn[acc], Gn[acc]
            K, U = torch.where(acc, Kn, K), torch.where(acc, Un, U)
            ar = float(acc.double().mean())
            if it < burn:
                h *= math.exp(0.5 * (ar - 0.8))
            else:
                accs.append(ar)
                sums += t * K
                cnt += 1
        pc = (sums / cnt).numpy()
        out.append(dict(t=float(t), lam_hat=float(pc.mean()),
                        lam_std=float(pc.std(ddof=1) / math.sqrt(n_chains)),
                        acc=float(np.mean(accs))))
    return out


def run_moments(H: int, a_star, b_star, *, ts=(100, 1000, 10000, 100000),
                gamma: float = 1.0, n_iter: int = 2000, verbose: bool = True):
    res = dict(H=H, model="moments", stages=[])
    for k in range(H + 1):
        lam, m = predict(H, k)
        a0, b0 = stage_point(H, k, a_star, b_star)
        pts = hmc_moments(a0, b0, ts=ts, gamma=gamma, n_iter=n_iter)
        # 正則部分の局所化の偏りを閉形式で補正 (Fisher = Jᵀ J、J = ∂m/∂θ)
        J = 2 * H + 2
        Jm = np.zeros((J, 2 * H))
        for j in range(J):
            e = 2 * j + 1
            Jm[j, :H] = b0 ** e
            Jm[j, H:] = a0 * e * b0 ** (e - 1) if e > 1 else a0
        ev = [x for x in np.linalg.eigvalsh(Jm.T @ Jm) if x > 1e-10]
        for p in pts:
            p["corr"] = 0.5 * sum(gamma / (p["t"] * e + gamma) for e in ev)
            p["lam_corr"] = p["lam_hat"] + p["corr"]
        f = fit_m(float(lam), [dict(t=p["t"], lam_hat=p["lam_corr"]) for p in pts])
        row = dict(k=k, lam_pred=str(lam), m_pred=m, hmc=pts,
                   basis="Aoyagi の定理" if k == 0 else
                   ("正則点" if k == H else "予想 (段階の漸化式)"),
                   m_fit=round(f[0], 2) if f else None,
                   r_fit=round(f[1], 2) if f else None)
        res["stages"].append(row)
        if verbose:
            print(f"  [moments] H={H} k={k} 予想 λ={lam} m={m} [{row['basis']}]")
            for p in pts:
                print(f"      t={p['t']:8.0f}  tE[K]={p['lam_hat']:.3f} ± {p['lam_std']:.3f}"
                      f"  補正後 {p['lam_corr']:.3f}  受理 {p['acc']:.2f}")
            print(f"      当てはめ m≈{row['m_fit']} (r={row['r_fit']})", flush=True)
    return res


def fisher(a, b, x):
    """K のθでの Hessian (realizable なので J^T J / n)。"""
    T = np.tanh(np.outer(x, b))
    J = np.hstack([T, (1 - T * T) * x[:, None] * a])
    return J.T @ J / len(x)


def fit_m(lam: float, pts):
    from exp.stage_dln import fit_m as _f
    return _f(lam, pts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--a-star", default="3,-3,3")
    ap.add_argument("--b-star", default="0.4,1.2,4")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ts", default="30,100,300,1000,3000,10000,30000")
    ap.add_argument("--iter", type=int, default=3000)
    ap.add_argument("--no-symbolic", action="store_true")
    ap.add_argument("--moments", action="store_true",
                    help="tanh 網と同じ局所イデアルの多項式モデルで測る")
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--xscale", type=float, default=3.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    H = a.H
    a_star = [float(v) for v in a.a_star.split(",")]
    b_star = [float(v) for v in a.b_star.split(",")]
    if a.moments:
        res = run_moments(H, a_star, b_star, gamma=a.gamma,
                          ts=[float(v) for v in a.ts.split(",")], n_iter=a.iter)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, default=str, ensure_ascii=False, indent=1)
        return 0
    x = a.xscale * np.random.default_rng(1).standard_normal(a.n)
    ts = [float(v) for v in a.ts.split(",")]
    res = dict(H=H, a_star=a_star, b_star=b_star, stages=[])
    print(f"== 2 層 tanh 網 H={H}  学習済みユニット (a*,b*)="
          f"{list(zip(a_star, b_star))[:H]}")
    for k in range(H + 1):
        lam, m = predict(H, k)
        a0, b0 = stage_point(H, k, a_star, b_star)
        row = dict(k=k, lam_pred=str(lam), m_pred=m,
                   basis="Aoyagi の定理" if k == 0 else
                   ("正則点" if k == H else "予想 (段階の漸化式)"))
        if not a.no_symbolic and H - k == 1:
            try:
                rep = symbolic(H, k, a_star, b_star)
                row.update(sym_lam=str(rep.rlct), sym_m=rep.multiplicity,
                           sym_core=[str(g) for g in rep.core_gens][:4])
            except Exception as e:                       # noqa: BLE001
                row["sym_err"] = f"{type(e).__name__}: {str(e)[:80]}"
        pts = hmc_mlp(a0, b0, x, ts=ts, n_iter=a.iter, gamma=a.gamma)
        # 正則部分の局所化の偏り (gamma_bias の閉形式) で補正した値も出す
        F = fisher(a0, b0, x)
        ev = [e for e in np.linalg.eigvalsh(F) if e > 1e-12 * max(1.0, float(np.max(np.abs(F))))]
        for p in pts:
            p["corr"] = 0.5 * sum(a.gamma / (p["t"] * e + a.gamma) for e in ev)
            p["lam_corr"] = p["lam_hat"] + p["corr"]
        row["fisher_ev"] = [float(e) for e in sorted(ev, reverse=True)]
        row["hmc"] = pts
        f = fit_m(float(lam), [dict(t=p["t"], lam_hat=p["lam_corr"]) for p in pts])
        if f:
            row["m_fit"], row["r_fit"] = round(f[0], 2), round(f[1], 2)
        res["stages"].append(row)
        print(f"  k={k} 予想 λ={lam} m={m} [{row['basis']}]"
              + (f" | 記号 λ={row.get('sym_lam')} m={row.get('sym_m')}"
                 if "sym_lam" in row else
                 (f" | 記号 失敗 {row.get('sym_err')}" if "sym_err" in row else "")))
        for p in pts:
            print(f"      HMC t={p['t']:8.0f}  tE[K]={p['lam_hat']:.3f} ± "
                  f"{p['lam_std']:.3f}  補正後 {p['lam_corr']:.3f}  受理 {p['acc']:.2f}")
        if "m_fit" in row:
            print(f"      当てはめ m≈{row['m_fit']} (r={row['r_fit']})", flush=True)
    # 学習
    rec = train_mlp(H, a_star[:H], b_star[:H], x, lr=0.01, steps=40000, record_every=40)
    lo = np.array(rec["loss"])
    traj = []
    for q in range(0, len(lo), max(1, len(lo) // 25)):
        d = describe_point(rec["a"][q], rec["b"][q])
        traj.append(dict(t=rec["t"][q], loss=float(lo[q]), **d))
    res["train"] = traj
    print("  学習の軌跡 (活性ユニット数 / b の相異なる値の数):")
    for r in traj:
        print(f"    step {r['t']:6d} 損失 {r['loss']:.3e}  活性 {r['active']}  "
              f"相異なる b {r['distinct']}  {r['groups']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, default=str, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
