r"""
llc_compare.py
==============

**厳密 λ と SGLD による λ̂ の突き合わせ。**

`cnn_sweep` と同じ学習済み θ\* に対して

  - `torch_rlct.torch_local_rlct` で**厳密な (λ, m)**
  - `llc_estimate.llc_sgld` で**数値推定 λ̂**

を出し、同じ部分空間・同じ損失・同じ realizable 設定で比べる。

------------------------------------------------------------------
比較が成立する条件 (詳細は `llc_estimate` の docstring)
------------------------------------------------------------------
1. 教師データは θ\* の出力 (realizable)
2. SGLD も**厳密側と同じ座標だけ**を動かす
3. L_n は負の対数尤度 (回帰は出力次元を和、係数 1/2)
4. データ点

   - `--sgld-n 0` なら**厳密側と同じ点**を使う。イデアルが同一なので
     推定対象は厳密 λ そのもの。ただし n が小さいと β\*=1/log n の漸近が
     効かず、局所化による偏り (`llc_estimate.gamma_bias`) が大きい
   - `--sgld-n N` (N > 0) なら同じ分布から N 点を取る。**λ がデータ点数で
     飽和していることを確認した場合のみ**正当 (`--check-saturation`)。
     飽和していなければ別のイデアルの λ を推定していることになる

------------------------------------------------------------------
出力の読み方
------------------------------------------------------------------
厳密 λ は**部分空間に制限した下界**、λ̂ も同じ部分空間に制限してあるので、
両者は同じ量を指している。したがって差は

  (a) SGLD の未収束・離散化
  (b) 局所化 γ による系統的な偏り (正則モデルでは閉形式で分かる)
  (c) 有限 n による漸近の誤差

のいずれか。(b) は `gamma_bias` で切り分けられる。

使い方
------
    python llc_compare.py --depths 1,2 --acts relu,tanh --losses both
    python llc_compare.py --sgld-n 256 --gammas 1,10 --epss 2e-4,5e-4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Dict, List, Optional, Sequence

__all__ = ["compare_one", "run_grid", "summarize"]


def _data_for(ds, n_data: int):
    from train_rlct import _stratified
    return ds.X[_stratified(ds, n_data)]


def inject_degeneracy(model, kind: str) -> List[str]:
    r"""学習済み θ\* に**退化を仕込む**。返り値は仕込んだ内容の説明。

    なぜ必要か: 動かす変数が 10 本で データ点が十分にあると、制限した
    部分空間では θ\* が**正則点**になり λ = d/2 = 5 と自明に決まってしまう
    (実測: n_data=16 で自由方向 0、正則方向 10、コア空)。正則点は推定器が
    最も得意な場所なので、それだけでは校正にならない。

    **realizable 性は壊れない。**教師データを θ\* の出力そのものとして作る
    (`realizable_targets`) ので、θ\* をどう動かしても**その新しい θ\* が
    損失 0 の大域解**になる。したがって退化を仕込んでから教師を作れば、
    特異点における λ と λ̂ を正当に比べられる。

    kind:
      'none'         何もしない (正則点の対照実験)
      'dup_channel'  第 1 畳み込みの 2 つの出力チャネルを同一にする
                     (置換対称性が立ち、ファイバーが特異になる)
      'dead_channel' 第 1 畳み込みのチャネル 1 を 0 にする (死んだユニット)
      'half_scale'   チャネル 1 の重みを 1/2 にし、下流の対応する重みを 2 倍
                     (ReLU の正斉次性によるスケール対称性。出力は不変)
    """
    import torch
    import torch.nn as nn

    notes: List[str] = []
    convs = [m for m in model if isinstance(m, nn.Conv2d)]
    lins = [m for m in model if isinstance(m, nn.Linear)]
    if kind == "none" or not convs:
        return notes
    c0 = convs[0]
    with torch.no_grad():
        if kind == "dup_channel":
            if c0.weight.shape[0] < 2:
                return ["チャネルが 1 本なので dup_channel は適用せず"]
            c0.weight[1].copy_(c0.weight[0])
            c0.bias[1].copy_(c0.bias[0])
            notes.append("第 1 conv の出力チャネル 0 と 1 を同一にした")
        elif kind == "dead_channel":
            c0.weight[1].zero_()
            # バイアスも負にして、どの入力でも ReLU が立たないようにする
            c0.bias[1].fill_(-1.0)
            notes.append("第 1 conv のチャネル 1 を死んだユニットにした")
        elif kind == "half_scale":
            c0.weight[0].mul_(0.5)
            c0.bias[0].mul_(0.5)
            if lins:
                # Flatten 後の並びは (c, h, w) で c=0 が先頭ブロック
                W = lins[0].weight          # (out, c*h*w)
                blk = W.shape[1] // c0.weight.shape[0]
                W[:, :blk].mul_(2.0)
                notes.append("conv チャネル 0 を 1/2 倍し、下流 Linear の"
                             "対応ブロックを 2 倍 (出力不変)")
        else:
            raise ValueError(f"未知の退化 {kind}")
    return notes


def _fresh_points(spec: Dict, n: int):
    """学習と同じ分布から n 点を取る (SGLD 用、学習データとは別の seed)。"""
    from train_rlct import synthetic_classification, synthetic_regression

    shape = (spec.get("channels_in", 1), spec["size"], spec["size"])
    if spec["loss"] == "regression":
        ds = synthetic_regression(n=n, shape=shape, n_out=spec["n_out"],
                                  seed=spec["seed"] + 991)
    else:
        ds = synthetic_classification(n=n, shape=shape,
                                      n_classes=spec["n_classes"],
                                      seed=spec["seed"] + 991)
    return ds.X


def toy_cases() -> List[Dict]:
    r"""**真値が独立に分かる特異なトイモデル。**

    CNN の実験は結局すべて「コアが空、λ = 正則方向/2」になり、Fisher の
    階数の半分で説明できてしまう。推定器の本当の試験になるのは
    **λ が階数の半分でない特異点**なので、ここでは原点に置いた深層線形網を使う。

    深層線形網は LLC 推定の論文が校正に使っている唯一の族でもある
    (Lau et al., AISTATS 2025: "DLNs the most realistic setting where
    theoretical learning coefficients are available")。

    | モデル | ファイバーイデアル | λ | m | 根拠 |
    |---|---|---|---|---|
    | y = b(ax), θ\*=0 | ⟨ab⟩ | 1/2 | 2 | 単項式、Howald: τ-距離 1 → 1/(2·1) |
    | y = c(b(ax)), θ\*=0 | ⟨abc⟩ | 1/2 | 3 | 同、τ-距離 1 |
    | y = Wx (1 層), θ\* 任意 | ⟨x_i 方向⟩ | d/2 | 1 | 正則点 |

    λ = 1/2 は **Fisher の階数では説明できない** (Fisher は原点で恒等的に 0、
    階数 0)。正則点の公式 rank/2 なら 0 になるが真の λ は 1/2。
    **ここが推定器の本当の試験。**

    さらに **位数 m は推定量が原理的に出せない量**である (λ̂ は
    `nβ(E[L_n]−L_n(w*))` の 1 つの数しか返さず、m は漸近展開の
    log log n の項に入る)。m=2 と m=3 を区別できるのは厳密計算だけ。
    """
    import torch.nn as nn

    def dln(depth: int):
        def make():
            import torch
            mods = [nn.Linear(1, 1, bias=False) for _ in range(depth)]
            net = nn.Sequential(*mods)
            with torch.no_grad():
                for m in mods:
                    m.weight.zero_()          # θ* = 0 (特異点)
            return net, [f"{i}.weight" for i in range(depth)]
        return make

    def reg_linear(d: int):
        def make():
            import torch
            net = nn.Sequential(nn.Linear(d, 1))
            torch.manual_seed(0)
            return net, ["0.weight", "0.bias"]
        return make

    return [
        dict(name="DLN 1-1-1 at 0 (⟨ab⟩)", make=dln(2), in_dim=1,
             truth=sp_half(), truth_m=2,
             why="単項式イデアル、Howald の τ-距離 1 → λ=1/2"),
        dict(name="DLN 1-1-1-1 at 0 (⟨abc⟩)", make=dln(3), in_dim=1,
             truth=sp_half(), truth_m=3,
             why="同、τ-距離 1 → λ=1/2、位数 3"),
        dict(name="regular linear d=4", make=reg_linear(3), in_dim=3,
             truth=2.0, truth_m=1, why="正則点 λ=d/2=2"),
        dict(name="regular linear d=6", make=reg_linear(5), in_dim=5,
             truth=3.0, truth_m=1, why="正則点 λ=d/2=3"),
    ]


def sp_half() -> float:
    return 0.5


def run_toy(*, n: int = 512, epss: Sequence[float] = (2e-4, 5e-4, 1e-3),
            gammas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0),
            n_steps: int = 10000, chains: int = 4, seed: int = 0,
            verbose: bool = True) -> List[Dict]:
    r"""トイモデルで厳密 λ (記号) と λ̂ (SGLD) を突き合わせる。"""
    import sympy as sp
    import torch

    from llc_estimate import llc_sgld
    from torch_rlct import torch_local_rlct

    out: List[Dict] = []
    for case in toy_cases():
        net, params = case["make"]()
        X = torch.randn(n, case["in_dim"],
                        generator=torch.Generator().manual_seed(seed))
        rec = dict(name=case["name"], truth=case["truth"],
                   truth_m=case["truth_m"], why=case["why"])
        # 厳密 λ (記号)
        t = time.time()
        try:
            rep = torch_local_rlct(net, X[:8], loss="regression",
                                   params=params, max_vars=20,
                                   max_denominator=10 ** 6, zero_tol=1e-12,
                                   max_depth=10)
            rec.update(exact_rlct=str(rep.local.rlct),
                       exact_float=float(rep.local.rlct),
                       exact_mult=rep.local.multiplicity,
                       exact_ok=(sp.nsimplify(rep.local.rlct)
                                 == sp.nsimplify(case["truth"])),
                       mult_ok=(rep.local.multiplicity == case["truth_m"]),
                       t_exact=round(time.time() - t, 1))
        except Exception as e:                            # noqa: BLE001
            rec.update(exact_rlct=None,
                       exact_err=f"{type(e).__name__}: {str(e)[:70]}")
        # λ̂ (SGLD)
        ests = []
        for gamma in gammas:
            for eps in epss:
                net2, params2 = case["make"]()
                try:
                    r = llc_sgld(net2, X, params=params2, loss="regression",
                                 gamma=gamma, eps=eps, n_steps=n_steps,
                                 chains=chains, seed=seed)
                    ests.append(dict(eps=eps, gamma=gamma,
                                     lam_hat=r.lam_hat, lam_std=r.lam_std,
                                     split_drift=r.split_drift,
                                     diverged=r.diverged))
                except Exception as e:                    # noqa: BLE001
                    ests.append(dict(eps=eps, gamma=gamma,
                                     error=str(e)[:60]))
        rec["estimates"] = ests
        good = [e for e in ests if "lam_hat" in e
                and math.isfinite(e["lam_hat"])]
        if good:
            best = min(good, key=lambda e: abs(e["lam_hat"] - case["truth"]))
            rec.update(best_lam_hat=best["lam_hat"], best_eps=best["eps"],
                       best_gamma=best["gamma"],
                       best_rel_err=(best["lam_hat"] - case["truth"])
                       / case["truth"],
                       spread=(max(e["lam_hat"] for e in good)
                               - min(e["lam_hat"] for e in good))
                       / case["truth"])
        out.append(rec)
        if verbose:
            ex = (f"厳密 λ={rec.get('exact_rlct')} m={rec.get('exact_mult')}"
                  f" {'OK' if rec.get('exact_ok') else '**不一致**'}"
                  f"/{'m OK' if rec.get('mult_ok') else '**m 不一致**'}")
            es = (f"λ̂={rec.get('best_lam_hat', float('nan')):.3f} "
                  f"(γ={rec.get('best_gamma')}) "
                  f"誤差 {rec.get('best_rel_err', float('nan')):+.1%} "
                  f"振れ幅 {rec.get('spread', float('nan')):.0%}")
            print(f"  {case['name']:<26} 真値 λ={case['truth']} "
                  f"m={case['truth_m']}  {ex}  {es}", flush=True)
    return out


def compare_one(spec: Dict, *, sgld_n: int = 0,
                epss: Sequence[float] = (5e-4,),
                gammas: Sequence[float] = (1.0,),
                n_steps: int = 8000, chains: int = 4,
                precondition: str = "none",
                hard_labels: bool = False,
                exact: bool = True, verbose: bool = False) -> Dict:
    """1 ケースで厳密 λ と λ̂ を両方出す。"""
    from cnn_sweep import _build_and_train, _params_for
    from llc_estimate import llc_sgld
    from torch_rlct import torch_local_rlct

    rec: Dict = dict(spec)
    t0 = time.time()
    model, ds, trep, tinfo = _build_and_train(spec)
    rec.update(tinfo)
    # 退化は**学習後**に仕込む。教師は θ* の出力から作るので realizable は保たれる
    deg = spec.get("degeneracy", "none")
    rec["degeneracy_notes"] = inject_degeneracy(model, deg) if deg != "none" \
        else []
    rec["fit"] = round(float(trep.final_acc), 4)
    rec["n_params"] = sum(p.numel() for p in model.parameters())
    names = _params_for(model, spec["params"], spec["max_vars"])
    rec["moved"] = names
    nm = dict((n, p.numel()) for n, p in model.named_parameters())
    rec["n_moved"] = sum(nm[n] for n in (names or nm))
    Xe = _data_for(ds, spec["n_data"])

    # --- 厳密 λ ---
    if exact:
        t = time.time()
        try:
            rep = torch_local_rlct(
                model, Xe.reshape(Xe.shape[0], -1), loss=spec["loss"],
                params=names, input_shape=ds.shape,
                max_vars=spec["max_vars_hard"],
                max_denominator=spec["max_denominator"],
                taylor_order=spec["taylor_order"],
                zero_tol=spec["zero_tol"], max_depth=spec["max_depth"])
            L = rep.local
            rec.update(exact_rlct=str(L.rlct), exact_mult=L.multiplicity,
                       exact_float=float(L.rlct),
                       free=len(L.free_vars), gauge=len(L.gauge_fixed),
                       regular=len(L.regular_vars), core=len(L.core_vars),
                       exact_status="ok", t_exact=round(time.time() - t, 1))
        except Exception as e:                            # noqa: BLE001
            rec.update(exact_status=f"{type(e).__name__}",
                       exact_note=str(e)[:120],
                       t_exact=round(time.time() - t, 1))
    # --- λ̂ ---
    Xs = Xe if sgld_n <= 0 else _fresh_points(spec, sgld_n)
    rec["sgld_n"] = int(Xs.shape[0])
    rec["same_points"] = bool(sgld_n <= 0)
    ests: List[Dict] = []
    for gamma in gammas:
        for eps in epss:
            t = time.time()
            try:
                # **平坦化しない。**厳密側は torch_local_rlct が平坦な X と
                # input_shape を取るが、SGLD は model(X) を直に呼ぶので
                # Conv2d が要求する (N, C, H, W) のままで渡す
                r = llc_sgld(model, Xs,
                             params=names, loss=spec["loss"],
                             gamma=gamma, eps=eps, n_steps=n_steps,
                             chains=chains, precondition=precondition,
                             hard_labels=hard_labels, seed=spec["seed"],
                             verbose=verbose)
                ests.append(dict(
                    eps=eps, gamma=gamma, lam_hat=r.lam_hat,
                    lam_std=r.lam_std, chains=r.lam_chains,
                    split_drift=r.split_drift, frac_below=r.frac_below,
                    diverged=r.diverged, notes=r.notes,
                    t=round(time.time() - t, 1)))
            except Exception as e:                        # noqa: BLE001
                ests.append(dict(eps=eps, gamma=gamma,
                                 error=f"{type(e).__name__}: {str(e)[:90]}"))
    rec["estimates"] = ests
    # 厳密値に最も近い推定 (= 最良条件での誤差)
    if rec.get("exact_status") == "ok":
        good = [e for e in ests if "lam_hat" in e
                and math.isfinite(e["lam_hat"])]
        if good:
            best = min(good, key=lambda e: abs(e["lam_hat"]
                                               - rec["exact_float"]))
            rec["best_eps"] = best["eps"]
            rec["best_gamma"] = best["gamma"]
            rec["best_lam_hat"] = best["lam_hat"]
            rec["best_rel_err"] = ((best["lam_hat"] - rec["exact_float"])
                                   / rec["exact_float"])
            sp = [e["lam_hat"] for e in good]
            rec["spread"] = (max(sp) - min(sp)) / rec["exact_float"]
    rec["wall"] = round(time.time() - t0, 1)
    return rec


# ----------------------------------------------------------------------
def _line(r: Dict) -> str:
    head = (f"  {r['loss'][:3]:<4} {r['act']:<4} "
            f"n={r['taylor_order'] if r['act'] != 'relu' else '-':<2} "
            f"d={r['depth']:<2} {str(r.get('degeneracy','none'))[:12]:<12}")
    if r.get("exact_status") != "ok":
        ex = f"厳密 {r.get('exact_status','?')}"
    else:
        ex = f"厳密 λ={r['exact_rlct']:<5} m={r['exact_mult']}"
    if "best_lam_hat" in r:
        es = (f"λ̂={r['best_lam_hat']:6.3f} "
              f"(ε={r['best_eps']:.0e} γ={r['best_gamma']:g}) "
              f"誤差 {r['best_rel_err']:+6.1%} "
              f"格子内の振れ幅 {r.get('spread', float('nan')):.0%}")
    else:
        ok = [e for e in r.get("estimates", []) if "lam_hat" in e]
        es = (f"λ̂={ok[0]['lam_hat']:.3f}" if ok else "λ̂ 失敗")
    return f"{head} {ex}  {es}  fit={r.get('fit', float('nan')):.2f}"


def run_grid(specs: Sequence[Dict], *, verbose: bool = True,
             **kw) -> List[Dict]:
    out = []
    for s in specs:
        r = compare_one(dict(s), **kw)
        out.append(r)
        if verbose:
            print(_line(r), flush=True)
    return out


def summarize(recs: List[Dict]) -> str:
    L: List[str] = []
    both = [r for r in recs if r.get("exact_status") == "ok"
            and "best_lam_hat" in r]
    L.append("\n--- 厳密 λ 対 λ̂ ---")
    if not both:
        L.append("  両方そろったケースがありません。")
        return "\n".join(L)
    hdr = (f"  {'設定':<28} {'厳密 λ':>8} {'m':>2} {'λ̂(最良)':>9} "
           f"{'誤差':>8} {'振れ幅':>7}")
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))
    for r in both:
        o = "" if r["act"] == "relu" else f" n={r['taylor_order']}"
        dg = ("" if r.get("degeneracy", "none") == "none"
              else " " + r["degeneracy"])
        name = f"{r['loss'][:3]} {r['act']}{o} d={r['depth']}{dg}"
        L.append(f"  {name:<28} {r['exact_rlct']:>8} {r['exact_mult']:>2} "
                 f"{r['best_lam_hat']:>9.3f} {r['best_rel_err']:>+7.1%} "
                 f"{r.get('spread', float('nan')):>6.0%}")
    errs = [abs(r["best_rel_err"]) for r in both]
    L.append(f"\n  最良条件での相対誤差: 中央値 "
             f"{sorted(errs)[len(errs)//2]:.1%}, 最大 {max(errs):.1%}")
    spreads = [r.get("spread") for r in both if r.get("spread") is not None]
    if spreads:
        L.append(f"  ε, γ の格子内での λ̂ の振れ幅: 中央値 "
                 f"{sorted(spreads)[len(spreads)//2]:.0%}, "
                 f"最大 {max(spreads):.0%}")

    # 順序の保存 — 絶対値が当たらなくても順位が合うかは別の問題
    pairs = 0
    agree = 0
    for i in range(len(both)):
        for j in range(i + 1, len(both)):
            a, b = both[i], both[j]
            if abs(a["exact_float"] - b["exact_float"]) < 1e-9:
                continue
            pairs += 1
            if ((a["exact_float"] - b["exact_float"]) *
                    (a["best_lam_hat"] - b["best_lam_hat"]) > 0):
                agree += 1
    if pairs:
        L.append(f"  順序の保存: {agree}/{pairs} 組 ({agree/pairs:.0%})")

    neg = [r for r in both if r["best_lam_hat"] < 0]
    if neg:
        L.append(f"\n  [注意] λ̂ < 0 が {len(neg)} 件。"
                 "θ* が損失の最小点になっていない兆候です")
    div = [r for r in recs for e in r.get("estimates", [])
           if e.get("diverged")]
    if div:
        L.append(f"  [注意] 発散した推定が {len(div)} 件")
    drift = [e for r in recs for e in r.get("estimates", [])
             if e.get("notes") and any("未収束" in n for n in e["notes"])]
    if drift:
        L.append(f"  [注意] 未収束の疑いが {len(drift)} 件 "
                 "(残した鎖の前半と後半で λ̂ が 15% 以上動いた)")
    L.append("\n  [注意] 厳密 λ も λ̂ も**同じ部分空間に制限した値**です。"
             "全パラメータ空間の λ の下界であり、公表されている "
             "LLC 推定値と直接比べられる量ではありません。")
    return "\n".join(L)


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="1,2")
    ap.add_argument("--acts", default="relu,tanh")
    ap.add_argument("--orders", default="1")
    ap.add_argument("--losses", default="both",
                    choices=("classification", "regression", "both"))
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--kernel", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=4)
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--out-dim", type=int, default=1)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--n-data", type=int, default=8)
    ap.add_argument("--params", default="conv",
                    choices=("all", "head", "conv"))
    ap.add_argument("--max-vars", type=int, default=10)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--degeneracies", default="none",
                    help="none / dup_channel / dead_channel / half_scale "
                         "をカンマ区切りで")
    ap.add_argument("--max-denominator", type=int, default=16)
    ap.add_argument("--zero-tol", type=float, default=1e-2)
    ap.add_argument("--max-depth", type=int, default=6)
    # SGLD
    ap.add_argument("--sgld-n", type=int, default=0,
                    help="0 なら厳密側と同じ点。>0 なら同分布から取り直す")
    ap.add_argument("--epss", default="2e-4,5e-4,1e-3")
    ap.add_argument("--gammas", default="1,10")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--precondition", default="none",
                    choices=("none", "rmsprop"))
    ap.add_argument("--hard-labels", action="store_true",
                    help="θ* から標本抽出した整数ラベルを使う (λ̂ が負に振れる確認)")
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--toy", action="store_true",
                    help="真値が分かる特異なトイモデル (深層線形網) で比べる")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import rlct_cache
    rlct_cache.disable()

    if args.toy:
        print("===== トイモデル: 厳密 λ 対 λ̂ =====")
        print("  深層線形網を原点 (特異点) に置いた場合と正則点の対照\n")
        recs = run_toy(epss=[float(v) for v in args.epss.split(",") if v],
                       gammas=[float(v) for v in args.gammas.split(",") if v],
                       n_steps=args.steps, chains=args.chains)
        print("\n===== まとめ =====")
        ok = [r for r in recs if r.get("exact_ok")]
        mok = [r for r in recs if r.get("mult_ok")]
        print(f"  厳密 λ が真値と一致: {len(ok)}/{len(recs)}")
        print(f"  位数 m が真値と一致: {len(mok)}/{len(recs)}")
        for r in recs:
            if "best_rel_err" in r:
                print(f"  {r['name']:<26} λ̂ の誤差 "
                      f"{r['best_rel_err']:+7.1%}  "
                      f"γ, ε の格子内の振れ幅 {r.get('spread', 0):.0%}")
        print("\n  [注意] 位数 m は λ̂ が原理的に返せない量です "
              "(推定量は数 1 つしか返さない)。")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(recs, fh, ensure_ascii=False, indent=1, default=str)
            print(f"\n-> {args.out}")
        return 0

    losses = (["classification", "regression"] if args.losses == "both"
              else [args.losses])
    base = dict(degeneracy="none", size=args.size, channels=args.channels, kernel=args.kernel,
                hidden=args.hidden, n_classes=args.classes,
                n_out=args.out_dim, n=args.n, seed=0,
                seeds=[int(v) for v in args.seeds.split(",") if v],
                fit_min=0.95, n_data=args.n_data, params=args.params,
                max_vars=args.max_vars, max_vars_hard=80,
                max_denominator=args.max_denominator,
                zero_tol=args.zero_tol, max_depth=args.max_depth)
    specs: List[Dict] = []
    for loss in losses:
        ep, lr = ((600, 0.05) if loss == "classification" else (2000, 0.02))
        for act in [a for a in args.acts.split(",") if a]:
            for d in [int(v) for v in args.depths.split(",") if v]:
                orders = ([0] if act == "relu"
                          else [int(v) for v in args.orders.split(",") if v])
                for o in orders:
                    for dg in [v for v in args.degeneracies.split(",") if v]:
                        specs.append(dict(base, loss=loss, epochs=ep, lr=lr,
                                          act=act, depth=d, taylor_order=o,
                                          degeneracy=dg))

    print("===== 厳密 λ 対 SGLD の λ̂ =====")
    print(f"  画像 1x{args.size}x{args.size}, conv ch={args.channels}, "
          f"kernel={args.kernel}, hidden={args.hidden}, 変数={args.params}")
    print(f"  厳密側のデータ点 {args.n_data}, SGLD 側 "
          f"{'同じ点' if args.sgld_n <= 0 else str(args.sgld_n) + ' 点 (取り直し)'}"
          f", 鎖 {args.chains} 本 x {args.steps} 歩, "
          f"前処理={args.precondition}")
    print(f"  {len(specs)} ケース\n")
    recs = run_grid(specs, sgld_n=args.sgld_n,
                    epss=[float(v) for v in args.epss.split(",") if v],
                    gammas=[float(v) for v in args.gammas.split(",") if v],
                    n_steps=args.steps, chains=args.chains,
                    precondition=args.precondition,
                    hard_labels=args.hard_labels, exact=not args.no_exact)
    print("\n===== まとめ =====")
    print(summarize(recs))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(recs, fh, ensure_ascii=False, indent=1, default=str)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
