r"""
stage_report.py — 段階 1 の結果 (result/stage1/*.json) を表と図にまとめる

    python stage_report.py            # result/stage1/summary.md と stage1.png
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from fractions import Fraction

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # 参照パレットの先頭 4 色
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _load(path):
    with open(path) as f:
        return json.load(f)


def _rows(res, kind):
    out = []
    for st in res["stages"]:
        pts = st.get("hmc", [])
        if kind == "dln":
            lam, m = float(Fraction(st["lam"])), st["m"]
            vals = [(p["t"], p["lam_hat"], p["lam_std"]) for p in pts]
        else:
            lam, m = float(Fraction(st["lam_pred"])), st["m_pred"]
            vals = [(p["t"], p.get("lam_corr", p["lam_hat"]), p["lam_std"])
                    for p in pts]
        out.append(dict(k=st["k"], lam=lam, lam_s=st.get("lam", st.get("lam_pred")),
                        m=m, vals=vals, m_fit=st.get("m_fit"),
                        r_fit=st.get("r_fit"), row=st))
    return out


def table(name, res, kind) -> str:
    L = [f"### {name}", "",
         "| 段階 k | 厳密/予想 λ | m | 根拠 | 最大の t での HMC | 相対誤差 | m の当てはめ |",
         "|---|---|---|---|---|---|---|"]
    for r in _rows(res, kind):
        st = r["row"]
        basis = ("Aoyagi + 層の公式 (一致)" if kind == "dln" else st.get("basis", ""))
        if kind == "mlp" and "sym_lam" in st:
            basis += f"; 記号 λ={st['sym_lam']}"
        if r["vals"]:
            t, v, s = r["vals"][-1]
            err = (v - r["lam"]) / r["lam"]
            hv = f"{v:.3f} ± {s:.3f}"
            es = f"{err:+.1%}"
        else:
            hv, es = "—", "—"
        L.append(f"| {r['k']} | {r['lam_s']} | {r['m']} | {basis} | {hv} | {es} | "
                 f"{r['m_fit'] if r['m_fit'] is not None else '—'} |")
    return "\n".join(L)


def figure(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.3 * rows),
                             facecolor=SURF, squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (title, res, kind) in zip(axes.flat, panels):
        ax.set_facecolor(SURF)
        for i, r in enumerate(_rows(res, kind)):
            c = SERIES[i % len(SERIES)]
            if not r["vals"]:
                continue
            ts = [v[0] for v in r["vals"]]
            ys = [v[1] for v in r["vals"]]
            es = [v[2] for v in r["vals"]]
            ax.axhline(r["lam"], color=c, lw=1, ls=(0, (4, 3)), alpha=0.9)
            ax.errorbar(ts, ys, yerr=es, color=c, lw=2, marker="o", ms=4,
                        mec=SURF, mew=1.5, capsize=0,
                        label=f"k={r['k']}: λ={r['lam_s']}, m={r['m']}")
            ax.annotate(f"k={r['k']}", (ts[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("温度 t = nβ", fontsize=8, color=INK2)
        ax.set_ylabel("t·E_t[K]  (破線 = 厳密 λ)", fontsize=8, color=INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(True, color=GRID, lw=0.6)
        for sp_ in ax.spines.values():
            sp_.set_color(GRID)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK, ncol=2,
                  loc="upper left", bbox_to_anchor=(0.0, -0.2))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


def main() -> int:
    base = "result/stage1"
    spec = [("DLN H=(3,3,3)", "dln333.json", "dln"),
            ("DLN H=(3,3,3,3)", "dln3333.json", "dln"),
            ("DLN H=(3,3,3) データ階数 q=2", "dln333_q2.json", "dln"),
            ("tanh MLP H=2", "mlp2.json", "mlp"),
            ("tanh MLP H=3", "mlp3.json", "mlp"),
            ("モーメント多項式 H=2 (tanh と同じ局所イデアル)", "mom2.json", "mlp")]
    panels = []
    md = ["# 段階 1: 段階ごとの厳密 (λ, m) と HMC", ""]
    for title, fn, kind in spec:
        p = os.path.join(base, fn)
        if not os.path.exists(p):
            continue
        res = _load(p)
        panels.append((title, res, kind))
        md.append(table(title, res, kind))
        md.append("")
    with open(os.path.join(base, "summary.md"), "w") as f:
        f.write("\n".join(md))
    try:
        import matplotlib
        from matplotlib import font_manager
        for fp in glob.glob("/usr/share/fonts/**/*CJK*", recursive=True) + \
                glob.glob("/usr/share/fonts/**/*Noto*JP*", recursive=True):
            font_manager.fontManager.addfont(fp)
        names = {f.name for f in font_manager.fontManager.ttflist}
        for cand in ("Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic"):
            if cand in names:
                matplotlib.rcParams["font.family"] = cand
                break
    except Exception:
        pass
    figure(panels, os.path.join(base, "stage1.png"))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
