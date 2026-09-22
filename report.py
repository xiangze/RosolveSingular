r"""bench/report.py — 結果の集計。proved 率と失敗クラスの分布を出す。"""
from __future__ import annotations

import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="bench_results.json")
    args = ap.parse_args()
    recs = json.load(open(args.path, encoding="utf-8"))

    n = len(recs)
    cls = collections.Counter(r["class"] for r in recs)
    print(f"# 件数 {n}")
    print("# クラス別")
    for k, v in cls.most_common():
        print(f"    {k:<20} {v:>4}  ({100*v/n:5.1f}%)")

    proved = [r for r in recs if r["class"] == "proved"]
    print(f"# proved {len(proved)}/{n} = {100*len(proved)/n:.1f}%")
    routes = collections.Counter(r.get("route", "-") for r in proved)
    print("# proved の経路")
    for k, v in routes.most_common():
        print(f"    {k:<40} {v:>4}")

    print("# 変数の数ごとの proved 率")
    by_n = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        b = by_n[r["n_vars"]]
        b[1] += 1
        b[0] += r["class"] == "proved"
    for k in sorted(by_n):
        a, b = by_n[k]
        print(f"    n={k}: {a}/{b} ({100*a/b:5.1f}%)")

    print("# 次数ごとの proved 率")
    by_d = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        b = by_d[r["degree"]]
        b[1] += 1
        b[0] += r["class"] == "proved"
    for k in sorted(by_d):
        a, b = by_d[k]
        print(f"    d={k}: {a}/{b} ({100*a/b:5.1f}%)")

    print("# mu = oo (複素で非孤立) かどうかと proved 率の関係")
    print("#   mu は lambda の値にも判定にも使っていない。予測指標としての相関だけを見る。")
    for key, label in ((True, "mu = oo "), (False, "mu 有限")):
        sub = [r for r in recs if r.get("mu_infinite") is key]
        if sub:
            a = sum(1 for r in sub if r["class"] == "proved")
            print(f"    {label}: {a}/{len(sub)} ({100*a/len(sub):5.1f}%)")
    unk = [r for r in recs if r.get("mu_infinite") is None]
    if unk:
        print(f"    mu 未計算: {len(unk)} 件")
    print("# 擬斉次かどうかと proved 率")
    for key, label in ((True, "擬斉次  "), (False, "非擬斉次")):
        sub = [r for r in recs if r.get("quasihomogeneous") is key]
        if sub:
            a = sum(1 for r in sub if r["class"] == "proved")
            print(f"    {label}: {a}/{len(sub)} ({100*a/len(sub):5.1f}%)")

    # --- 実行時間 (サイズ・次数との関係) --------------------------
    try:
        from timing import timing_summary
        print("# 実行時間 (秒): 変数の数ごと")
        print(timing_summary(recs, by=("n_vars",)))
        print("# 実行時間 (秒): 次数ごと")
        print(timing_summary(recs, by=("degree",)))
        print("# 実行時間 (秒): (変数, 次数) ごと")
        print(timing_summary(recs, by=("n_vars", "degree")))
    except Exception as e:                            # noqa: BLE001
        print(f"# 実行時間の集計に失敗: {e}")

    nw = [r for r in recs if r.get("n_newton")]
    if nw:
        print(f"# ニュートン高速パスで閉じた chart を含むケース: {len(nw)}/{n}")
        tot = sum(r["n_newton"] for r in nw)
        print(f"#   打ち切った chart 数の合計 {tot}")

    bad = [r for r in recs if r.get("inconsistent")]
    if bad:
        print("# !! 独立な値と食い違うケース (赤信号)")
        for r in bad:
            print(f"    {r['name']}: {r['inconsistent']}")
    else:
        print("# 独立な値との食い違い: なし")

    print("# 失敗の代表例")
    seen = set()
    for r in recs:
        if r["class"] in ("proved",) or r["class"] in seen:
            continue
        seen.add(r["class"])
        print(f"    [{r['class']}] {r['name']}  f={r['f'][:60]}")


if __name__ == "__main__":
    main()
