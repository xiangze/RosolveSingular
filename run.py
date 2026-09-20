r"""
bench/run.py — ベンチマークを回して失敗を分類する。

各ケースについて
  1. 族ごとの専用経路 (families.family_rlct)
  2. だめなら重み付きブローアップ + certify
を試し、結果と失敗の分類を JSON に書き出す。

失敗の分類 (症状 -> 当てるべきパッチの場所):
  resolve_max_depth : 中心の選び方の限界        -> 重み / 中心の選択則
  resolve_other     : 解消の内部エラー
  smt_unknown       : 判定の規模                -> 面の分解 / timeout
  nc_refuted        : 正規交差が壊れる点あり    -> localize / 定義域の切り分け
  identity_refuted  : 帳簿のバグ                -> 解消側の実装
  covering_unknown  : 被覆が確立しない          -> 補題の追加
  inconsistent      : 独立な値と食い違う (赤信号)
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import sympy as sp

from certify import certify
from families import family_rlct
from oracles import check_consistency, complexity_hints, cross_oracles
from resolve_singularity import ResolutionFailure, resolve_singularities

import cases as cases_mod


def classify(rec: Dict) -> str:
    if rec.get("inconsistent"):
        return "inconsistent"
    st = rec.get("status")
    if st == "proved":
        return "proved"
    if rec.get("resolve_fail"):
        return ("resolve_max_depth" if rec["resolve_fail"] in
                ("max_depth", "no-progress", "prune-inconsistent", "max_charts")
                else "resolve_other")
    if rec.get("identity_bad"):
        return "identity_refuted"
    if rec.get("nc_refuted"):
        return "nc_refuted"
    if rec.get("covering_bad"):
        return "covering_unknown"
    return "smt_unknown"


def run_case(name, f, gens, family, *, max_depth=20, timeout_ms=8000,
             cross=False, montecarlo=False, covering=False) -> Dict:
    rec: Dict = {"name": name, "family": family, "n_vars": len(gens),
                 "degree": int(sp.Poly(sp.expand(f), *gens).total_degree()),
                 "n_terms": len(sp.Poly(sp.expand(f), *gens).monoms()),
                 "f": str(f)}
    t0 = time.time()

    fam = family_rlct(f, gens)
    if fam.status == "proved":
        rec.update(status="proved", route=f"family:{fam.route}",
                   rlct=str(fam.rlct), mult=fam.multiplicity)
    else:
        try:
            res = resolve_singularities(f, gens, prune=False, weighted=True,
                                        max_depth=max_depth)
        except ResolutionFailure as e:
            rec.update(status="unknown", route="resolve", rlct=None,
                       resolve_fail=e.reason)
            rec["time"] = round(time.time() - t0, 2)
            rec["class"] = classify(rec)
            return rec
        except Exception as e:                       # noqa: BLE001
            rec.update(status="unknown", route="resolve", rlct=None,
                       resolve_fail="exception:" + str(e)[:60])
            rec["time"] = round(time.time() - t0, 2)
            rec["class"] = classify(rec)
            return rec
        cert = certify(res, timeout_ms=timeout_ms)
        rec.update(status=cert.status, route="blowup+certify",
                   rlct=str(res.rlct), mult=res.multiplicity,
                   n_charts=len(res.charts))
        rec["identity_bad"] = any(not lf.identity_ok for lf in cert.leaves)
        rec["nc_refuted"] = any(lf.unit_status == "refuted"
                                or lf.jac_status == "refuted"
                                for lf in cert.leaves)
        rec["covering_bad"] = any(c[1] != "proved" for c in cert.covering)

    # 難しさの予測指標 (共変量)。判定には使わない。
    try:
        rec.update(complexity_hints(f, gens))
    except Exception:
        pass

    lam = sp.sympify(rec["rlct"]) if rec.get("rlct") else None
    msgs = check_consistency(f, gens, lam)
    if msgs:
        rec["inconsistent"] = msgs
    if cross:
        rec["cross"] = {k: str(v) for k, v in cross_oracles(f, gens).items()}
    if montecarlo and lam is not None:
        # 解消の結果を一切使わない独立な検査
        try:
            from independent import monte_carlo_disagrees
            bad = monte_carlo_disagrees(f, gens, lam, n_samples=80000)
            if bad:
                rec.setdefault("inconsistent", []).append(bad)
        except Exception:
            pass
    if covering and rec.get("route") == "blowup+certify" and len(gens) <= 4:
        try:
            from independent import covering_sample_check
            from resolve_singularity import resolve_singularities as _rs
            r2 = _rs(f, gens, prune=False, weighted=True, max_depth=max_depth)
            cv = covering_sample_check(r2)
            rec["covering_sample"] = cv["status"]
            if cv["status"] == "gap":
                rec.setdefault("inconsistent", []).append(
                    f"被覆の標本検証で {cv['miss']}/{cv['cells']} セルが未被覆")
        except Exception:
            pass
    rec["time"] = round(time.time() - t0, 2)
    rec["class"] = classify(rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--max-vars", type=int, default=4)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--timeout-ms", type=int, default=8000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--failures", action="store_true",
                    help="蓄積した失敗例をレグレッションとして含める")
    ap.add_argument("--hard", action="store_true",
                    help="真値が独立に分かる難しい族 (known_families) を含める")
    ap.add_argument("--montecarlo", action="store_true",
                    help="体積の漸近による独立な数値推定と突き合わせる (遅い)")
    ap.add_argument("--covering", action="store_true",
                    help="被覆の前向き標本検証を行う (n <= 4)")
    args = ap.parse_args()

    cs = cases_mod.structured_cases(args.max_vars)
    if args.random:
        cs += cases_mod.random_cases(n_list=tuple(range(2, args.max_vars + 1)))
    if args.failures:
        from failures import load_failures
        cs += load_failures()
    if args.hard:
        from known_families import all_cases
        cs += [(nm, f, g, fam) for nm, f, g, fam, _ in all_cases()]
    if args.limit:
        cs = cs[:args.limit]

    recs: List[Dict] = []
    for i, (name, f, gens, family) in enumerate(cs):
        rec = run_case(name, f, gens, family, max_depth=args.max_depth,
                       timeout_ms=args.timeout_ms, cross=args.cross,
                       montecarlo=args.montecarlo, covering=args.covering)
        recs.append(rec)
        print(f"[{i+1}/{len(cs)}] {name:<34} {rec['class']:<18} "
              f"lambda={rec.get('rlct')} ({rec['time']}s)"
              + ("  !! " + "; ".join(rec["inconsistent"])
                 if rec.get("inconsistent") else ""), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=1)
    print(f"-> {args.out} に {len(recs)} 件")


if __name__ == "__main__":
    main()
