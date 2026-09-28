r"""
bench/regression.py — 全レグレッションを一括で回す。

ここまでに作った検査を一つのコマンドでまとめて実行する。

  guards      : 判定器が甘くなっていないかの門番 (採用条件)
                負のテスト / 変異検査 / 独立検査 / Aoyagi の補題 /
                Vandermonde の真値 / 座標不変性、既知の未修正バグの報告
  structured  : 構造化された多項式の族 (単項式/Brieskorn/交差項/形式のべき/
                低ランク/退化例)
  random      : ランダムなスパース多項式 (n, 次数, 項数, 係数ビット長の格子)
  hard        : 真値が独立に分かり、かつ難しい族
                (一般の位置の超平面配置 / 斉次形式 / Vandermonde)
  nn          : ニューラルネット (MLP x {relu, tanh} x 退化の型)
  failures    : 蓄積した失敗例の再測

使い方:

    export PYTHONPATH=..:.
    python regression.py                 # guards + structured + nn (既定)
    python regression.py --all           # 全部
    python regression.py --suites guards,nn --out r.json
    python regression.py --quick         # 小さい範囲だけ

終了コードは、guards に赤があるか、独立な値との食い違い (inconsistent) が
あれば 1。それ以外は 0。CI に載せる前提。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List

SUITES = ("guards", "structured", "random", "hard", "nn", "failures")
DEFAULT = ("guards", "structured", "nn")


def _run_guards(verbose: bool) -> Dict:
    # ガードは正しさの門なので、**必ずキャッシュ抜き**で走らせる。
    # キャッシュを有効にしたまま回すと「自分が貯めた値で自分を検査する」
    # ことになり、検査の意味が無くなる (cache_tests だけは自前で
    # 一時キャッシュを作って使う)。
    import rlct_cache as _rc
    _saved = _rc.current()
    _rc.disable()
    try:
        return _run_guards_inner(verbose)
    finally:
        if _saved is not None and _saved.path:
            _rc.enable(_saved.path)


def _run_guards_inner(verbose: bool) -> Dict:
    import guards
    t0 = time.time()
    res = guards.run_all(verbose=verbose)
    bad = [r for r in res if not r.ok]
    known = guards.known_issues()
    return {"suite": "guards", "total": len(res), "failed": len(bad),
            "names_failed": [r.name for r in bad],
            "known_issues": [{"name": k.name, "fixed": k.ok, "detail": k.detail}
                             for k in known],
            "time": round(time.time() - t0, 1)}


def _run_poly(suite: str, args, verbose: bool) -> Dict:
    import cases as cases_mod
    import run as run_mod
    if suite == "structured":
        cs = cases_mod.structured_cases(args.max_vars)
    elif suite == "random":
        cs = cases_mod.random_cases(
            n_list=tuple(range(2, args.max_vars + 1)),
            d_list=(3, 4, 6) if not args.quick else (3, 4),
            t_list=(2, 3, 5) if not args.quick else (2, 3))
    elif suite == "hard":
        from known_families import all_cases
        cs = [(n, f, g, fam) for n, f, g, fam, _ in all_cases()]
    elif suite == "failures":
        from failures import load_failures
        cs = load_failures()
    else:
        raise ValueError(suite)
    if args.limit:
        cs = cs[:args.limit]
    recs: List[Dict] = []
    t0 = time.time()
    for i, (name, f, gens, fam) in enumerate(cs):
        rec = run_mod.run_case(name, f, gens, fam, max_depth=args.max_depth,
                               timeout_ms=args.timeout_ms)
        recs.append(rec)
        if verbose:
            print(f"  [{i+1}/{len(cs)}] {name:<34} {rec['class']:<18} "
                  f"lambda={rec.get('rlct')}"
                  + ("  !! " + "; ".join(rec["inconsistent"])
                     if rec.get("inconsistent") else ""), flush=True)
    import collections
    cls = collections.Counter(r["class"] for r in recs)
    return {"suite": suite, "total": len(recs), "classes": dict(cls),
            "proved": cls.get("proved", 0),
            "inconsistent": [r["name"] for r in recs if r.get("inconsistent")],
            "records": recs, "time": round(time.time() - t0, 1)}


def _run_nn(args, verbose: bool) -> Dict:
    import nn_cases as N
    archs = [(1, 1, 1), (1, 2, 1), (1, 3, 1), (2, 2, 1), (1, 2, 1, 1)]
    if not args.quick:
        archs += [(1, 4, 1), (2, 3, 1), (1, 2, 2), (1, 3, 2, 1)]
    cs = []
    for w in archs:
        for act in ("relu", "tanh"):
            for deg in ("redundant", "duplicate", "dead", "zero"):
                if deg == "dead" and act != "relu":
                    continue
                if act == "tanh" and sum(w) > (4 if args.quick else 5):
                    continue
                c = N.mlp_case_retry(w, act, deg, 2 * w[0] + 4, 2)
                if c:
                    cs.append(c)
    recs = []
    t0 = time.time()
    for i, c in enumerate(cs):
        r = N.run_case(c, timeout_s=args.nn_timeout,
                       timeout_ms=args.timeout_ms)
        recs.append(r)
        if verbose:
            print(f"  [{i+1}/{len(cs)}] {c.name:<32} core={r.get('core')} "
                  f"lambda={r.get('rlct')} {r.get('status')} ({r.get('time')}s)",
                  flush=True)
    import collections
    cls = collections.Counter(r.get("status", "?") for r in recs)
    return {"suite": "nn", "total": len(recs), "classes": dict(cls),
            "proved": cls.get("proved", 0), "records": recs,
            "summary": N.summarize(recs), "time": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default=",".join(DEFAULT),
                    help=f"カンマ区切り。使えるのは {','.join(SUITES)}")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-vars", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--timeout-ms", type=int, default=4000)
    ap.add_argument("--nn-timeout", type=int, default=30)
    ap.add_argument("--cache", default="rlct_cache.json",
                    help="解いた局所データの永続キャッシュ ('off' で無効)")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="キャッシュを 1 から作り直す")
    ap.add_argument("--verify-cache", action="store_true",
                    help="キャッシュの当たりを毎回計算し直して突き合わせる")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import rlct_cache
    if str(args.cache).lower() in ("off", "none", ""):
        rlct_cache.disable()
        print("# キャッシュ: 無効")
    else:
        rlct_cache.enable(args.cache, rebuild=args.rebuild_cache,
                          verify=args.verify_cache)
        print(f"# キャッシュ: {rlct_cache.current()}")
        print("#   (guards は必ずキャッシュ抜きで走ります。"
              "多項式スイートを独立な検査として使いたいときは "
              "--cache off か --verify-cache を付けてください)")

    suites = SUITES if args.all else tuple(
        s.strip() for s in args.suites.split(",") if s.strip())
    verbose = not args.quiet

    out: List[Dict] = []
    bad = False
    for s in suites:
        print(f"\n===== {s} =====", flush=True)
        if s == "guards":
            r = _run_guards(verbose)
            print(f"  guards: {r['total'] - r['failed']}/{r['total']} 緑 "
                  f"({r['time']}s)")
            for k in r["known_issues"]:
                print(f"  [既知] {k['name']}: "
                      + ("修正された -> 昇格させること" if k["fixed"] else "未修正"))
            if r["failed"]:
                bad = True
                print(f"  ** 赤: {r['names_failed']}")
        elif s == "nn":
            r = _run_nn(args, verbose)
            print(r["summary"])
        else:
            r = _run_poly(s, args, verbose)
            print(f"  {s}: proved {r['proved']}/{r['total']} "
                  f"{r['classes']} ({r['time']}s)")
            if r["inconsistent"]:
                bad = True
                print(f"  ** 独立な値と食い違い: {r['inconsistent']}")
        # 段階別の実行時間 (多項式のサイズ・次数との関係)
        if r.get("records"):
            try:
                from timing import timing_summary
                print("  -- 実行時間 (秒) --")
                print(timing_summary(r["records"], by=("n_vars",)))
            except Exception as e:                    # noqa: BLE001
                print(f"  (実行時間の集計に失敗: {e})")
        out.append(r)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
        print(f"\n-> {args.out}")

    import rlct_cache as _rc2
    if _rc2.current() is not None:
        _rc2.save()
        print(f"\n# {_rc2.current()}")

    print("\n===== 判定 =====")
    print("  NG (guards が赤、または独立な値との食い違いあり)" if bad
          else "  OK (guards 全緑、食い違いなし)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
