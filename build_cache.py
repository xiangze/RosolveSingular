r"""
bench/build_cache.py — 永続キャッシュ (rlct_cache) に値を貯める。

regression を速くするのが目的ではなく、**中小規模の ReLU ネットのコア
イデアルを大量に解いて貯めておく**のが主目的。ニューラルネットでは
アーキテクチャが違っても同じコアイデアルが何度も現れるので、根の値
(K, core_vars) を入れておくとあとの計算で当たる。

各ケースは別プロセスで走らせる。1 件が止まってもスイープ全体は進み、
その都度キャッシュを保存するので途中で落ちても貯まった分は残る。

使い方
------
    python build_cache.py --relu                    # ReLU 格子を貯める
    python build_cache.py --relu --tanh --poly      # 全部
    python build_cache.py --relu --rebuild          # 1 から作り直す
    python build_cache.py --relu --limit 50 --timeout 20
    python build_cache.py --stats                   # 中身を見るだけ

**証明できた値しか入らない** (certify が proved、または族の閉じた式)。
コードの指紋が変わると過去のエントリは読み込み時に捨てられるので、
バグを直したあとは貯め直すことになる。
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]


# ----------------------------------------------------------------------
# 子プロセス
# ----------------------------------------------------------------------
def _tf_worker(spec, cache_path: str, opts: Dict, q) -> None:
    """Transformer 1 層の仕様から式を作って 1 件解く。"""
    sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    import rlct_cache
    rlct_cache.enable(cache_path)
    import nn_cases as N
    t = time.time()
    try:
        c = N.transformer_case_from_spec(spec)
    except Exception as e:                            # noqa: BLE001
        q.put({"name": str(spec), "status": "build-error",
               "note": str(e)[:50], "wall": round(time.time() - t, 2)})
        return
    if c is None:
        q.put({"name": str(spec), "status": "skip"}); return
    try:
        r = N.run_case(c, timeout_ms=opts["timeout_ms"],
                       max_depth=opts["max_depth"],
                       timeout_s=opts["timeout"] - 3)
    except Exception as e:                            # noqa: BLE001
        r = {"name": c.name, "status": "error", "note": str(e)[:50]}
    cache = rlct_cache.current()
    if cache is not None:
        cache.save()
        r["entries"] = len(cache.entries)
        r["hit"] = cache.n_hit
    r["wall"] = round(time.time() - t, 2)
    q.put({k: r.get(k) for k in
           ("name", "status", "rlct", "core", "core_route", "n_cache_put",
            "entries", "hit", "wall", "n_params", "arch", "degeneracy")})


def _spec_worker(spec, cache_path: str, opts: Dict, q) -> None:
    """仕様 (widths, deg, n_data, seed) から式を作って 1 件解く。

    格子は数千件になるので親で全部組み立てず、仕様だけ配って子で作る。
    1 件の構築は ReLU なら 0.1 秒程度で、解消のコストに比べて無視できる。
    """
    sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    import rlct_cache
    rlct_cache.enable(cache_path)
    import nn_cases as N
    t = time.time()
    c = N.case_from_spec(spec)
    if c is None:
        q.put({"name": str(spec), "status": "skip"}); return
    try:
        r = N.run_case(c, timeout_ms=opts["timeout_ms"],
                       max_depth=opts["max_depth"],
                       timeout_s=opts["timeout"] - 3)
    except Exception as e:                            # noqa: BLE001
        r = {"name": c.name, "status": "error", "note": str(e)[:50]}
    cache = rlct_cache.current()
    if cache is not None:
        cache.save()
        r["entries"] = len(cache.entries)
        r["hit"] = cache.n_hit
    r["wall"] = round(time.time() - t, 2)
    q.put({k: r.get(k) for k in
           ("name", "status", "rlct", "core", "core_route", "n_cache_put",
            "entries", "hit", "wall", "n_params", "arch", "degeneracy")})


def _nn_worker(blob: bytes, cache_path: str, opts: Dict, q) -> None:
    """ケースは親で 1 度だけ作って pickle で渡す。

    子プロセスで格子を作り直すと、記号 MLP を数百個組み立て直すことになり
    (実測で 1 件あたり数十秒) 本来の計算より高くつく。
    """
    import pickle
    sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    import rlct_cache
    rlct_cache.enable(cache_path)
    import nn_cases as N
    c = pickle.loads(blob)
    t = time.time()
    try:
        r = N.run_case(c, timeout_ms=opts["timeout_ms"],
                       max_depth=opts["max_depth"],
                       timeout_s=opts["timeout"] - 3)
    except Exception as e:                            # noqa: BLE001
        r = {"name": c.name, "status": "error", "note": str(e)[:50]}
    cache = rlct_cache.current()
    if cache is not None:
        cache.save()
        r["entries"] = len(cache.entries)
        r["hit"] = cache.n_hit
    r["wall"] = round(time.time() - t, 2)
    q.put({k: r.get(k) for k in
           ("name", "status", "rlct", "core", "core_route", "n_cache_put",
            "entries", "hit", "wall", "n_params", "arch", "degeneracy",
            "activation")})


def _poly_worker(idx: int, cache_path: str, opts: Dict, q) -> None:
    sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    import rlct_cache
    rlct_cache.enable(cache_path)
    import cases as C
    from run import run_case
    cs = C.structured_cases(4) + C.random_cases(n_list=(2, 3, 4))
    nm, f, g, fam = cs[idx]
    t = time.time()
    try:
        r = run_case(nm, f, g, fam, max_depth=20,
                     timeout_ms=opts["timeout_ms"])
    except Exception as e:                            # noqa: BLE001
        r = {"name": nm, "class": "error", "note": str(e)[:50]}
    cache = rlct_cache.current()
    if cache is not None:
        cache.save()
        r["entries"] = len(cache.entries)
        r["hit"] = cache.n_hit
    r["wall"] = round(time.time() - t, 2)
    r["status"] = r.get("class")
    q.put({k: r.get(k) for k in
           ("name", "status", "rlct", "n_cache_put", "entries", "hit",
            "wall", "n_vars")})


def _nn_cases(kind: str, opts: Dict):
    import nn_cases as N
    if kind == "relu":
        return N.relu_grid(max_params=opts["max_params"],
                           n_data=opts["n_data"],
                           seeds=opts["seeds"])
    return N.grid()


def _run(target, args, limit: int):
    q = mp.Queue()
    p = mp.Process(target=target, args=args + (q,))
    p.start(); p.join(limit)
    if p.is_alive():
        p.terminate(); p.join()
        return {"status": "timeout", "wall": float(limit)}
    try:
        return q.get_nowait()
    except Exception:
        return {"status": "crash", "wall": 0.0}


# ----------------------------------------------------------------------
# 本体
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="rlct_cache.json")
    ap.add_argument("--rebuild", action="store_true",
                    help="キャッシュを 1 から作り直す")
    ap.add_argument("--relu", action="store_true", help="ReLU の格子を貯める")
    ap.add_argument("--transformer", action="store_true",
                    help="Transformer 1 層 (Self-Attention + MLP(ReLU))")
    ap.add_argument("--tanh", action="store_true",
                    help="既存の nn_cases.grid() (relu + tanh) も貯める")
    ap.add_argument("--poly", action="store_true", help="多項式スイートも貯める")
    ap.add_argument("--stats", action="store_true", help="中身を見るだけ")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 件だけ")
    ap.add_argument("--start", type=int, default=0,
                    help="N 件目から再開する (途中で止めたとき用)")
    ap.add_argument("--skip-after-timeouts", type=int, default=3,
                    help="同じ (アーキテクチャ, 退化の型) が N 回続けて "
                         "打ち切られたら、その組は以降飛ばす。"
                         "0 で無効。大きいネットで時間を溶かさないため")
    ap.add_argument("--budget", type=int, default=0,
                    help="全体の時間予算 (秒)。超えたら打ち切って保存する")
    ap.add_argument("--timeout", type=int, default=30, help="1 件の上限 (秒)")
    ap.add_argument("--timeout-ms", type=int, default=4000, help="SMT の上限")
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--max-params", type=int, default=48)
    ap.add_argument("--n-data", default="6,8,10,14")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="", help="1 件ごとの記録を JSON で出す")
    args = ap.parse_args()

    import rlct_cache
    rlct_cache.enable(args.cache, rebuild=args.rebuild)
    if args.stats:
        c = rlct_cache.current()
        print(json.dumps(rlct_cache.stats(), ensure_ascii=False, indent=1))
        by = collections.Counter(e.route.split(":")[0] for e in
                                 c.entries.values())
        print("経路別:", dict(by.most_common()))
        byn = collections.Counter(e.n_vars for e in c.entries.values())
        print("変数の数別:", dict(sorted(byn.items())))
        hot = sorted(c.entries.values(), key=lambda e: -e.hits)[:10]
        print("よく当たるエントリ:")
        for e in hot:
            print(f"  hits={e.hits:<4} lambda={e.rlct:<8} {e.route:<22} "
                  f"{e.desc[:70]}")
        return 0
    print(f"# キャッシュ: {rlct_cache.current()}")

    mp.set_start_method("fork")
    opts = dict(timeout=args.timeout, timeout_ms=args.timeout_ms,
                max_depth=args.max_depth, max_params=args.max_params,
                n_data=tuple(int(v) for v in args.n_data.split(",") if v),
                seeds=tuple(int(v) for v in args.seeds.split(",") if v))

    import pickle
    recs: List[Dict] = []
    jobs: List[tuple] = []
    if args.relu:
        import nn_cases as N
        specs = N.relu_specs(max_params=args.max_params,
                             n_data=opts["n_data"], seeds=opts["seeds"])
        print(f"# ReLU 格子: {len(specs)} 件 (仕様のみ; 式は子プロセスで作る)")
        jobs += [("spec", s_, "") for s_ in specs]
    if args.transformer:
        import nn_cases as N
        specs = N.transformer_specs()
        print(f"# Transformer 1 層: {len(specs)} 件")
        jobs += [("tf", s_, "") for s_ in specs]
    if args.tanh:
        t = time.time()
        cs = _nn_cases("grid", opts)
        print(f"# nn_cases.grid(): {len(cs)} 件 (構築 {time.time()-t:.1f}s)")
        jobs += [("nn", pickle.dumps(c), "") for c in cs]
    if args.poly:
        import cases as C
        n = len(C.structured_cases(4) + C.random_cases(n_list=(2, 3, 4)))
        print(f"# 多項式スイート: {n} 件")
        jobs += [("poly", i, "") for i in range(n)]
    # 小さいものから順に解く。大きいネットは打ち切られることが多く、
    # 打ち切られたケースはキャッシュに何も足さないので後回しにする。
    def _cost(job):
        kind, payload, _ = job
        if kind == "spec":
            w = payload[0]
            return (sum(w[i + 1] * w[i] + w[i + 1] for i in range(len(w) - 1)),
                    payload[2])
        if kind == "tf":
            T, d, dk, dff, H = payload[:5]
            return (T * (H * (2 * d * dk + 2 * dk) + d * dff), payload[6])
        return (0, 0)
    jobs.sort(key=_cost)
    if args.start:
        jobs = jobs[args.start:]
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        print("何も指定されていません "
              "(--relu / --transformer / --tanh / --poly / --stats)")
        return 1

    t0 = time.time()
    bad = collections.Counter()      # (widths, deg) -> 連続打ち切り回数
    n_skipped = 0
    for n, (kind, idx, sub) in enumerate(jobs):
        if args.budget and time.time() - t0 > args.budget:
            print(f"  ... 時間予算 {args.budget}s に達したので打ち切り "
                  f"(残り {len(jobs)-n} 件)", flush=True)
            break
        fam = ((idx[0], idx[1]) if kind == "spec"
               else (tuple(idx[:5]), idx[5]) if kind == "tf" else None)
        if (args.skip_after_timeouts and fam is not None
                and bad[fam] >= args.skip_after_timeouts):
            n_skipped += 1
            continue
        if kind == "tf":
            r = _run(_tf_worker, (idx, args.cache, opts), args.timeout)
        elif kind == "spec":
            r = _run(_spec_worker, (idx, args.cache, opts), args.timeout)
        elif kind == "nn":
            r = _run(_nn_worker, (idx, args.cache, opts), args.timeout)
        else:
            r = _run(_poly_worker, (idx, args.cache, opts), args.timeout)
        recs.append(r)
        if fam is not None:
            if str(r.get("status")) in ("timeout", "crash", "error"):
                bad[fam] += 1
            else:
                bad[fam] = 0
        print(f"  [{n+1}/{len(jobs)}] {str(r.get('name'))[:44]:<44} "
              f"{str(r.get('status')):<9} core={r.get('core')} "
              f"lam={r.get('rlct')} put={r.get('n_cache_put')} "
              f"entries={r.get('entries')} {r.get('wall')}s", flush=True)

    rlct_cache.enable(args.cache)       # 子が保存した最終状態を読み直す
    print(f"\n# {rlct_cache.current()}")
    print("# status:", dict(collections.Counter(
        str(r.get("status")) for r in recs).most_common()))
    print(f"# 総時間 {time.time()-t0:.1f}s, "
          f"登録 {sum(int(r.get('n_cache_put') or 0) for r in recs)} 件"
          + (f", 打ち切りが続いて飛ばした {n_skipped} 件" if n_skipped else ""))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(recs, fh, ensure_ascii=False, indent=1, default=str)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
