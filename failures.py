r"""
bench/failures.py — 失敗例の蓄積と自動縮約。

失敗を JSON に落として終わりにせず、

  1. 再現可能な形で保存する (式・変数・設定・失敗クラス)
  2. 最小の失敗例に自動で縮約する (項を落とす / 次数を下げる / 係数を単純に
     する / 変数を減らす)
  3. 次回以降のレグレッションに自動編入する

今回の会話でも、(xy+z^2)^2 を縮約して「反例がファイバー外の点だった」と
特定できたのが修正の決め手だった。縮約は人間が手でやると時間がかかるので
自動化する価値が高い。

使い方:
    from failures import record_failure, load_failures, minimize
    record_failure(f, gens, cls="resolve_max_depth", note="...")
    for case in load_failures():   # run.py がレグレッションに編入する
        ...
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import sympy as sp

FAIL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "failures")


class _Timeout(Exception):
    pass


def _alarm(sec):
    def handler(*_):
        raise _Timeout()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(sec)


# ----------------------------------------------------------------------
# 保存 / 読み出し
# ----------------------------------------------------------------------
def record_failure(f, gens, *, cls: str, note: str = "",
                   settings: Optional[Dict] = None, minimized: bool = False,
                   directory: str = FAIL_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    rec = {"f": sp.sstr(sp.expand(f)),
           "gens": [str(v) for v in gens],
           "class": cls, "note": note,
           "settings": settings or {}, "minimized": minimized}
    key = hashlib.sha1((rec["f"] + "|" + ",".join(rec["gens"])).encode()
                       ).hexdigest()[:12]
    path = os.path.join(directory, f"{key}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    return path


def load_failures(directory: str = FAIL_DIR) -> List[Tuple[str, sp.Expr,
                                                           Tuple[sp.Symbol, ...],
                                                           str]]:
    """蓄積した失敗例を cases.py と同じ形式で返す。"""
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        rec = json.load(open(os.path.join(directory, name), encoding="utf-8"))
        gens = tuple(sp.Symbol(v, real=True) for v in rec["gens"])
        f = sp.sympify(rec["f"], locals={str(v): v for v in gens})
        tag = "failed" + ("/min" if rec.get("minimized") else "")
        out.append((f"[{rec['class']}] {name[:-5]}", f, gens, tag))
    return out


# ----------------------------------------------------------------------
# 自動縮約
# ----------------------------------------------------------------------
def _variants(f, gens):
    """1 手で小さくする候補を返す (項を落とす / 次数を下げる / 係数を単純化)。"""
    p = sp.Poly(sp.expand(f), *gens)
    monoms, coeffs = p.monoms(), p.coeffs()
    out = []
    # 項を 1 つ落とす
    for i in range(len(monoms)):
        if len(monoms) <= 1:
            break
        terms = [c * sp.prod([v ** e for v, e in zip(gens, m)])
                 for j, (m, c) in enumerate(zip(monoms, coeffs)) if j != i]
        out.append(sp.expand(sum(terms)))
    # 指数を 1 つ下げる
    for i, m in enumerate(monoms):
        for j, e in enumerate(m):
            if e > 1:
                mm = list(m)
                mm[j] = e - 1
                terms = [(coeffs[k] * sp.prod([v ** ee for v, ee
                                               in zip(gens, (mm if k == i
                                                             else monoms[k]))]))
                         for k in range(len(monoms))]
                out.append(sp.expand(sum(terms)))
    # 係数を +-1 に単純化
    if any(abs(c) != 1 for c in coeffs):
        terms = [sp.sign(c) * sp.prod([v ** e for v, e in zip(gens, m)])
                 for m, c in zip(monoms, coeffs)]
        out.append(sp.expand(sum(terms)))
    # 変数を 1 つ 0 にする
    for v in gens:
        out.append(sp.expand(sp.expand(f).subs(v, 0)))
    seen, uniq = set(), []
    for g in out:
        if g == 0 or g.is_number:
            continue
        k = sp.sstr(g)
        if k not in seen and k != sp.sstr(sp.expand(f)):
            seen.add(k)
            uniq.append(g)
    return uniq


def _size(f, gens) -> Tuple[int, int, int]:
    p = sp.Poly(sp.expand(f), *gens)
    return (len(p.free_symbols), p.total_degree(), len(p.monoms()))


def minimize(f, gens, still_fails: Callable[[sp.Expr, Sequence], bool], *,
             max_steps: int = 60, per_case_timeout: int = 20, verbose=False):
    """失敗が保たれる範囲で f をできるだけ小さくする (貪欲)。

    still_fails(f, gens) が True を返す間だけ縮約する。判定には時間制限を
    掛け、時間内に終わらない場合は「失敗が保たれている」とは見なさない。
    """
    cur = sp.expand(f)
    cur_gens = tuple(gens)
    if not still_fails(cur, cur_gens):
        if verbose:
            print("    [warn] 出発点が述語を満たしていません "
                  "(縮約で見つかるのは別の失敗例になります)")
    steps = 0
    improved = True
    while improved and steps < max_steps:
        improved = False
        for cand in _variants(cur, cur_gens):
            if steps >= max_steps:
                break
            steps += 1
            cg = tuple(v for v in cur_gens if v in cand.free_symbols) or cur_gens
            if _size(cand, cg) >= _size(cur, cur_gens):
                continue
            try:
                _alarm(per_case_timeout)
                ok = still_fails(cand, cg)
            except Exception:
                ok = False
            finally:
                signal.alarm(0)
            if ok:
                cur, cur_gens = cand, cg
                improved = True
                if verbose:
                    print(f"    縮約 -> {cur}")
                break
    return cur, cur_gens


def resolve_fails_predicate(*, max_depth: int = 12, weighted=True,
                            prune=False):
    """「解消が止まらない」ことだけを見る軽い述語。

    run_case を丸ごと呼ぶと族の経路 (z3 を含む) まで走って遅いので、
    resolve_max_depth の縮約にはこちらを使う。
    """
    from resolve_singularity import ResolutionFailure, resolve_singularities

    def pred(f, gens):
        try:
            resolve_singularities(f, tuple(gens), prune=prune,
                                  weighted=weighted, max_depth=max_depth)
            return False
        except ResolutionFailure:
            return True
        except Exception:
            return False

    return pred


def make_predicate(cls: str, *, max_depth: int = 12, timeout_ms: int = 3000):
    """run.run_case の分類が cls と一致するかを返す述語を作る (重い)。

    証明側の失敗 (nc_refuted, smt_unknown など) の縮約に使う。
    解消が止まらない例には resolve_fails_predicate のほうが速い。
    """
    import run as run_mod

    def pred(f, gens):
        rec = run_mod.run_case("probe", f, tuple(gens), "probe",
                               max_depth=max_depth, timeout_ms=timeout_ms)
        return rec["class"] == cls

    return pred


def _demo():
    import sympy as sp
    x, y, z = sp.symbols("x y z", real=True)

    # 重み付きを切ると x^3+y^4+z^5 は止まらない。そこから最小の失敗例を探す。
    f = x**3 + y**4 + z**5 + x * y * z
    print("元の失敗例:", f, " (weighted=False では解消が止まらない)")
    pred = resolve_fails_predicate(max_depth=8, weighted=False)
    print("  元が失敗するか:", pred(f, (x, y, z)))
    g, gg = minimize(f, (x, y, z), pred, max_steps=40, per_case_timeout=6,
                     verbose=True)
    print("縮約後:", g, " 変数", list(gg))
    p = record_failure(g, gg, cls="resolve_max_depth",
                       note="自動縮約 (weighted=False で非停止)", minimized=True)
    print("保存:", p)
    for n, ff, gs, tag in load_failures():
        print(f"  蓄積: {n}  f={ff}  [{tag}]")
    print("\n蓄積した失敗例は run.py --failures でレグレッションに編入される。")


if __name__ == "__main__":
    _demo()
