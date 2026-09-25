r"""
timing.py — RLCT 計算の段階別 (phase) 実行時間を記録する軽量プロファイラ。

目的
----
「ネットワーク / 多項式のサイズ・次数と計算時間の関係」を測るために、
1 つの多項式に対する RLCT 計算を段階に分けて計時する。

段階 (phase) の名前は固定文字列:
    family        族ごとの閉じた式の判定 (families.family_rlct)
    newton        ニュートン非退化の判定 + LP (certify.rlct_via_newton)
    newton_chart  chart ごとのニュートン高速パス (resolve_singularity)
    ideal         イデアルを運ぶ解消 (ideal_resolve.resolve_ideal)
    resolve       ブローアップによる解消 (resolve_singularities)
    certify       証明書の検査 (certify.certify)
    lp            LP / ILP の呼び出し
    smt           Z3 の呼び出し
    total         rlct_certified 全体

使い方
------
>>> from timing import timed, record, last_record, timing_summary
>>> with record() as rec:              # 1 多項式ぶんの記録を開始
...     with timed("resolve"):
...         ...
>>> rec.phases            # {'resolve': 1.23, ...}
>>> rec.counts            # {'resolve': 1, ...}

ネストしても二重計上しない: 内側の phase の時間は内側にだけ入る
(外側は「自分の時間 + 内側」= 経過時間で持つので、合計したいときは
`rec.exclusive` を使う)。

スレッドローカルなので、並列に回しても混ざらない。
記録が始まっていない (record() の外の) 呼び出しは黙って無視される。
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "TimingRecord", "record", "timed", "add_meta", "bump",
    "current", "last_record", "timing_summary", "enabled", "set_enabled",
]

_local = threading.local()
_ENABLED = True


def set_enabled(flag: bool) -> None:
    """計時を止める (オーバーヘッドを完全に消したいとき)。"""
    global _ENABLED
    _ENABLED = bool(flag)


def enabled() -> bool:
    return _ENABLED


@dataclass
class TimingRecord:
    """1 つの多項式に対する計時結果。"""

    label: str = ""
    phases: Dict[str, float] = field(default_factory=dict)   # 経過 (ネスト込み)
    exclusive: Dict[str, float] = field(default_factory=dict)  # 自分だけの時間
    counts: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)
    wall: float = 0.0
    _stack: List[List] = field(default_factory=list, repr=False)
    _t0: float = 0.0

    # -- 記録 ---------------------------------------------------------
    def _enter(self, name: str) -> None:
        self._stack.append([name, time.perf_counter(), 0.0])

    def _exit(self) -> None:
        name, t0, child = self._stack.pop()
        dt = time.perf_counter() - t0
        self.phases[name] = self.phases.get(name, 0.0) + dt
        self.exclusive[name] = self.exclusive.get(name, 0.0) + max(dt - child, 0.0)
        self.counts[name] = self.counts.get(name, 0) + 1
        if self._stack:
            self._stack[-1][2] += dt

    def bump(self, name: str, n: int = 1) -> None:
        """時間ではなく回数だけ数える (LP 呼び出し回数など)。"""
        self.counts[name] = self.counts.get(name, 0) + n

    # -- 出力 ---------------------------------------------------------
    def as_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {
            "wall": round(self.wall, 4),
            "phases": {k: round(v, 4) for k, v in sorted(self.phases.items())},
            "exclusive": {k: round(v, 4)
                          for k, v in sorted(self.exclusive.items())},
            "counts": dict(sorted(self.counts.items())),
        }
        if self.meta:
            d["meta"] = dict(self.meta)
        if self.label:
            d["label"] = self.label
        return d

    def __str__(self) -> str:
        parts = ", ".join(f"{k}={v:.3f}s" for k, v in
                          sorted(self.exclusive.items(), key=lambda t: -t[1]))
        return f"{self.label or 'timing'}: wall={self.wall:.3f}s [{parts}]"


def current() -> Optional[TimingRecord]:
    """いま記録中の TimingRecord (無ければ None)。"""
    return getattr(_local, "rec", None)


def last_record() -> Optional[TimingRecord]:
    """直前に閉じた記録。rlct_certified の後に呼べば段階別の時間が取れる。"""
    return getattr(_local, "last", None)


@contextmanager
def record(label: str = "", **meta):
    """1 多項式ぶんの記録を開始する。ネストした場合は外側だけが有効。"""
    outer = getattr(_local, "rec", None)
    if outer is not None or not _ENABLED:
        # 既に記録中 (または無効) なら何もしない = 入れ子でも壊れない
        yield outer if outer is not None else TimingRecord(label=label)
        return
    rec = TimingRecord(label=label, meta=dict(meta))
    rec._t0 = time.perf_counter()
    _local.rec = rec
    try:
        yield rec
    finally:
        rec.wall = time.perf_counter() - rec._t0
        while rec._stack:                       # 例外で抜けても帳尻を合わせる
            rec._exit()
        _local.rec = None
        _local.last = rec


@contextmanager
def timed(name: str):
    """phase `name` の時間を測る。記録中でなければ何もしない。"""
    rec = getattr(_local, "rec", None)
    if rec is None or not _ENABLED:
        yield
        return
    rec._enter(name)
    try:
        yield
    finally:
        rec._exit()


def bump(name: str, n: int = 1) -> None:
    rec = getattr(_local, "rec", None)
    if rec is not None:
        rec.bump(name, n)


def add_meta(**kw) -> None:
    """サイズ・次数などの共変量を記録に足す。"""
    rec = getattr(_local, "rec", None)
    if rec is not None:
        rec.meta.update(kw)


# ----------------------------------------------------------------------
# 集計
# ----------------------------------------------------------------------
def timing_summary(records, *, by=("n_vars",), top: int = 8) -> str:
    """計時レコード (dict の列) を共変量ごとに集計して表にする。

    records の各要素は {'time': float, 'timing': {...}, 'n_vars': int, ...}
    という形 (bench/run.py, bench/nn_cases.py が書き出す形) を想定する。
    """
    import statistics as _st

    rows: Dict[tuple, List[dict]] = {}
    for r in records:
        key = tuple(r.get(b) for b in by)
        rows.setdefault(key, []).append(r)

    head = " / ".join(by)
    out = [f"{head:<18} {'n':>4} {'median':>9} {'mean':>9} {'max':>9}  内訳 (中央値)"]
    out.append("-" * 92)
    for key in sorted(rows, key=lambda k: tuple((x is None, x) for x in k)):
        rs = rows[key]
        ts = [float(r.get("time") or 0.0) for r in rs]
        ph: Dict[str, List[float]] = {}
        for r in rs:
            for k, v in (r.get("timing", {}) or {}).get("exclusive", {}).items():
                ph.setdefault(k, []).append(float(v))
        inner = ", ".join(
            f"{k}={_st.median(v):.2f}"
            for k, v in sorted(ph.items(), key=lambda t: -_st.median(t[1]))[:top]
        )
        out.append(f"{str(key):<18} {len(rs):>4} {_st.median(ts):>9.3f} "
                   f"{_st.mean(ts):>9.3f} {max(ts):>9.3f}  {inner}")
    return "\n".join(out)
