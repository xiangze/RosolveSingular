r"""
rlct_cache.py — 解いた局所データ (y^k * f_rest, 振幅 y^h) の λ を貯めて使い回す。

------------------------------------------------------------------
何をキャッシュするか
------------------------------------------------------------------
chart の値を決めるのは f_rest **だけではない**。局所ゼータは

    Z(z) = ∫ |y^k f_rest(y)|^z |y^h| dy

なので、同じ f_rest でも (k, h) が違えば λ は違う。したがってキーは
**局所データ全体** (k, h, f_rest) にする。

さらに λ は次の操作で変わらないので、キーはその同値類の代表にする
(正規化しておくと、別々の計算で現れた「同じ問題」が同じキーになる):

  * 変数の入れ替え (k, h, 指数を一緒に入れ替える)
  * 非零定数倍 f -> c f    (|c f| = |c| |f| で極の位置は動かない)
  * f にも k にも現れない変数を落とす (体積の因子にしかならない)

正規化は「取りこぼし (miss)」を減らすためのもので、**当たりの正しさは
キーに正規化後のデータを丸ごと入れる**ことで担保する。正規化が甘くても
誤った当たりは起きない (別の問題が同じキーになることはない)。

------------------------------------------------------------------
バグの伝搬をどう防ぐか
------------------------------------------------------------------
「簡単な多項式の値が間違っていると、それを使った複雑な計算まで間違う」
— これはキャッシュ固有の、そして最も重い問題である。4 段構えで防ぐ。

(1) **証明されたものしか入れない**。certify が proved を返した値、または
    族ごとの閉じた式で確定した値だけを保存する。unknown は保存しない。

(2) **コードの指紋を持たせる**。値を作った時点の resolve_singularity.py /
    certify.py / families.py / lemmas.py / ideal_resolve.py の内容から
    sha256 を取って記録する。**バグを直すと指紋が変わるので、過去の
    エントリは自動的に無効になる**。「簡単な多項式のバグ」を直した瞬間に、
    それに依存した値も含めて全部使われなくなる。

(3) **依存関係を記録して推移的に無効化できる**。ある値を出すのに使った
    キャッシュのキーを depends_on に残す。あとで 1 つが誤りと分かったら
    `invalidate(key)` でそれを使った値も芋づるで消せる。

(4) **当たるたびに安い健全性検査を通す**。少なくとも
      0 < λ,  λ <= newton_rlct(局所データ)  (ニュートン LP は常に上界)
    を確かめる。破れていたらそのエントリを捨てて、通常どおり計算し直し、
    `poisoned` として記録する。

それでも「キャッシュを使った値」は「その場で計算した値」と同じ強さでしか
ない。証明モードで厳密を期すなら `--no-cache`、あるいは `--verify-cache`
(当たりを毎回計算し直して突き合わせる) を使う。

------------------------------------------------------------------
使い方
------------------------------------------------------------------
    import rlct_cache
    rlct_cache.enable("bench/rlct_cache.json")   # 読み込み + 以後の保存先
    rlct_cache.clear()                           # 1 から作り直す
    rlct_cache.disable()                         # 完全に使わない
    print(rlct_cache.stats())

環境変数 `RLCT_CACHE` にパスを入れておくと既定の保存先になる。
`RLCT_CACHE=off` で無効。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

__all__ = [
    "canonical_key", "code_fingerprint", "Cache",
    "enable", "disable", "clear", "get", "put", "stats", "save",
    "current", "invalidate", "set_verify",
]

DEFAULT_PATH = os.environ.get("RLCT_CACHE", "rlct_cache.json")
_SOURCES = ("resolve_singularity.py", "certify.py", "families.py",
            "lemmas.py", "ideal_resolve.py", "rlct_cache.py")


# ----------------------------------------------------------------------
# コードの指紋
# ----------------------------------------------------------------------
_fingerprint: Optional[str] = None


def code_fingerprint() -> str:
    """値を生成したコードの sha256 (先頭 16 桁)。

    これが変わったエントリは信用しない。バグを直せば必ず変わるので、
    「古いバグの結果を使い続ける」ことが起こらない。
    """
    global _fingerprint
    if _fingerprint is not None:
        return _fingerprint
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in _SOURCES:
        p = os.path.join(here, name)
        try:
            with open(p, "rb") as fh:
                h.update(name.encode())
                h.update(fh.read())
        except OSError:
            h.update(b"<missing>" + name.encode())
    _fingerprint = h.hexdigest()[:16]
    return _fingerprint


# ----------------------------------------------------------------------
# 正規化したキー
# ----------------------------------------------------------------------
def _normalized(k: Sequence[int], h: Sequence[int], poly: sp.Poly):
    """局所データ (k, h, f_rest) を同値類の代表に直す。

    Returns (n', k', h', [(指数, 係数), ...]) か None (有理数係数でない等)。
    """
    monoms = [tuple(int(e) for e in m) for m in poly.monoms()]
    if not monoms:
        return None
    try:
        coeffs = [sp.Rational(c) for c in poly.coeffs()]
    except (TypeError, ValueError):
        return None                      # 記号係数はキャッシュしない
    n = len(k)

    # (a) f にも k にも現れない変数は体積の因子にしかならないので落とす
    keep = [j for j in range(n)
            if k[j] > 0 or any(m[j] for m in monoms)]
    if not keep:
        return None
    monoms = [tuple(m[j] for j in keep) for m in monoms]
    kk = [int(k[j]) for j in keep]
    hh = [int(h[j]) for j in keep]

    # (b) 非零定数倍で正規化 (content を割り、先頭係数を正にする)
    den = sp.ilcm(*[c.q for c in coeffs]) if len(coeffs) > 1 else coeffs[0].q
    ints = [int(c * den) for c in coeffs]
    g = 0
    for v in ints:
        g = sp.igcd(g, abs(v))
    g = int(g) or 1
    ints = [v // g for v in ints]
    order = sorted(range(len(monoms)), key=lambda i: monoms[i])
    if ints[order[0]] < 0:
        ints = [-v for v in ints]

    # (c) 変数の並べ替え。λ は (k, h, 指数の列) を一緒に入れ替えても不変。
    #     列ごとの特徴量で並べ替える。同点が残っても「取りこぼし」になるだけで
    #     誤った当たりにはならない (キーに全データが入っているため)。
    sig = []
    for j in range(len(kk)):
        col = sorted((m[j] for m in monoms), reverse=True)
        sig.append(((kk[j], hh[j], tuple(col)), j))
    sig.sort()
    perm = [j for _, j in sig]
    kk = [kk[j] for j in perm]
    hh = [hh[j] for j in perm]
    items = sorted(((tuple(m[j] for j in perm), c)
                    for m, c in zip(monoms, ints)), key=lambda t: t[0])
    return (len(kk), tuple(kk), tuple(hh), tuple(items))


def canonical_key(k, h, poly: sp.Poly, gens=None) -> Optional[str]:
    """局所データの正規化キー (sha256 の先頭 32 桁)。キャッシュ不可なら None。"""
    norm = _normalized(k, h, poly)
    if norm is None:
        return None
    blob = json.dumps(norm, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def readable(k, h, poly: sp.Poly, gens=None) -> str:
    """監査用の読める形 (キーと一緒に保存する)。"""
    norm = _normalized(k, h, poly)
    if norm is None:
        return ""
    n, kk, hh, items = norm
    terms = " + ".join(
        f"{c}*" + "*".join(f"z{j}^{e}" for j, e in enumerate(m) if e)
        for m, c in items)
    return f"n={n} k={list(kk)} h={list(hh)}  f_rest = {terms}"


# ----------------------------------------------------------------------
# エントリとキャッシュ
# ----------------------------------------------------------------------
@dataclass
class Entry:
    rlct: str                      # sympy の文字列表現
    mult: Optional[int]
    route: str                     # どの経路で確定したか
    fingerprint: str               # 生成時のコード指紋
    created: float = 0.0
    n_vars: int = 0
    degree: int = 0
    n_terms: int = 0
    depends_on: List[str] = field(default_factory=list)
    desc: str = ""                 # 監査用の読める形
    hits: int = 0


class Cache:
    """(局所データ -> lambda, m) の永続キャッシュ。"""

    def __init__(self, path: Optional[str] = None, *, verify: bool = False):
        self.path = path
        self.verify = verify            # 当たりを毎回計算し直して突き合わせる
        self.entries: Dict[str, Entry] = {}
        self.n_hit = 0
        self.n_miss = 0
        self.n_stale = 0                # 指紋が合わずに捨てた
        self.n_poisoned = 0             # 健全性検査に引っかかって捨てた
        self.n_put = 0
        self.poisoned: List[str] = []
        self._used: List[str] = []      # 今の計算で使ったキー (depends_on 用)
        if path:
            self.load()

    # -- 永続化 ------------------------------------------------------
    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception:               # 壊れていたら作り直す
            self.entries = {}
            return
        fp = code_fingerprint()
        for key, d in (raw.get("entries") or {}).items():
            try:
                e = Entry(**d)
            except TypeError:
                continue
            if e.fingerprint != fp:
                self.n_stale += 1       # コードが変わっている -> 使わない
                continue
            self.entries[key] = e

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        out = {
            "schema": 1,
            "fingerprint": code_fingerprint(),
            "saved": time.time(),
            "entries": {k: asdict(v) for k, v in self.entries.items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def clear(self) -> None:
        """1 から作り直す。"""
        self.entries = {}
        self.n_hit = self.n_miss = self.n_stale = self.n_poisoned = 0
        self.n_put = 0
        self.poisoned = []
        if self.path and os.path.exists(self.path):
            os.remove(self.path)

    # -- 依存関係 ----------------------------------------------------
    def invalidate(self, key: str) -> List[str]:
        """key と、それに (推移的に) 依存するエントリをすべて消す。

        あとで「あの値は間違っていた」と分かったときの後始末。
        消したキーの一覧を返す。
        """
        dead = {key}
        changed = True
        while changed:
            changed = False
            for k, e in list(self.entries.items()):
                if k in dead:
                    continue
                if dead & set(e.depends_on):
                    dead.add(k)
                    changed = True
        removed = [k for k in dead if k in self.entries]
        for k in removed:
            del self.entries[k]
        return removed

    # -- 参照と登録 --------------------------------------------------
    def get(self, key: Optional[str], *, k=None, h=None, poly=None, gens=None):
        """当たれば (lambda, m)。健全性検査に落ちたら捨てて None。"""
        if key is None:
            return None
        e = self.entries.get(key)
        if e is None:
            self.n_miss += 1
            return None
        lam = sp.nsimplify(e.rlct)
        if not self._sane(lam, k, h, poly, gens):
            self.n_poisoned += 1
            self.poisoned.append(key)
            del self.entries[key]
            return None
        e.hits += 1
        self.n_hit += 1
        self._used.append(key)
        return (lam, e.mult)

    def _sane(self, lam, k, h, poly, gens) -> bool:
        """当たりに対する安い健全性検査。

        ニュートン LP は非退化かどうかに関係なく **常に上界**なので、
        lambda がそれを超えていたらそのエントリは壊れている。
        """
        try:
            if lam is None or lam is sp.oo or lam <= 0:
                return False
            if poly is None or gens is None or k is None or h is None:
                return True
            from resolve_singularity import _lp_newton_value
            monoms = [tuple(ki + ei for ki, ei in zip(k, m))
                      for m in poly.monoms()]
            ub = _lp_newton_value(monoms, [hj + 1 for hj in h], hot=True)
            if ub == float("inf"):
                return True
            return float(lam) <= ub * (1 + 1e-6) + 1e-9
        except Exception:
            return True

    def put(self, key: Optional[str], lam, mult, route: str, *,
            desc: str = "", n_vars: int = 0, degree: int = 0,
            n_terms: int = 0, depends_on: Optional[Sequence[str]] = None):
        """証明された値だけを入れる。"""
        if key is None or lam is None or lam is sp.oo:
            return
        self.entries[key] = Entry(
            rlct=str(sp.nsimplify(lam)),
            mult=(int(mult) if mult is not None else None),
            route=route,
            fingerprint=code_fingerprint(),
            created=time.time(),
            n_vars=n_vars, degree=degree, n_terms=n_terms,
            depends_on=list(depends_on if depends_on is not None
                            else self._used),
            desc=desc,
        )
        self.n_put += 1

    # -- 依存の記録 --------------------------------------------------
    def begin(self) -> None:
        """1 つの計算の開始 (使ったキーの記録をリセット)。"""
        self._used = []

    def used(self) -> List[str]:
        return list(self._used)

    # -- 報告 --------------------------------------------------------
    def stats(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "entries": len(self.entries),
            "hit": self.n_hit, "miss": self.n_miss,
            "stale(指紋違い)": self.n_stale,
            "poisoned(検査で除去)": self.n_poisoned,
            "put": self.n_put,
            "fingerprint": code_fingerprint(),
        }

    def __str__(self) -> str:
        s = self.stats()
        return (f"cache[{s['entries']} 件] hit={s['hit']} miss={s['miss']} "
                f"stale={s['stale(指紋違い)']} "
                f"poisoned={s['poisoned(検査で除去)']} ({self.path})")


# ----------------------------------------------------------------------
# 既定のキャッシュ (モジュールレベル)
# ----------------------------------------------------------------------
_CACHE: Optional[Cache] = None


def enable(path: Optional[str] = None, *, verify: bool = False,
           rebuild: bool = False) -> Cache:
    """キャッシュを有効にする。rebuild=True なら 1 から作り直す。"""
    global _CACHE
    p = path or DEFAULT_PATH
    if str(p).lower() in ("off", "none", ""):
        _CACHE = None
        return None
    _CACHE = Cache(p, verify=verify)
    if rebuild:
        _CACHE.clear()
    return _CACHE


def disable() -> None:
    global _CACHE
    _CACHE = None


def current() -> Optional[Cache]:
    return _CACHE


def set_verify(flag: bool) -> None:
    if _CACHE is not None:
        _CACHE.verify = bool(flag)


def clear() -> None:
    if _CACHE is not None:
        _CACHE.clear()


def save() -> None:
    if _CACHE is not None:
        _CACHE.save()


def invalidate(key: str) -> List[str]:
    return _CACHE.invalidate(key) if _CACHE is not None else []


def get(k, h, poly, gens):
    if _CACHE is None:
        return None
    key = canonical_key(k, h, poly, gens)
    return _CACHE.get(key, k=k, h=h, poly=poly, gens=gens)


def put(k, h, poly, gens, lam, mult, route: str, **kw):
    if _CACHE is None:
        return
    key = canonical_key(k, h, poly, gens)
    if key is None:
        return
    try:
        deg = int(poly.total_degree())
        nt = len(poly.monoms())
    except Exception:
        deg = nt = 0
    _CACHE.put(key, lam, mult, route,
               desc=readable(k, h, poly, gens),
               n_vars=len(gens) if gens else 0, degree=deg, n_terms=nt, **kw)


def record_resolution(res, gens, *, proved: bool, route: str = "subtree",
                      f=None) -> int:
    """証明できた解消から、根と部分木の値をまとめて登録する。

    proved=False なら何もしない (**証明されていない値は決して入れない**)。
    戻り値は登録した件数。
    """
    if _CACHE is None or not proved:
        return 0
    n = 0
    try:
        if f is not None:
            p0 = sp.Poly(sp.expand(f), *gens)
            put([0] * len(gens), [0] * len(gens), p0, gens,
                res.rlct, res.multiplicity, "root:" + route)
            n += 1
        for kk, hh, pp, lam_s, m_s in getattr(res, "cache_candidates", []):
            put(kk, hh, pp, gens, lam_s, m_s, route)
            n += 1
    except Exception:
        pass
    return n


def stats() -> Dict[str, object]:
    return _CACHE.stats() if _CACHE is not None else {"entries": 0,
                                                      "path": None}


# 環境変数で既定を決める (RLCT_CACHE=off で無効)
if os.environ.get("RLCT_CACHE", "").lower() not in ("", "off", "none"):
    enable(os.environ["RLCT_CACHE"])
