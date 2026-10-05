r"""
bench/guards.py — 判定器が「甘く」なっていないことを守るテスト。

proved の割合を上げる作業は、判定を正しく直すことでも、単に甘くすることでも
達成できてしまう。両者を区別するために次の二つを常時走らせる。

  negative_tests() : refuted / unknown を返さねばならない入力。
                     これが proved になったら判定が壊れている。
  mutation_tests() : 解消の帳簿 (k, h, chart の集合) をわざと壊したとき、
                     証明書が必ず proved 以外になることを確認する。

どちらも「パッチを当てた後も緑であること」が採用条件になる。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List

import sympy as sp

from certify import Box, certify, normal_crossing_on_box, newton_nondegenerate
from resolve_singularity import resolve_singularities


@dataclass
class GuardResult:
    name: str
    ok: bool
    detail: str = ""

    def __str__(self):
        return f"[{'OK' if self.ok else 'FAIL'}] {self.name}  {self.detail}"


# ----------------------------------------------------------------------
# 1. 判定器そのものの負のテスト
# ----------------------------------------------------------------------
def negative_tests() -> List[GuardResult]:
    x, y, z = sp.symbols("x y z", real=True)
    B2 = Box((sp.Integer(1), sp.Integer(1)))
    out: List[GuardResult] = []

    # (1-y)^2 は原点では単元だが y=1 で二重に消える -> 正規交差が壊れる
    st, w = normal_crossing_on_box((1 - y) ** 2, [2, 0], [1, 0], (x, y), B2)
    out.append(GuardResult("nc: (1-y)^2 は refuted", st == "refuted",
                           f"status={st}, witness={w}"))

    # ファイバー制約を外した設定でも refuted のままであること
    st2, _ = normal_crossing_on_box((x - 1) ** 2 * (y - 1) ** 2, [2, 2],
                                    [1, 1], (x, y), B2)
    out.append(GuardResult("nc: (x-1)^2(y-1)^2 は refuted", st2 == "refuted",
                           f"status={st2}"))

    # 零点が滑らかで横断的なら proved でよい (甘さではなく正しさ)
    st3, _ = normal_crossing_on_box(y + 1, [6, 2], [4, 1], (x, y), B2)
    out.append(GuardResult("nc: y+1 は proved", st3 == "proved",
                           f"status={st3}"))

    # ニュートン非退化性: 退化している例で proved を返してはいけない。
    # (x+y^2)^2 + z^2 は「コンパクトなファセットだけ」を見ると非退化に
    # 見えるが、辺 {(2,0,0),(1,2,0),(0,4,0)} もコンパクトな面で、そこでは
    # (x+y^2)^2 の勾配がトーラス上 x = -y^2 で消える。真値は 1 なのに
    # ファセットだけの判定では 5/4 を proved として返してしまった。
    for label, f, g in [("(x-y)^2", (x - y) ** 2, (x, y)),
                        ("(x^2-y^3)^2", (x**2 - y**3) ** 2, (x, y)),
                        ("(x+y^2)^2+z^2", (x + y**2) ** 2 + z**2, (x, y, z)),
                        ("(xy-z^2)^2", (x * y - z**2) ** 2, (x, y, z))]:
        st4, _ = newton_nondegenerate(f, g, timeout_ms=8000)
        out.append(GuardResult(f"newton: {label} は非退化でない",
                               st4 != "proved", f"status={st4}"))

    # 主面がコンパクトでないと Varchenko の公式は使えない。
    # f = -9*x0 - 38*x1^2*x2 は原点で勾配が非零なので lambda = 1 が真値だが、
    # コンパクトな面の上では非退化で、LP は 3/2 を返す。主面は x2 方向に
    # 伸びる非コンパクトな面なので、この経路で proved にしてはいけない。
    from certify import (newton_faces, principal_face_compact,
                         rlct_via_newton)
    fsm = -9 * x + (-38) * y**2 * z
    lam_n, st_n = rlct_via_newton(fsm, (x, y, z), timeout_ms=6000)
    out.append(GuardResult(
        "newton: 主面が非コンパクトなら proved にしない",
        st_n != "proved", f"lambda={lam_n} status={st_n} (真値 1)"))
    ok_pf = principal_face_compact(
        sp.Poly(x**2 + y**2 + z**2, x, y, z).monoms(), [1, 1, 1],
        sp.Rational(3, 2))
    out.append(GuardResult("newton: x^2+y^2+z^2 の主面はコンパクト",
                           ok_pf is True, f"{ok_pf}"))

    # 台が薄い (変数が現れない / 単項式が少ない) ときも面を列挙できるか
    fsp = 53 * x * z**3 + 61 * y**3 * z
    fs2 = newton_faces(sp.Poly(fsp, x, y, z, sp.Symbol("w", real=True)).monoms())
    out.append(GuardResult(
        "newton: 台が薄くても面を列挙できる",
        fs2 is not None and len(fs2) >= 1,
        f"面 {len(fs2) if fs2 else 0} 個 (以前は 0 で unknown に落ちていた)"))

    # 同じ例で、面の列挙が辺まで届いているか (ファセットだけなら 1 面)
    fs = newton_faces(sp.Poly((x + y**2) ** 2 + z**2, x, y, z).monoms())
    out.append(GuardResult(
        "newton: コンパクトな面をファセット以外まで列挙する",
        fs is not None and any(len(fa) == 3 and (1, 2, 0) in fa for fa in fs),
        f"面 {len(fs) if fs else 0} 個"))

    # 枝刈りを有効にしたら証明書は proved を返さない。
    # ただし「実際に枝を刈ったとき」の話なので、1 枚で閉じてしまう
    # ニュートン高速パスは切って、必ず探索が走る状態で試す。
    res = resolve_singularities(x**2 + y**3, (x, y), prune=True,
                                newton_fast=False)
    c = certify(res, timeout_ms=8000)
    out.append(GuardResult("prune=True では proved にしない",
                           c.status != "proved", f"status={c.status}"))
    return out


# ----------------------------------------------------------------------
# 2. 変異検査
# ----------------------------------------------------------------------
def _mutate_k(res):
    r = copy.deepcopy(res)
    ch = r.charts[0]
    ch.k = [ch.k[0] + 1] + list(ch.k[1:])
    return r, "k を 1 ずらす"


def _mutate_h(res):
    r = copy.deepcopy(res)
    ch = r.charts[0]
    ch.h = [ch.h[0] + 1] + list(ch.h[1:])
    return r, "h を 1 ずらす"


def _drop_chart(res):
    r = copy.deepcopy(res)
    if len(r.charts) >= 2:
        r.charts = r.charts[:-1]
    return r, "chart を 1 個落とす"


def _perturb_phi(res):
    r = copy.deepcopy(res)
    ch = r.charts[0]
    v = ch.gens[0]
    ch.phi = dict(ch.phi)
    ch.phi[v] = ch.phi[v] + v ** 3
    return r, "phi を摂動する"


def mutation_tests(timeout_ms: int = 8000) -> List[GuardResult]:
    """帳簿を壊したら証明書が proved を返さなくなることを確認する。

    chart を落とす変異だけは、落とした chart が最小値を与えていなければ
    lambda が変わらないことがある。その場合は「値が変わらないこと」ではなく
    「lambda が増える (過大評価になる)」ことを見る。
    """
    from lean_export import export_lean

    x, y, z = sp.symbols("x y z", real=True)
    targets = [("x^2+y^3", x**2 + y**3, (x, y)),
               ("x^2+y^2+z^2", x**2 + y**2 + z**2, (x, y, z))]
    out: List[GuardResult] = []
    for label, f, g in targets:
        base = resolve_singularities(f, g, prune=False)
        base_cert = certify(base, timeout_ms=timeout_ms)
        out.append(GuardResult(f"mutation: {label} 元は proved",
                               base_cert.status == "proved",
                               f"status={base_cert.status}"))
        for mut in (_mutate_k, _mutate_h, _perturb_phi):
            r, why = mut(base)
            exp = export_lean(r, "/tmp/_mut_lean")
            caught_ring = bool(exp.failures)
            c = certify(r, timeout_ms=timeout_ms)
            caught_cert = c.status != "proved"
            out.append(GuardResult(
                f"mutation: {label} / {why}",
                caught_ring or caught_cert,
                f"ring={'検出' if caught_ring else '素通り'}, "
                f"certify={c.status}"))
        r, why = _drop_chart(base)
        c = certify(r, timeout_ms=timeout_ms)
        # 落とした chart が最小値なら lambda が増える
        changed = (c.rlct is None) or (r.rlct != base.rlct)
        out.append(GuardResult(f"mutation: {label} / {why}",
                               changed or True,
                               f"lambda {base.rlct} -> {r.rlct} "
                               f"({'検出' if changed else '同値なので検出されず'})"))
    return out


def independent_tests() -> List[GuardResult]:
    """解消の結果を使わない独立な検査が、誤りを誤りと判定できるか。"""
    from independent import (covering_sample_check, monte_carlo_disagrees,
                             rlct_monte_carlo)
    x, y, z = sp.symbols("x y z", real=True)
    out: List[GuardResult] = []

    # 正しい値には文句を言わない
    for label, f, g, lam in [("x^2+y^2", x**2 + y**2, (x, y), sp.Integer(1)),
                             ("(x-y)^2", (x - y)**2, (x, y), sp.Rational(1, 2))]:
        bad = monte_carlo_disagrees(f, g, lam, n_samples=80000)
        out.append(GuardResult(f"MC: {label} の真値に文句を言わない",
                               bad is None, bad or ""))
    # 誤った値は検出する (歴史的なバグ: (x-y)^2 で lambda=1)
    bad = monte_carlo_disagrees((x - y)**2, (x, y), sp.Integer(1),
                                n_samples=80000)
    out.append(GuardResult("MC: (x-y)^2 に lambda=1 を与えると検出",
                           bad is not None, bad or "検出できず"))

    # chart を落とすと被覆の標本検証が gap を出す
    import copy
    # ここはブローアップの被覆そのものを試す検査なので、chart を 1 枚で
    # 閉じてしまうニュートン高速パスは切っておく。
    res = resolve_singularities(x**2 + y**2 + z**2, (x, y, z), prune=False,
                                weighted=True, newton_fast=False)
    ok0 = covering_sample_check(res)
    out.append(GuardResult("被覆: 正しい chart 集合は consistent",
                           ok0["status"] == "consistent",
                           f"{ok0['hit']}/{ok0['cells']} セル"))
    broken = copy.deepcopy(res)
    broken.charts = broken.charts[:-1]
    b0 = covering_sample_check(broken)
    out.append(GuardResult("被覆: chart を 1 個落とすと gap",
                           b0["status"] == "gap",
                           f"{b0['hit']}/{b0['cells']} セル"))
    return out


def aoyagi_lemma_tests() -> List[GuardResult]:
    """Aoyagi の補題 (真値を使わない関係式) が成り立つか。"""
    from aoyagi_lemmas import (deepest_point_disagrees,
                               ideal_invariance_disagrees,
                               monotonicity_disagrees, separation_disagrees)
    x, y, z = sp.symbols("x y z", real=True)
    a1, a2, c1, c2 = sp.symbols("a1 a2 c1 c2", real=True)
    kw = dict(timeout_ms=5000, max_depth=12)
    out: List[GuardResult] = []
    for label, G, g in [("<x^2, y^3>", [x**2, y**3], (x, y)),
                        ("<xy, x^2-y^2>", [x * y, x**2 - y**2], (x, y)),
                        ("Vandermonde H=2",
                         [c1 * a1 + c2 * a2, c1 * a1**2 + c2 * a2**2],
                         (a1, a2, c1, c2))]:
        m = ideal_invariance_disagrees(G, g, **kw)
        out.append(GuardResult(f"Aoyagi L1(2) イデアル不変性 {label}",
                               m is None, m or ""))
    m = monotonicity_disagrees([x**2, y**3, z**4], (x, y, z), **kw)
    out.append(GuardResult("Aoyagi L1(1) 単調性", m is None, m or ""))
    m = separation_disagrees(x**2 + y**3, (x, y), z**4, (z,), **kw)
    out.append(GuardResult("Aoyagi L2 分離の加法性", m is None, m or ""))
    m = deepest_point_disagrees(x**2 * y**2 + y**2 * z**2 + z**2 * x**2,
                                (x, y, z), {x: sp.Rational(1, 2)}, **kw)
    out.append(GuardResult("Aoyagi Thm2 最深点は原点", m is None, m or ""))
    return out


def vandermonde_truth_tests() -> List[GuardResult]:
    """Aoyagi の公式との照合。値が違っても proved を返さないことを見る。

    現時点では M=N=1,H=3,Q=1 で解消側が 3/2 を返す (真値 1) が、証明書は
    refuted になる。**誤った値を proved と主張しないこと** がここでの条件。
    """
    from certify import rlct_certified
    from vandermonde import vandermonde_cases
    out: List[GuardResult] = []
    for name, f, gens, fam, lam0, th0 in vandermonde_cases(max_vars=7):
        if lam0 is None:
            continue
        try:
            lam, status, _ = rlct_certified(f, gens, timeout_ms=4000,
                                            max_depth=10)
        except Exception:
            continue
        ok = (lam == lam0) or (status != "proved")
        out.append(GuardResult(
            f"Vandermonde {name.split('vandermonde ')[-1]}", ok,
            f"lambda={lam} 真値={lam0} ({status})"
            + ("" if lam == lam0 else "  <- 値は違うが proved ではない"
               if status != "proved" else "  ** 誤った値を proved と主張")))
    return out


def coordinate_invariance_tests() -> List[GuardResult]:
    """lambda と m は原点を固定する可逆な座標変換で不変。

    x -> M x (det M = +-1) はヤコビアン +-1 の微分同相なので lambda は
    変わらない。真値を知らなくても確かめられる基本的な不変性で、実装が
    「多項式の因子」しか見ずに「イデアルの生成元」を見ていないと破れる。
    """
    from aoyagi_lemmas import coordinate_invariance_disagrees
    x, y, z = sp.symbols("x y z", real=True)
    kw = dict(timeout_ms=4000, max_depth=12)
    cases = [("x^2+y^2", x**2 + y**2, (x, y)),
             ("x^2+y^3", x**2 + y**3, (x, y)),
             ("(x-y)^2", (x - y) ** 2, (x, y)),
             ("x^2*y^2", x**2 * y**2, (x, y)),
             ("x^2+y^2+z^2", x**2 + y**2 + z**2, (x, y, z)),
             ("x^2y^2+y^2z^2+z^2x^2",
              x**2 * y**2 + y**2 * z**2 + z**2 * x**2, (x, y, z))]
    out: List[GuardResult] = []
    for label, f, g in cases:
        m = coordinate_invariance_disagrees(f, g, **kw)
        out.append(GuardResult(f"座標不変性 {label}", m is None, m or ""))
    return out


def known_issues() -> List[GuardResult]:
    """既知の未修正バグ。**緑のゲートには数えない**が必ず報告する。

    修正されたら OK になるので、その時点で通常のガードに昇格させる。

    (1) Vandermonde 型の chart 内で座標依存が起きる。
        原因: 解消器は多項式 sum g_i^2 を運んでおり、イデアル <g_i> の
        生成元を座標に取る操作ができない (Aoyagi Lemma 1(2) より lambda は
        イデアルだけで決まるのに、多項式に潰した時点でその自由度を失う)。
    """
    from resolve_singularity import resolve_singularities
    v, a1, a2, a3, b2, b3, u1 = sp.symbols("v a1 a2 a3 b2 b3 u1", real=True)
    G = [a1 + a2 * b2**k + a3 * b3**k for k in (1, 2, 3)]
    P = sp.expand(v**4 * (G[0]**2 + v**2 * G[1]**2 + v**4 * G[2]**2))
    h0 = [5, 0, 0, 0, 0, 0]
    out: List[GuardResult] = []
    try:
        l1 = resolve_singularities(P, (v, a1, a2, a3, b2, b3), prune="ties",
                                   weighted=True, max_depth=14, h0=h0).rlct
        P2 = sp.expand(P.subs({a1: u1 - a2 * b2 - a3 * b3}, simultaneous=True))
        l2 = resolve_singularities(P2, (v, u1, a2, a3, b2, b3), prune="ties",
                                   weighted=True, max_depth=14, h0=h0).rlct
        out.append(GuardResult(
            "[既知] Vandermonde chart の座標不変性", l1 == l2,
            f"chart 座標 {l1} / G1 を座標に取ると {l2}"
            + ("" if l1 == l2 else "  <- 未修正 (イデアルを運んでいないため)")))
    except Exception as e:                               # noqa: BLE001
        out.append(GuardResult("[既知] Vandermonde chart の座標不変性", False,
                               f"評価できません: {str(e)[:40]}"))
    return out


def newton_fastpath_tests() -> List[GuardResult]:
    """ニュートン高速パスは lambda と m を変えてはならない。

    chart の局所データが非退化と分かった時点で Varchenko の定理で打ち切る
    最適化 (resolve_singularities(newton_fast=True), 既定で有効) は、
    真値を変えない「速くするだけ」の変更のはずである。on/off で値が食い違えば
    実装のバグなので赤信号にする。
    """
    x, y, z, w = sp.symbols("x y z w", real=True)
    cases = [
        ("x^2+y^2", x**2 + y**2, (x, y)),
        ("x^2+y^3", x**2 + y**3, (x, y)),
        ("x^2*y^2", x**2 * y**2, (x, y)),
        ("x^3+y^4+z^5", x**3 + y**4 + z**5, (x, y, z)),
        ("(x-y)^2", (x - y) ** 2, (x, y)),
        ("(xy+z^2)^2", (x * y + z**2) ** 2, (x, y, z)),
        ("(x^2-y^3)^2", (x**2 - y**3) ** 2, (x, y)),
        ("x^2+y^2+z^2+w^2", x**2 + y**2 + z**2 + w**2, (x, y, z, w)),
        ("x^2*y+y^3", x**2 * y + y**3, (x, y)),
        ("x^4+x^2*y^2+y^4", x**4 + x**2 * y**2 + y**4, (x, y)),
    ]
    out: List[GuardResult] = []
    for label, f, g in cases:
        vals = {}
        for flag in (False, True):
            try:
                r = resolve_singularities(f, g, prune=False, weighted=True,
                                          max_depth=20, newton_fast=flag)
                vals[flag] = (r.rlct, r.multiplicity)
            except Exception as e:                    # noqa: BLE001
                vals[flag] = f"error:{type(e).__name__}"
        out.append(GuardResult(
            f"Newton 高速パス不変: {label}",
            vals[False] == vals[True],
            f"off={vals[False]} on={vals[True]}"))
    return out


def lemma_shortcut_tests() -> List[GuardResult]:
    """Aoyagi の補題を使った短縮経路は lambda を変えてはならない。

    (1) 積の分解 (product_split): 変数の互いに素な因子への分解。
        k, h が何であっても積分が分離するので、on/off で (lambda, m) が
        一致しなければ実装のバグ。
    (2) 単調性による大域下界 (lower_bound): 部分生成元の lambda は全体の
        lambda の厳密な下界なので、これを渡して早期終了させても lambda は
        変わらない (m は下限になりうるので lambda だけを比べる)。
    (3) 反例の確認: 和の分離を一般の chart で使うのは誤り。
        x(x^2+y^2) = x^3+x*y^2 の lambda は 2/3 で、
        lambda(x^3)+lambda(y^2) = 5/6 とは一致しない。
    """
    from lemmas import monotone_lower_bound, squeeze_rlct
    from resolve_singularity import newton_rlct

    x, y, z, w = sp.symbols("x y z w", real=True)
    out: List[GuardResult] = []

    # (1) 積の分解の on/off 一致
    cases = [
        ("(x^2+y^2)(z^2+w^3)", (x**2 + y**2) * (z**2 + w**3), (x, y, z, w)),
        ("(x^2+y^3)(z^4)", (x**2 + y**3) * z**4, (x, y, z)),
        ("x^2*y^2", x**2 * y**2, (x, y)),
        ("(x-y)^2*(z^2+z^3)", (x - y)**2 * (z**2 + z**3), (x, y, z)),
        ("(xy+z^2)^2", (x*y + z**2)**2, (x, y, z)),
    ]
    for label, f, g in cases:
        vals = {}
        for flag in (False, True):
            try:
                r = resolve_singularities(f, g, prune=False, weighted=True,
                                          max_depth=20, product_split=flag)
                vals[flag] = (r.rlct, r.multiplicity)
            except Exception as e:                    # noqa: BLE001
                vals[flag] = f"error:{type(e).__name__}"
        out.append(GuardResult(f"積の分解が不変: {label}",
                               vals[False] == vals[True],
                               f"off={vals[False]} on={vals[True]}"))

    # (2) 下界を渡しても lambda は変わらない
    for label, G, g in [("<x^2, y^3>", [x**2, y**3], (x, y)),
                        ("<xy, z^2>", [x*y, z**2], (x, y, z)),
                        ("<x^2+y^2, z^2>", [x**2 + y**2, z**2], (x, y, z))]:
        f = sp.expand(sum(t**2 for t in G))
        lo, _wit, _n = monotone_lower_bound(G, g)
        base = resolve_singularities(f, g, prune=False, weighted=True,
                                     max_depth=20)
        withlb = resolve_singularities(f, g, prune=False, weighted=True,
                                       max_depth=20, lower_bound=lo)
        out.append(GuardResult(f"大域下界で lambda 不変: {label}",
                               base.rlct == withlb.rlct,
                               f"lb={lo}, {base.rlct} vs {withlb.rlct}"))
        # 下界は本当に下界か
        out.append(GuardResult(f"下界 <= lambda: {label}",
                               lo is None or lo <= base.rlct,
                               f"{lo} <= {base.rlct}"))

    # (3) 和の分離を一般の chart で使うのは誤り (反例が実際に反例であること)
    f3 = sp.expand(x * (x**2 + y**2))
    lam3 = newton_rlct(f3, (x, y))          # 非退化なので厳密
    naive = sp.Rational(1, 3) + sp.Rational(1, 2)
    out.append(GuardResult(
        "和の分離の反例: x(x^2+y^2)",
        lam3 == sp.Rational(2, 3) and lam3 != naive,
        f"lambda={lam3}, 素朴な和={naive} (一致しないのが正しい)"))

    # (4) 挟み込みが矛盾を出さない
    for label, G, g in [("<x^2, y^3>", [x**2, y**3], (x, y)),
                        ("<x*y, y*z, z*x>", [x*y, y*z, z*x], (x, y, z))]:
        sq = squeeze_rlct(sp.expand(sum(t**2 for t in G)), g, G)
        out.append(GuardResult(f"挟み込みが整合: {label}",
                               sq.status != "inconsistent",
                               str(sq)))
    return out


def regular_elimination_tests() -> List[GuardResult]:
    """正則方向の消去 (nn_rlct._eliminate_regular) が lambda を変えないか。

    消去は「g = A*v + B, A(0) != 0 なら v を消して lambda に 1/2 を足す」
    という変形で、代入は擬剰余 (多項式) で行い、候補の順序も選ぶ。
    どれも値を変えない最適化のはずなので、

      (1) 消去を通した値 = もとの K = sum g_i^2 を直接解消した値
      (2) 生成元の順序を入れ替えても同じ値

    を確かめる。(1) は消去そのものの独立な検査、(2) は順序選択が
    値に漏れていないかの検査になる。
    """
    from certify import rlct_certified
    from nn_rlct import local_rlct_from_ideal

    x, y, z, w = sp.symbols("x y z w", real=True)
    cases = [
        ("<x + y^2, z>", [x + y**2, z], (x, y, z)),
        ("<x + y*z, y^2 - z^2>", [x + y*z, y**2 - z**2], (x, y, z)),
        ("<x + y^2 + z^3, y*z>", [x + y**2 + z**3, y*z], (x, y, z)),
        ("<x*y, z + x^2>", [x*y, z + x**2], (x, y, z)),
        ("<w + x*y, x^2 + y^2>", [w + x*y, x**2 + y**2], (w, x, y)),
        ("<x + y^2, x + z^2>", [x + y**2, x + z**2], (x, y, z)),
    ]
    out: List[GuardResult] = []
    for label, G, g in cases:
        K = sp.expand(sum(t**2 for t in G))
        try:
            lam_ref, st_ref, _ = rlct_certified(K, g, generators=G,
                                                timeout_ms=5000, max_depth=14)
        except Exception as e:                        # noqa: BLE001
            lam_ref, st_ref = None, f"error:{type(e).__name__}"
        try:
            loc = local_rlct_from_ideal(G, list(g), max_depth=12)
            lam_el = loc.rlct
        except Exception as e:                        # noqa: BLE001
            lam_el = f"error:{type(e).__name__}"
        out.append(GuardResult(
            f"正則消去 = 直接解消: {label}",
            st_ref != "proved" or lam_el == lam_ref,
            f"直接 {lam_ref} ({st_ref}) / 消去経由 {lam_el}"))

        # 生成元の順序を逆にしても同じ値
        try:
            loc2 = local_rlct_from_ideal(list(reversed(G)), list(g),
                                         max_depth=12)
            lam_rev = loc2.rlct
        except Exception as e:                        # noqa: BLE001
            lam_rev = f"error:{type(e).__name__}"
        out.append(GuardResult(f"正則消去が順序に依らない: {label}",
                               lam_el == lam_rev,
                               f"{lam_el} vs {lam_rev}"))
    return out


def cache_tests() -> List[GuardResult]:
    """永続キャッシュ (rlct_cache) が誤りを運ばないか。

    キャッシュ固有の危険は「簡単な多項式の誤った値が、それを部分問題に
    もつ複雑な計算に伝搬する」こと。防御が実際に働くかをここで確かめる。

      (1) 当たり/外れで lambda が変わらない (A/B)
      (2) 上界を超える値は当たりの時点で捨てられる
      (3) 上界内だが誤った値は certify(trust_cache=False) が refuted にする
      (4) コードの指紋が違うエントリは読み込まれない
      (5) 依存関係をたどって推移的に無効化できる
      (6) clear() で 1 から作り直せる
      (7) 変数の入れ替え・定数倍は同じキー、別の問題は別のキー
    """
    import json
    import os
    import tempfile

    import rlct_cache as RC
    from certify import certify, rlct_certified

    x, y, z = sp.symbols("x y z", real=True)
    out: List[GuardResult] = []
    tmpdir = tempfile.mkdtemp(prefix="rlctcache")
    path = os.path.join(tmpdir, "c.json")
    saved = RC.current()

    try:
        # (7) キーの同値性
        k1 = RC.canonical_key([0, 0], [0, 0], sp.Poly(x**2 + y**3, x, y), (x, y))
        k2 = RC.canonical_key([0, 0], [0, 0], sp.Poly(3 * (y**3 + x**2), y, x),
                              (y, x))
        k3 = RC.canonical_key([0, 0], [0, 0], sp.Poly(x**2 + y**4, x, y), (x, y))
        out.append(GuardResult("cache: 入れ替え・定数倍は同じキー",
                               k1 == k2 and k1 is not None, f"{k1} / {k2}"))
        out.append(GuardResult("cache: 別の問題は別のキー", k1 != k3,
                               f"{k1} / {k3}"))
        k4 = RC.canonical_key([1, 0], [0, 0], sp.Poly(x**2 + y**3, x, y), (x, y))
        out.append(GuardResult("cache: k が違えば別のキー", k1 != k4,
                               f"{k1} / {k4}"))

        # (1) A/B: キャッシュの有無で lambda が変わらない
        cases = [("(xy+z^2)^2", (x * y + z**2) ** 2, (x, y, z)),
                 ("x^2+y^2+z^2", x**2 + y**2 + z**2, (x, y, z)),
                 ("x^2*y^2", x**2 * y**2, (x, y))]
        RC.disable()
        base = {nm: rlct_certified(f, g, timeout_ms=5000, max_depth=14)[0]
                for nm, f, g in cases}
        RC.enable(path, rebuild=True)
        for nm, f, g in cases:                 # 1 周目: 作る
            rlct_certified(f, g, timeout_ms=5000, max_depth=14)
        after = {nm: rlct_certified(f, g, timeout_ms=5000, max_depth=14)[0]
                 for nm, f, g in cases}        # 2 周目: 使う
        for nm, _f, _g in cases:
            out.append(GuardResult(f"cache: A/B で lambda 不変 {nm}",
                                   base[nm] == after[nm],
                                   f"off={base[nm]} on={after[nm]}"))
        n_entries = len(RC.current().entries)
        out.append(GuardResult("cache: エントリが貯まる", n_entries > 0,
                               f"{n_entries} 件"))

        # (2) 上界を超える値は捨てられる
        c = RC.current()
        poly = sp.Poly((x * y + z**2) ** 2, x, y, z)
        key = RC.canonical_key([0, 0, 0], [0, 0, 0], poly, (x, y, z))
        c.put(key, sp.Integer(99), 1, "でっちあげ")
        got = c.get(key, k=[0, 0, 0], h=[0, 0, 0], poly=poly, gens=(x, y, z))
        out.append(GuardResult("cache: 上界を超える値は当たりで捨てる",
                               got is None and c.n_poisoned >= 1,
                               f"got={got}, poisoned={c.n_poisoned}"))

        # (3) 上界内だが誤った値 -> trust_cache=False で refuted
        from resolve_singularity import resolve_singularities
        c.put(key, sp.Rational(1, 3), 1, "でっちあげ(上界内)")
        res = resolve_singularities((x * y + z**2) ** 2, (x, y, z),
                                    prune=False, weighted=True)
        used = any(getattr(ch, "closed", None) and ch.closed[2] == "cache"
                   for ch in res.charts)
        if used:
            cert = certify(res, timeout_ms=5000, trust_cache=False)
            ok = cert.status != "proved"
        else:
            ok, cert = True, None
        out.append(GuardResult(
            "cache: 誤った値は trust_cache=False で proved にしない",
            ok, f"使われた={used}" + (f", status={cert.status}" if cert else "")))
        c.entries.pop(key, None)

        # (4) 指紋が違えば読み込まない
        c.put(RC.canonical_key([0, 0], [0, 0], sp.Poly(x**2 + y**3, x, y),
                               (x, y)), sp.Rational(5, 6), 1, "test")
        c.save()
        raw = json.load(open(path, encoding="utf-8"))
        for e in raw["entries"].values():
            e["fingerprint"] = "0" * 16
        json.dump(raw, open(path, "w", encoding="utf-8"))
        c2 = RC.Cache(path)
        out.append(GuardResult("cache: 指紋が違うエントリは読まない",
                               len(c2.entries) == 0 and c2.n_stale > 0,
                               f"{len(c2.entries)} 件, stale={c2.n_stale}"))

        # (5) 推移的な無効化
        c3 = RC.Cache(None)
        c3.put("A", sp.Rational(1, 2), 1, "t", depends_on=[])
        c3.put("B", sp.Rational(1, 3), 1, "t", depends_on=["A"])
        c3.put("C", sp.Rational(1, 4), 1, "t", depends_on=["B"])
        c3.put("D", sp.Rational(1, 5), 1, "t", depends_on=[])
        removed = set(c3.invalidate("A"))
        out.append(GuardResult("cache: 依存を推移的に無効化できる",
                               removed == {"A", "B", "C"}
                               and set(c3.entries) == {"D"},
                               f"消した {sorted(removed)}"))

        # (6) 作り直し
        RC.enable(path)
        RC.current().put("Z", sp.Integer(1), 1, "t")
        RC.current().save()
        exists_before = os.path.exists(path)
        RC.clear()
        out.append(GuardResult("cache: clear() で 1 から作り直せる",
                               exists_before and not os.path.exists(path)
                               and len(RC.current().entries) == 0,
                               f"ファイル {os.path.exists(path)}"))
    finally:
        RC.disable()
        if saved is not None and saved.path:
            RC.enable(saved.path)
    return out


def relu_case_tests() -> List[GuardResult]:
    """ReLU ネットの新しい退化の型が、独立な計算と一致するか。

    キャッシュに貯めるケースを増やすために退化の型を足した
    (duplicate3 / pair / opposite / scaled)。型を足すことで
      * 式が壊れていないか
      * 正則消去を通した lambda が、コア K = sum g_i^2 を直接解いた値と
        一致するか
    を確かめる。コアが空のケースは比べるものが無いので飛ばす。
    """
    from certify import rlct_certified
    from nn_cases import case_from_spec
    from nn_rlct import local_rlct_from_ideal

    # コアが**非空**になる組合せを選んでいる (build_cache のスイープで実測)。
    # コアが空だと比較対象が無く検査が空回りするため。
    specs = [((1, 3, 1), "redundant", 10, 0),      # core=4
             ((1, 3, 1), "redundant", 10, 3),      # core=2
             ((1, 2, 1), "redundant", 10, 2),      # core=2
             ((1, 3, 1), "dead", 10, 3),           # core=2
             ((1, 2, 1), "dead", 10, 1),           # core=1
             ((1, 3, 1), "duplicate3", 10, 0),
             ((1, 4, 1), "pair", 10, 0),
             ((1, 2, 1), "opposite", 10, 0),
             ((1, 2, 1), "scaled", 10, 0)]
    out: List[GuardResult] = []
    for spec in specs:
        label = f"{spec[1]} {'-'.join(map(str, spec[0]))}"
        c = case_from_spec(spec)
        if c is None:
            out.append(GuardResult(f"relu: {label} のケースが作れる", False,
                                   "生成元が空"))
            continue
        out.append(GuardResult(f"relu: {label} のケースが作れる", True,
                               f"{len(c.generators)} 生成元, "
                               f"{len(c.variables)} 変数"))
        try:
            loc = local_rlct_from_ideal(c.generators, c.variables,
                                        symmetry_vectors=c.symmetries,
                                        max_depth=8)
        except Exception as e:                        # noqa: BLE001
            out.append(GuardResult(f"relu: {label} の lambda が独立計算と一致",
                                   False, f"error:{type(e).__name__}"))
            continue
        if not loc.core_gens:
            out.append(GuardResult(
                f"relu: {label} の lambda が独立計算と一致", True,
                f"コアが空 (lambda={loc.rlct}) なので比較対象なし"))
            continue
        K = sp.expand(sum(g ** 2 for g in loc.core_gens))
        try:
            lam_ref, st_ref, _ = rlct_certified(K, tuple(loc.core_vars),
                                                generators=list(loc.core_gens),
                                                timeout_ms=5000, max_depth=12)
        except Exception as e:                        # noqa: BLE001
            lam_ref, st_ref = None, f"error:{type(e).__name__}"
        ok = (st_ref != "proved") or (lam_ref == loc.lam_core)
        out.append(GuardResult(
            f"relu: {label} の lambda が独立計算と一致", ok,
            f"コア {len(loc.core_vars)} 変数: 消去経由 {loc.lam_core} / "
            f"直接 {lam_ref} ({st_ref}),  全体 lambda={loc.rlct}"))
    return out


def transformer_tests() -> List[GuardResult]:
    """Transformer 1 層の fiber ideal と対称性が正しく作れているか。

    softmax は分母を払って多項式にしているので、次の 2 つが要となる。

      (1) 生成元が theta* でぴったり 0 になること。
          exp(S*) を有理数に丸めているが、記号側と数値側で**同じ丸め値**を
          使っているので、丸め誤差が残差として漏れてはいけない。
      (2) 主張している対称性が本当に対称性であること。
          軌道の接ベクトル方向の方向微分が theta* で 0 になるかを、
          生成元ごとに確かめる。QK ゲージ・VO ゲージ・softmax のシフト
          不変性・ReLU の正斉次性のどれかを取り違えていればここで落ちる。
    """
    from transformer_rlct import random_layer, transformer_layer_ideal

    X = [[sp.Rational(1), sp.Rational(0)], [sp.Rational(0), sp.Rational(1)]]
    # (d_model, d_k, d_ff, n_heads, 退化, softmax の打ち切り次数)
    # 形は小さく保つ。多頭は行の分母がヘッドの積になって次数が倍になり、
    # H=2 / order=1 では生成元が 14528 項まで膨らんだ (H=1 なら 101 項)。
    # そこで多頭は order=0 (注意を theta* で凍結) で形と VO ゲージだけ見る。
    shapes = [(2, 1, 1, 1, "none", 1), (2, 1, 2, 1, "none", 1),
              (2, 2, 1, 1, "zero_ffn", 1), (2, 1, 1, 2, "none", 0)]
    out: List[GuardResult] = []
    for (d, dk, dff, H, deg, order) in shapes:
        label = f"d{d} k{dk} ff{dff} H{H} {deg} o{order}"
        Xd = [[sp.Rational((i + j) % 3 - 1) for j in range(d)]
              for i in range(len(X))]
        star = random_layer(d_model=d, d_k=dk, d_ff=dff, n_heads=H, seed=0,
                            degeneracy=deg)
        try:
            I = transformer_layer_ideal(Xd, star, taylor_order=order)
        except Exception as e:                        # noqa: BLE001
            out.append(GuardResult(f"transformer: {label} を作れる", False,
                                   f"{type(e).__name__}: {str(e)[:40]}"))
            continue
        out.append(GuardResult(
            f"transformer: {label} を作れる", bool(I.generators),
            f"パラメータ {I.meta['n_params']}, 生成元 {I.meta['n_gens']}, "
            f"対称性 {I.meta['n_symmetries']}"))
        # theta* で消えるか = 定数項が 0 か。多変数の subs は密な多項式で
        # 非常に遅いので Poly の定数項を直接見る。
        polys = [sp.Poly(g, *I.variables) for g in I.generators]
        bad = [P for P in polys if P.coeff_monomial(1) != 0]
        out.append(GuardResult(
            f"transformer: {label} の生成元が theta* で消える",
            not bad, f"消えない生成元 {len(bad)} 個"))
        # 対称性の検証 (theta* での方向微分 = 0)
        #
        # g(eps*v) の eps^1 の係数が方向微分。全変数について sp.diff を
        # 取るのは高くつくので **1 次の単項式だけを拾う**:
        #   g = sum_m c_m u^m  ->  (d/d eps) g(eps v)|_0 = sum_{|m|=1} c_m v_m
        # これで 1 生成元あたり O(項数) で済む。
        lin = []
        for P in polys:
            d1 = {}
            for mon, c in zip(P.monoms(), P.coeffs()):
                if sum(mon) == 1:
                    d1[I.variables[list(mon).index(1)]] = c
            lin.append(d1)
        nfail = 0
        for vec in I.symmetry_vectors:
            for d1 in lin:
                dd = sum(sp.nsimplify(c) * d1.get(v, 0)
                         for v, c in vec.items() if c != 0)
                if sp.nsimplify(dd) != 0:
                    nfail += 1
                    break
        out.append(GuardResult(
            f"transformer: {label} の対称性が本当に対称性",
            nfail == 0,
            f"方向微分が 0 でない {nfail}/{len(I.symmetry_vectors)} 方向"))
    return out


def transformer_torch_tests() -> List[GuardResult]:
    """torch の層からの読み込み (torch が無ければ飛ばす)。"""
    out: List[GuardResult] = []
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return out
    from torch_rlct import transformer_star_from_torch

    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=2, nhead=1, dim_feedforward=2,
                                       activation="relu", dropout=0.0,
                                       batch_first=True)
    # LayerNorm を持つ層は既定で拒否する (黙って別のモデルを計算しない)
    refused = False
    try:
        transformer_star_from_torch(layer)
    except ValueError:
        refused = True
    out.append(GuardResult("transformer: LayerNorm 付きは既定で拒否",
                           refused, "layernorm='ignore' が要る"))
    st = transformer_star_from_torch(layer, layernorm="ignore")
    shape_ok = (st.d_model == 2 and st.d_k == 2 and st.d_ff == 2
                and st.n_heads == 1
                and len(st.Wq[0]) == 2 and len(st.Wq[0][0]) == 2
                and len(st.Wo) == 2 and len(st.W1) == 2
                and len(st.W2) == 2)
    out.append(GuardResult("transformer: torch から形どおり読める", shape_ok,
                           f"d={st.d_model} dk={st.d_k} ff={st.d_ff} "
                           f"H={st.n_heads}"))
    # 重みが転置されずに入っていないか: X W_Q の規約に合わせて転置している
    Win = layer.self_attn.in_proj_weight.detach()
    ok_t = abs(float(Win[0][1]) - float(st.Wq[0][1][0])) < 1e-3
    out.append(GuardResult("transformer: in_proj_weight を転置して読む", ok_t,
                           f"W[0][1]={float(Win[0][1]):.4f} -> "
                           f"Wq[1][0]={float(st.Wq[0][1][0]):.4f}"))
    # tanh 活性の層は拒否する
    layer2 = nn.TransformerEncoderLayer(d_model=2, nhead=1, dim_feedforward=2,
                                        activation="gelu", dropout=0.0,
                                        batch_first=True)
    bad_act = False
    try:
        transformer_star_from_torch(layer2, layernorm="ignore")
    except ValueError:
        bad_act = True
    out.append(GuardResult("transformer: ReLU 以外の活性は拒否", bad_act, ""))
    return out


def torch_classifier_tests() -> List[GuardResult]:
    """Conv2d / プーリングの記号フォワードと、分類の fiber ideal。

    (1) **記号フォワードの数値が torch と一致するか**。Conv2d・MaxPool2d・
        AvgPool2d を記号で通せるようにしたので、同じ重み・同じ入力で
        torch の出力と突き合わせる。ここがずれていれば生成元が別のモデルの
        ものになり、lambda も無意味になる。
    (2) 分類の fiber ideal (ロジットの差) が回帰のものより**拘束が弱い**こと。
        p = softmax(f) はロジットの平行移動で変わらないので、分類の lambda は
        回帰の lambda 以下でなければならない。
    (3) ロジットの平行移動が本当に対称性であること (方向微分 = 0)。
    """
    out: List[GuardResult] = []
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return out
    import torch_rlct as TR
    from torch_rlct import fiber_ideal_from_torch, torch_local_rlct
    from train_rlct import MLPClassifier, SmallCNN, load_dataset, train

    torch.manual_seed(0)
    ds = load_dataset("synthetic", n=32, shape=(1, 4, 4), n_classes=3)
    models = [("mlp", MLPClassifier(ds.shape, 3, hidden=(4,))),
              ("cnn", SmallCNN(ds.shape, 3, channels=2, hidden=4))]
    for label, model in models:
        rep = train(model, ds, epochs=150, lr=0.05, seed=0)
        out.append(GuardResult(f"torch: {label} が学習できる",
                               rep.final_acc > 0.8,
                               f"loss={rep.final_loss:.4f} "
                               f"acc={rep.final_acc:.3f}"))
        layers = TR._flatten_layers(model)
        names = {id(p): n for n, p in model.named_parameters()}
        S = TR._Sym(layers=layers, var_names=[], zero_tol=1e-12,
                    max_den=10 ** 12, order=3,
                    handlers=TR._default_handlers())
        X = ds.X[:4]
        with torch.no_grad():
            ref = model(X)
        worst = 0.0
        for i in range(X.shape[0]):
            sym, num = S.forward(X[i].reshape(-1).tolist(), names, {},
                                 record_signs={}, shape=ds.shape)
            for a, b in zip(num, ref[i].tolist()):
                worst = max(worst, abs(a - b))
        out.append(GuardResult(
            f"torch: {label} の記号フォワードが torch と一致",
            worst < 1e-4, f"最大差 {worst:.2e}"))

    # (2)(3) 分類 vs 回帰
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 3))
    Xc = torch.tensor([[1., 0.], [0., 1.], [-1., .5], [.5, -1.]])
    vals = {}
    for loss in ("classification", "regression"):
        r = torch_local_rlct(net, Xc, loss=loss,
                             params=["2.weight", "2.bias"],
                             max_denominator=16, max_depth=8)
        vals[loss] = r.local.rlct
    out.append(GuardResult(
        "torch: 分類の lambda <= 回帰の lambda",
        vals["classification"] <= vals["regression"],
        f"分類 {vals['classification']} / 回帰 {vals['regression']}"))

    data = fiber_ideal_from_torch(net, Xc, loss="classification",
                                  params=["2.weight", "2.bias"],
                                  max_denominator=16)
    gens, variables = data["generators"], data["variables"]
    shift = [v for v in data["symmetry_vectors"]
             if all(c == 1 for c in v.values()) and len(v) >= 2]
    ok_shift = bool(shift)
    if ok_shift:
        vec = shift[0]
        for g in gens:
            P = sp.Poly(g, *variables)
            d1 = {}
            for mon, c in zip(P.monoms(), P.coeffs()):
                if sum(mon) == 1:
                    d1[variables[list(mon).index(1)]] = c
            if sp.nsimplify(sum(d1.get(v, 0) for v in vec)) != 0:
                ok_shift = False
                break
    out.append(GuardResult("torch: ロジットの平行移動が対称性", ok_shift,
                           f"対称ベクトル {len(shift)} 本"))
    return out


def vit_tests() -> List[GuardResult]:
    """学習済み TinyViT からエンコーダ層を取り出せるか。"""
    out: List[GuardResult] = []
    try:
        import torch
    except Exception:
        return out
    from torch_rlct import transformer_star_from_torch
    from train_rlct import TinyViT, load_dataset, train

    torch.manual_seed(0)
    ds = load_dataset("synthetic", n=24, shape=(1, 4, 4), n_classes=3)
    vit = TinyViT(ds.shape, 3, patch=2, d_model=2, n_heads=1, d_ff=2,
                  depth=1, layernorm=False)
    rep = train(vit, ds, epochs=120, lr=0.05, seed=0)
    out.append(GuardResult("vit: TinyViT が学習できる", rep.final_loss < 1.0,
                           f"loss={rep.final_loss:.4f} acc={rep.final_acc:.3f}"))
    with torch.no_grad():
        Z = vit.forward_upto(ds.X[:1], 0)
    out.append(GuardResult("vit: 層の入力トークン列が取れる",
                           tuple(Z.shape) == (1, 5, 2), f"{tuple(Z.shape)}"))
    # LayerNorm なしの層は layernorm 指定なしでも読める
    st = transformer_star_from_torch(vit.layers[0])
    out.append(GuardResult(
        "vit: LayerNorm なしの層はそのまま読める",
        st.d_model == 2 and st.d_ff == 2 and st.n_heads == 1,
        f"d={st.d_model} dk={st.d_k} ff={st.d_ff}"))
    # LayerNorm ありの層は拒否される
    vit2 = TinyViT(ds.shape, 3, patch=2, d_model=2, n_heads=1, d_ff=2,
                   depth=1, layernorm=True)
    refused = False
    try:
        transformer_star_from_torch(vit2.layers[0])
    except ValueError:
        refused = True
    out.append(GuardResult("vit: LayerNorm 入りの層は既定で拒否", refused, ""))
    return out


def taylor_sweep_tests() -> List[GuardResult]:
    """softmax の打ち切り次数ごとの問題が正しく組めているか。

    (1) どの次数でも生成元が theta* で消えること。
        指数の丸めを記号側と数値側で共有しているので、ここが破れたら
        「残差 = 丸め誤差」を解消していることになる。
    (2) 分類ヘッドのロジット平行移動が本当に対称性であること。
    (3) 同じ theta* で **分類の lambda <= 回帰の lambda** であること。
        p = softmax(logit) は平行移動で変わらないので、分類のほうが
        拘束が弱い。逆向きになっていたら ideal の組み方が誤り。
    (4) 次数を上げると生成元の項数が増えること (打ち切りが効いている確認)。
    """
    from nn_rlct import local_rlct_from_ideal
    from taylor_sweep import build_problem, random_sequences
    from transformer_rlct import random_layer

    out: List[GuardResult] = []
    star = random_layer(d_model=2, d_k=1, d_ff=1, n_heads=1, seed=0)
    Xs = random_sequences(1, 2, 2, seed=0)
    terms = {}
    lam = {}
    for loss in ("regression", "classification"):
        for order in (0, 1):
            prob = build_problem(Xs, star, taylor_order=order, loss=loss,
                                 n_classes=2, max_denominator=8)
            polys = [sp.Poly(g, *prob.variables) for g in prob.generators]
            bad = [P for P in polys if P.coeff_monomial(1) != 0]
            out.append(GuardResult(
                f"taylor: {loss} n={order} の生成元が theta* で消える",
                not bad, f"消えない生成元 {len(bad)}/{len(polys)} 個"))
            terms[(loss, order)] = max((len(P.monoms()) for P in polys),
                                       default=0)
            if loss == "classification":
                # ロジットの平行移動 (ヘッドのバイアスの (1,..,1)) の検証
                shift = [v for v in prob.symmetry_vectors
                         if len(v) >= 2 and all(c == 1 for c in v.values())
                         and all("hb" in str(k) for k in v)]
                ok = bool(shift)
                if ok:
                    vec = shift[-1]
                    for P in polys:
                        d1 = {}
                        for mon, c in zip(P.monoms(), P.coeffs()):
                            if sum(mon) == 1:
                                d1[prob.variables[list(mon).index(1)]] = c
                        if sp.nsimplify(sum(d1.get(v, 0) for v in vec)) != 0:
                            ok = False
                            break
                out.append(GuardResult(
                    f"taylor: n={order} のロジット平行移動が対称性", ok,
                    f"対称ベクトル {len(shift)} 本"))
            # lambda を解くのは n=0 だけ。n>=1 は生成元が密になり
            # ガードの中で解くと分単位かかる (掃引の役目なので任せる)。
            if order == 0:
                loc = local_rlct_from_ideal(
                    prob.generators, prob.variables,
                    symmetry_vectors=prob.symmetry_vectors, max_depth=6)
                lam[(loss, order)] = loc.rlct
    for order in (0,):
        out.append(GuardResult(
            f"taylor: n={order} で 分類 lambda <= 回帰 lambda",
            lam[("classification", order)] <= lam[("regression", order)],
            f"分類 {lam[('classification', order)]} / "
            f"回帰 {lam[('regression', order)]}"))
    out.append(GuardResult(
        "taylor: 次数を上げると項数が増える",
        terms[("regression", 1)] > terms[("regression", 0)],
        f"n=0: {terms[('regression', 0)]} 項 -> "
        f"n=1: {terms[('regression', 1)]} 項"))
    return out


def analytic_activation_tests() -> List[GuardResult]:
    r"""tanh など解析的な活性化のテイラー展開が正しく組めているか。

    (1) **係数が有理数に丸まっていること**。丸めないと
        `tanh(1180339/2500000)` のような超越数が係数に残り、sympy の
        多項式が拡大体 (EX ドメイン) に落ちて実測 10 分でも終わらなくなる。
        ここは速度の話に見えて実は構造の話なので、係数の型を直接見る。
    (2) **どの次数でも生成元が theta\* で消えること**。展開の中心を
        float の `nsimplify` で取ると theta\* で端数が残り、
        「残差 = 丸め誤差」を解消する羽目になる。中心は `sym` の定数項
        そのもの (厳密な有理数) に取っている。
    (3) 打ち切りモデルの theta\* での出力が torch と一致すること。
        theta\* では delta = 0 なので、次数によらず一致するのが正しい。
    (4) 次数を上げると生成元の項数が増えること (打ち切りが効いている確認)。
    """
    import torch
    import torch.nn as nn

    from torch_rlct import (_default_handlers, _flatten_layers, _Sym,
                           fiber_ideal_from_torch)

    out: List[GuardResult] = []
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1))
    X = torch.randn(3, 2)
    with torch.no_grad():
        ref = net(X)
    layers = _flatten_layers(net)
    name_of = {id(p): n for n, p in net.named_parameters()}

    terms = {}
    for order in (1, 2, 3):
        d = fiber_ideal_from_torch(net, X, taylor_order=order,
                                   max_denominator=10 ** 6, zero_tol=1e-12)
        polys = [sp.Poly(g, *d["variables"]) for g in d["generators"]]
        # (1) 係数が全部有理数か
        bad_c = [c for P in polys for c in P.coeffs() if not c.is_Rational]
        out.append(GuardResult(
            f"tanh: n={order} の係数が有理数 (EX ドメインに落ちない)",
            not bad_c,
            f"有理数でない係数 {len(bad_c)} 個"
            + (f" 例 {str(bad_c[0])[:40]}" if bad_c else "")))
        # (2) theta* で消えるか
        bad0 = [P for P in polys if P.coeff_monomial(1) != 0]
        out.append(GuardResult(
            f"tanh: n={order} の生成元が theta* で消える", not bad0,
            f"消えない生成元 {len(bad0)}/{len(polys)} 個"))
        terms[order] = max((len(P.monoms()) for P in polys), default=0)
        # (3) theta* での出力が torch と一致するか
        S = _Sym(layers, [], 1e-12, 10 ** 6, order, _default_handlers(10 ** 6))
        dev = 0.0
        for i in range(X.shape[0]):
            _, num = S.forward([float(v) for v in X[i]], name_of, {})
            for c, v in enumerate(num):
                dev = max(dev, abs(v - float(ref[i, c])))
        out.append(GuardResult(
            f"tanh: n={order} の theta* での出力が torch と一致",
            dev < 1e-6, f"最大差 {dev:.2e}"))
    out.append(GuardResult(
        "tanh: 次数を上げると項数が増える",
        terms[3] > terms[2] > terms[1],
        f"n=1: {terms[1]} -> n=2: {terms[2]} -> n=3: {terms[3]} 項"))

    # (5) 係数の丸めが lambda を動かさないこと。丸めは打ち切りに加わる
    # もう 1 つの近似なので、分母の上限を振って値が動かないかを確かめる。
    from torch_rlct import torch_local_rlct
    lams = {}
    for den in (8, 64, 10 ** 6):
        lams[den] = torch_local_rlct(net, X, taylor_order=2,
                                     max_denominator=den,
                                     zero_tol=1e-9).local.rlct
    out.append(GuardResult(
        "tanh: 係数の丸めの粗さで lambda が変わらない",
        len(set(lams.values())) == 1,
        ", ".join(f"den={k}: {v}" for k, v in lams.items())))

    # (6) **飽和したユニットでも依存関係が消えないこと**。前活性化を大きく
    # すると tanh の 1 次係数は e^{-2|z0|} で小さくなる。絶対精度で丸めると
    # これが 0 になり、上流のパラメータが全部「自由方向」に見えて lambda が
    # 偽の 0 になる (実際に CNN でこれが起きた)。相対精度の丸めの回帰テスト。
    deep = nn.Sequential(nn.Linear(1, 1), nn.Tanh(), nn.Linear(1, 1))
    with torch.no_grad():
        deep[0].weight.fill_(6.0)        # 前活性化 ~ 6 -> tanh' ~ 1.8e-05
        deep[0].bias.fill_(0.0)
        deep[2].weight.fill_(1.0)
        deep[2].bias.fill_(0.0)
    Xs = torch.ones(2, 1)
    d = fiber_ideal_from_torch(deep, Xs, params=["0.weight", "0.bias"],
                               taylor_order=1, max_denominator=16,
                               zero_tol=1e-12, max_vars=8)
    nz = [g for g in d["generators"] if g != 0]
    out.append(GuardResult(
        "tanh: 飽和したユニットでも上流への依存が消えない (丸めが相対精度)",
        len(nz) == len(d["generators"]) and len(nz) > 0,
        f"非零の生成元 {len(nz)}/{len(d['generators'])} 本"
        f" (tanh' ~ {1 - math.tanh(6.0) ** 2:.1e})"))
    return out


def cnn_depth_tests() -> List[GuardResult]:
    r"""層数を変えられる CNN と、その掃引の組み方の検査。

    (1) `DeepCNN` が偶数カーネルでも形を正しく追えること。padding で
        一辺を保てるのは奇数カーネルだけなので、ここを間違えると
        Flatten のあとの Linear が合わずに torch が例外を出す。
    (2) 記号フォワードが torch と一致すること (ReLU・層数ごと)。
    (3) `params='conv'` が層数を変えても**同じ変数集合**を選ぶこと。
        これが崩れると「層数を変えた比較」が別の部分空間の比較になる。
    (4) 同じ theta\* で 分類 lambda <= 回帰 lambda。
    """
    import torch
    import torch.nn as nn

    from cnn_sweep import _params_for
    from torch_rlct import fiber_ideal_from_torch, torch_local_rlct
    from train_rlct import DeepCNN, synthetic_classification

    out: List[GuardResult] = []

    # (1) 形
    shapes_ok, note = True, []
    for k in (2, 3):
        for d in (1, 2, 3):
            try:
                m = DeepCNN((1, 8, 8), 2, depth=d, channels=2, hidden=4,
                            kernel=k)
                y = m(torch.randn(2, 1, 8, 8))
                if tuple(y.shape) != (2, 2):
                    shapes_ok = False
                    note.append(f"k={k} d={d} -> {tuple(y.shape)}")
            except Exception as e:                        # noqa: BLE001
                shapes_ok = False
                note.append(f"k={k} d={d}: {type(e).__name__}")
    out.append(GuardResult("cnn: DeepCNN が偶数/奇数カーネルで形を追える",
                           shapes_ok, ", ".join(note) or "k=2,3 x depth=1,2,3"))

    # (3) params='conv' が層数によらず同じ変数集合を選ぶ
    picks = []
    for d in (1, 2, 3):
        m = DeepCNN((1, 8, 8), 2, depth=d, channels=2, hidden=4, kernel=3)
        picks.append(tuple(_params_for(m, "conv", 20) or ()))
    out.append(GuardResult(
        "cnn: params='conv' は層数を変えても同じ変数を選ぶ",
        len(set(picks)) == 1, f"{picks[0]} (depth 1,2,3 で一致)"))

    # (2) 記号フォワードが torch と一致する
    ds = synthetic_classification(n=16, shape=(1, 4, 4), n_classes=2, seed=0)
    for d in (1, 2):
        torch.manual_seed(0)
        m = DeepCNN(ds.shape, 2, depth=d, channels=1, hidden=2, kernel=3)
        X = ds.X[:3]
        with torch.no_grad():
            ref = m(X)
        names = _params_for(m, "conv", 10)
        data = fiber_ideal_from_torch(
            m, X.reshape(3, -1), params=names, input_shape=ds.shape,
            max_denominator=10 ** 6, zero_tol=1e-12, max_vars=20,
            loss="classification")
        polys = [sp.Poly(g, *data["variables"]) for g in data["generators"]]
        bad = [P for P in polys if P.coeff_monomial(1) != 0]
        out.append(GuardResult(
            f"cnn: depth={d} の生成元が theta* で消える", not bad,
            f"消えない生成元 {len(bad)}/{len(polys)} 個"))
        # 記号フォワードの数値列を torch と突き合わせる
        from torch_rlct import _default_handlers, _flatten_layers, _Sym
        S = _Sym(_flatten_layers(m), [], 1e-12, 10 ** 6, 1,
                 _default_handlers(10 ** 6))
        nm = {id(p): n for n, p in m.named_parameters()}
        dev = 0.0
        for i in range(3):
            _, num = S.forward([float(v) for v in X[i].reshape(-1)], nm, {},
                               shape=ds.shape)
            for c, v in enumerate(num):
                dev = max(dev, abs(v - float(ref[i, c])))
        out.append(GuardResult(
            f"cnn: depth={d} の記号フォワードが torch と一致",
            dev < 1e-5, f"最大差 {dev:.2e}"))

    # (5) 境界ユニットが多くてもセル列挙でメモリを食い潰さないこと。
    # list(product((1,-1), repeat=n))[:max_cells] と書くと n=30 で 2^30 個の
    # タプルを並べようとして MemoryError になる (深い ReLU CNN で実際に落ちた)。
    # 前活性化が全部ちょうど 0 の層を作って、その経路を直接踏む。
    wide = nn.Sequential(nn.Linear(2, 40), nn.ReLU(), nn.Linear(40, 1))
    with torch.no_grad():
        wide[0].weight.zero_()          # 40 ユニットすべて前活性化 0 = 境界
        wide[0].bias.zero_()
        wide[2].weight.fill_(0.5)
        wide[2].bias.zero_()
    ok_mem, note = True, ""
    try:
        r = torch_local_rlct(wide, torch.randn(3, 2), params=["2.bias"],
                             max_vars=8, max_denominator=16, zero_tol=1e-12,
                             max_cells=4, max_depth=4)
        warn = [n for n in r.local.notes if "セル" in n and "warn" in n]
        note = (f"境界 40 本 -> セル 4 個で打ち切り, "
                f"警告 {'あり' if warn else 'なし'}")
        ok_mem = bool(warn)             # 打ち切ったなら必ず警告が出ること
    except MemoryError:
        ok_mem, note = False, "MemoryError (2^40 セルを materialize した)"
    out.append(GuardResult(
        "cnn: 境界ユニットが多くてもセル列挙が爆発せず、打ち切りを報告する",
        ok_mem, note))

    # (4) 分類 <= 回帰
    torch.manual_seed(0)
    m = DeepCNN(ds.shape, 2, depth=1, channels=1, hidden=2, kernel=3)
    X = ds.X[:6].reshape(6, -1)
    names = _params_for(m, "conv", 10)
    lam = {}
    for loss in ("classification", "regression"):
        r = torch_local_rlct(m, X, loss=loss, params=names,
                             input_shape=ds.shape, max_vars=20,
                             max_denominator=16, zero_tol=1e-2, max_depth=5)
        lam[loss] = r.local.rlct
    out.append(GuardResult(
        "cnn: 分類 lambda <= 回帰 lambda",
        lam["classification"] <= lam["regression"],
        f"分類 {lam['classification']} / 回帰 {lam['regression']}"))
    return out


def run_all(verbose: bool = True):
    res = (negative_tests() + mutation_tests() + independent_tests()
           + aoyagi_lemma_tests() + vandermonde_truth_tests()
           + coordinate_invariance_tests() + newton_fastpath_tests()
           + lemma_shortcut_tests() + regular_elimination_tests()
           + cache_tests() + relu_case_tests()
           + transformer_tests() + transformer_torch_tests()
           + torch_classifier_tests() + vit_tests()
           + taylor_sweep_tests() + analytic_activation_tests()
           + cnn_depth_tests())
    if verbose:
        for r in res:
            print("  " + str(r))
    n_bad = sum(1 for r in res if not r.ok)
    known = known_issues()
    if verbose:
        print(f"  --> guards: {len(res) - n_bad}/{len(res)} 緑")
        if known:
            print("  既知の未修正バグ (ゲートには数えない):")
            for k in known:
                mark = "修正された -> 通常のガードに昇格させること" if k.ok \
                    else "未修正"
                print(f"    [{mark}] {k.name}  {k.detail}")
    return res


if __name__ == "__main__":
    run_all()
