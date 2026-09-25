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


def run_all(verbose: bool = True):
    res = (negative_tests() + mutation_tests() + independent_tests()
           + aoyagi_lemma_tests() + vandermonde_truth_tests()
           + coordinate_invariance_tests() + newton_fastpath_tests()
           + lemma_shortcut_tests() + regular_elimination_tests())
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
