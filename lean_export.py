r"""
lean_export.py
==============

特異点解消の結果を Lean 4 (Mathlib) の証明義務として書き出し、可能なら
検証する。狙いは「chart 族が原点の近傍を覆っている」ことと、
lambda を読み取る際に使った代数的な帳簿が正しいことを機械的に確かめること。

------------------------------------------------------------------
何を証明するのか
------------------------------------------------------------------
lambda = min_chart min_j (h_j+1)/k_j が正しいためには、二層の条件が要る。

  (上層) 各 chart で  f(phi(y)) = y^k * (原点で非零な f_rest),
         |det D phi| = y^h * (単元)  になっていること
  (下層) chart 族が原点の近傍を過不足なく覆っていること

上層は各 chart で閉じた代数的等式なので、多項式の恒等式として `ring` で
決着する。下層は解析的な主張だが、ブローアップの chart に限れば
「全射性」という一階の命題に落ちる。

  中心 C = {x_j = 0 : j in J} のブローアップの chart i は
      sigma_i(y)_i = y_i,  sigma_i(y)_j = y_i * y_j (j in J, j != i),
      sigma_i(y)_k = y_k   (k not in J)
  このとき任意の x に対し、|x_j| (j in J) が最大になる i を選び
      y_i = x_i,  y_j = x_j / x_i (x_i != 0 のとき、そうでなければ 0)
  とおけば sigma_i(y) = x かつ |y_j| <= 1。すなわち

      forall x, exists i in J, exists y, sigma_i(y) = x
                                   and y_i = x_i and |y_j| <= 1

  が成り立つ。y_i = x_i なので x -> 0 のとき y も原点に近づき、
  近傍が覆われていることが言える。この補題を一度だけ手で証明しておき
  (LeanCover.lean)、各ブローアップ節点はその具体化として出力する。

------------------------------------------------------------------
出力されるファイル
------------------------------------------------------------------
  <dir>/LeanCover.lean       手書きの一般補題 (chart の全射性)
  <dir>/Resolution.lean      解消ごとに生成される証明義務
  <dir>/lakefile.lean        Mathlib に依存する lake プロジェクト雛形
  <dir>/lean-toolchain

生成される定理は 4 種類:

  subst_<chart>   ブローアップ/座標変換/平行移動の 1 ステップの代入等式
                  f_before(sigma(y)) = (括り出した単項式) * f_after(y)   [ring]
  chart_<leaf>    葉ごとの合成写像の等式
                  f(phi(y)) = y^k * f_rest(y)                            [ring]
  unit_<leaf>     f_rest(0) != 0 (正規交差になっている確認)               [norm_num]
  cover_<node>    そのブローアップの chart 族が全射であること              [一般補題]
  jac_<leaf>      ヤコビ行列式 = y^h * (単元)  (n <= 3、任意)              [ring]

------------------------------------------------------------------
生成器そのものが検査器になる
------------------------------------------------------------------
出力する前に、すべての等式を sympy で厳密に検算する。等式が成り立たない
場合は Lean を待たずにその場で報告する (strict=True なら例外)。
つまり Lean 環境が無くても、この機能は k, h, f_rest, phi の帳簿の
整合性チェックとして機能する。

------------------------------------------------------------------
使い方
------------------------------------------------------------------
    res = resolve_singularities(x**2 + y**3, (x, y))
    rep = export_lean(res, "lean_out")
    print(rep.summary())
    rep.verify()          # lake / lean があれば実際にコンパイルする
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

__all__ = ["LeanExport", "Obligation", "export_lean", "lean_expr"]


# ----------------------------------------------------------------------
# sympy -> Lean の式プリンタ
# ----------------------------------------------------------------------
_LEAN_KEYWORDS = {
    "fun", "let", "have", "show", "from", "with", "do", "if", "then", "else",
    "match", "end", "theorem", "lemma", "def", "by", "at", "in", "open",
}


def lean_name(sym) -> str:
    """sympy の記号名を Lean の識別子に直す。"""
    t = str(sym)
    out = []
    for ch in t:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    t = "".join(out)
    if not t or not (t[0].isalpha() or t[0] == "_"):
        t = "v_" + t
    if t in _LEAN_KEYWORDS:
        t = t + "_"
    return t


def lean_expr(e) -> str:
    """sympy 式を Lean の項に変換する (実数体上)。"""
    e = sp.sympify(e)
    if isinstance(e, sp.Symbol):
        return lean_name(e)
    if e.is_Integer:
        return f"({int(e)} : ℝ)" if e < 0 else str(int(e))
    if e.is_Rational:
        p, q = e.p, e.q
        num = f"({p} : ℝ)" if p < 0 else f"({p} : ℝ)"
        return f"({num} / {q})"
    if e.is_Float:
        r = sp.Rational(e).limit_denominator(10 ** 9)
        return lean_expr(r)
    if e.is_Add:
        return "(" + " + ".join(lean_expr(a) for a in e.args) + ")"
    if e.is_Mul:
        return "(" + " * ".join(lean_expr(a) for a in e.args) + ")"
    if e.is_Pow:
        b, x = e.args
        if x.is_Integer and x >= 0:
            return f"({lean_expr(b)} ^ {int(x)})"
        if x.is_Integer and x < 0:
            return f"({lean_expr(b)} ^ ({int(x)} : ℤ))"
        raise ValueError(f"Lean に落とせない指数: {e}")
    if e == sp.S.Zero:
        return "0"
    if e == sp.S.One:
        return "1"
    raise ValueError(f"Lean に落とせない式: {e} ({type(e)})")


def _monomial(gens, exps) -> sp.Expr:
    return sp.prod([v ** int(k) for v, k in zip(gens, exps)])


# ----------------------------------------------------------------------
# 証明義務
# ----------------------------------------------------------------------
@dataclass
class Obligation:
    name: str
    kind: str                 # 'subst' | 'chart' | 'unit' | 'cover' | 'jac'
    statement: str            # Lean の定理文 (全体)
    ok: Optional[bool]        # sympy による検算結果 (None = 検算対象外)
    detail: str = ""

    def __str__(self) -> str:
        mark = {True: "OK", False: "NG", None: "--"}[self.ok]
        return f"[{mark}] {self.kind:<6} {self.name}  {self.detail}"


@dataclass
class LeanExport:
    directory: str
    obligations: List[Obligation]
    gens: Tuple[sp.Symbol, ...]
    files: List[str] = field(default_factory=list)

    @property
    def failures(self) -> List[Obligation]:
        return [o for o in self.obligations if o.ok is False]

    def summary(self) -> str:
        n_ok = sum(1 for o in self.obligations if o.ok is True)
        n_ng = len(self.failures)
        n_na = sum(1 for o in self.obligations if o.ok is None)
        lines = [f"# Lean 出力: {self.directory}",
                 f"#   証明義務 {len(self.obligations)} 件 "
                 f"(sympy 検算: OK {n_ok} / NG {n_ng} / 対象外 {n_na})"]
        for o in self.obligations:
            lines.append("  " + str(o))
        if n_ng:
            lines.append("  ** sympy の段階で成り立たない等式があります。"
                         "解消の帳簿 (k, h, f_rest, phi) にバグがあります。**")
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    # --- Lean の実行 ---------------------------------------------------
    def verify(self, timeout: int = 1800) -> Dict[str, object]:
        """lake build を実行して検証する。ツールチェインが無ければ報告のみ。"""
        if shutil.which("lake") is None and shutil.which("lean") is None:
            return {"available": False,
                    "message": "lean / lake が見つかりません。elan で導入し、"
                               f"{self.directory} で `lake update && lake build` "
                               "を実行してください。"}
        cmd = ["lake", "build"] if shutil.which("lake") else \
              ["lean", os.path.join(self.directory, "Resolution.lean")]
        try:
            p = subprocess.run(cmd, cwd=self.directory, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"available": True, "ok": False, "message": "タイムアウト"}
        out = p.stdout + p.stderr
        return {"available": True, "ok": p.returncode == 0,
                "returncode": p.returncode,
                "sorry": out.count("declaration uses 'sorry'"),
                "output": out[-4000:]}


# ----------------------------------------------------------------------
# 手書きの一般補題 (chart の全射性)
# ----------------------------------------------------------------------
_COVER_LEAN = r'''/-
LeanCover.lean — ブローアップの chart 族が全射であることの一般補題。

中心 C = {x j = 0 | j ∈ J} のブローアップの chart i は

    chartMap J i y j = if j = i then y i
                       else if j ∈ J then y i * y j
                       else y j

この補題は「任意の x に対し、ある i ∈ J と y が存在して chartMap J i y = x、
しかも y i = x i かつ j ∈ J \ {i} では |y j| ≤ 1」を主張する。
y i = x i なので x → 0 のとき y → 0 の側も原点に近づき、原点の近傍が
chart たちの像で覆われることが従う。
-/
import Mathlib

open Finset

variable {n : ℕ}

/-- ブローアップの chart i の座標変換。 -/
noncomputable def chartMap (J : Finset (Fin n)) (i : Fin n)
    (y : Fin n → ℝ) : Fin n → ℝ :=
  fun j => if j = i then y i else if j ∈ J then y i * y j else y j

/-- chart 族の全射性 (近傍を覆うこと)。 -/
theorem blowup_cover (J : Finset (Fin n)) (hJ : J.Nonempty) (x : Fin n → ℝ) :
    ∃ i ∈ J, ∃ y : Fin n → ℝ,
      chartMap J i y = x ∧ y i = x i ∧ ∀ j ∈ J, j ≠ i → |y j| ≤ 1 := by
  -- |x j| が J 上で最大になる i を取る
  obtain ⟨i, hiJ, hmax⟩ := J.exists_max_image (fun j => |x j|) hJ
  refine ⟨i, hiJ, fun j => if j = i then x i else if x i = 0 then 0 else x j / x i,
          ?_, ?_, ?_⟩
  · funext j
    by_cases hji : j = i
    · subst hji; simp [chartMap]
    · by_cases hjJ : j ∈ J
      · by_cases hx : x i = 0
        · -- x i = 0 なら最大性から x j = 0
          have hle : |x j| ≤ |x i| := hmax j hjJ
          have : |x j| ≤ 0 := by simpa [hx] using hle
          have hxj : x j = 0 := abs_eq_zero.mp (le_antisymm this (abs_nonneg _))
          simp [chartMap, hji, hjJ, hx, hxj]
        · field_simp [chartMap, hji, hjJ, hx]
      · simp [chartMap, hji, hjJ]
  · simp
  · intro j hjJ hji
    by_cases hx : x i = 0
    · simp [hji, hx]
    · have hle : |x j| ≤ |x i| := hmax j hjJ
      have : |x j / x i| = |x j| / |x i| := abs_div _ _
      simp only [hji, hx, if_false, if_neg hji, this]
      exact (div_le_one (abs_pos.mpr hx)).mpr hle

/-- chart の像が元の箱からはみ出さないこと (被覆の「過剰がない」側)。 -/
theorem blowup_image (J : Finset (Fin n)) (i : Fin n) {b : ℝ} (hb : b ≤ 1)
    (y : Fin n → ℝ) (hi : |y i| ≤ b) (hj : ∀ j ∈ J, j ≠ i → |y j| ≤ 1)
    (hk : ∀ j, j ∉ J → |y j| ≤ b) :
    ∀ j, |chartMap J i y j| ≤ b := by
  intro j
  by_cases hji : j = i
  · subst hji; simpa [chartMap] using hi
  · by_cases hjJ : j ∈ J
    · have h1 : |y i * y j| = |y i| * |y j| := abs_mul _ _
      have h2 : |y j| ≤ 1 := hj j hjJ hji
      have : |y i| * |y j| ≤ b * 1 := by
        exact mul_le_mul hi h2 (abs_nonneg _) (le_trans (abs_nonneg _) hi)
      simpa [chartMap, hji, hjJ, h1] using this.trans_eq (mul_one b)
    · simpa [chartMap, hji, hjJ] using hk j hjJ
'''

_README = """# 自動生成された証明義務

    f = {f}
    lambda = {lam},  位数 m = {m},  証明義務 {n} 件

## 検証のしかた

Lean 4 と Mathlib が必要。elan を入れてから:

```sh
lake update                                  # Mathlib を取得
cp .lake/packages/mathlib/lean-toolchain .   # ツールチェインを合わせる
lake build
```

`lean-toolchain` は仮の値なので、上のように Mathlib のものに合わせること。
`lake build` がエラーなく終われば全ての義務が証明されたことになる。

## 何が証明されるか

| 定理 | 主張 | 戦術 |
|---|---|---|
| `chart_*` | 合成写像に沿って f が単項式 x^k と f_rest の積になる | `ring` |
| `unit_*`  | f_rest(0) != 0、すなわち正規交差になっている | `norm_num` |
| `jac_*`   | ヤコビ行列式 = x^h * (原点で非零な因子) | `ring` |
| `inv_*`   | 座標変換・平行移動が全単射 (逆写像を明示) | `ring` |
| `cover_*` | ブローアップの chart 族が全射 (不足がない) | `LeanCover.blowup_cover` |
| `image_*` | chart の像が元の箱に収まる (過剰がない) | `LeanCover.blowup_image` |

`chart_*`, `unit_*`, `jac_*` が k, h, f_rest の帳簿を、`cover_*`, `inv_*` が
「場合分けに漏れがない」ことを担保する。この二つが揃うと
lambda = min_chart min_j (h_j+1)/k_j の入力がすべて検証されたことになる。

## 証明されないこと

* ヤコビ行列の成分が本当に phi の偏微分であること (sympy が計算した値を
  そのまま定数として渡している)。
* ゼータ関数の極の位置が min_j (h_j+1)/k_j であるという解析的な主張。
  これは Watanabe の定理そのもので、ここでは形式化していない。
* 座標変換 `inv_*` が定義域全体で全単射であること (原点近傍での局所的な
  主張のみ)。
"""

_LAKEFILE = '''import Lake
open Lake DSL

package resolution where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib Resolution where
  roots := #[`LeanCover, `Resolution]
'''


# ----------------------------------------------------------------------
# 生成本体
# ----------------------------------------------------------------------
def _subst_of_step(step, gens) -> Optional[Dict[sp.Symbol, sp.Expr]]:
    """Step.detail からは復元しないので、代入は phi の差分から作る。"""
    return None


def _compose_phi(chart) -> Dict[sp.Symbol, sp.Expr]:
    return dict(chart.phi)


def _check_chart_identity(f, chart) -> Tuple[bool, sp.Expr, sp.Expr]:
    """f(phi(y)) == y^k * f_rest(y) を sympy で検算する。"""
    gens = chart.gens
    lhs = sp.expand(f.subs({v: chart.phi[v] for v in gens}, simultaneous=True))
    rhs = sp.expand(_monomial(gens, chart.k) * chart.f_rest())
    return sp.simplify(sp.expand(lhs - rhs)) == 0, lhs, rhs


def _jacobian_det(chart) -> sp.Expr:
    gens = chart.gens
    M = sp.Matrix([[sp.diff(chart.phi[v], w) for w in gens] for v in gens])
    return sp.expand(M.det())


def _invert_substitution(v, e, gens):
    """置換 v ↦ e(v, ...) の逆写像を求める。

    e が v について 1 次なら厳密に解ける (座標変換と平行移動はこの形)。
    分母が 0 でない仮定が要るときは (仮定名, 式) を第 2 成分で返す。
    """
    e = sp.together(sp.expand(e))
    num, den = sp.fraction(e)
    try:
        p = sp.Poly(sp.expand(num), v)
    except sp.PolynomialError:
        return None, None
    if p.degree() != 1:
        return None, None
    A = sp.cancel(p.coeff_monomial(v) / den)
    B = sp.cancel(p.coeff_monomial(1) / den)
    inv = sp.cancel((v - B) / A)
    hyp = None if A.is_number else ("hA", A)
    return inv, hyp


def _image_theorem(name: str, n: int, J: Sequence[int]) -> str:
    js = ", ".join(str(j) for j in J)
    return (f"/-- {name}: chart の像は元の箱 {{|x_j| <= b}} からはみ出さない "
            f"(b <= 1)。 -/\n"
            f"theorem image_{name} (i : Fin {n}) {{b : ℝ}} (hb : b ≤ 1)\n"
            f"    (y : Fin {n} → ℝ) (hi : |y i| ≤ b)\n"
            f"    (hj : ∀ j ∈ ({{{js}}} : Finset (Fin {n})), j ≠ i → |y j| ≤ 1)\n"
            f"    (hk : ∀ j, j ∉ ({{{js}}} : Finset (Fin {n})) → |y j| ≤ b) :\n"
            f"    ∀ j, |chartMap {{{js}}} i y j| ≤ b :=\n"
            f"  blowup_image _ i hb y hi hj hk\n")


def _cover_theorem(name: str, n: int, J: Sequence[int]) -> str:
    js = ", ".join(str(j) for j in J)
    return (f"/-- {name}: 中心 {{x_j = 0 | j ∈ {{{js}}}}} のブローアップの "
            f"chart 族は全射。 -/\n"
            f"theorem cover_{name} (x : Fin {n} → ℝ) :\n"
            f"    ∃ i ∈ ({{{js}}} : Finset (Fin {n})), ∃ y : Fin {n} → ℝ,\n"
            f"      chartMap {{{js}}} i y = x ∧ y i = x i ∧\n"
            f"      ∀ j ∈ ({{{js}}} : Finset (Fin {n})), j ≠ i → |y j| ≤ 1 :=\n"
            f"  blowup_cover _ (by decide) x\n")


def export_lean(
    resolution,
    directory: str = "lean_out",
    *,
    include_jacobian: bool = True,
    include_image: bool = True,
    max_jacobian_vars: int = 3,
    strict: bool = False,
    certificate=None,
    max_expr_terms: int = 400,
) -> LeanExport:
    """解消の結果を Lean 4 の証明義務として書き出す。

    resolution : resolve_singularities の返り値
    strict     : sympy の検算に落ちた等式があれば例外を投げる
    certificate: certify() の結果。渡すと各 chart の定義域と検証状況を
                 ヘッダのコメントに書き込む。
    """
    gens = resolution.gens
    n = len(gens)
    f = sp.expand(resolution.f)
    os.makedirs(directory, exist_ok=True)

    obligations: List[Obligation] = []
    body: List[str] = []

    def binder(exprs, extra=()):
        used = set()
        for e in exprs:
            used |= sp.sympify(e).free_symbols
        names = [lean_name(v) for v in gens if v in used]
        names = list(extra) + names
        return (" (" + " ".join(names) + " : ℝ)") if names else ""

    var_decl = " ".join(lean_name(v) for v in gens)

    # --- 葉ごとの合成等式と単元条件 ---------------------------------
    for ch in resolution.charts:
        nm = ch.name.replace("-", "_").replace(".", "_")
        ok, lhs, rhs = _check_chart_identity(f, ch)
        mono = _monomial(gens, ch.k)
        try:
            lhs_sub = f.subs({v: ch.phi[v] for v in gens}, simultaneous=True)
            stmt = (f"/-- chart {ch.name}: f ∘ phi = (単項式 x^{tuple(ch.k)}) "
                    f"* f_rest -/\n"
                    f"theorem chart_{nm}{binder([lhs_sub, mono, ch.f_rest()])} :\n"
                    f"    {lean_expr(lhs_sub)}\n"
                    f"      = {lean_expr(mono)} * ({lean_expr(ch.f_rest())}) := by\n"
                    f"  ring\n")
        except ValueError as e:
            obligations.append(Obligation(f"chart_{nm}", "chart", "", None,
                                          f"出力できません: {e}"))
            continue
        obligations.append(Obligation(f"chart_{nm}", "chart", stmt, ok,
                                      f"k={tuple(ch.k)}"))
        body.append(stmt)

        # f_rest(0) != 0 : Lean 側で実際に 0 を代入して評価させる
        subs0 = ch.f_rest().subs({v: 0 for v in gens})
        okv = subs0 != 0
        lam_binder = " ".join(lean_name(v) for v in gens)
        zeros = " ".join("0" for _ in gens)
        stmt2 = (f"/-- chart {ch.name}: f_rest(0) = {subs0} ≠ 0 (正規交差) -/\n"
                 f"theorem unit_{nm} :\n"
                 f"    (fun {lam_binder} : ℝ => {lean_expr(ch.f_rest())}) {zeros} "
                 f"≠ 0 := by\n  norm_num\n")
        obligations.append(Obligation(f"unit_{nm}", "unit", stmt2, okv,
                                      f"f_rest(0) = {subs0}"))
        body.append(stmt2)

        # --- ヤコビ行列式 -------------------------------------------
        if include_jacobian and n <= max_jacobian_vars:
            det = _jacobian_det(ch)
            hmono = _monomial(gens, ch.h)
            q = sp.cancel(sp.together(det / hmono)) if hmono != 0 else sp.S.Zero
            num, den = sp.fraction(q)
            unit_ok = (den.subs({v: 0 for v in gens}) != 0
                       and num.subs({v: 0 for v in gens}) != 0)
            okj = sp.simplify(sp.expand(det - hmono * q)) == 0 and unit_ok
            try:
                rows = [[sp.diff(ch.phi[v], w) for w in gens] for v in gens]
                entries = ";\n         ".join(
                    ", ".join(lean_expr(e) for e in row) for row in rows)
                detfn = {1: "Matrix.det_fin_one", 2: "Matrix.det_fin_two",
                         3: "Matrix.det_fin_three"}[n]
                jexprs = [e for row in rows for e in row] + [hmono, q]
                stmt3 = (f"/-- chart {ch.name}: ヤコビ行列式 = x^{tuple(ch.h)} "
                         f"* (単元)。行列の成分は phi の偏微分 (sympy で計算)。 -/\n"
                         f"theorem jac_{nm}{binder(jexprs)} :\n"
                         f"    Matrix.det (!![{entries}]"
                         f" : Matrix (Fin {n}) (Fin {n}) ℝ)\n"
                         f"      = {lean_expr(hmono)} * ({lean_expr(q)}) := by\n"
                         f"  rw [{detfn}]\n  ring\n")
                obligations.append(Obligation(f"jac_{nm}", "jac", stmt3, okj,
                                              f"h={tuple(ch.h)}, 単元部 = {q}"))
                body.append(stmt3)
            except (ValueError, KeyError) as e:
                obligations.append(Obligation(f"jac_{nm}", "jac", "", None,
                                              f"出力できません: {e}"))

    # --- ブローアップ節点ごとの被覆 ----------------------------------
    for name, ch in sorted(resolution.nodes.items()):
        if ch.status != "blowup":
            continue
        # 子の入口ステップに記録された中心を使う
        J, wts = None, ()
        for cch in resolution.nodes.values():
            if cch.parent == name:
                st = next((t for t in cch.own_steps() if t.kind == "blowup"), None)
                if st is not None and st.center:
                    J = [gens.index(v) for v in st.center]
                    wts = tuple(st.weights)
                    break
        if not J or len(J) < 2:
            continue
        if wts and any(v != 1 for v in wts):
            # 重み付きブローアップの被覆補題は LeanCover に未形式化
            nm = name.replace("-", "_").replace(".", "_")
            obligations.append(Obligation(
                f"cover_{nm}", "cover", "", None,
                f"重み {list(wts)} のブローアップ: 一般補題が未形式化のため出力しません"))
            continue
        nm = name.replace("-", "_").replace(".", "_")
        stmt = _cover_theorem(nm, n, sorted(J))
        obligations.append(Obligation(f"cover_{nm}", "cover", stmt, True,
                                      f"J={[str(gens[j]) for j in sorted(J)]}"))
        body.append(stmt)
        if include_image:
            stmt_i = _image_theorem(nm, n, sorted(J))
            obligations.append(Obligation(f"image_{nm}", "image", stmt_i, True,
                                          "像が箱からはみ出さない (過剰がない側)"))
            body.append(stmt_i)

    # --- 座標変換・平行移動が全単射であること --------------------------
    for name, ch in sorted(resolution.nodes.items()):
        st = next((t for t in ch.own_steps()
                   if t.kind in ("coordchg", "recenter")), None)
        if st is None or not st.subs:
            continue
        nm = name.replace("-", "_").replace(".", "_")
        for v, e in st.subs:
            inv, hyp = _invert_substitution(v, e, gens)
            if inv is None:
                obligations.append(Obligation(f"inv_{nm}_{lean_name(v)}", "inv",
                                              "", None, "逆写像を構成できません"))
                continue
            ok = sp.simplify(sp.expand(e.subs(v, inv) - v)) == 0
            hyp_txt = f" ({hyp[0]} : {lean_expr(hyp[1])} ≠ 0)" if hyp else ""
            tac = "  field_simp\n  ring\n" if hyp else "  ring\n"
            # 合成をそのまま書き下す (sympy に約分させない)
            t = sp.Symbol("t_placeholder", real=True)
            lhs_txt = lean_expr(e.subs(v, t))
            inv_txt = lean_expr(inv.subs(v, t))
            lhs_txt = re.sub(r"\bt_placeholder\b", f"({inv_txt})", lhs_txt)
            lhs_txt = re.sub(r"\bt_placeholder\b", "t", lhs_txt)
            used_names = ["t"] + [lean_name(w) for w in gens
                                  if re.search(rf"\b{lean_name(w)}\b", lhs_txt)]
            stmt = (f"/-- {ch.name}: 座標変換 {v} ↦ {e} は全単射 "
                    f"(逆写像 {v} ↦ {inv}) -/\n"
                    f"theorem inv_{nm}_{lean_name(v)}"
                    f" ({' '.join(used_names)} : ℝ){hyp_txt} :\n"
                    f"    {lhs_txt} = t := by\n"
                    f"{tac}")
            obligations.append(Obligation(f"inv_{nm}_{lean_name(v)}", "inv",
                                          stmt, ok, f"{v} ↦ {e}"))
            body.append(stmt)

    # --- ファイル出力 -------------------------------------------------
    files = []
    p_cover = os.path.join(directory, "LeanCover.lean")
    with open(p_cover, "w", encoding="utf-8") as fh:
        fh.write(_COVER_LEAN)
    files.append(p_cover)

    header = [
        "/-",
        f"  Resolution.lean — 自動生成 (resolve_singularity.py)",
        f"  f = {f}",
        f"  変数: {list(gens)}",
        f"  lambda = {resolution.rlct}, 位数 m = {resolution.multiplicity}",
        "",
        "  chart_* : 合成写像に沿った f の正規交差化 (ring)",
        "  unit_*  : f_rest が原点で単元 (norm_num)",
        "  jac_*   : ヤコビ行列式 = x^h * 単元 (ring)",
        "  cover_* : ブローアップの chart 族が全射 (LeanCover.blowup_cover)",
        "  image_* : chart の像が元の箱からはみ出さない (LeanCover.blowup_image)",
        "-/",
    ]
    if certificate is not None:
        header += ["/-", f"  certify() の判定: {certificate.status}"]
        for lf in certificate.leaves:
            header.append(f"    {lf}")
        for nm2, st2, why2 in certificate.covering:
            header.append(f"    [{st2}] 被覆 {nm2}: {why2}")
        header += ["-/"]
    header += [
        "import Mathlib",
        "import LeanCover",
        "",
    ]
    p_res = os.path.join(directory, "Resolution.lean")
    with open(p_res, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n" + "\n".join(body))
    files.append(p_res)

    p_lake = os.path.join(directory, "lakefile.lean")
    with open(p_lake, "w", encoding="utf-8") as fh:
        fh.write(_LAKEFILE)
    files.append(p_lake)

    p_readme = os.path.join(directory, "README.md")
    with open(p_readme, "w", encoding="utf-8") as fh:
        fh.write(_README.format(f=f, lam=resolution.rlct,
                                m=resolution.multiplicity,
                                n=len(obligations)))
    files.append(p_readme)

    p_tc = os.path.join(directory, "lean-toolchain")
    with open(p_tc, "w", encoding="utf-8") as fh:
        fh.write("leanprover/lean4:v4.15.0\n")
    files.append(p_tc)

    exp = LeanExport(directory=directory, obligations=obligations,
                     gens=gens, files=files)
    if strict and exp.failures:
        raise ValueError("sympy の検算に失敗した証明義務があります:\n"
                         + "\n".join(str(o) for o in exp.failures))
    return exp


# ----------------------------------------------------------------------
# デモ
# ----------------------------------------------------------------------
def _demo():
    from resolve_singularity import resolve_singularities

    x, y, z = sp.symbols("x y z", real=True)

    for label, f, g, kw in [
        ("x^2+y^3", x**2 + y**3, (x, y), dict(prune=False)),
        ("x^2*y^2", x**2 * y**2, (x, y), {}),
        ("(x-y)^2", (x - y) ** 2, (x, y), {}),
        ("x^2+y^2+z^2", x**2 + y**2 + z**2, (x, y, z), dict(prune=False)),
    ]:
        res = resolve_singularities(f, g, **kw)
        out = f"lean_{label.replace('^','').replace('*','').replace('+','_').replace('(','').replace(')','').replace('-','m')}"
        exp = export_lean(res, out)
        print("=" * 66)
        print(f"{label}:  lambda = {res.rlct}, m = {res.multiplicity}")
        print(exp.summary())
    # --- 生成器そのものが検査器になることの確認 ---------------------
    print("=" * 66)
    print("帳簿をわざと壊すと sympy の段階で検出される:")
    res = resolve_singularities(x**2 + y**3, (x, y), prune=False)
    bad = res.charts[0]
    bad.k = [bad.k[0] + 1] + list(bad.k[1:])       # k を 1 つずらす
    exp = export_lean(res, "lean_broken")
    for o in exp.obligations:
        if o.ok is False:
            print("  " + str(o))
    print(f"  -> NG {len(exp.failures)} 件 (strict=True なら例外)")

    print("=" * 66)
    print("生成した Lean を検証するには、出力ディレクトリで")
    print("  lake update && lake build")
    print("を実行する (elan で Lean 4 と Mathlib が必要)。")
    v = exp.verify()
    print("verify():", v.get("message", v))


if __name__ == "__main__":
    _demo()
