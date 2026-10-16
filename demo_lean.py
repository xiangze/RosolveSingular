r"""
demo_lean.py — `lean_export` のデモ

`lean_export` 本体に置くと `resolve_singularity` を import する必要があり、
`resolve_singularity -> lean_export` と合わせて循環になる。デモだけを上の層に
出すことで、`lean_export` は Resolution を受け取るだけの**消費側**に保てる
(REORG_PLAN.ja.md §2)。

    python demo_lean.py
"""
from __future__ import annotations

import sympy as sp

# ----------------------------------------------------------------------
# デモ
# ----------------------------------------------------------------------
def main():
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
    main()
