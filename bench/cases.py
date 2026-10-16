r"""
bench/cases.py — ベンチマークの入力を作る。

場当たりに大きくするのではなく、次元 n・全次数 d・項数 t を独立に振る格子に
する。失敗したときにどの軸が効いたのかを特定できるようにするため。

  structured_cases() : 構造が分かっている族 (オラクルが閉じた式で持てる)
  random_cases()     : スパースなランダム多項式 (seed 固定で再現可能)
"""
from __future__ import annotations

import itertools
import random
from typing import List, Tuple

import sympy as sp

Case = Tuple[str, sp.Expr, Tuple[sp.Symbol, ...], str]   # (名前, f, 変数, 族)


def V(n, prefix="x"):
    return sp.symbols(f"{prefix}0:{n}", real=True)


def structured_cases(max_vars: int = 5) -> List[Case]:
    out: List[Case] = []
    # --- 単項式 ---------------------------------------------------------
    for n in range(1, max_vars + 1):
        v = V(n)
        for exps in ([2] * n, list(range(1, n + 1)), [1] + [4] * (n - 1)):
            f = sp.prod([vv ** e for vv, e in zip(v, exps)])
            out.append((f"monomial n={n} {exps}", f, v, "monomial"))
    # --- Brieskorn (対角型) ---------------------------------------------
    for n in range(2, max_vars + 1):
        v = V(n)
        for exps in ([2] * n, [2, 3] + [4] * (n - 2), list(range(2, n + 2)),
                     [3, 4, 5][:n] + [2] * max(0, n - 3)):
            exps = exps[:n]
            f = sum(vv ** e for vv, e in zip(v, exps))
            out.append((f"brieskorn {exps}", f, v, "brieskorn"))
    # --- 交差項つき ------------------------------------------------------
    for n in range(2, max_vars + 1):
        v = V(n)
        out.append((f"cross n={n}",
                    sum(v[i] ** 2 * v[(i + 1) % n] ** 2 for i in range(n)),
                    v, "cross"))
        out.append((f"sq+cross n={n}",
                    sum(vv ** 2 for vv in v) + sp.prod(v) ** 2, v, "sq+cross"))
    # --- 形式のべき -----------------------------------------------------
    for n in range(2, max_vars + 1):
        v = V(n)
        lin = sum(v)
        out.append((f"(sum x)^2 n={n}", lin ** 2, v, "power_form"))
        out.append((f"(sum x)^3 n={n}", lin ** 3, v, "power_form"))
        q = sum(v[i] * v[i + 1] for i in range(n - 1))
        if q != 0:
            out.append((f"q^2 indefinite n={n}", q ** 2, v, "power_form"))
    # --- 変数が互いに素な積・和 -----------------------------------------
    for n in (2, 3):
        u, w = V(n, "u"), V(n, "w")
        out.append((f"prod of sums n={n}",
                    sum(uu ** 2 for uu in u) * sum(ww ** 2 for ww in w),
                    u + w, "product"))
    # --- 低ランク型 -----------------------------------------------------
    for h in (1, 2, 3):
        a, b = V(h, "a"), V(h, "b")
        out.append((f"(a.b)^2 h={h}",
                    (sum(ai * bi for ai, bi in zip(a, b))) ** 2, a + b,
                    "lowrank"))
        out.append((f"||ab^T||^2 h={h}",
                    sum((ai * bj) ** 2 for ai in a for bj in b), a + b,
                    "lowrank"))
    # --- 退化した例 (証明経路が無いはずのもの) ---------------------------
    x, y, z = sp.symbols("x y z", real=True)
    out += [
        ("(x^2-y^3)^2", (x**2 - y**3) ** 2, (x, y), "degenerate"),
        ("(x^2-y^2*z)^2", (x**2 - y**2 * z) ** 2, (x, y, z), "degenerate"),
        ("x^2+x*y^4+y^6", x**2 + x * y**4 + y**6, (x, y), "degenerate"),
        ("(xy-z^2)^3", (x * y - z**2) ** 3, (x, y, z), "degenerate"),
    ]
    return out


def random_cases(n_list=(2, 3, 4), d_list=(3, 4, 6), t_list=(2, 3, 5),
                 coeff_bits=(1, 6), per_combo: int = 2,
                 seed: int = 20260912) -> List[Case]:
    """スパースなランダム多項式。原点で消えるよう定数項は入れない。"""
    rng = random.Random(seed)
    out: List[Case] = []
    for n, d, t, cb in itertools.product(n_list, d_list, t_list, coeff_bits):
        v = V(n)
        for rep in range(per_combo):
            terms = []
            for _ in range(t):
                while True:
                    e = [rng.randint(0, d) for _ in range(n)]
                    if 1 <= sum(e) <= d:
                        break
                c = rng.randint(1, 2 ** cb) * rng.choice([1, -1])
                terms.append(c * sp.prod([vv ** ee for vv, ee in zip(v, e)]))
            f = sp.expand(sum(terms))
            if f == 0 or f.is_number:
                continue
            out.append((f"rand n={n} d={d} t={t} bits={cb} #{rep}", f, v,
                        "random"))
    return out
