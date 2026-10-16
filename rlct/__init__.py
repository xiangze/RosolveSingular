"""rlct — 実対数閾値 (RLCT) の厳密計算。

階層 (依存は常に下向き、循環なし。REORG_PLAN.ja.md):

    core/     多面体と LP (newton) / 解消 (resolve_singularity, ideal_resolve)
              / 不変量 / キャッシュ / 判定器 (certify)
    closers/  族ごとに閉じる経路 (families, lemmas, dln_oracle, lc_filter,
              strata_closer)。**解き手を import せず注入で受け取る**
    models/   モデルからファイバーイデアルを作る層
    export/   Lean への書き出し

`import rlct` で closer への結線が済む (core.certify が
closers.lemmas.install_subsolver を呼ぶ)。
"""
from rlct.core import certify as _certify   # noqa: F401  (closer の結線)
from rlct.core.certify import (  # noqa: F401
    certify, rlct_certified, rlct_interval,
)
from rlct.core.newton import (  # noqa: F401
    newton_multiplicity, newton_nondegenerate, newton_rlct,
)
from rlct.core.resolve_singularity import (  # noqa: F401
    ResolutionFailure, resolve_singularities,
)

__all__ = [
    "certify", "rlct_certified", "rlct_interval",
    "resolve_singularities", "ResolutionFailure",
    "newton_rlct", "newton_nondegenerate", "newton_multiplicity",
]
