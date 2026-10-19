# RLCT toolkit / 実対数閾値ツール群

Resolution of singularities for computing real log canonical thresholds (RLCT),
with machine-checked certificates and a regression-test harness.

- 日本語ドキュメント: [DOCUMENTATION.ja.md](DOCUMENTATION.ja.md)
- English documentation: [DOCUMENTATION.en.md](DOCUMENTATION.en.md)
- 構成と階層化の経緯: [REORG_PLAN.ja.md](REORG_PLAN.ja.md)
- 高速化の調査: [SPEEDUP_SURVEY.ja.md](SPEEDUP_SURVEY.ja.md)

## インストール

```sh
pip install sympy scipy pulp z3-solver graphviz python-flint   # z3 が無いと証明できません
apt-get install singular graphviz                              # 任意
pip install torch                                              # models/ を使う場合のみ
```

## 最短の使い方

```python
import sympy as sp
from rlct import rlct_certified, rlct_interval

x, y, z = sp.symbols("x y z", real=True)
rlct_certified(x**2 + y**3, (x, y))      # -> (5/6, 'proved', 'newton')
rlct_interval(f, gens)                   # 厳密値が出ないときは健全な区間
```

`import rlct` で closer への結線 (判定器 -> closer の注入) が済みます。

## 構成

**依存は常に下向きで、循環はありません** (`bench.guards.layering_tests` が毎回検査)。

```
rlct/                    ライブラリ
  core/
    timing.py            段階別の実行時間
    newton.py            多面体・面の列挙・LP・非退化判定 (**葉**)
    resolve_singularity.py  ブローアップ + 正規化、重み付き、分枝限定、Graphviz
    ideal_resolve.py     イデアルを運ぶ解消 (生成元を座標に取れる)
    invariants.py        重複度・擬斉次性・Milnor/Tjurina
    rlct_cache.py        解いた局所データの永続キャッシュ
    certify.py           三値の証明書 (proved/unknown/refuted)、健全な区間
  closers/               族ごとに閉じる経路。**解き手を import せず注入で受け取る**
    families.py          単項式/積/和/形式のべき/ニュートン非退化
    lemmas.py            Aoyagi の補題 (挟み込み、積の分解、単調性の下界)
    dln_oracle.py        深層線形網の閉形式 (detect -> reconstruct -> compare)
    lc_filter.py         codim/2 の健全な上下界 (有理点 + b 関数)
    strata_closer.py     対称性の層からの上界
  models/                モデルからファイバーイデアルを作る
    nn_rlct.py           一般のイデアルと 1 隠れ層 ReLU 網
    torch_rlct.py        PyTorch の読み込み
    transformer_rlct.py  Transformer 1 層
    resnet_rlct.py       ResNet18/50 (BN 融合込み)
  export/
    lean_export.py       chart 木の証明義務を Lean 4 に書き出す
  data.py                画像データの読み込み (学習と評価で共有)
exp/                     学習と掃引 (ライブラリではない)
  train_rlct.py train_resnet.py llc_estimate.py llc_compare.py
  cnn_sweep.py taylor_sweep.py random_regression.py
  stage_dln.py stage_mlp.py stage_report.py
bench/                   ガード・ケース・オラクル・回帰テスト
  guards.py cases.py nn_cases.py known_families.py oracles.py agreement.py
  aoyagi_lemmas.py failures.py independent.py regression.py run.py report.py
  build_cache.py vandermonde.py dln.py atw_invariant.py bench_strata.py
```

層の順序 (下から):

```
timing, newton          -> families, invariants, rlct_cache
                        -> lc_filter, lemmas -> resolve_singularity
                        -> ideal_resolve, nn_rlct -> certify
                        -> torch_rlct, resnet_rlct, transformer_rlct -> exp, bench
```

## パラメータ掃引

`exp/cnn_sweep.py` と `exp/taylor_sweep.py` は層数・活性化・テイラー次数を
リストで受け取りますが、**形状はスカラー**です。形状も振る場合は
`exp/shape_sweep.py` を外側から回します。

```sh
python -m exp.shape_sweep --list                      # 格子の一覧と最悪時間の見積り
python -m exp.shape_sweep --grid cnn-small --dry-run  # 計画だけ (計算しない)
python -m exp.shape_sweep --grid cnn-small --out runs/cnn.jsonl
python -m exp.shape_sweep --report runs/cnn.jsonl --csv runs/cnn.csv
```

* 形状の組ごとに子プロセスで内側の掃引を起動します。内側もさらに次数ごとに
  プロセスを分けているので、1 つ固まっても掃引は進みます。
* **再開できます。**同じコマンドを再実行すると、揃っている形状は飛ばします
  (揃っていない形状はやり直し)。
* **タイムアウトと失敗を落としません。**`status` が `ok` / `timeout` /
  `error` のいずれかで必ず記録されます。λ が出なかったこともデータです。
* `-- ` の後ろの引数は内側の掃引にそのまま渡ります
  (例: `-- --orders 0,1` で次数を絞る)。

## 回帰テスト

**リポジトリの根から**実行します (パッケージなので `-m` を使う)。

```sh
python -m bench.regression              # guards + structured + nn
python -m bench.regression --all        # ランダム多項式・難しい族・失敗例も
python -m bench.guards                  # 門番だけ (213 件)
python -m bench.bench_strata --quick    # 層の公式と Aoyagi の閉形式の照合
python -m exp.train_rlct --model mlp --size 4 --epochs 300 --rlct
```

終了コードは、guards に赤があるか独立な値との食い違いがあれば 1。
