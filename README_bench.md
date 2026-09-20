# bench — 逐次改善と回帰テストの土台

## 何のためか

proved の割合を上げる作業は、判定を正しく直すことでも、単に甘くすることでも
達成できてしまう。両者を区別しながら改善を回すための最小構成。

## 使い方

```sh
export PYTHONPATH=..:.
python regression.py                      # 一括 (guards + structured + nn)
python regression.py --all --out r.json   # 全スイート
python guards.py                          # 門番だけ
python run.py --max-vars 4 --random --out r.json
python report.py r.json
python run.py --failures                  # 蓄積した失敗例を再測
python run.py --hard --montecarlo --covering
python nn_cases.py                        # ニューラルネットの格子
python vandermonde.py                     # Aoyagi の真値との比較
python aoyagi_lemmas.py                   # 真値を使わない不変性の検査
python independent.py                     # 独立な検査 (体積の漸近、被覆の標本)
```

## ファイル

| | 役割 |
|---|---|
| `regression.py` | 全スイートの一括実行。CI 用の終了コード |
| `cases.py` | 構造化された族とランダム族 (n, 次数, 項数, 係数ビット長を独立に振る) |
| `nn_cases.py` | ニューラルネットの格子 (MLP x {relu, tanh} x 退化の型)。torch 非依存 |
| `known_families.py` | 真値が独立に分かり、かつ難しい族 |
| `vandermonde.py` | Vandermonde 行列型の真値 (Aoyagi 2019 Theorem 6 と N=1 の厳密式) |
| `oracles.py` | 族の閉じた式、設定違いの突き合わせ、1/m <= lambda <= n/m |
| `aoyagi_lemmas.py` | Aoyagi の補題を真値なしの検査に (イデアル不変性/単調性/分離/最深点/座標不変性) |
| `independent.py` | 解消の結果を使わない検査 (体積の漸近による lambda 推定、被覆の前向き標本) |
| `guards.py` | 門番。負のテスト + 変異検査 + 独立検査 + 真値照合 + 座標不変性、既知バグの報告 |
| `failures.py` | 失敗例の蓄積と自動縮約 |
| `run.py` / `report.py` | 個別実行と集計 |

## 失敗の分類とパッチの当て先

| クラス | 意味 | 当てる場所 |
|---|---|---|
| `resolve_max_depth` | 解消が止まらない | 中心の選択則、重み |
| `smt_unknown` | 判定が決まらない | 面の分解、ファイバー制約、timeout |
| `nc_refuted` | 正規交差が壊れる点あり | localize、定義域の切り分け |
| `identity_refuted` | 恒等式が不成立 | 解消側の帳簿 |
| `covering_unknown` | 被覆が確立しない | 補題の追加 |
| `inconsistent` | 独立な値と食い違う | **最優先で調査** |

## 運用ルール

1. パッチを当てる前に、必ず最小の失敗例に縮約する (`failures.minimize`)
2. `guards.py` が全件緑でなければ採用しない
3. 主指標は proved 率ではなく「既知の値と一致した proved の数」
4. 失敗は保存し、修正後に `run.py --failures` で再測する

## 既知の未修正バグ

`guards.known_issues()` が毎回報告する (緑のゲートには数えない)。修正されたら
OK に変わるので、その時点で通常のガードに昇格させる。

- Vandermonde 型の chart 内で座標依存: chart 座標で 3/2、G_1 を座標に取ると 7/6。
  解消器が多項式を運んでいてイデアルを運んでいないため。`ideal_resolve.py` は
  この問題を持たないが phi を追跡していない。

## 測定値 (2026-09 時点)

* 構造化 51 件 (n<=6, d<=13): proved 51/51
* ランダム 111 件 (n<=3, d<=6): proved 111/111
* ランダム 36 件 (n=4-5, d=4-8): proved 22/36、失敗は全て `resolve_max_depth`
* ニューラルネット 54 件: proved 39、コア 0-3 変数なら 100%、4 変数で 50%、5 以上で 0%
* guards 44 件: 全緑
