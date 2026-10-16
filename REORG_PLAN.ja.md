# ファイル構成の整理・階層化 — 実施記録 (2026-10-13 計画 / 2026-10-17 実施)

対象: `xiangze/RosolveSingular`。

> ## 実施結果 (2026-10-17)
>
> **循環依存を 0 にし、46 本を 3 つのパッケージに階層化した。ガードは全段で
> 213/213 緑、λ と m は 1 件も変わっていない。**
>
> | | 実施前 | 実施後 |
> |---|---|---|
> | 最大の強連結成分 | **9 モジュール** (6,000 行超) | **0** |
> | 層の深さ | 9 段 + 塊 | 12 段 (塊なし) |
> | ルート直下の .py | 46 本 | 0 本 (`rlct/` `exp/` `bench/`) |
> | 関数内に隠れた循環 import | 20 本以上 | 0 |
> | ガード | 213/213 | **216/216** (階層の検査 3 本を追加) |
>
> ### 段ごとの実施内容
>
> **段 1: `rlct/core/newton.py` の切り出し** (コミット `refactor(1)`)
> `certify.py` が「判定器」と「ニュートン多面体の道具」を同居させていた。
> 後者と、`resolve_singularity.py` の LP/ILP を `newton.py` に分離した。
>
> | | 前 | 後 |
> |---|---|---|
> | `certify.py` | 1633 行 | 1064 行 |
> | `resolve_singularity.py` | 2027 行 | 1865 行 |
> | `newton.py` | — | 899 行 |
>
> あわせて `_demo()` の中だけで上の層を import していた 3 箇所を解消
> (`invariants` の RLCT 節を除去、`ideal_resolve` の Vandermonde 比較を
> `vandermonde.demo_ideal_route()` へ、`lean_export._demo` を
> `bench/demo_lean.py` へ)。**強連結成分 9 -> 4。**
>
> **段 2: 依存の逆転 3 箇所** (コミット `refactor(2)`)
> * `lemmas` の相互再帰 -> `install_subsolver()` で上の層から解き手を注入。
>   既定の解き手は `certify` の import 時に入る。注入が無ければ `None` を
>   返して分解をあきらめる (**健全側に倒れる**)。
> * `resnet_rlct <-> train_resnet` -> データ読み込みを `rlct/data.py` に分離。
> * `failures <-> run` -> `make_predicate` が `run_case` を引数で受け取る。
>
> **強連結成分 4 -> 0。**
>
> **段 3: パッケージ化** (コミット `refactor(3)`)
> 46 本を `rlct/{core,closers,models,export}` `exp/` `bench/` に移し、
> import 231 箇所を書き換えた。モジュール名は変えていない
> (`nn_rlct` は `rlct.models.nn_rlct`)。
>
> **段 4: 循環を禁止するガード** (`bench.guards.layering_tests`)
> (1) 強連結成分がサイズ 2 以上でないこと、(2) `newton` が葉であること・
> `closers/*` が判定器を import しないこと・`rlct/` が `bench/` `exp/` を
> import しないこと、(3) closer への結線が効いていること。
> **関数内 import も数える。**故意に循環を仕込むと 2 本が赤になることを確認済み。
>
> ### 副産物として直したもの
>
> * `exp/cnn_sweep.py` と `exp/taylor_sweep.py` に **開発環境の絶対パス
>   `/home/claude` が埋め込まれていた** (子プロセスの `sys.path` 細工)。
>   他の環境では通らないので `__file__` から求める形に直した。
> * `bench/guards.py` の `sys.path[:0] = ["."]` 5 箇所、`bench/build_cache.py`
>   の同種 5 箇所、`bench/dln.py` と `bench/atw_invariant.py` の
>   ディレクトリ探索 bootstrap を、パッケージ化により不要にして除去。
> * `rlct_cache` のコード指紋のパスを新構成に合わせた。
>
> ### 残っている課題
>
> * `resolve_singularities` 656 行 / 分岐 138 の分解 (§H2 の closer 列への
>   分解)。**この整理とは別の作業**として切り離すこと — 混ぜると、どちらが
>   値を変えたのか分からなくなる。
> * `closers/protocol.py` (closer の契約を型で表す) は未実装。現状は
>   `install_subsolver` の 1 本だけで、契約は docstring にある。
> * ドキュメントが記述する `bench/` 構成と実体が食い違っていた問題は解消したが、
>   `DOCUMENTATION.ja.md` / `.en.md` の節番号は旧構成のまま。
>
> 以下は実施前の計画 (記録として残す)。実測値はすべて実施前のもの。

対象: `xiangze/RosolveSingular`。手元の作業コピー (ルート 20 本 13,973 行 +
`bench/` 16 本 5,506 行 = 36 本 19,479 行) を静的に解析した結果と、その上での提案。

> **前提 2 つ。**
> (1) 公開 HEAD は作業コピーより遅れている (別セッションの報告では
> `lemmas.py` と `bench/` が公開側に無い)。この session からは当該リポジトリへの
> GitHub アクセスが無いので、公開側の実際の木は未確認。
> (2) 別セッションが `strata_closer.py` / `bench_strata.py` のパッチを
> **ベース 359a038 に対して未 push** で持っている。移行はこれと衝突しうるので
> §4 に手当てを書いた。

---

## 1. 診断 — 問題はファイル数ではなく、核心が 1 つの塊になっていること

ファイルが増えたこと自体は問題ではない。静的解析で出た本当の問題は**相互依存**。

### 1.1 核心に 9 モジュールの強連結成分がある [実測]

```
certify ↔ resolve_singularity ↔ lemmas ↔ families ↔ invariants
        ↔ ideal_resolve ↔ lean_export ↔ rlct_cache ↔ bench/vandermonde
```

合計 6,000 行超が互いに依存していて、**どれが下でどれが上かが定義されていない**。
他に小さい循環が 2 つ: `resnet_rlct ↔ train_resnet`、`bench/failures ↔ bench/run`。

### 1.2 循環は全て「関数の中の import」で回避されている [実測]

| モジュール | トップレベル import | 関数内 import |
|---|---|---|
| `certify` | なし | families, ideal_resolve, invariants, lemmas, resolve_singularity, rlct_cache |
| `resolve_singularity` | なし | certify, invariants, lean_export, lemmas, rlct_cache |
| `lemmas` | なし | certify, families, resolve_singularity |
| `families` | なし | certify, resolve_singularity |
| `invariants` | なし | certify |
| `ideal_resolve` | resolve_singularity | certify, resolve_singularity, vandermonde |
| `lean_export` | なし | resolve_singularity |
| `rlct_cache` | なし | resolve_singularity |

核心 9 本のうち**トップレベル import は 1 本だけ**で、残り 20 本以上は全て
関数の中に押し込んで循環を避けている。つまり:

* **動いてはいる**。どんなディレクトリ構成にしても壊れない (遅延 import は
  配置に鈍感)。だから移行の機械的リスクは低い。
* しかし**依存の向きがファイルの先頭を読んでも分からない**。新しい循環を
  防ぐ仕組みも無い。`import` を関数に隠すのは循環の*回避*であって*解消*ではない。

### 1.3 ライブラリが bench に依存している [実測]

`ideal_resolve` → `bench/vandermonde` (`lambda_N1`, `vandermonde_ideal`)。
**本体がテストデータを import している**。これが `bench/vandermonde` を
核心の強連結成分に引き込んでいる原因。

### 1.4 入口が散っている [実測]

36 本のうち **30 本に `if __name__ == "__main__"` がある**。
どれが利用者向けの入口で、どれが開発時のデモなのかが区別できない。

### 1.5 巨大な関数が高速化の衝突点になっている [既測、`SPEEDUP_SURVEY.ja.md` §H1]

```
resolve_singularities  656 行, 分岐 138, ネスト深さ 6
certify                222 行, 分岐 57
fiber_ideal_from_torch 202 行
local_rlct_from_ideal  203 行
```

提案済みの高速化のほぼ全部が `resolve_singularities` を触る。
しかもここは**健全性バグ 4 件が「読んで」見つかった**場所。

---

## 2. 鍵になる発見 — `newton.py` を切り出すと循環の 4 辺が消える [実測]

循環の各辺が**実際に何を使っているか**を調べると、`certify` への依存の大半が
ニュートン多面体の関数群だけだった:

| 辺 | `certify` から使っているもの | newton に移した後 |
|---|---|---|
| `resolve_singularity` → `certify` | `newton_multiplicity`, `newton_nondegenerate`, `principal_face_compact` | **辺が消える** |
| `families` → `certify` | 同上 3 つ | **辺が消える** |
| `ideal_resolve` → `certify` | `newton_multiplicity` | **辺が消える** |
| `rlct_cache` → `resolve_singularity` | `_lp_newton_value` のみ | **辺が消える** |
| `invariants` → `certify` | `rlct_certified`, `_HAS_Z3` | 報告層に上げる (§3) |
| `lemmas` → `certify` | `certify` 本体 | **本物の再帰** → 注入にする (§3) |

つまり `certify.py` は今 2 つの異なるものを抱えている:

* **判定器** (`certify`, `rlct_certified`, `rlct_interval`, `compute_domains`) — 上の層
* **ニュートン多面体の道具** (`newton_faces`, `newton_facet_normals`,
  `newton_multiplicity`, `newton_nondegenerate`, `principal_face_compact`,
  `_to_z3`, `_groebner_no_torus_zero`) — 下の層

後者を `core/newton.py` に出すだけで、核心の塊が崩れる。**これが最小の一手。**

---

## 3. 提案する階層

### 3.1 ディレクトリ

```
rlct/
  core/
    chart.py         Chart, Step, Resolution, ResolutionFailure, Box   (型だけ、依存なし)
    timing.py        位相ごとの計測                                      (既に依存なし)
    newton.py        多面体・面の列挙・LP・非退化 (z3/Gröbner)          ← certify から分離
    resolve.py       解消の探索 (ブローアップ・中心選択・分枝限定)
    ideal_resolve.py イデアル経路
    invariants.py    重複度・擬斉次重み・Milnor/Tjurina
    cache.py         永続キャッシュ
    certify.py       判定器 (三値証明書)。closer を束ねる               ← 判定だけ残す
  closers/
    protocol.py      closer の契約 (入力 chart / 出力 (λ, m, 証明書) / 健全性の向き)
    families.py      単項式・積・和・冪形・Newton
    lemmas.py        Aoyagi の補題 (squeeze, 積分解, 単調性)
    dln.py           深層線形網 (Aoyagi 閉形式 + detect→reconstruct→compare)
    vandermonde.py   Vandermonde 族の閉形式                            ← bench から昇格
    strata.py        対称性の層からの上界                              ← 別セッションの成果
  models/
    ideal.py         一般のイデアルから局所 RLCT (旧 nn_rlct)
    torch_io.py      PyTorch の読み込み (旧 torch_rlct)
    transformer.py   Transformer 1 層
    resnet.py        ResNet18/50 (BN 融合込み)
  io/
    lean_export.py   Resolution を Lean に書き出す (型だけ受け取る)
    report.py        報告の整形 (旧 bench/report + invariants.singularity_report)
  exp/
    train.py         MLP/CNN の学習 (旧 train_rlct)
    train_resnet.py  ResNet の学習
    llc.py           SGLD による LLC 推定
    llc_compare.py   厳密 λ と λ̂ の突き合わせ
    sweeps.py        taylor_sweep + cnn_sweep の統合
  cli.py             単一の入口 (サブコマンド)
bench/
  guards/            健全性ガード (今の 1,943 行を群ごとに分割)
  cases/             テストケースとデータ (vandermonde の*ケース*、nn_cases, …)
  oracles.py         参照オラクル
  agreement.py       閉形式との一致表
  regression.py      回帰の束ね
  atw_invariant.py   調査用の試作 (組み込まない旨を明記済み)
```

### 3.2 これが本当に DAG になることの検算 [実測]

提案後の辺を実際に書き下して強連結成分を取ると **循環なし、10 層**になる
(現状は 9 段 + 9 モジュールの塊):

```
層 0: core/chart, core/timing, exp/llc
層 1: closers/protocol, core/newton, io/lean_export
層 2: closers/dln, closers/vandermonde, core/cache, core/invariants, core/resolve
層 3: closers/families, closers/strata, core/ideal_resolve
層 4: closers/lemmas
層 5: core/certify
層 6: io/report, models/ideal
層 7: models/resnet, models/torch_io, models/transformer
層 8: exp/train, exp/train_resnet
層 9: exp/llc_compare, exp/sweeps
```

### 3.3 機械的に移せない 2 箇所

**(a) `lemmas → certify` は本物の再帰。** squeeze と積分解は部分イデアルを
判定器に投げ直す。これは**依存の逆転**で解く: closer は `certify` を import せず、
`certify` が自分 (あるいは `solve_sub` という呼び出し可能オブジェクト) を
closer に**渡す**。

```python
# closers/protocol.py
class Closer(Protocol):
    name: str
    def try_close(self, chart: Chart, *, solve_sub: SubSolver) -> ClosedOrNone:
        """閉じられたら (λ, m, 証明書) を返す。閉じられなければ None。
        **証明書を出せない限り proved を返してはならない。**"""
```

これは `SPEEDUP_SURVEY.ja.md` §H2 で提案した「各段を契約つきの closer にする」
のと同じ継ぎ目であり、別セッションの `strata_closer.py` も実質この形。
**2 つの独立した方向から同じ継ぎ目が出てきているので、ここで固めるのが妥当。**

**(b) `invariants → certify`。** `singularity_report` が `rlct_certified` を
呼んでいるだけなので、報告を `io/report.py` に上げれば `invariants` は
下の層に落ちる。

---

## 4. 移行の順序 — 各段で 203 本のガードを緑に保つ

**全段を通じて値 (λ, m) が変わらないことがゲート。**ガードは既に
`PYTHONPATH=.:bench python bench/guards.py` で 203/203 緑。

| 段 | 内容 | 壊れるリスク | 確認 |
|---|---|---|---|
| 0 | **互換シムを置く**: 旧モジュール名を新実装の re-export にする (`certify.py` が `core/certify.py` と `core/newton.py` から再公開) | 低 | 旧 import が全部通る |
| 1 | `core/newton.py` を切り出す (循環 4 辺が消える) | 低 | 203/203 + 循環検査 |
| 2 | `core/chart.py` に型を集める (`lean_export` の辺が消える) | 低 | 203/203 |
| 3 | `vandermonde` の**閉形式**を `closers/` に昇格 (ケースは bench に残す) | 低 | 本体→bench の依存が 0 になる |
| 4 | `closers/protocol.py` を定義し、既存 5 本を順に適合させる | **中** | closer ごとに 203/203 |
| 5 | `invariants` の報告を `io/report.py` へ | 低 | 203/203 |
| 6 | `models/` `exp/` にファイルを移動 | 低 | CLI を手で 1 回ずつ叩く |
| 7 | `cli.py` に入口を集約、各ファイルの `__main__` をデモ扱いに格下げ | 低 | — |
| 8 | **シムを外す** | 低 | 203/203 |
| — | (別途) `resolve_singularities` 656 行を closer の列に分解 | **高** | §H2 の約束 3 つを守る |

**段 0 のシムが要点。**別セッションの未 push パッチ (`strata_closer.py`、
ベース 359a038) と、外部のノートブックや手元のスクリプトを壊さずに進められる。
シムを外すのは最後 (段 8) で、それまで旧名は全部生きている。

**段 4 だけがリスク中。**closer の契約を変えると健全性の向きを取り違える
余地があるので、1 本ずつ適合させて都度ガードを回す。段 8 以降の
`resolve_singularities` の分解はリスク高なので、この整理とは**別の作業**として
切り離すこと (混ぜると、どちらが値を変えたのか分からなくなる)。

---

## 5. 同時に入れる価値のある小さな仕掛け

1. **循環を禁止するガード。** 提案後のグラフが DAG であることを
   `bench/guards.py` に 1 本足す (`ast` で import を読んで強連結成分を取るだけ。
   §3.2 の検算スクリプトがそのまま使える)。これが無いと、関数内 import で
   また静かに循環が戻る。
2. **トップレベル import を正典にする。** 段 1〜3 で循環が消えたら、
   関数内 import を順にトップレベルへ戻す。戻せないものが残ったら、
   それは設計がまだ間違っているという信号。
3. **`__init__.py` で公開 API を宣言する。** 現状 `__all__` があるのは
   一部だけ。層ごとに公開面を決めると、closer を足すときに触る範囲が確定する。

---

## 6. やらないほうがよいこと

* **ファイル数を減らすこと自体を目的にしない。** 36 本という数は
  19,479 行に対して多すぎない (平均 541 行)。問題は数ではなく依存の向き。
* **`bench/guards.py` (1,943 行) を 1 本のまま放置しない**が、
  分割は**群ごと** (negative, mutation, independent, …) にする。
  19 モジュールを import しているのは性質上そうなるので、そこは許容する。
* **整理と高速化・機能追加を同じコミットに混ぜない。**
  値が変わったときに原因が特定できなくなる。整理のコミットは
  「λ と m が 1 件も変わらない」を不変条件にする。
