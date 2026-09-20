# 特異点解消による実対数閾値 (RLCT) の計算

多項式の特異点を解消して実対数閾値 λ とその位数 m を求め、**結果が正しいことの
証明書まで付ける**ツール群です。特異学習理論 (SLT) の学習係数の計算・検証を
想定しています。

---

## 1. 全体像

```
                    f (多項式) と変数
                          |
        +-----------------+------------------+
        |                                    |
  families.py                         invariants.py
  族ごとの閉じた式                    m, 擬斉次性, mu, tau
  (単項式/積/和/形式のべき/非退化)     (トリアージと検算)
        |                                    |
        | 当たれば厳密値                      | 重みの選択則へ
        v                                    v
  certify.rlct_certified  <----  resolve_singularity.py
        |                        ブローアップ + 正規化の反復
        |                        (重み付き / 分枝限定 / Graphviz)
        v                                    |
  certify.py                                 |
  三値の証明書 (proved/unknown/refuted)  <----+
  SMT で正規交差と被覆を検証
        |
        +--> lean_export.py   Lean 4 の証明義務を生成
        +--> rlct_interval    厳密値が出ないとき区間 [lo, hi]

  応用:  nn_rlct.py (一般のモデル/ReLU)  torch_rlct.py (PyTorch)
  検証:  bench/ (ケース・オラクル・ガード・失敗の蓄積)
```

### 依存関係

| パッケージ | 必須 | 用途 |
|---|---|---|
| `sympy` | **必須** | 多項式演算全般 |
| `scipy` | 推奨 | 探索ループ内の高速な LP (HiGHS) |
| `pulp` | 推奨 | LP/ILP (CBC などの外部ソルバー) |
| `z3-solver` | 推奨 | 正規交差性・非退化性の判定 (これが無いと証明できない) |
| `graphviz` (と `dot`) | 任意 | chart 木の描画 |
| `torch` | 任意 | PyTorch モデルの読み込み |
| Singular | 任意 | 局所順序での Milnor/Tjurina 数 |

```sh
pip install sympy scipy pulp z3-solver graphviz
apt-get install singular graphviz     # 任意
```

---

## 2. 使い方 (最短)

```python
import sympy as sp
from certify import rlct_certified, rlct_interval

x, y, z = sp.symbols("x y z", real=True)

lam, status, route = rlct_certified(x**2 + y**3, (x, y))
# -> (5/6, 'proved', 'newton')

iv = rlct_interval(f, gens)      # 厳密値が出ないときは区間を返す
iv.status, iv.lo, iv.hi
```

各モジュールは単体で実行するとデモが走ります。

```sh
python resolve_singularity.py    # 解消・分枝限定・Graphviz
python certify.py                # 証明書と非退化判定
python families.py               # 族ごとの閉じた式
python invariants.py             # m, mu, tau
python lean_export.py            # Lean 4 の出力
python nn_rlct.py                # ReLU ネットの局所 RLCT
python torch_rlct.py             # PyTorch モデル
```

---

## 3. 理論的な枠組み

ゼータ関数

    zeta(z) = ∫ |f(x)|^z φ(x) dx

に対し、特異点解消定理により固有写像 φ が存在して局所座標で

    f(φ(y)) = y^k · U(y) · u(y),      |det Dφ(y)| = |v(y)| · y^h

(U, u, v は原点で非零) と書けます。このとき

    λ = min_chart min_{j : k_j > 0} (h_j + 1) / k_j
    m = 最小値を達成する座標の個数 (chart 間では最大)

コードはこの `(k, h, u, U, φ)` の四つ組を**厳密に**追跡します。`U` は途中で
落とした単元を溜めるためのもので、これを追跡しないと恒等式が成り立ちません
(実際にそのバグがありました)。

### λ が正しいために必要な条件

| | 条件 | 実装での扱い |
|---|---|---|
| (a) | f∘φ = y^k·U·u の恒等式 | `certify` が sympy で厳密検算、Lean の `chart_*` が `ring` |
| (b) | det Dφ = y^h·v の恒等式 | `jac_*` (n≤3)。成分が偏微分であることは未証明 |
| (c) | chart 族が近傍を覆う (不足なし) | Lean の `cover_*` + `blowup_cover` |
| (d) | 像が元の箱に収まる (過剰なし) | Lean の `image_*` + `blowup_image` |
| (e) | u·v が定義域全体で正規交差 | `certify` が Z3 で判定 |
| (f) | φ が測度零を除いて微分同相 | 座標変換は `inv_*` で確認、未形式化の部分あり |
| (g) | 台の関数 ψ ≥ 0, ψ(0) > 0 | 前提 |
| (h) | 極が min (h+1)/k にある | **Watanabe の定理。前提として置く** |

`certify` が `proved` を返すのは (a)(c)(d)(e) が確認できたときです。(h) は
形式化していません。

---

## 4. モジュール別の解説

### 4.1 `resolve_singularity.py` — 解消の本体

```python
res = resolve_singularities(f, gens, prune=False, weighted=True, max_depth=20)
res.rlct, res.multiplicity, res.charts, res.nodes
res.print_report()
res.render_tree("tree", "png")    # chart の場合分けを Graphviz で
res.to_lean("lean_out")
```

#### アルゴリズム

各 chart で `f = x^k · U · f_rest`、`|Jacobian| = 単元 · x^h` を保ちながら:

1. **正規化** — 単項式の最大公約因子を括り出し、原点で消えない因子 (単元) を
   除去して `U` に溜める
2. **停止判定** — `f_rest(0) ≠ 0` なら正規交差型になっており解消済み
3. **滑らかな因子の座標化** — `f_rest` が原点で滑らかな因子 g をもち、g が
   ある変数について 1 次で、その変数が例外因子に使われていないとき、g 自身を
   座標に取る。`(x-y)^2` のように特異点集合が正の次元をもつ場合の無限反復を防ぐ
4. **ブローアップ** — 中心 `{x_j = 0 : j ∈ J}`。J は「f_rest の全単項式が J の
   変数を少なくとも 1 つ含む」最小の集合 (最小ヒッティング集合、分枝限定 or ILP)
5. **中心の付け替え** — 例外因子上の原点以外の非正規交差点へ平行移動
6. 停止しない場合は `ResolutionFailure` (理由: `max_depth` / `no-progress` /
   `max_charts` / `prune-inconsistent`)

単項式の置換は指数ベクトルの線形変換なので、多項式展開なしで実行できます。
20 変数の `Σx_i²` が 1 秒以下で終わります。

#### 重み付きブローアップ (`weighted=True`)

    x_i = s·y_i^{w_i},   x_j = y_i^{w_j}·y_j   (j ∈ J, j ≠ i)

指数は `e_i' = Σ_{j∈J} w_j e_j`、ヤコビアンは `w_i · y_i^{(Σw_j) − 1}`。
重みは**擬斉次な場合にのみ**使います (連立一次方程式で厳密に判定)。
擬斉次でない場合に使うと反復が増えて止まらなくなるためです。

`w_i` が偶数の chart は `y_i ↦ y_i^{w_i}` が実数体で負の側を覆えないので、
符号 `s = ±1` の 2 chart に分けます。

これにより `x³+y⁴+z⁵` が 5 chart・0.07 秒で解けます (通常のブローアップでは
何回反復しても止まりません)。

#### 分枝限定 (`prune=True` / `'ties'`)

λ は全 chart の最小値なので、最小化問題として枝刈りできます。

- **上界 (primal)**: ニュートン多面体の LP `max Σν_a s.t. Σν_a·a_j ≤ h_j+1`
  — どの節点で計算しても大域最小値の上界
- **下界 (dual)**: メディアント不等式
  `Σ(h_j+1)/(Σk_j + a) ≥ min_j (h_j+1)/(k_j + a)` から
  `λ(部分木) ≥ min_j (h_j+1)/(k_j + M)`
- LB > UB なら打ち切り。厳密な不等号なので λ も m も保存される
- `'ties'` は到達済みの値と同値の部分木も刈る (λ は正しいが m は下限になりうる)

**下界の予算 M は「残り max_depth 回で解消しきる」ことを前提にしている**ので、
証明モードでは `prune=False` を使ってください。矛盾を検出した場合は
`ResolutionFailure(reason='prune-inconsistent')` を送出します。

#### Graphviz

`res.render_tree()` で chart の場合分けを木に描きます。各ノードに
ブローアップの中心と chart、変換直後の式、括り出した単項式と正規化後の式、
指数ベクトル `k, h`、解消済みなら λ と m が入ります。色は
解消済み=緑 / 枝刈り=灰 / 未解消=赤 / 座標変換=青 / 付け替え=橙。
`ResolutionFailure` にも途中までの木が入ります。

### 4.2 `certify.py` — 三値の証明書

```python
cert = certify(res, eps=1, timeout_ms=10000)
cert.status    # 'proved' | 'unknown' | 'refuted'
cert.rlct      # proved のときだけ値、そうでなければ None
cert.print_report()
```

#### 正規交差の判定 (要点)

`u` が消えること自体は問題ではありません。零点が滑らかで例外因子と横断的なら
正規交差のままです。壊れるのは `grad u(p)` が
`{e_j : y_j(p) = 0, j ∈ supp(k)∪supp(h)}` の張る空間に入る点なので、
量化子なしで書けます。

    u = 0 ∧ (j ∉ E) ∂u/∂y_j = 0 ∧ (j ∈ E) [y_j = 0 ∨ ∂u/∂y_j = 0]

これが Z3 で UNSAT なら、箱全体で正規交差であることの**証明**になります。

さらに二つの局所化が本質的です。

- **ファイバーへの限定** (`localize=True`): 台の関数 ψ は原点近傍にしか台を
  持たないので、`φ(p) = 0` を満たす点だけ見ればよい。正規交差性は開条件で
  ファイバー∩箱はコンパクトなので、ファイバー上で成り立てば近傍でも成り立つ
- **付け替え chart の担当領域**: 付け替えた chart は δ 球だけを担当し、親は
  その球を除いた領域を担当する

#### 定義域の追跡

根を `{|x_j| ≤ ε}` とし、被覆補題が与える境界で伝播します。

| ステップ | 子の箱 |
|---|---|
| ブローアップ chart i | `b'_i = b_i`, `b'_j = 1` (j ∈ J\{i}), 他は継承 |
| 座標変換 `x_v = (y_v−B)/A` | `b'_v = |A|max·b_v + |B|max` (区間演算) |
| 中心の付け替え | 全方向 `δ` |

#### 位数 m のニュートン多面体からの決定

```python
newton_multiplicity(f, gens, h=None)
```

対角線が境界に当たる点 `p = t*(1,...,1)` を含む最小の面の次元 d に対し
`m = n - d`。これは `m = rank{p を通るファセットの法線}` と書けます。
**トーリック解消で扇を細分しなくても、得られるはずの位数が分かります。**

#### ニュートン非退化性

```python
newton_nondegenerate(f, gens)   # ('proved'|'refuted'|'unknown', 面)
rlct_via_newton(f, gens)        # (λ, status)
```

各コンパクト面 σ について `∃x (全 x_i ≠ 0) ∧ ∇f_σ(x) = 0` が UNSAT かを見ます
(Euler の関係式から `f_σ = 0` も従うので勾配だけで足りる)。
**実数体上で直接判定している**点が重要で、複素で退化していても実で非退化という
ことがあり、RLCT に必要なのは後者です。非退化なら Varchenko の定理で
λ = 1/(ニュートン距離) が厳密になります。

#### 厳密値が出ないとき

```python
iv = rlct_interval(f, gens)     # 健全な区間 [lo, hi]
```

- 下界: `1/m` (`1/m ≤ lct_ℂ ≤ λ_ℝ`)
- 上界: `n/m`、ニュートン LP、途中まで解消できた chart の値の最小、
  枝刈りありの解消の到達値 (いずれも「ある chart の値」なので上界)

区間が一点に潰れれば値は確定します。潰れない場合は SGLD などの数値推定に
切り替え、推定値がこの区間に入るかで推定のバグと真の難しさを切り分けます。

### 4.2.5 `ideal_resolve.py` — イデアルを運ぶ解消 (既定で併用)

```python
rlct_certified(f, gens, ideal=True, generators=[g1, g2, ...])   # 既定 on
rlct_interval(f, gens, ideal=True)
resolve_ideal(generators, gens)                                  # 直接
```

多項式 `sum g_i^2` ではなく **生成元の組** を運ぶ。各節点で

1. 定数係数による線形簡約 (イデアル不変)
2. 共通単項式の括り出し (f = sum g^2 なので k には 2c)
3. **生成元を座標に取る** — 原点で `dg != 0` の生成元があれば、その変数
   (k=h=0 のもの) を g に取り替える。多項式の *因子* しか見ない
   `_smooth_factor_change` では扱えない「和の項」を使える
4. 停止判定: 全生成元が単項式 x 単元 -> 単項式イデアルなので Howald の
   LP で **厳密に** 閉じる (従来の「単一の単項式まで掘る」より早く終わる)
5. ブローアップ

注意: 「lambda += 1/2 して変数を消す」分離は k = h = 0 のときしか使えない
(f = x^k(v^2 + ...) は v について分離しない)。ここでは分離せず座標変換
だけを行い、v を生成元として残す。

効果 (Vandermonde 型、真値は Aoyagi の公式):

| ケース | 真値 | イデアル版 | 多項式版 |
|---|---|---|---|
| M=1 H=2 Q=1 | 3/4 | **3/4** | 1 |
| M=1 H=2 Q=2 | 2/3 | **2/3** | 解消できず |
| M=2 H=2 Q=1 | 1 | **1** | 1 |
| M=1 H=3 Q=1 | 1 | 解消できず | 3/2 (誤り) |

**未実装**: phi を追跡していないので `certify` / `lean_export` に渡せない。
そのためイデアル版の値を採用した場合の status は必ず `unknown` になる
(誤った値を proved と主張しないため)。`ideal=False` で従来どおり。

### 4.3 `families.py` — 族ごとの専用経路

上から順に試し、当たれば閉じた式で λ と m を確定させます。

| 経路 | 条件 | 値 |
|---|---|---|
| monomial | `f = c·x^k` | `λ = min 1/k_j`, `m` = 達成数 |
| product | 変数が互いに素な積 | `λ = min λ_i`, `m` = 最小を達成する `m_i` の和 |
| sum | 変数が互いに素な和、各項が非負 | `λ = Σλ_i`, `m = Σm_i − (成分数−1)` |
| power_form | `f = g^p`、g が 1 次/2 次形式 | 1 次: `1/p`。2 次で階数 r: 定値 `r/2p`、不定値 `min(1/p, r/2p)` |
| newton | 実数体上で非退化 | `1/(ニュートン距離)`、`m` = 対角線が当たる点を通るファセット法線の階数 |

積の規則に符号条件は不要 (`|f₁f₂| = |f₁||f₂|`)、和の規則には非負性が要ります。

これらは**証明経路であると同時にオラクル**で、汎用経路の値と突き合わせます。

### 4.4 `invariants.py` — 局所不変量

```python
multiplicity(f, gens)                  # ord_0 f
quasihomogeneous_weights(f, gens)      # (重み w, 次数 d) or None
milnor_number(f, gens, backend="auto") # (mu, 経路)
tjurina_number(f, gens)                # (tau, 経路)
singularity_report(f, gens).print_report()
```

- 擬斉次判定は**連立一次方程式で厳密に**行います (LP 近似ではない)。
  これが `resolve_singularity` の重み選択則の根拠です
- μ, τ は Singular があれば**局所順序 ds** で計算し、原点だけの値を返します。
  無ければ sympy のグレブナー基底で大域的な `Σ_p μ_p` になります
  (例: `x⁵+y⁵+x²y²` は局所 μ=11、大域 16)
- 孤立でない場合は μ = τ = `oo`。μ=0 は「原点で滑らか」

**注意**: μ と τ は代数閉体上の不変量です。実数上の孤立性判定では不十分で、
`(x²+y²)²` は実点では原点しか特異でないのに複素では非孤立なので μ = ∞ です。
実 RLCT の検算に使えるのは μ ではなく `1/m ≤ λ ≤ n/m` のほうです。

### 4.5 `lean_export.py` — Lean 4 の証明義務

```python
exp = res.to_lean("lean_out", certificate=cert)
exp.print_summary()
exp.verify()        # lake/lean があればコンパイル
```

出力されるファイル: `LeanCover.lean` (手書きの一般補題)、`Resolution.lean`
(生成物)、`lakefile.lean`、`README.md`。

| 定理 | 主張 | 戦術 |
|---|---|---|
| `chart_*` | `f∘φ = y^k · 単元 · f_rest` | `ring` |
| `unit_*` | `f_rest(0) ≠ 0` | `norm_num` |
| `jac_*` | ヤコビ行列式 `= y^h · 単元` (n≤3) | `Matrix.det_fin_*` + `ring` |
| `inv_*` | 座標変換が全単射 (逆写像を明示) | `ring` / `field_simp` |
| `cover_*` | chart 族が全射 (不足なし) | `blowup_cover` |
| `image_*` | 像が箱に収まる (過剰なし) | `blowup_image` |

**出力前にすべての等式を sympy で検算する**ので、Lean が無くても帳簿の整合性
チェックとして機能します (デモで k をわざとずらすと NG が出ます)。

**未検証**: この環境には Lean が無いため、生成した Lean は一度もコンパイルして
いません。特に手書きの `blowup_cover` は Mathlib の補題名に依存するので、
初回は手直しが要る可能性があります。重み付きブローアップの被覆補題は未形式化で、
該当ノードは出力を省略して記録します。

### 4.6 `nn_rlct.py` — 一般のモデル / ReLU ネット

```python
local_rlct_from_ideal(generators, variables, symmetry_vectors=[...])
relu_local_rlct(X, A_star, b_star, c_star, d_star)
```

三段階:

1. **セルの固定** — θ\* での前活性化の符号でモデルを多項式化。θ\* がセル内部なら
   厳密。境界なら接するセルを全列挙して最小を取る (錐への制限はしていないので
   下界)
2. **商** — 生成元の線形簡約 → スケール対称性のゲージ固定 (自由な作用なら λ も m
   も不変で次元だけ落ちる) → 正則方向の消去 (`λ += 1/2` して変数を 1 つ落とす)
   → 自由方向の除去
3. **コアイデアルの解消**

観察: θ\* がセル内部にあるとコアが空 (=正則) になることが多く、**ReLU の本質的な
特異性はセルの境界に集中**しています。

### 4.7 `torch_rlct.py` — PyTorch の読み込み

```python
rep = torch_local_rlct(net, X, params=["2.weight"], zero_tol=1e-2)
rep.print_report()
```

- 直列構造を自前で辿り、各パラメータを `θ*+u` または定数として記号展開
- 対応レイヤ: Linear / ReLU / LeakyReLU / Identity / Flatten、および Tanh /
  Sigmoid / SiLU / Softplus / GELU (前活性化まわりのテイラー展開)
- スケール対称性を自動抽出。ただし**関係するパラメータを一部でも凍結していると
  それは対称性ではない**ので商にしません
- `zero_tol` で小さい重みを 0 に丸めます。丸めないとほぼ必ず正則点になります
- `targets` は θ\* が realizable な点かの検査にだけ使い、fiber ideal は常に
  「θ\* のモデルが真」として作ります

### 4.8 `bench/` — 逐次改善の土台

```sh
cd bench && export PYTHONPATH=..:.
python guards.py                          # 判定器が甘くなっていないか (採用条件)
python run.py --max-vars 4 --random --out r.json
python report.py r.json
python run.py --failures                  # 蓄積した失敗例を再測
python run.py --hard --montecarlo --covering   # 難しい族 + 独立な検査
python regression.py --all                     # 全スイート一括 (CI 用)
```

| ファイル | 役割 |
|---|---|
| `cases.py` | 構造化された族とランダム族 (n, d, 項数, 係数ビット長を独立に振る) |
| `oracles.py` | 族の閉じた式、設定違いの突き合わせ、`1/m ≤ λ ≤ n/m` の検査 |
| `guards.py` | 負のテスト・変異検査・独立検査・Aoyagi 補題・Vandermonde 真値・**座標不変性**、および既知の未修正バグ (ゲート外で報告) |
| `run.py` | 実行と失敗の分類、JSON 出力 |
| `report.py` | 集計。クラス分布、経路別、n/次数ごとの proved 率、μ との相関 |
| `failures.py` | 失敗例の蓄積と自動縮約 |
| `nn_cases.py` | ニューラルネットの格子 (MLP x {relu, tanh} x 退化の型)。torch 非依存 |
| `regression.py` | 全スイートの一括実行 (guards / structured / random / hard / nn / failures) |
| `independent.py` | 解消の結果を使わない独立な検査 (体積の漸近による λ の数値推定、被覆の前向き標本検証) |
| `known_families.py` | 真値が独立に分かり、かつ難しい族 (一般の位置の超平面配置、斉次形式、Vandermonde 型) |
| `aoyagi_lemmas.py` | Aoyagi (Entropy 2019) の補題を真値を使わない検査にしたもの (イデアル不変性・単調性・分離・最深点) |
| `vandermonde.py` | Vandermonde 行列型の真値。Aoyagi (2019) Theorem 6 (H<=3) と N=1 の厳密式 (lambda と位数 theta) |

#### 失敗の分類とパッチの当て先

| クラス | 意味 | 当てる場所 |
|---|---|---|
| `resolve_max_depth` | 解消が止まらない | 中心の選択則、重み |
| `smt_unknown` | 判定が決まらない | 面の分解、ファイバー制約、timeout |
| `nc_refuted` | 正規交差が壊れる点あり | localize、定義域の切り分け |
| `identity_refuted` | 恒等式が不成立 | 解消側の帳簿 |
| `covering_unknown` | 被覆が確立しない | 補題の追加 |
| `inconsistent` | 独立な値と食い違う | **最優先で調査** |

#### 運用ルール

proved 率を上げる作業は、判定を正しく直すことでも**単に甘くすること**でも
達成できてしまいます。両者を区別するために:

1. パッチを当てる前に、必ず最小の失敗例に縮約する
2. `guards.py` が全件緑でなければ採用しない
3. 主指標は proved 率ではなく「**既知の値と一致した** proved の数」
4. 失敗は保存・自動縮約し、修正後に `run.py --failures` で再測する

---

## 5. 現在の到達点

| 集合 | proved |
|---|---|
| 構造化 51 件 (n ≤ 6, d ≤ 13) | 51/51 |
| ランダム 111 件 (n ≤ 3, d ≤ 6) | 111/111 |
| ランダム 36 件 (n = 4–5, d = 4–8) | 22/36 (残りは全て `resolve_max_depth`) |
| guards 38 件 (負のテスト・変異・独立検査・Aoyagi 補題・Vandermonde 真値) | 38/38 緑 |
| Vandermonde 型で値が真値と異なる例 | 1 件 (M=N=1,H=3,Q=1: 3/2 対 真値 1)。証明書は refuted |
| guards 17 件 | 17/17 緑 |
| 独立な値との食い違い | 0 件 |

n=4,5 の失敗は**全て解消側**で、証明側 (SMT・被覆・恒等式) の失敗はゼロです。
`max_depth` を 30 に上げても止まらないので、深さ不足ではなく中心の選び方の限界です。

---

## 6. 開発中に見つかった主なバグ

このコードは何度も誤った λ を返しており、そのたびに検査を足してきました。

| バグ | 症状 | 修正 |
|---|---|---|
| 例外因子上の非正規交差点の見落とし | `(x−y)²` が λ=1 (真値 1/2) | 中心の付け替えを追加 |
| 付け替えが過剰 | 原点に写らない点まで候補にしていた | `φ(p)=0` の確認を追加 |
| 単元を帳簿から捨てていた | 恒等式が不成立 | `Chart.unit` で追跡 |
| certify が恒等式を検算していなかった | k を壊しても proved | 恒等式検査を追加 |
| 枝刈りの予算が max_depth 依存 | λ=∞ を静かに返す | 上界との矛盾検出 |
| 解消済み+付け替えの chart を捨てていた | λ を過大評価しうる | `done` に追加 |
| 実数上の孤立性で μ の公式を使った | `(x²+y²)²` で μ=9 (真値 ∞) | 複素のゼロ次元性で判定 |
| CBC の精度 | 全 chart を刈って λ=∞ | 上界に相対 1e-6 の余裕 |

「間違いが間違いに見えない」のがこの問題の本質的な難しさです。対策は、
既知の閉じた式との照合、同じ量を別経路で二度出すこと、不変量の内部検査、
そして変異検査の四つです。

---

## 6.5 未修正の既知バグ: 座標依存

同じ関数を、ヤコビアン 1 の可逆な座標変換で書き換えると答えが変わります。
Vandermonde 型 (M=N=1, H=3, Q=1) の第 1 回ブローアップ chart の局所問題

    P = v^4 (G_1^2 + v^2 G_2^2 + v^4 G_3^2),  G_k = a_1 + a_2 b_2^k + a_3 b_3^k

に対し、chart 座標のままだと lambda = 3/2、G_1 を座標に取ってからだと 7/6。

原因は、解消器が **多項式 sum g_i^2 を運んでいて、イデアル <g_i> を運んで
いない**ことです。lambda はイデアルだけで決まる (Aoyagi Lemma 1(2)) のに、
多項式に潰した時点で「生成元を座標に取る」操作ができなくなります。
`_smooth_factor_change` は f_rest の *因子* しか見ないので、
G_1 のような *和の項* は座標化できません。

修正の方向は、`Chart` に生成元の組を持たせ、各節点で
(1) 定数係数による線形簡約 (2) **正則方向の消去** (3) 単元因子の除去
(4) ブローアップ の順に試すこと。(2) は現在 `nn_rlct` の前処理として
根で一度だけ行われており、節点ごとに繰り返す必要があります。

`guards.py` の `known_issues()` がこの不一致を毎回報告します
(緑のゲートには数えません)。修正されたら OK になるので、その時点で
通常のガードに昇格させてください。

## 7. 既知の限界

- **(h) の未形式化**: 極が `min (h_j+1)/k_j` にあるという Watanabe の定理は前提
- **Lean は未コンパイル**: この環境に Lean が無いため、生成物の検証は未実施
- **重み付きの被覆補題が未形式化**: 該当ノードの `cover_*` は出力されない
- **`x³+y⁴+z⁵` 型の一部**: 重み付きで解決したが、n=4,5 の非擬斉次な例では
  依然として止まらない。正しい一般化はトーリック解消 (双対扇の正則細分)
- **自動縮約が大きい例に力不足**: 判定 1 回のコストが大きく、シグナルによる
  時間制限が C レベルの呼び出しで遅延する
- **被覆の完全性を検査する独立な手段がない**: 「chart を 1 個落とす」変異は、
  落とした chart が最小値を与えていないと検出できない

---

## 8. 参考文献 (抜粋)

- Watanabe, *Algebraic Geometry and Statistical Learning Theory*, Cambridge, 2009
- Lin, *Algebraic Methods for Evaluating Integrals in Bayesian Statistics*, PhD thesis, 2011
- Howald, "Multiplier ideals of monomial ideals", TAMS 353 (2001)
- Varchenko (1976);Arnold–Gusein-Zade–Varchenko
- Aoyagi & Watanabe, IEICE Trans. 88(10), 2005 (Vandermonde 行列型特異点)
- Abramovich, Temkin, Włodarczyk, "Functorial embedded resolution via weighted blowings up" (arXiv:1906.07106)
- Bierstone & Milman, Invent. Math. 128 (1997);Encinas–Villamayor, Acta Math. 181 (1998)
- Blanco & Frühbis-Krüger (解消アルゴリズムの実装比較)
- Lau, Furman, Wang, Murfet, Wei, "The Local Learning Coefficient" (arXiv:2308.12108)
- Şimşek et al., "Geometry of the Loss Landscape in Overparameterized Neural Networks", ICML 2021
- Grigsby, Lindsey, Rolnick, "Hidden symmetries of ReLU networks", ICML 2023
