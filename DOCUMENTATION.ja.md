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

#### chart ごとのニュートン高速パス (`newton_fast=True`, 既定)

chart の局所ゼータは

    Z(z) = ∫ |y^k · f_rest(y)|^z · |y^h| dy      (chart の原点近傍)

なので、`P = y^k · f_rest` が原点で**実数体上ニュートン多面体に関して
非退化**であれば、Varchenko の定理から λ は LP (振幅付きニュートン距離) で
厳密に決まります。つまり **f_rest が単元になるまでブローアップを続ける必要が
ありません**。各 chart で

1. 凸包からコンパクト面を列挙
2. 各面 σ について `∃y, (∀i) y_i ≠ 0 ∧ ∇P_σ(y) = 0` が UNSAT かを Z3 で判定
3. すべて UNSAT なら `newton_rlct(P, h=h)` と `newton_multiplicity` で
   (λ, m) を確定し、その chart を閉じる (`status='resolved(newton)'`)

を行います。非退化を**証明できたときだけ**採用するので真値は変わりません
(`bench/guards.py` の「Newton 高速パス不変」10 件が on/off の一致を常時監視)。

中心の付け替え候補がある chart では使いません。高速パスが確定するのは
chart の**原点**における値で、例外因子上の他の点は付け替え chart が
担当するからです (単元で閉じる場合とまったく同じ扱い)。

`newton_fast_max_terms` (既定 60)、`newton_fast_max_vars` (既定 24) を
超える chart では、凸包と Z3 のコストが見合わないので判定しません。
`newton_fast=False` で従来どおり単元まで解消します。

`certify()` はこの葉を単元条件ではなく「非退化の再判定 + LP と位数の再計算」で
検査します (解消側のキャッシュを信用しない)。食い違えば `refuted`。
Varchenko の定理自体は (h) Watanabe の定理と同様、前提として仮定します
(Lean 側は `newton_*` として `sorry` 相当で出力)。

計測 (ランダム多項式 54 本、n=2..4, d=3..5):

| | 合計時間 | 中央値 | chart 合計 | 解消成功 |
|---|---|---|---|---|
| `newton_fast=False` | 14.7s | 0.062s | 302 | 46/54 |
| `newton_fast=True` | 6.0s | 0.019s | 146 | 49/54 |

#### 積の分解 (`product_split=True`, 既定)

chart の残差 `f_rest` が**変数の互いに素な因子**の積 `A(u)·B(v)` に分かれるとき、
局所データの積分が完全に分離します。

    ζ(z) = ∫|u^{k_u}A|^z u^{h_u} du · ∫|v^{k_v}B|^z v^{h_v} dv

したがって `λ = min(λ_A, λ_B)`、位数は最小を与える側の和。**`k, h` が何であっても
成り立つ**ので、木のどのノードでも使えます。どの因子にも現れない変数 `y_j` は
局所データが単項式 `y_j^{k_j}` だけなので、`(h_j+1)/k_j` を持つ 1 変数の部分問題
として同じ規則で合流させます。

部分問題はそれぞれ `proved` になることを確認したうえでのみ採用し、`certify()` は
分解と合流を計算し直して検査します（食い違えば `refuted`）。

`(x²+y²)(z²+w³)`: 6 chart・0.22 秒 → **1 chart・0.07 秒**（λ=5/6, m=1 で一致）。

#### 大域下界による早期終了 (`lower_bound=`)

λ の厳密な**下界**が分かっているとき渡すと、到達値がその下界に達した時点で
残りの chart をすべて打ち切ります。下界は Aoyagi Lemma 1(1) から作れます
(`lemmas.py`)。λ は厳密なままですが、打ち切った時点で**位数 m は下限**に
なりうるので、その旨が `warnings` に入ります。

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


#### 主面がコンパクトであること (非退化だけでは足りない)

Varchenko の公式 `λ = 1/(ニュートン距離)` には、非退化性のほかに
**主面 (ニュートン距離を与える点 p = t·(h+1) を含む Γ₊ の最小の面) が
コンパクトである**という条件が要ります。

> `f = −9x₀ − 38x₁²x₂` は原点で勾配が非零（∇f(0) = (−9,0,0)）なので
> `u = f` と座標を取れば **λ = 1, m = 1** が真値です。
> コンパクトな面は辺 `{(1,0,0), (0,2,1)}` だけで、その上では `∂f/∂x₀ = −9 ≠ 0`
> なので**非退化**。それでも LP は **3/2** を返します。
> `p = (2/3, 2/3, 2/3)` は `conv(台)` の外にあり、主面は x₂ 方向に伸びる
> 非コンパクトな面だからです。

コンパクトな面は `conv(台)` に含まれるので、判定は **`p ∈ conv(台)`**
（凸結合が存在するかの LP）で済みます。`certify.principal_face_compact()` が
これを行い、`rlct_via_newton` / `families._route_newton` /
`resolve_singularity._chart_newton_close` の 3 か所すべてで
「非退化 **かつ** 主面がコンパクト」のときだけ proved にします。

この条件を入れると、158 件のベンチで **2 件の proved が消えました**。
どちらも `family:newton` 経由で主面が非コンパクトだった、つまり
**証明になっていなかった**値です（能力ではなく誤りを失いました）。

#### 台が薄いときの面の列挙

凸包からコンパクトなファセットを拾う方法は、台が薄いと 1 枚も見つかりません。
`f = 53x₀x₂³ + 61x₁³x₂` を 4 変数で渡すと、`x₃` 方向に Γ₊ が筒になっていて
法線が空になり、非退化判定が `unknown` に落ちて解消が無限反復していました
（ベンチの `resolve_max_depth` 9 件のうち複数がこれ）。

対策は 2 つです。

1. **現れない変数を落として計算する**（法線はその成分を 0 にして戻す）。
2. それでもファセットが取れないときは、**部分集合を直接 LP で篩う**:
   σ がコンパクトな面 ⟺ `∃w, d: ⟨w,m⟩ = d (m∈σ), ⟨w,m⟩ ≥ d+1 (m∉σ), w ≥ 1`。
   成分ごとの順序で支配される単項式は `w > 0` では決して argmin に入らないので
   先に落とし、残りの反鎖が 12 個以下のときだけ 2^|·| 回の LP を回します。

速い凸包の方を先に試し、失敗したときだけ LP に落とすようにしてあります
（LP を常用すると chart ごとに数千回の LP が走り、解消が数分単位で遅くなりました）。

#### コンパクトな面の列挙 (ファセットだけでは足りない)

Varchenko の非退化条件は Γ₊(f) の**コンパクトな面すべて**についての条件で、
ファセットだけを見るのでは足りません。

> `f = (x + y²)² + z² = x² + 2xy² + y⁴ + z²`
> コンパクトなファセットは 1 枚 (法線 (2,1,2)) だけで、その上では
> `∇f_σ = 0` が `z ≠ 0` と両立しないので「非退化」に見えます。
> しかし辺 `{(2,0,0), (1,2,0), (0,4,0)}` もコンパクトな面で、その面多項式
> `(x+y²)²` はトーラス上 `x = -y²` で勾配が消えます。真値は λ = 1 ですが、
> ファセットだけの判定では **5/4 を proved として返していました**。

正しい列挙は、多面体の面について `face(w₁ + w₂) = face(w₁) ∩ face(w₂)`
(交わりが空でないとき) が成り立つことを使います。Γ₊ の面は

    face(w) = ∩_{i∈S} face(n_i) ∩ ∩_{j∈T} face(e_j)

(`n_i` はコンパクトなファセット法線、`face(e_j) = {m : m_j が最小}`) の形で
尽くされ、`w > 0` になるのは **S が空でないとき**、すなわちコンパクトな面です。
したがって「コンパクトなファセットの面から始めて、他のファセット面・座標面との
交わりで閉じる」だけで全部得られます。面の数が上限を超えたら `None` を返して
`unknown` に倒します。

この誤りは `bench/guards.py::regular_elimination_tests()` — 正則方向の消去を
通した値と、もとの `K = Σg_i²` を直接解消した値の突き合わせ — で見つかりました。
`<x + y², z>` が直接 5/4 (proved) / 消去経由 1 と食い違ったのが端緒です。
修正後、既存の 158 件のベンチでは proved 率も値も**まったく変わりません**
(149/158)。失っていたのは正しさだけで、能力ではありませんでした。

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

#### 正則方向の消去を擬剰余で行う

`g = A·v + B` で `A(0) ≠ 0` のとき `v` を消去します。以前は
`h.subs(v, -B/A)` を `sp.cancel(sp.together(...))` で有理式として整理して
いましたが、`A` が多項式の場合 `sol = -B/A` は有理関数になり、係数が 28 桁まで
膨らんで `cancel` が支配的コストになっていました (tanh 1-2-1 の 19.7 秒のうち
`cancel` が 10.7 秒)。

いまは**擬剰余**を使います。`h = Σ_e c_e(others)·v^e` に対し

    A^d · h(-B/A) = Σ_e c_e (-B)^e A^{d-e}      (d = deg_v h)

を Horner 法で多項式のまま組み立てます。`A` は原点で単元なので `A^d` を掛けても
局所環でイデアルは変わらず、RLCT も変わりません。仕上げに整数 content を割ります
(非零の有理数倍なのでイデアル不変)。さらに候補を「`A` が定数」「`A, B` が小さい」
順に選びます。

実測 (tanh 1-2-1, 7 変数, 6 生成元): **19.7 秒 → 1.50 秒 (13 倍)**。
コア生成元も 139/137/136 項・次数 10 → 49/47/46 項・次数 7 と小さくなり、
その後の解消も速くなります (局所 RLCT 全体で 6.8 秒 → 2.9 秒)。

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

### 4.7.4 `lemmas.py` — Aoyagi の補題を「計算の短縮」に使う

`bench/aoyagi_lemmas.py` が同じ補題を*検査*に使うのに対し、こちらは*計算*に
使います。**使えるものと使えないものの区別が肝心**なので、根拠と反例を
モジュールの docstring に全部書いてあります。

**使える (1): Lemma 1(1) 単調性による挟み込み**

`Σ_{i∈S} g_i² ≤ Σ_i g_i²` が各点で成り立つので `λ(部分生成元) ≤ λ(全体)`。
つまり部分生成元の λ は全体の λ の**厳密な下界**です。一方 `newton_rlct` は
非退化かどうかに関係なく常に上界なので

    λ(部分生成元) ≤ λ ≤ newton_rlct(f)

で挟めます。両端が一致すれば**ブローアップなしで λ が確定**します。

```python
from lemmas import squeeze_rlct, monotone_lower_bound
sq = squeeze_rlct(f, gens, generators)
sq.status   # 'proved' なら sq.rlct が厳密
sq.lo, sq.hi
```

部分生成元は変数も次数も小さいので、族ごとの閉じた式 (`families.py`) だけで
厳密に出ることが多いのが効きます。**位数 m は挟み込みでは決まらない**ので、
この経路を採った場合 m は下限として返します。一致しなくても `sq.lo` は
`resolve_singularities(lower_bound=...)` の早期終了に使えます。

`certify.rlct_certified` と `nn_rlct.local_rlct_from_ideal` が自動で使います
(経路名 `squeeze`、`LocalRLCT.core_route == 'squeeze'`)。

**コストの上限が必須** — 族経路は内部でニュートン非退化判定 (凸包 + Z3) を
呼ぶので、大きな生成元に当てると探索本体より高くつきます。実測 (tanh 1-2-1,
7 変数, 6 生成元):

| | 挟み込み | 積の分解 |
|---|---|---|
| 上限なし | 7s → **146s** | 7s → **52s** |
| 上限あり | 7s → 6.8s (0.06s) | 7s → 6.7s (0.003s) |

効いた上限は次の 2 つです。

1. **展開前に大きさを見積もる**。`Σ_{i∈S} g_i²` を展開してから項数を見ると、
   その展開自体が支配的コストになります (tanh のコアでは 20 項の生成元が
   2 乗で 235 項)。生成元の項数 `n_i` から `Σ n_i(n_i+1)/2` を上限として
   先に弾きます。加えて呼び出し回数と壁時計の予算で打ち切ります。
2. **pay-when-needed**。1 回目は安い設定 (0.5 秒、単独の生成元のみ) で試し、
   **解消が失敗したときだけ**予算を上げて (8 秒、2 個組まで) 呼び直します。

積の分解の側は `factor_list` が高くつくので、**健全な O(1) のふるい**を
前に置きます。正規化で単項式因子は括り出し済みなので、分解
`f_rest = A(u)·B(v)` があれば A も B も 2 項以上をもち、指数ベクトルの台は
直積 `S = S_u × S_v` になります。したがって

> 単項式の個数は 4 以上の合成数でなければならない

が必要条件で、素数や 3 以下なら `factor_list` を呼ばずに済みます。

**使える (2): 積の分解** — 上の 4.1 を参照。`product_split_value()`。

**使えない (3): Lemma 2 (和の分離) を一般の chart で**

`f_rest = A(u) + B(v)` と変数分離できても、局所データは

    P = y^k(A + B) = u^{k_u}v^{k_v}A + u^{k_u}v^{k_v}B

で単項式因子が**両方の項に共通に掛かる**ため、P 自体は変数分離しません。

> 反例: `x(x²+y²) = x³ + xy²` は (非退化なので) λ = 2/3。
> 一方 `λ(x³) + λ(y²) = 1/3 + 1/2 = 5/6` で一致しません。

使えるのは `k = 0` のノードだけ（実測で chart ノードの 3.8%。構造上分離できる
のは 29.3% あります）で、そこは `families.py` の和の経路が既に担当しています。

**使えない (4): Theorem 2 を「付け替えを省く」根拠に**

「斉次なら最深点は原点だから中心の付け替えは要らない」は**誤り**です。

> 反例: `x² + y² − z²` は斉次ですが、原点の λ は 1、錐上の原点以外の点は
> 滑らかなので λ = 1/2 で、原点は最深ではありません。

Theorem 2 は「f が変数群 w₁..w_j について斉次なら、その群を 0 に落としても λ は
増えない」という**別々の変数群**についての主張で、1 つの chart の中で原点と
例外因子上の点を比べる根拠にはなりません。

これらの区別は `bench/guards.py` の `lemma_shortcut_tests()` が常時監視します
（on/off 一致 5 件、下界の健全性 6 件、反例 1 件、挟み込みの整合 2 件）。

### 4.7.5 `timing.py` — 段階別の実行時間

多項式・ネットワークのサイズや次数と計算時間の関係を測るための軽量
プロファイラです。スレッドローカルなので並列に回しても混ざりません。

```python
from timing import record, timed, last_record, timing_summary

with record("case-1", n_vars=4, degree=6):
    with timed("resolve"):
        res = resolve_singularities(f, gens)
    with timed("certify"):
        cert = certify(res)
rec = last_record()
rec.phases      # {'resolve': 1.23, 'certify': 0.31}  (ネスト込みの経過)
rec.exclusive   # 自分だけの時間 (二重計上しない)
rec.counts      # 呼び出し回数
rec.as_dict()   # JSON に落とせる形
```

段階名は固定です: `family` / `newton` / `newton_chart` / `ideal` /
`resolve` / `certify` / `eliminate_regular` / `nn_reduce+resolve` / `lp` / `smt`。

`certify.rlct_certified()` は内部で `record()` を開いているので、呼んだ後に
`timing.last_record()` で内訳が取れます。

```
>>> rlct_certified((x*y + z**2)**2, (x, y, z))
(1/2, 'proved', 'blowup+certify')
>>> print(timing.last_record())
rlct_certified: wall=0.248s [resolve=0.157s, ideal=0.046s, certify=0.022s,
                             newton=0.008s, newton_chart=0.003s]
```

`bench/run.py` と `bench/nn_cases.py` は各レコードに `time`
(ケース全体の秒数) と `timing` (段階別の内訳・共変量) を書き出し、
`bench/report.py` / `nn_cases.summarize()` / `bench/regression.py` が
`timing_summary()` で変数の数・次数・アーキテクチャ別に集計します。

```
# 実行時間 (秒): (変数, 次数) ごと
n_vars / degree       n    median      mean       max  内訳 (中央値)
(4, 4)               19     0.060     2.258    36.510  resolve=0.42, certify=0.03, ...
```

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
| `guards.py` | 負のテスト・変異検査・独立検査・Aoyagi 補題・Vandermonde 真値・**座標不変性**・**Newton 高速パス / 補題による短縮経路の on-off 一致**、および既知の未修正バグ (ゲート外で報告) |
| `run.py` | 実行と失敗の分類、JSON 出力 |
| `report.py` | 集計。クラス分布、経路別、n/次数ごとの proved 率、μ との相関 |
| `failures.py` | 失敗例の蓄積と自動縮約 |
| `nn_cases.py` | ニューラルネットの格子 (MLP x {relu, tanh} x 退化の型)。torch 非依存 |
| `regression.py` | 全スイートの一括実行 (guards / structured / random / hard / nn / failures) |
| `independent.py` | 解消の結果を使わない独立な検査 (体積の漸近による λ の数値推定、被覆の前向き標本検証) |
| `known_families.py` | 真値が独立に分かり、かつ難しい族 (一般の位置の超平面配置、斉次形式、Vandermonde 型) |
| `aoyagi_lemmas.py` | Aoyagi (Entropy 2019) の補題を真値を使わない検査にしたもの (イデアル不変性・単調性・分離・最深点)。**計算の短縮**に使うほうは本体の `lemmas.py` |
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
| **コンパクトな面をファセットしか見ていなかった** | `(x+y²)²+z²` が λ=5/4 を **proved** で返す (真値 1) | 面の交わりで閉じて全コンパクト面を列挙 |
| **主面のコンパクト性を見ていなかった** | `−9x₀−38x₁²x₂` が λ=3/2 を **proved** で返す (真値 1、原点で滑らか) | `p ∈ conv(台)` を LP で確認してから proved にする |
| 台が薄いと面が列挙できない | `53x₀x₂³+61x₁³x₂` (4 変数) で法線が空になり解消が無限反復 | 現れない変数を落とす + 部分集合 LP へのフォールバック |
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
