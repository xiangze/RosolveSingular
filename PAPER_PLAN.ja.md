# RLCT 厳密計算ツールの論文化 — 独自性の検証と構成案

作成日 2026-10-06。先行研究は 2026-10 時点の調査に基づく。

---

## 0. 結論

**論文になります。ただし今のままでは出せません。**

- 主張できる新規性は**ある**。ただし当初の「RLCT の厳密計算は未開拓」という形では
  **過大主張**になる。2024〜2026 年に厳密計算の論文が 3 本出ており、そのうち
  1 本 (Sergeant-Perthuis–Tsigaridas–Tsukahara 2026) は**同じ
  `min_i (h_i+1)/k_i` の式を使った厳密アルゴリズム**である。
- 我々の位置は「**任意次元・イデアル・重複度・実装**」の 4 点に絞られる。
  この 4 点はいずれも既存研究が持っていないので、主張は成立する。
- 機械学習側の貢献は「**非線形アーキテクチャでの LLC 推定器の校正用真値**」が
  最も強い。LLC の論文自身が「ResNet18 は真値が分からないので校正できない」と
  書いており、そこがそのまま空いている。
- **出す前に必須の作業が 3 件**ある (§6)。とくに座標依存バグの修正と、
  LLC 推定器との実測比較は、どちらが欠けても査読を通らない。

論文は 2 本に分けるのが自然 (§4)。1 本に詰めると、記号計算側の査読者には
実験が余分に見え、ML 側の査読者にはアルゴリズムが読めない。

---

## 1. 先行研究の地図

調査して確認した範囲。**太字が我々の主張に直接刺さるもの。**

### 1.1 RLCT の厳密計算

| 研究 | 与えるもの | 制約 |
|---|---|---|
| Lin, arXiv:1003.5338 (2010) + 博士論文 (2011)。`rlct.lib` (Singular) / `asymptotics.m2` | fiber ideal の定義、Newton 多面体の τ-距離、sos 非退化判定 | **単項式イデアルのみ厳密、他は上界**。ブローアップは手動の補助関数 (`blowupMap`/`strictTransform`) |
| **Kosta & Windisch, arXiv:2411.13392 (2024)**。SageMath 実装あり | **(λ, m) を厳密に** | **実超平面配置のみ** (線形形式の積)。ブローアップを使わない組合せ論 |
| **Sergeant-Perthuis, Tsigaridas & Tsukahara, arXiv:2608.20183 (2026)**。SCML 2026 | **局所 RLCT を厳密に、決定的、計算量上界つき**。「初の決定的アルゴリズム」 | **2 変数のみ**。**単一多項式のみ** (イデアル不可)。**重複度なし**。**実装なし** |
| Collins, Ann. Inst. Fourier 68 (2018) | 実 LCT (構成的) | 2 次元のみ、実装なし |
| Paemurru, arXiv:2404.18238 (2024) | 重み付きブローアップ + Newton 多面体で LCT | **複素** LCT。実装なし。SPTT に「一部のクラスで停止しない」と批判されている |
| Kurumadani, arXiv:2408.13030 / arXiv:2603.12785 | 非特異点での RLCT、3 層 NN の LLC **上界** | 上界のみ |

**確認した「空白」の根拠 2 つ。**

> "Currently, there are no programs for computing real log canonical thresholds"
> — Lin, arXiv:1003.5338 (2010)

> "Although computing a resolution of singularities is infeasible at the scale of
> practical statistical problems, learning coefficients have been derived for
> several families of models"
> — Drton, Gross, Kosta, Leykin, Sullivant, Windisch, arXiv:2511.15419 (2025)

15 年を挟んでこの 2 文が並ぶのが、一般次元での空白の証拠になる。ただし
上の表の 2 件が反例なので、**「一般の多項式・イデアルに対して、任意次元で」**
と必ず限定すること。

### 1.2 RLCT ≠ 複素 LCT — ここが我々の存在理由

複素 LCT は既存の D-module アルゴリズムで計算できる。Macaulay2 の
`BernsteinSato` パッケージには **`lct(Ideal)` がすでにある**。査読者は必ず
「それで済むのでは」と訊く。答えは済まない:

> "**However, this number may be greater than 1** if the codimension of the real
> zero locus of the function is greater than 1. **So it does not necessarily
> coincide with the maximal root of the b-function up to a sign, nor with the log
> canonical threshold of the complexification.** In fact, the real jumping numbers
> can be even disjoint from the non-integral jumping numbers of the complexification."
> — M. Saito, *On real log canonical thresholds*, arXiv:0707.2308 (2007)

同論文の不等式: `p(f,ω) ≥ rlct(f) ≥ lct(f_C)`。

Lin も明記している:

> "The complex threshold is often different from the RLCT." — arXiv:1003.5338, Remark 3.1

**この 1 節が論文の冒頭に必要。**Bernstein–Sato 経路が使えない理由を
先に潰しておかないと、related work で落ちる。

> 注: Kosta–Windisch は実超平面配置では `rlct = lct` が成り立つことを
> 証明している (Thm 4.3)。「常に異なる」とは書かないこと。
> また `lct(x²+y²+z²)` 等の具体値は**自分で検算してから**書くこと
> (調査では Saito §3 の一般式からの推定にとどまった)。

### 1.3 「既存部品の組み合わせでは」という反論への備え

ここが最大の弱点なので、先に書いておく。Singular には既に:

- `resolve.lib` (Frühbis-Krüger & Pfister): Villamayor アルゴリズムの実装。
  chart 木を返す
- `reszeta.lib`: **`computeN` が全変換の因子の重複度** (= 我々の `k_j`)、
  **`computeV` が体積形式の引き戻しの重複度** (= 我々の `h_j`) を返す
- `resweighted.lib` (Lee, Abramovich, Frühbis-Krüger, arXiv:2008.02169):
  **ATW の重み付きブローアップの実装**。Villamayor より「著しく効率的」

つまり **`min_j (h_j+1)/k_j` を読むための指数は、今日の Singular で取れる。**
我々の貢献を「解消から指数を取れる」と書くと、この 3 ライブラリで否定される。

**正しい主張の切り方**: 貢献は指数ではなく

1. **実・半代数的な層** — `Σ g_i²` への還元、各 chart の実点のうち定義域 Ω の
   上にあるものの管理、実の重複度の集約
2. **(λ, m) の組としての集約** と三値証明書 (proved/unknown/refuted)
3. **高速路** (Newton 非退化、積分解、単調性、キャッシュ) と**学習済み NN から
   fiber ideal までの経路**

### 1.4 機械学習側 — LLC 数値推定の現状

- 正しい引用は **Lau, Furman, Wang, Murfet & Wei, "The Local Learning
  Coefficient: A Singularity-Aware Complexity Measure", AISTATS 2025**
  (= arXiv:2308.12108 **v2**)。
  - v1 は著者 3 名・別題。**v1 を引くと取り下げ前の版を引くことになる。**
  - **arXiv:2402.03698 (Furman & Lau) は取り下げ済み** (2308.12108 に統合)。
    独立の文献として引かないこと。
- 推定量: `λ̂(w*) = nβ*(E_{w|w*,β*,γ}[L_n(w)] − L_n(w*))`, `β* = 1/log n`、
  局所化は Gauss 事前分布 `exp(−γ‖w‖²/2)`。SGLD でサンプリング。
- **校正は実質 deep linear network (DLN) 1 系統のみ。**

> "The recent clarification of theoretical LLCs in DLNs by Aoyagi (2024), making
> DLNs **the most realistic setting where theoretical learning coefficients are
> available**."

> "In contrast to the LLC estimates for DLN, the LLC estimates for **ResNet18
> cannot be calibrated as we do not know the true LLC values**."
> — 両方とも arXiv:2308.12108v2

- 専用ベンチマーク **Hitchcock & Hoogland, arXiv:2507.21449 (2025)** も真値は
  DLN のみ。報告されている失敗モード:
  - "For SGLD, at the step size values where the relative error is close to zero,
    **a significant fraction of the LLC estimates are also diverging**"
  - "LLC estimates which are **orders of magnitude larger than the maximum
    theoretical value of d/2**"
  - Theorem 2.1 で、標準的な SGMCMC の収束仮定が **DLN で成り立たないことを証明**
- **重複度 m は推定されていない** (探したが見つからなかった)。推定量の構成上
  λ しか出ない。**我々は m を出す。**

### 1.5 NN の閉形式 λ — ここは当初の想定より埋まっている

| モデル | 結果 | 文献 |
|---|---|---|
| Deep linear network | **λ, θ 厳密** | Aoyagi, *Neural Networks* 172:106132 (2024) |
| **3 層 ReLU** | **λ, θ 厳密** (Thm 6) | **Aoyagi, arXiv:2501.12747 (2025)** |
| Softmax 出力 | 線形網の結果に帰着 (Thm 7) | 同上 |
| 3 層 tanh (入出力 1 ユニット) | 厳密 (Vandermonde 還元) | Aoyagi, RIMS Kôkyûroku 1694 (2010) |
| Toy Model of Superposition | k-gon 臨界点で厳密 | Chen, Lau, Mendel, Wei, Murfet, arXiv:2310.06301 |
| 一般の解析的活性化 (3 層) | **上界のみ** | Kurumadani, arXiv:2603.12785 (2026) |
| **CNN** | **なし** | — |
| **Attention / Transformer** | **なし** | — |

**「ReLU の厳密 λ は無い」は誤り。**Aoyagi Thm 6 を無視すると即座に落とされる。
ただし Thm 6 の前提を読むと、我々の居場所は残る:

- 深さは**厳密に 3 層** (隠れ層 1 枚、重み行列 2 枚)
- λ は `W₀ = {w | L(w) = min L}` の**最適パラメータで**定義される。
  `p₀(x,y) = p(x,y|w₀)` と置くので **realizable が構成上の仮定**
- ReLU は符号セル分解で**厳密に**扱われている (多項式近似ではない)。
  ここは正確に書くこと
- **隠れ層の群分割 `H₁⁽²⁾ + … + H_{k⁽²⁾}⁽²⁾` と各群の階数 `r_i` は「与えられた
  もの」として扱われ、与えられた `w*` からどう決まるかは書かれていない**

したがって正しい言い方は:

> Aoyagi (2501.12747, Thm 6) は、realizable な回帰モデルの最適パラメータにおいて、
> **隠れ層の群分割と各群の階数を人手で与えたとき**の 3 層 ReLU 網の λ, θ を
> 閉形式で与える。これは組合せ的に添字づけられた族に対する公式であり、
> **与えられた学習済み θ\* を入力として λ を返す手続きではない**。
> 深さ 4 以上、畳み込み構造、attention はいずれも対象外。

---

## 2. 主張できること / 譲るべきこと

### 2.1 主張できる (4 点)

1. **任意次元での厳密な (λ, m)**。2 変数制限 (SPTT) も超平面配置制限
   (Kosta–Windisch) もない。
2. **イデアルを直接扱う**。SPTT は単一多項式のみ。fiber ideal は本来
   多生成元なので、これは本質的な差。
3. **重複度 m を返す**。SPTT は λ のみ。LLC 推定器も λ のみ。
4. **動く実装がある**。SPTT は実装なし。Lin の実装は単項式のみ厳密。
   さらに**三値証明書**により、値が信用できない場合を silent に返さない。

これに加えて、ML 側で:

5. **学習済み PyTorch モデル (MLP / CNN / Transformer 1 層) から fiber ideal を
   作り λ に至る経路**。ReLU セル固定、対称性のゲージ固定、正則方向の消去、
   自由方向の除去。Aoyagi の公式が要求する「群分割」を人手で与える必要がない。
6. **CNN と attention に対する初の厳密 λ** (ただし部分空間への制限つき = 下界)。
7. **非線形アーキテクチャでの LLC 推定器の校正用真値** (λ と m の両方)。

### 2.2 譲るべきこと

- Newton 多面体の高速路は **Lin Thm 1.3 そのもの**。既知理論の実装として書く。
- 重み付きブローアップ + Newton 多面体という設計の複素版は
  **Paemurru arXiv:2404.18238**。引用して実の場合の差分を明示する。
- 重み付きブローアップのエンジン自体は **`resweighted.lib`** にある。
- 解消からの指数抽出は **`reszeta.lib` の `computeN`/`computeV`** にある。
- 統計モデルの閉形式 λ は 20 年分蓄積がある (Aoyagi, Watanabe, Drton, Zwiernik,
  Rusakov–Geiger, Yamazaki, Kurumadani)。「厳密値が無い」とは書けない。
- 「部分空間に制限した λ は全体の λ の下界」。**毎回書くこと。**
- Taylor 打ち切りは**次数ごとに別のモデル**。「λ が次数で変わる」ではなく
  「どの次数で止まるか」が問い。

---

## 3. 我々が持っている実験結果

論文の証拠として使える、現時点の実測。

| 項目 | 結果 |
|---|---|
| 構造化多項式 51 件 (n≤6, d≤13) | proved 51/51 |
| ランダム 111 件 (n≤3, d≤6) | proved 111/111 |
| ランダム 36 件 (n=4–5, d=4–8) | proved 22/36 (失敗は全て `resolve_max_depth`) |
| NN 54 件 | proved 39。コア 0–3 変数なら 100%、4 変数で 50%、5 以上で 0% |
| ガード | 169/169 緑 (既知バグ 1 件はゲート外) |
| 独立経路との値の食い違い | 0 件 |
| キャッシュ 158 件 | 値の不一致 0 件 |
| Transformer 1 層 softmax 打ち切り | 非退化 θ\*: n=1,2,3 で λ=2 一致。退化 θ\*: n=0→1 で 1→3/2 と動き、n≥2 は未到達 |
| 数層 CNN (分類) | ReLU: 層数 1,2,3 で λ = 4, 2, 4 (**非単調**)。tanh n=1: 4, 4, timeout |
| 数層 CNN (回帰) | ReLU: 4, 4, 7/2。λ(回帰) ≥ λ(分類) を実測 (depth 2 で 4 対 2) |
| 学習済み TinyViT 層 0 | λ = 5 = 9/2 + 1/2 |

**発見として論文に書けるもの**:

- **ReLU では λ が層数に対して単調でない** (4 → 2 → 4)。λ = 正則方向/2 なので、
  これは「第 1 畳み込み層のパラメータのうち出力から識別できる本数」が
  層数に対して単調でないことを意味する。深さを足すと死んだユニットとセル境界が
  増え、識別可能な方向が**減ることがある**。
  → SPTT も「深さとともに実効複雑度が**減る**ことがある」と報告しており、
    独立に同じ向きの現象を見ていることになる。**相互に補強できる。**
- **tanh では層数に鈍い** (解けた範囲で λ = 4 一定)。死んだユニットを作らないため。
- **λ(回帰) ≥ λ(分類)**。分類はロジットの差しか拘束しないため。回帰の fiber ideal
  を分類に使うと λ を過大評価する、という実測例。
- **打ち切り次数の収束**: 非退化 θ\* では softmax 3 次まで λ 不変。
  退化 θ\* では 1 次で既に動く。**本当に知りたい退化点が一番解けない。**
- 計算の律速は**係数の大きさではなく生成元の項数**
  (`exp` の分母上限を 10⁴ → 16 に落としても 93.5s → 93.5s)。

---

## 4. 論文の構成案 — 2 本に分ける

### 論文 A (記号計算): "Exact computation of real log canonical thresholds via symbolic resolution of singularities"

想定読者: 計算代数幾何。JSC / ISSAC / MEGA / SIAGA。

1. **Introduction**: RLCT の定義 (Watanabe, Lin)。**なぜ複素 LCT では駄目か**
   (Saito の不等式)。既存実装の地図と空白。
2. **Background**: fiber ideal、`λ = min_j (h_j+1)/k_j` と重複度、
   不変量 `f(φ(y)) = y^k U(y) u(y)`, `|det Dφ| = |v(y)| y^h`。
3. **Algorithm**:
   - chart 木、最小 hitting set による中心選択 (ILP)、重み付きブローアップ
   - Newton 多面体による上界と mediant による下界、分枝限定
   - **主面のコンパクト性判定** — これが無いと偽の proved が出る (§6 のバグ表)
   - 全コンパクト面の列挙 (面の交叉による閉包)
   - 擬除算による正則方向の消去
   - **実・半代数的な層** (ここが新規)
4. **Certificates**: proved / unknown / refuted の三値。恒等式の厳密検査、
   正規交叉性の SMT 判定、Lean 4 への証明義務の書き出し。
5. **Fast paths**: Newton 非退化 (= Lin Thm 1.3 の実装)、積分解、
   単調性による下界、永続キャッシュ (コード指紋 + 依存関係の無効化)。
6. **Experiments**: 上の表。**既知閉形式との一致**
   (Vandermonde, DLN, 正則点, 単項式イデアル)。**変異検査の検出力**。
7. **Limitations**: n=4,5 の非擬斉次例、計算量上界が無いこと、座標依存。

**SPTT との差分を 1 段落で明示する**こと: 任意次元 / イデアル / 重複度 / 実装。
逆に SPTT の持つ**計算量上界は我々に無い**ので、それは正直に書く。

### 論文 B (機械学習): "Exact learning coefficients for small trained networks: ground truth for LLC estimation"

想定読者: SLT / interpretability。NeurIPS Datasets & Benchmarks track が第一候補。
次点 AISTATS、*Neural Networks*、JMLR (MLOSS)。

1. **Introduction**: LLC 推定は 100M パラメータまで届いたが、**DLN 以外では
   校正できていない** (Lau et al. の自己申告を引用)。重複度は誰も推定していない。
2. **Method**: 学習済みモデル → fiber ideal。
   - 分類は**ロジットの差** (回帰の ideal を使うと λ を過大評価)
   - ReLU / MaxPool のセル固定 (境界では錐に制限しないので下界)
   - softmax / tanh の分母払いと Taylor 打ち切り (**次数ごとに別モデル**)
   - 対称性のゲージ固定、正則方向の消去
   - 部分空間に制限 → **下界**
3. **Validation**: 記号フォワードが torch と一致 (最大差 ~1e-8)、
   生成元が θ\* で厳密に消えること、既知閉形式 (DLN, Aoyagi Thm 6 の特殊形) との一致。
4. **Benchmark**: 厳密 (λ, m) の表。アーキテクチャ × 深さ × 活性化 × タスク ×
   打ち切り次数。**←これが成果物**
5. **LLC 推定器との比較** (**未実施。§6 の必須項目**):
   同じ θ\* で SGLD 推定量を走らせ、厳密値との差、ステップ幅 ε・局所化 γ・
   鎖長への感度、符号と順序の保存を測る。
6. **Findings**: §3 の発見 (層数非単調、tanh の鈍さ、回帰 ≥ 分類、打ち切り収束)。
7. **Limitations**: 下界であること、規模、退化点が解けないこと。

> 単著 1 本にするなら、論文 B を本体にして記号計算側を付録に落とす。
> ICML/ICLR/NeurIPS 本会議に出すなら、背骨を
> 「既存の LLC 推定値は非線形アーキテクチャで X% 偏っている」という
> **ML の主張**にしないと「ML ではない」で落ちる。

---

## 5. 投稿先

| venue | 向き | 備考 |
|---|---|---|
| **NeurIPS Datasets & Benchmarks** | ◎ 論文 B | 真値ベンチマークはまさに欠けているもの。Hitchcock–Hoogland が DLN 限定の劣版を作っており、需要の証拠になる |
| **Neural Networks (Elsevier)** | ◎ 論文 B | Aoyagi の DLN 公式 (LLC 論文が校正に使っているもの) の掲載誌。査読者層が最も好意的 |
| **JMLR (MLOSS track)** | ○ 両方 | ソフトウェア論文の枠がある。Watanabe の WBIC, Aoyagi の RBM も JMLR |
| **AISTATS** | ○ 論文 B | Lau et al. の掲載先 |
| **JSC / ISSAC** | ◎ 論文 A | 正しさの証明・停止性・**計算量上界**を要求される。SPTT と比較される |
| **SIAGA** | ○ 論文 A | 実代数幾何 + 応用。きれいな定理と計算量が欲しい |
| **Algebraic Statistics (MSP)** | ○ 論文 A | fiber ideal / RLCT の本籍地。ソフトウェア論文を取るかは要確認 |
| MEGA | ○ | 隔年。JSC 特集号に回る |
| SCML | △ | 2026 は終了済み。主トラックは 1–2 頁の extended abstract で非 archival。露出用 |
| IEICE Trans. | △ | Aoyagi–Watanabe 2005 の掲載先だが、届けたい ML 読者が読まない |
| Information Geometry | × | 手法が情報幾何ではない |
| Journal of Algebraic Statistics | × | **現在の同名誌の出版元と評価を要確認**。意図しているのは MSP の *Algebraic Statistics* の可能性が高い |

---

## 6. 投稿前に必ずやること (優先順)

### 必須 1. 座標依存バグの修正 — **部分的に実施 (2026-10-08)**

**修正できた部分。** イデアルを運ぶ `ideal_resolve.resolve_ideal` に
**親の chart から引き継いだ指数 `k0, h0` を渡せるようにした**。これで
Vandermonde の chart 問題

```
P = v⁴(G₁² + v²G₂² + v⁴G₃²),  G_k = a₁ + a₂b₂^k + a₃b₃^k,  h = [5,0,0,0,0,0]
```

は、生成元 `[v²G₁, v³G₂, v⁴G₃]` を渡せば**どちらの座標系でも 7/6** になった
(多項式経路は chart 座標で 3/2、`G₁` を座標に取ると 7/6 と座標依存)。
しかも速い (0.5s 対 1.3s)。`h0` を渡さないと h=0 の別問題を解いて 1/4 に
なるので、**その受け渡し自体も回帰テストに入れた**。

ガード 3 本を `known_issues` から通常のガードに昇格:
`ideal_coordinate_invariance_tests()`。

**修正できていない部分 — 当初の見立てより深刻だった。** 族全体で調べたところ、
**多項式経路は Vandermonde 族で系統的に λ を過大評価する**:

| 設定 | 真値 | 多項式経路 | イデアル経路 |
|---|---|---|---|
| M=N=1, H=2, Q=1 | 3/4 | 1 | **3/4 (一致)** |
| M=N=1, H=3, Q=1 | 1 | 3/2 | 停止せず |
| M=N=1, H=4, Q=1 | 7/6 | 2 | 停止せず |
| M=N=1, H=3, Q=2 | 5/6 | 3/2 | 停止せず |
| M=2, N=1, H=2, Q=1 | 1 | 1 | 1 |

イデアル経路は H=2 では**真値 3/4 を厳密に当てる**が、H ≥ 3 では停止しない。
原因は中心の選び方で、`a1_1` の chart で 11 回続けてブローアップしても
不変量が落ちない。正しい一般化は **Aoyagi の rational blowing-up**
(Entropy 21(6):561, 2019) か**トーリック解消**で、これは小さな修正ではなく
研究課題である。

**救い: 誤った値を proved として返してはいない。** `rlct_certified` は
これらすべてで `status='refuted'` を返す:

```
M=1 H=2 Q=1  真値 3/4  ->  (3/4, 'refuted')   ← 値は正しいが保守的に refuted
M=1 H=3 Q=1  真値 1    ->  (3/2, 'refuted')
M=1 H=3 Q=2  真値 5/6  ->  (3/2, 'refuted')
```

新ガード `vandermonde_never_proved_wrong_tests()` が
「**値が真値と違うなら status は proved でない**」を族全体で検査する。
これは一致表より強い健全性の性質で、論文に書くべき不変量である。

> **論文での書き方**: 「三層ネットの背後にある Vandermonde 族は現状
> H ≥ 3 で解けず、値は過大評価になる。ただし証明書がそれを検出するので
> 誤った値が proved として出ることはない」と正直に書く。
> これを隠して一致表だけ出すと、査読で Aoyagi の公式と突き合わせられて
> 終わる。先に書いて、`refuted` の設計を売りにするのが唯一の筋。

### 必須 2. LLC 推定器との実測比較 — **実施済み (2026-10-08)**

`llc_estimate.py` / `llc_compare.py` を実装し、同じ θ\* で厳密 λ と SGLD の
λ̂ を突き合わせた。詳細は `DOCUMENTATION.ja.md` §4.6.8。**論文の主張が
当初案から変わる結果が出た。**

**(a) 推定器自身は正しく動く。** 正則モデル (λ = d/2) で検算して
d=2,4,8,12 で誤差 −9.9%, −4.4%, −0.9%, −0.4%。ここに至るまでに実装の
落とし穴が 3 つあり、いずれも論文に書く価値がある:

- 誤差棒を 1 鎖の標準誤差で出すと自己相関のぶん過小 (±0.075 と出るのに
  ε を変えると λ̂ が 3.5〜6.2 まで動いた)。鎖間のばらつきにして ±15〜20%
- **局所化 γ による偏りには閉形式がある**:
  `E[nβ(L_n−L_n*)] = (1/2) Σ_i nβf_i/(nβf_i+γ)`。厳密な Gauss 事後からの
  直接標本抽出と**小数 4 桁まで一致** (1.9768 対 1.9760)。つまり偏りは
  解析的に補正・検算できる。γ=100 (nβ≈82) で真値の 45% まで落ちる
- RMSProp 前処理は +10〜30% 過大評価する (発散項の補正を省くため)

**(b) 分類では推定器が機能しない (新しい発見)。**

| 設定 | 厳密 λ | λ̂ | 誤差 |
|---|---|---|---|
| 分類 CNN depth1 (16 点) | 5 | 0.713 | **−85.7%** |
| 分類 CNN depth1 dead_channel | 5/2 | 1.661 | −33.6% |
| 分類 CNN depth1 (8 点) | 4 | 0.004 | **−99.9%** |

順序の保存すら 0/1 組。原因は **softmax の飽和**で、制限した部分空間の
Fisher 固有値を直接測ると分類は **7.8e-25 〜 8.0e-21** (epoch を 600 → 60 に
減らしても同じ)、回帰は 9.2e-02 〜 9.7e+00。学習しきった分類器はロジットが
大きく softmax が飽和し、**KL の曲率が数値的に消える**。λ̂ は
`E[L_n]−L_n(w*)` の差を見るので信号が無くなる。

**一方、厳密計算は影響を受けない** — ロジットの差のイデアルを扱い、RLCT は
正の定数倍で不変なので、曲率が 1e-24 でも λ は変わらない。

> これは**厳密計算でしか見えない現象**であり、論文 B の中心的な主張に
> できる。「λ̂ が小さい = 退化が小さい」と読んではいけない場合がある。
> ただし公表されている LLC の実験は分類器 (ResNet18, 言語モデル) が多いので、
> **この主張は慎重に書く必要がある**: 我々が見たのは「制限した部分空間を
> 持つ、小さく、過学習させた分類器」であって、彼らの設定そのものではない。
> 主張を一般化する前に、(i) 温度・weight decay・ラベル平滑化で飽和を
> 緩めた場合、(ii) 部分空間を制限しない場合、を確かめること。

**(c) 回帰では独立な 2 経路がぴったり一致する。**

| 設定 | 厳密 λ (記号) | Fisher の階数 | 解析形の予測 (γ=0.01) |
|---|---|---|---|
| 回帰 CNN depth1 | 5 | 10 | **4.998** |
| 回帰 CNN depth1 dead_channel | 5/2 | **5** | **2.500** |

`dead_channel` では Fisher の固有値がちょうど 5 本ゼロになり、記号計算が
「自由方向 5 本」と言ったのと一致する。

**(d) 特異点での試験。** 上の CNN はすべて「コアが空、λ = 正則方向/2」で
Fisher の階数の半分で説明できてしまう。**λ が階数の半分でない特異点**での
試験として、原点に置いた深層線形網を使った (LLC の論文が校正に使っている
唯一の族でもある):

| モデル | 真値 | 厳密計算 (0.0s) | λ̂ |
|---|---|---|---|
| `y=b(ax)`, θ\*=0, ⟨ab⟩ | λ=1/2, m=2 | **λ=1/2, m=2 両方一致** | 0.438±0.169 (**−12.3%**) |
| `y=c(b(ax))`, θ\*=0, ⟨abc⟩ | λ=1/2, m=3 | **λ=1/2, m=3 両方一致** | 0.286±0.146 (**−42.7%**) |

原点では Fisher が恒等的に 0 (階数 0) なので `rank/2` は 0 を与えるが
真の λ は 1/2。

**これが論文 B の一番鋭い結果になる**: λ はどちらも 1/2 で同じなのに、
λ̂ の誤差は −12% から −43% へ悪化する。両者の違いは**位数だけ** (m=2 対 m=3)
で、位数は λ̂ が原理的に返せない量である。つまり
**λ̂ の誤差は、λ̂ が表現できない量とともに増大する。**
深い (より退化した) 特異点で推定が悪くなるという、定量的で再現可能な主張。

**(e) 方法論上の注意 (論文に必ず書く)。** 動かす変数が 10 本でデータ点が
十分あると、制限した部分空間では θ\* が**正則点**になり λ = d/2 = 5 と
自明に決まる (実測: n_data=4,8,16 で λ = 2, 4, 5 と増え、5 = d/2 で止まる。
λ ≤ d/2 は常に成り立つので**この飽和は証明済み**)。
したがって「部分空間に制限して λ を測る」実験は、**退化を仕込まないと
特異性を見ていない**ことになる。退化は学習後に仕込めばよい
(`llc_compare.inject_degeneracy`): 教師を θ\* の出力から作るので、
どんな θ\* でもそれが損失 0 の大域解になり realizable 性は壊れない。
なお**チャネルの複製は、下流の重みを凍結していると対称性にならない**ので
λ は下がらなかった。効いたのは死んだユニット (自由方向 5 本 → λ=5/2)。

### 追加: ResNet18 / ResNet50 への拡張 — **実施済み (2026-10-08)**

小さい CNN では λ = 正則方向/2 ばかりで「適切な解で止まっていない可能性」が
あったため、実データで学習した ResNet に拡張した (`resnet_rlct.py` /
`train_resnet.py`)。**11,176,842 パラメータの resnet18 で動く** (`fc.bias`
を動かす場合 0.8 秒)。

**鍵は 2 つ。**

1. **BatchNorm を畳み込みに融合する。** BN は 1/sqrt を含むので多項式でないが、
   評価モードでは統計量が定数なのでアフィン写像であり、直前の畳み込みに
   厳密に畳み込める。融合後のネットワークは元と**数値的に同一**かつ多項式
   (実測の出力差 resnet18 4.8e-07 / resnet50 8.1e-06)。
2. **「ずれ」だけを疎に運ぶ。** 密に記号式を持つと layer4 の 1 ブロックで
   1800 万回の記号演算になり終わらない。各活性化を
   `値 = num (float64) + d (疎な sympy 式、θ* で 0)` と分け、`d` は変数に
   依存するエントリだけ持つ。副産物として**生成元は構成上 θ* で厳密に 0**
   になり、丸め誤差が残差として漏れる問題 (softmax の `e*` や tanh の係数で
   手当てが必要だった) が消える。

**正しさの検査**: 生成元が θ\* で消えるのは必要条件にすぎないので、
「パラメータを δ 動かしたときの torch の実際の出力変化」と記号式の値を
突き合わせた。BasicBlock・Bottleneck・残差・downsample・layer4 内の conv を
踏んで**相対誤差 6e-15 〜 1.5e-10**。

> 最初 float32 で比べて相対 1e-3 と出て「不一致」と誤判定した。変化量 1e-5 に
> 対し float32 の丸めが 1e-7 なので雑音に埋もれていただけ。数値部分を
> float64 に統一して解決。**論文の検証節に書く価値がある落とし穴。**

**閉形式との一致**: `fc.bias` だけを動かした分類は、ロジットの平行移動が
対称性なので λ = (C−1)/2。実測 resnet18 C=10 で **9/2**、C=5 で **2**。

**費用の目安** (記号計算の量は変数の数 × 影響が広がる範囲):

| params / tail_from | 実測 |
|---|---|
| `fc.bias` / `layer4` | 0.8s |
| `fc.weight`, `layer4.2.conv3.weight` (スライス) | 秒 |
| `layer4.0.conv1.weight` (影響が 512 エントリに広がる) | 114s |
| `layer4` 全体 | 不可 |

**死んだ入力チャネルから読む重みは自由方向になる** (記号の寄与が恒等的に 0)。
実測で `layer4.1.conv2.weight` の入力チャネル 0 がこれに当たった。誤りでは
なく、そのパラメータが λ に効かないことを厳密計算が検出している。

### 必須 3. 既知閉形式との一致表 — **実施済み (2026-10-08)**

`bench/agreement.py` を実装。1 ケース 90s、キャッシュ無効。

| 族 | 件数 | 一致 | proved | 不一致を proved |
|---|---|---|---|---|
| 正則点 (λ=d/2) | 5 | 5/5 | 5 | 0 |
| 単項式イデアル (Howald/Lin Thm 1.3) | 14 | 13/14 | 13 | 0 |
| 斉次形式 (λ=n/d) | 11 | 11/11 | 11 | 0 |
| 超平面配置 | 11 | 11/11 | 10 | 0 |
| 行列分解 | 2 | 1/1 | 2 | 0 |
| Vandermonde | 12 | 9/12 | 5 | 0 |
| **合計** | **55** | **50/52** | **46** | **0** |

**値が真値と違うのに proved として返したケースは 0 件。**不一致 3 件は
いずれも `unknown` か `refuted`。これが論文で主張すべき健全性の性質。

実装中に見つけた注意点 2 つ:

- **Lin Thm 1.3 の係数 2。** Lin の RLCT は `ζ(z)=∫(Σf_i²)^{−z/2}` で
  定義されるので **Watanabe の λ の 2 倍**。正しくは λ = 1/(2ℓ)。
  手計算 4 例で確認した (`⟨x₁x₂⟩` λ=1/2、`⟨x₁³,x₂²⟩` λ=5/12 など)。
  最初 1/ℓ で実装して全件不一致になった
- **健全性の集計自体にバグがあった。** timeout のレコードに `honest` を
  設定し忘れ、「不一致を proved 1 件」と誤報した。健全性の集計が誤報するのは
  最悪なので、ここは論文にも書く

**追加 (2026-10-08): 深層線形網の閉形式を実装し、照合した。**
著者公開 PDF から Theorem 1 を取得できた
(https://www2.math.cst.nihon-u.ac.jp/aoyagi/wp/wp-content/uploads/2025/02/neuralnet.pdf)。
`bench/dln.py` に実装し、**同論文 Theorem 2 (L=2) と 657 件で不一致 0**、
さらに我々の厳密計算 (⟨ab⟩ → λ=1/2 θ=2、⟨abc⟩ → λ=1/2 θ=3) と**位数まで
一致**することを確認した。PDF 抽出で M̄ の上線や r² の指数が落ちている
可能性があったので、この三重の検算は必須だった。

厳密計算との一致は **9/11、不一致 2 件はいずれも proved でない**。
不一致は `H=(2,2,2), r=1` のように真の階数が 0 でない縮約ランク回帰で、
Vandermonde 族と同じ難所に当たる。

**これは論文 B にとって大きい**: 深層線形網は LLC 推定の論文が校正に使って
いる唯一の族なので、「同じ族で我々は厳密に一致する、しかも位数も出る」と
言える。任意の深さ・幅・階数でケースを生成できるので、ベンチマークの
規模も自由に取れる。

**残っている作業**: Aoyagi arXiv:2501.12747 Thm 6 (3 層 ReLU) との照合。
こちらは「我々の計算が Thm 6 の群分割を自動で再現するか」が最強の検証になる。

### (旧) 必須 3 の当初案

- Aoyagi の DLN 公式 (*Neural Networks* 172:106132) と**我々の計算の一致**を、
  複数の層数・階数で示す
- Aoyagi Thm 6 (3 層 ReLU, arXiv:2501.12747) の特殊形との一致。
  Thm 6 は群分割を人手で与えるので、**我々の計算がその分割を自動で再現するか**を
  見るのが最も強い検証になる
- Kosta–Windisch の Sage 実装 (超平面配置) との**クロスチェック**。
  独立実装との一致は査読で効く
- 正則点で λ = d/2、単項式イデアルで Lin Thm 1.3 と一致

### あると強い

- **計算量上界**。SPTT は持っている。無いなら「経験的コスト表」で代替し、
  持っていないことを明記する
- tanh の 2 次を届かせる (現状 600s でも未到達)。
  係数を有効桁 3 桁に揃えて分母の伸びを抑えるのが第一候補
- n=4,5 の非擬斉次例: トーリック解消 (双対扇の正則細分) に移す
- Lean 4 の証明義務を実際にコンパイルする (現状未コンパイル)

---

## 7. 引用必須の文献

**RLCT の定義と理論**
- S. Watanabe, *Algebraic Geometry and Statistical Learning Theory*, CUP, 2009
- S. Lin, *Ideal-Theoretic Strategies for Asymptotic Approximation of Marginal
  Likelihood Integrals*, arXiv:1003.5338, 2010
- S. Lin, *Algebraic Methods for Evaluating Integrals in Bayesian Statistics*,
  PhD thesis, UC Berkeley, 2011
- M. Saito, *On real log canonical thresholds*, arXiv:0707.2308, 2007 ← **RLCT ≠ LCT**

**厳密計算 (直接の競合)**
- D. Kosta, D. Windisch, *Classification of real hyperplane singularities by real
  log canonical thresholds*, arXiv:2411.13392, 2024
- G. Sergeant-Perthuis, E. Tsigaridas, Y. Tsukahara, *Exact Algebraic Computation
  of Learning Coefficients for Two-Dimensional Singular Models*, arXiv:2608.20183,
  SCML 2026 ← **最重要の競合**
- E. Paemurru, *Reading the log canonical threshold of a plane curve singularity
  from its Newton polyhedron*, arXiv:2404.18238, 2024
- T. Collins, *Log-canonical thresholds in real and complex dimension 2*,
  Ann. Inst. Fourier 68 (2018), DOI 10.5802/aif.3229

**複素 LCT の計算 (使えない理由の提示用)**
- D. Andres, V. Levandovskyy, J. Martín-Morales, *Effective Methods for the
  Computation of Bernstein-Sato polynomials...*, arXiv:1002.3644, 2010
- C. Berkesch, A. Leykin, *Algorithms for Bernstein–Sato polynomials and
  multiplier ideals*, arXiv:1002.1475, 2010
- T. Shibuta, *Algorithms for computing multiplier ideals*, arXiv:0807.4302, 2008
- Z. Teitler, *Software for multiplier ideals*, J. Softw. Algebra Geom. 7 (2015),
  DOI 10.2140/jsag.2015.7.1
- R. Blanco, S. Encinas, *A procedure for computing the log canonical threshold of
  a binomial ideal*, arXiv:1405.3942

**解消のソフトウェア (「組み合わせでは」への備え)**
- A. Frühbis-Krüger, G. Pfister, Singular `resolve.lib`, `reszeta.lib`
  (`computeN` / `computeV`)
- J. Lee, *Algorithmic resolution via weighted blowings up*, arXiv:2008.02169, 2020
  (`resweighted.lib`, with D. Abramovich and A. Frühbis-Krüger)
- D. Abramovich, M. Temkin, J. Włodarczyk, *Functorial embedded resolution via
  weighted blowings up*, arXiv:1906.07106
- G. Bodnár, J. Schicho, *Automated resolution of singularities for hypersurfaces*,
  J. Symbolic Comput., 2000

**NN の閉形式 λ**
- M. Aoyagi, *Consideration on the learning efficiency of multiple-layered neural
  networks with linear units*, Neural Networks 172:106132, 2024 ← DLN
- M. Aoyagi, *Singular learning coefficients and efficiency in learning theory*,
  arXiv:2501.12747, 2025 ← **3 層 ReLU, Thm 6**
- M. Aoyagi, S. Watanabe, IEICE Trans. J88-D-II(10):2112–2124, 2005 ← Vandermonde
- M. Aoyagi, K. Nagata, Neural Computation 24(6):1569–1610, 2012
- M. Aoyagi, JMLR 11:1243–1272, 2010 ← RBM
- Y. Kurumadani, *Upper Bounds for Local Learning Coefficients of Three-Layer
  Neural Networks*, arXiv:2603.12785, 2026
- Y. Kurumadani, *Learning Coefficients in Semi-Regular Models*, arXiv:2406.02646

**LLC 数値推定**
- E. Lau, Z. Furman, G. Wang, D. Murfet, S. Wei, *The Local Learning Coefficient:
  A Singularity-Aware Complexity Measure*, **AISTATS 2025** (= arXiv:2308.12108v2)
  ← v1 や arXiv:2402.03698 (取り下げ) を引かない
- R. Hitchcock, J. Hoogland, *From Global to Local: A Scalable Benchmark for Local
  Posterior Sampling*, arXiv:2507.21449, 2025 ← **失敗モードの報告**
- J. Hoogland et al., *Loss Landscape Degeneracy and Stagewise Development in
  Transformers*, TMLR 2025 (arXiv:2402.02364)
- G. Wang et al., *Differentiation and Specialization of Attention Heads via the
  Refined LLC*, ICLR 2025 Spotlight (arXiv:2410.02984)
- Z. Chen, E. Lau, J. Mendel, S. Wei, D. Murfet, *Dynamical versus Bayesian Phase
  Transitions in a Toy Model of Superposition*, arXiv:2310.06301
- S. Watanabe, *A widely applicable Bayesian information criterion*,
  JMLR 14:867–897, 2013
- M. Drton, M. Plummer, *A Bayesian information criterion for singular models*,
  JRSS-B 79(2):495–528, 2017
- M. Drton, E. Gross, D. Kosta, A. Leykin, S. Sullivant, D. Windisch,
  *Singular Learning Theory for Factor Analysis*, arXiv:2511.15419, 2025

---

## 8. 未確認事項 (投稿前に自分で確認)

1. **Lin の 2011 博士論文本体** — `math.berkeley.edu` がこの環境から読めず、
   arXiv:1003.5338 で代用した
2. **Lin の `rlct.lib` / `asymptotics.m2` の中身** — URL は生きている
   (`shaoweilin.github.io/public/rlct.html`) が、ファイル本体をテキストとして
   読めなかった。ページには "**exact for monomial ideals, upper bound otherwise**"
   と明記されている。**査読者はここを見るので、自分で落として確認すること**
3. **Saito arXiv:0707.2308 の §3 の具体値** — 一般式 `rlct = n/d` までは確認したが、
   `x²+y²+z²` 等の具体値は自分で検算してから書くこと。掲載誌が見つからず
   preprint のままの可能性
4. **Aoyagi (2024) の DLN 公式の完全形** と **reduced rank regression の場合分け** —
   原論文から取ること
5. **Lau et al. Fig. J.2 の相対誤差の数値** — 引用するなら図から直接読むこと
6. **Chen et al. (2310.06301) に理論 vs SGLD の定量誤差表があるか**
7. **MSP の *Algebraic Statistics* がソフトウェア論文を受けるか** (編集部に確認)
8. **SPTT の最終版** — 任意次元・イデアル・重複度・実装の 4 点が本当に
   彼らに無いことを、投稿直前にもう一度確認する
9. `resolve.lib` / `resweighted.lib` の性能限界の具体的数値 —
   Frühbis-Krüger の Dagstuhl 報告と Cambridge の章が出所の可能性
