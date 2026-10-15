# RLCT toolkit / 実対数閾値ツール群

Resolution of singularities for computing real log canonical thresholds (RLCT),
with machine-checked certificates and a regression-test harness.

- 日本語ドキュメント: [DOCUMENTATION.ja.md](DOCUMENTATION.ja.md)
- English documentation: [DOCUMENTATION.en.md](DOCUMENTATION.en.md)
- ベンチ/回帰テスト: [bench/README.md](bench/README.md)

## インストール

```sh
pip install sympy scipy pulp z3-solver graphviz     # z3 が無いと証明できません
apt-get install singular graphviz                   # 任意
pip install torch                                   # torch_rlct.py を使う場合のみ
```

## 最短の使い方

```python
import sympy as sp
from certify import rlct_certified, rlct_interval

x, y, z = sp.symbols("x y z", real=True)
rlct_certified(x**2 + y**3, (x, y))      # -> (5/6, 'proved', 'newton')
rlct_interval(f, gens)                   # 厳密値が出ないときは健全な区間
```

学習済みモデルと Transformer:

```sh
python train_rlct.py --model cnn --size 4 --epochs 300 --rlct --params 0.weight,0.bias
python taylor_sweep.py --loss classification --classes 2 --seqs 2 --orders 0,1,2
python cnn_sweep.py --depths 1,2,3 --acts relu,tanh --orders 1 --losses both \
    --size 8 --channels 2 --kernel 2 --hidden 2 --params conv --max-vars 10
```

## 本体

| ファイル | 役割 |
|---|---|
| `resolve_singularity.py` | ブローアップ + 正規化。重み付きブローアップ、分枝限定、Graphviz の木 |
| `ideal_resolve.py` | イデアルを運ぶ解消 (既定で併用)。生成元を座標に取れる |
| `certify.py` | 三値の証明書 (proved/unknown/refuted)、SMT による正規交差判定、健全な区間 |
| `families.py` | 族ごとの閉じた式 (単項式/積/和/形式のべき/ニュートン非退化) |
| `lemmas.py` | Aoyagi の補題を計算の短縮に使う (単調性の下界 + Newton 上界の挟み込み、積の分解)。使えない補題と反例も明記 |
| `timing.py` | 段階別の実行時間 (family / newton / resolve / certify …) の記録と集計 |
| `rlct_cache.py` | 解いた局所データ (k, h, f_rest) → λ の永続キャッシュ。部分木の値も貯め、コード指紋・依存関係・上界検査で誤りの伝搬を防ぐ |
| `invariants.py` | 重複度、擬斉次性、Milnor 数・Tjurina 数 (Singular 併用) |
| `lean_export.py` | chart 木の証明義務を Lean 4 に書き出す |
| `nn_rlct.py` | 一般のモデルと 1 隠れ層 ReLU ネットの局所 RLCT |
| `torch_rlct.py` | PyTorch モデルの読み込み (Transformer 1 層は `torch_transformer_rlct`) |
| `train_rlct.py` | 分類データで MLP / CNN / ViT を学習し、学習済み θ* の局所 RLCT を計算する CLI |
| `transformer_rlct.py` | Transformer 1 層 (Self-Attention + 残差 + MLP(ReLU) + 残差)。softmax は分母を払って多項式化 |
| `taylor_sweep.py` | softmax を n 次で打ち切ったときの λ を n = 0,1,2,3… と並べ、**止まるかどうか**を判定する CLI |
| `cnn_sweep.py` | 数層 CNN の学習解の λ を **層数 × 活性化 (relu/tanh) × テイラー次数 × タスク** 別にまとめる CLI |
| `llc_estimate.py` | SGLD による LLC (局所学習係数) の数値推定。局所化 γ の偏りの閉形式つき |
| `llc_compare.py` | 同じ θ* で**厳密 λ と λ̂ を突き合わせる**。特異なトイモデル (深層線形網) 込み |
| `resnet_rlct.py` | **ResNet18/50 の局所 RLCT**。BN を畳み込みに融合し、「ずれ」だけを疎に運ぶ |
| `train_resnet.py` | ResNet を CIFAR 等で学習し、学習済み θ* の RLCT と λ̂ を出す CLI (GPU 想定) |
| `dln_oracle.py` | **深層線形網のオラクル**。Aoyagi の閉形式 (λ, θ) と、イデアルを DLN と**確認できた場合だけ**解消を飛ばす経路 (detect → reconstruct → compare)。70〜5326x |
| `bench/dln.py` | 上の閉形式を bench 向けに再公開。Theorem 2 と 657 件で検算済み |
| `bench/agreement.py` | **既知の閉形式との一致表** (正則点・単項式・斉次・超平面・Vandermonde・行列分解・深層線形網) |

## 回帰テスト

```sh
cd bench && export PYTHONPATH=..:.
python regression.py                  # guards + structured + nn
python regression.py --all            # ランダム多項式・難しい族・失敗例も
python regression.py --quick          # 小さい範囲だけ
```

終了コードは、guards に赤があるか独立な値との食い違いがあれば 1。
