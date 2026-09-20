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

## 本体

| ファイル | 役割 |
|---|---|
| `resolve_singularity.py` | ブローアップ + 正規化。重み付きブローアップ、分枝限定、Graphviz の木 |
| `ideal_resolve.py` | イデアルを運ぶ解消 (既定で併用)。生成元を座標に取れる |
| `certify.py` | 三値の証明書 (proved/unknown/refuted)、SMT による正規交差判定、健全な区間 |
| `families.py` | 族ごとの閉じた式 (単項式/積/和/形式のべき/ニュートン非退化) |
| `invariants.py` | 重複度、擬斉次性、Milnor 数・Tjurina 数 (Singular 併用) |
| `lean_export.py` | chart 木の証明義務を Lean 4 に書き出す |
| `nn_rlct.py` | 一般のモデルと 1 隠れ層 ReLU ネットの局所 RLCT |
| `torch_rlct.py` | PyTorch モデルの読み込み |

## 回帰テスト

```sh
cd bench && export PYTHONPATH=..:.
python regression.py                  # guards + structured + nn
python regression.py --all            # ランダム多項式・難しい族・失敗例も
python regression.py --quick          # 小さい範囲だけ
```

終了コードは、guards に赤があるか独立な値との食い違いがあれば 1。
