"""closers — 族ごとに閉じる経路。

各 closer の契約:
  入力  chart / イデアル
  出力  (lambda, m, 証明書) で閉じる、または None (閉じられない)
  健全性の向き  **証明書を出せないものは proved を返さない**

closer は判定器や解消器を import しない (循環になる)。部分問題の解き手は
`install_subsolver()` で上の層から注入される。
"""
