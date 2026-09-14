# RLCT toolkit / 実対数閾値ツール群

Resolution of singularities for computing real log canonical thresholds (RLCT),
with machine-checked certificates.

- 日本語ドキュメント: [DOCUMENTATION.ja.md](DOCUMENTATION.ja.md)
- English documentation: [DOCUMENTATION.en.md](DOCUMENTATION.en.md)

## Files

| File | Purpose |
|---|---|
| `resolve_singularity.py` | Blow-up + renormalization; weighted blow-ups, branch & bound, Graphviz tree |
| `certify.py` | Three-valued certificate (proved/unknown/refuted), SMT checks, sound intervals |
| `families.py` | Closed-form RLCT for monomial / product / sum / power-of-form / nondegenerate families |
| `invariants.py` | Multiplicity, quasi-homogeneity, Milnor and Tjurina numbers |
| `lean_export.py` | Lean 4 proof obligations for the chart tree |
| `nn_rlct.py` | Local RLCT of general models and 1-hidden-layer ReLU networks |
| `torch_rlct.py` | PyTorch model import |
| `bench/` | Cases, oracles, guards, failure archive, reporting |

```sh
pip install sympy scipy pulp z3-solver graphviz
python certify.py          # demo
cd bench && PYTHONPATH=..:. python guards.py
```
