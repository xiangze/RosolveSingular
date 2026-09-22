# Real Log Canonical Thresholds via Resolution of Singularities

A toolkit that resolves singularities of polynomials to compute the real log
canonical threshold (RLCT) λ and its multiplicity m, **and produces a
certificate that the answer is correct**. Aimed at computing and validating
learning coefficients in singular learning theory (SLT).

---

## 1. Architecture

```
                 f (polynomial) and variables
                          |
        +-----------------+------------------+
        |                                    |
  families.py                         invariants.py
  closed forms per family             m, quasi-homogeneity, mu, tau
  (monomial/product/sum/form^p/       (triage and cross-checks)
   Newton-nondegenerate)                     |
        |                                    | feeds the weight rule
        | exact when it applies              v
        v                             resolve_singularity.py
  certify.rlct_certified  <----  iterated blow-up + renormalization
        |                        (weighted / branch & bound / Graphviz)
        v                                    |
  certify.py                                 |
  three-valued certificate  <----------------+
  (proved / unknown / refuted), SMT-checked
        |
        +--> lean_export.py   generates Lean 4 proof obligations
        +--> rlct_interval    sound interval [lo, hi] when not exact

  Applications: nn_rlct.py (general models / ReLU), torch_rlct.py (PyTorch)
  Validation:   bench/ (cases, oracles, guards, failure archive)
```

### Dependencies

| Package | Required | Purpose |
|---|---|---|
| `sympy` | **yes** | all polynomial algebra |
| `scipy` | recommended | fast in-loop LP (HiGHS) |
| `pulp` | recommended | LP/ILP via external solvers (CBC) |
| `z3-solver` | recommended | normal-crossing and nondegeneracy decisions (no certificates without it) |
| `graphviz` (+ `dot`) | optional | chart-tree rendering |
| `torch` | optional | importing PyTorch models |
| Singular | optional | Milnor/Tjurina numbers in a local ordering |

```sh
pip install sympy scipy pulp z3-solver graphviz
apt-get install singular graphviz     # optional
```

---

## 2. Quick start

```python
import sympy as sp
from certify import rlct_certified, rlct_interval

x, y, z = sp.symbols("x y z", real=True)

lam, status, route = rlct_certified(x**2 + y**3, (x, y))
# -> (5/6, 'proved', 'newton')

iv = rlct_interval(f, gens)      # sound interval when no exact value
iv.status, iv.lo, iv.hi
```

Every module runs a demo when executed directly:

```sh
python resolve_singularity.py    # resolution, branch & bound, Graphviz
python certify.py                # certificates and nondegeneracy
python families.py               # closed forms per family
python invariants.py             # m, mu, tau
python lean_export.py            # Lean 4 output
python nn_rlct.py                # local RLCT of ReLU networks
python torch_rlct.py             # PyTorch models
```

---

## 3. Theoretical framework

For the zeta function

    zeta(z) = ∫ |f(x)|^z φ(x) dx

resolution of singularities gives a proper map φ such that, in local
coordinates,

    f(φ(y)) = y^k · U(y) · u(y),      |det Dφ(y)| = |v(y)| · y^h

with U, u, v nonvanishing at the origin. Then

    λ = min over charts of min_{j : k_j > 0} (h_j + 1) / k_j
    m = number of coordinates attaining the minimum (max across charts)

The code tracks `(k, h, u, U, φ)` **exactly**. `U` accumulates unit factors
dropped along the way; without tracking it the identity simply fails (this was
an actual bug).

### Conditions required for λ to be correct

| | Condition | How it is handled |
|---|---|---|
| (a) | identity `f∘φ = y^k·U·u` | checked exactly in sympy by `certify`; Lean `chart_*` by `ring` |
| (b) | identity `det Dφ = y^h·v` | `jac_*` (n≤3); that the entries are partial derivatives is *not* proved |
| (c) | charts cover a neighbourhood (nothing missing) | Lean `cover_*` + `blowup_cover` |
| (d) | images stay inside the box (nothing extra) | Lean `image_*` + `blowup_image` |
| (e) | `u·v` normal crossing on the whole domain | decided by Z3 in `certify` |
| (f) | φ a diffeomorphism off a null set | coordinate changes checked by `inv_*`; partly unformalized |
| (g) | amplitude ψ ≥ 0, ψ(0) > 0 | assumed |
| (h) | the pole sits at `min (h+1)/k` | **Watanabe's theorem; assumed** |

`certify` returns `proved` when (a)(c)(d)(e) are established. (h) is not
formalized.

---

## 4. Modules

### 4.1 `resolve_singularity.py` — the resolver

```python
res = resolve_singularities(f, gens, prune=False, weighted=True, max_depth=20)
res.rlct, res.multiplicity, res.charts, res.nodes
res.print_report()
res.render_tree("tree", "png")    # chart case split as a Graphviz tree
res.to_lean("lean_out")
```

#### Algorithm

Maintaining `f = x^k · U · f_rest` and `|Jacobian| = unit · x^h` in each chart:

1. **Renormalize** — factor out the monomial gcd; strip factors that do not
   vanish at the origin (units) into `U`
2. **Stop test** — `f_rest(0) ≠ 0` means normal crossing form is reached
3. **Smooth-factor coordinate change** — if `f_rest` has a factor g smooth at
   the origin, linear in some variable not used by an exceptional divisor, take
   g itself as a coordinate. This prevents infinite iteration when the singular
   locus is positive dimensional (e.g. `(x-y)^2`)
4. **Blow-up** — centre `{x_j = 0 : j ∈ J}` where J is the smallest set such
   that every monomial of `f_rest` contains a variable of J (minimum hitting
   set, solved by branch & bound or ILP)
5. **Recentring** — translate to non-normal-crossing points on the exceptional
   divisor away from the origin
6. On failure, `ResolutionFailure` (`max_depth` / `no-progress` /
   `max_charts` / `prune-inconsistent`)

Monomial substitution is a linear map on exponent vectors, so no polynomial
expansion is needed: `Σx_i²` in 20 variables finishes in under a second.

#### Weighted blow-ups (`weighted=True`)

    x_i = s·y_i^{w_i},   x_j = y_i^{w_j}·y_j   (j ∈ J, j ≠ i)

Exponents map by `e_i' = Σ_{j∈J} w_j e_j`; the Jacobian is
`w_i · y_i^{(Σw_j) − 1}`. Weights are used **only when `f_rest` is
quasi-homogeneous** with respect to them (decided exactly by a linear system).
Using them otherwise makes the iteration diverge.

When `w_i` is even, `y_i ↦ y_i^{w_i}` cannot reach negative reals, so that
chart splits into two with sign `s = ±1`.

This resolves `x³+y⁴+z⁵` in 5 charts and 0.07 s; plain blow-ups never terminate
on it.

#### Per-chart Newton fast path (`newton_fast=True`, default)

The local zeta function of a chart is

    Z(z) = ∫ |y^k · f_rest(y)|^z · |y^h| dy      (near the chart's origin)

so if `P = y^k · f_rest` is **nondegenerate over ℝ with respect to its Newton
polyhedron** at the origin, Varchenko's theorem fixes λ exactly as an LP (the
Newton distance with amplitude `h`). In that case there is **no need to keep
blowing up until `f_rest` becomes a unit**. For each chart:

1. enumerate the compact faces of the Newton polyhedron (convex hull);
2. for each face σ, ask z3 whether `∃y, (∀i) y_i ≠ 0 ∧ ∇P_σ(y) = 0` is UNSAT;
3. if all faces are UNSAT, close the chart with `newton_rlct(P, h=h)` and
   `newton_multiplicity` (`status='resolved(newton)'`).

The value is adopted **only when nondegeneracy is proved**, so the answer never
changes; `bench/guards.py` runs 10 on/off agreement checks ("Newton 高速パス不変")
on every run.

The fast path is skipped on charts that have recentring candidates: like the
unit-termination case it fixes the value at the chart's **origin**, and other
points of the exceptional divisor belong to the recentred charts.

Charts with more than `newton_fast_max_terms` (60) monomials or
`newton_fast_max_vars` (24) variables are not tested — the hull and z3 cost
stops paying off. `newton_fast=False` restores the old behaviour.

`certify()` checks such a leaf by **re-deciding nondegeneracy and recomputing
the LP value and the multiplicity** rather than by the unit condition (it does
not trust the resolver's cache); a mismatch is `refuted`. Varchenko's theorem
itself is assumed, like (h) Watanabe's theorem, and is emitted as a `newton_*`
placeholder on the Lean side.

Measured on 54 random polynomials (n = 2..4, d = 3..5):

| | total | median | charts | resolved |
|---|---|---|---|---|
| `newton_fast=False` | 14.7s | 0.062s | 302 | 46/54 |
| `newton_fast=True` | 6.0s | 0.019s | 146 | 49/54 |

#### Branch and bound (`prune=True` / `'ties'`)

λ is a minimum over charts, so pruning is legitimate.

- **Upper bound (primal)**: the Newton polyhedron LP
  `max Σν_a s.t. Σν_a·a_j ≤ h_j+1` — valid at any node as a bound on the global
  minimum
- **Lower bound (dual)**: from the mediant inequality,
  `λ(subtree) ≥ min_j (h_j+1)/(k_j + M)`
- Prune when LB > UB. Strict inequality, so both λ and m are preserved
- `'ties'` also prunes subtrees equal to the incumbent (λ stays correct, m may
  become a lower bound)

**The budget M assumes the subtree resolves within the remaining `max_depth`**,
so use `prune=False` in proof mode. Inconsistencies raise
`ResolutionFailure(reason='prune-inconsistent')`.

#### Graphviz

`res.render_tree()` draws the case split. Each node carries the blow-up centre
and chart, the expression right after the substitution, the extracted monomial
and the expression after renormalization, the exponent vectors `k, h`, and λ/m
when resolved. Colours: resolved = green, pruned = grey, unresolved = red,
coordinate change = blue, recentring = orange. `ResolutionFailure` carries the
partial tree.

### 4.2 `certify.py` — three-valued certificate

```python
cert = certify(res, eps=1, timeout_ms=10000)
cert.status    # 'proved' | 'unknown' | 'refuted'
cert.rlct      # a value only when proved, otherwise None
cert.print_report()
```

#### The normal-crossing test

`u` vanishing is not itself a problem: if its zero set is smooth and transverse
to the exceptional divisors, normal crossing still holds. It fails exactly where
`grad u(p)` lies in the span of `{e_j : y_j(p) = 0, j ∈ supp(k)∪supp(h)}`,
which is quantifier-free:

    u = 0 ∧ (j ∉ E) ∂u/∂y_j = 0 ∧ (j ∈ E) [y_j = 0 ∨ ∂u/∂y_j = 0]

UNSAT from Z3 is a **proof** of normal crossing on the whole box.

Two localizations are essential:

- **Restriction to the fibre** (`localize=True`): the amplitude ψ is supported
  near the origin, so only points with `φ(p) = 0` matter. Normal crossing is an
  open condition and the fibre ∩ box is compact, so holding on the fibre implies
  holding on a neighbourhood
- **Territory of recentred charts**: a recentred chart owns only the δ-ball
  around its point; the parent owns the box minus that ball

#### Domain tracking

Root box `{|x_j| ≤ ε}`, propagated by the bounds the covering lemma provides:

| Step | Child box |
|---|---|
| blow-up chart i | `b'_i = b_i`, `b'_j = 1` (j ∈ J\{i}), rest inherited |
| coordinate change `x_v = (y_v−B)/A` | `b'_v = \|A\|max·b_v + \|B\|max` (interval arithmetic) |
| recentring | `δ` in every direction |

#### Multiplicity from the Newton polyhedron

```python
newton_multiplicity(f, gens, h=None)
```

If the diagonal meets the boundary at `p = t*(1,...,1)` and the minimal face
containing p has dimension d, then `m = n - d`, which equals
`rank{facet normals through p}`. **This gives the multiplicity a toric
resolution would produce, without subdividing any fan.**

#### Newton nondegeneracy

```python
newton_nondegenerate(f, gens)   # ('proved'|'refuted'|'unknown', face)
rlct_via_newton(f, gens)        # (λ, status)
```

For each compact face σ, decide whether `∃x (all x_i ≠ 0) ∧ ∇f_σ(x) = 0` is
UNSAT (Euler's relation makes `f_σ = 0` automatic). The test is run **directly
over the reals**, which matters: a polynomial can be degenerate over ℂ but
nondegenerate over ℝ, and the RLCT needs the latter. When nondegenerate,
Varchenko's theorem makes λ = 1/(Newton distance) exact.

#### When no exact value is available

```python
iv = rlct_interval(f, gens)     # sound interval [lo, hi]
```

- Lower: `1/m` (since `1/m ≤ lct_ℂ ≤ λ_ℝ`)
- Upper: `n/m`, the Newton LP value, the smallest value among charts resolved
  before the failure, and the value from a tie-pruned run (each is "a value
  attained by some chart", hence an upper bound)

If the interval collapses to a point the value is determined. Otherwise switch
to numerical estimation (SGLD) and use the interval to separate estimator bugs
from genuine difficulty.

### 4.2.5 `ideal_resolve.py` — ideal-carrying resolver (on by default)

```python
rlct_certified(f, gens, ideal=True, generators=[g1, g2, ...])   # default on
rlct_interval(f, gens, ideal=True)
resolve_ideal(generators, gens)                                  # direct
```

Carries the **generator set** instead of the polynomial `sum g_i^2`. At every
node it tries: linear reduction, common-monomial extraction (k += 2c),
**taking a generator as a coordinate** (impossible when only *factors* of the
polynomial are inspected), termination as soon as the ideal is monomial (exact
by Howald's LP), then blow-up.

Note: the split `lambda += 1/2` is only valid when k = h = 0, since
`f = x^k(v^2 + ...)` does not separate in v. The implementation therefore
performs the coordinate change only and keeps v as a generator.

Effect on Vandermonde cases (ground truth from Aoyagi's formula): H=2 cases go
from 1 (or failure) to the correct 3/4, 2/3, 1; H=3 still does not terminate.

**Not implemented**: phi is not tracked, so the result cannot be handed to
`certify` / `lean_export`. Whenever the ideal route's value is adopted the
status is therefore `unknown`, never `proved`. Use `ideal=False` for the
previous behaviour.

### 4.3 `families.py` — family-specific exact routes

Tried in order; the first that applies gives closed-form λ and m.

| Route | Condition | Value |
|---|---|---|
| monomial | `f = c·x^k` | `λ = min 1/k_j`, `m` = count attaining it |
| product | variable-disjoint product | `λ = min λ_i`, `m` = sum of `m_i` over minimizers |
| sum | variable-disjoint sum, each part nonneg | `λ = Σλ_i`, `m = Σm_i − (parts−1)` |
| power_form | `f = g^p`, g linear or quadratic form | linear: `1/p`; quadratic of rank r: definite `r/2p`, indefinite `min(1/p, r/2p)` |
| newton | nondegenerate over ℝ | `1/(Newton distance)`, `m` = rank of the facet normals through the diagonal point |

The product rule needs no sign hypothesis (`|f₁f₂| = |f₁||f₂|`); the sum rule
does need nonnegativity.

These double as **oracles**: the general route's answer is cross-checked
against them.

### 4.4 `invariants.py` — local invariants

```python
multiplicity(f, gens)                  # ord_0 f
quasihomogeneous_weights(f, gens)      # (weights w, degree d) or None
milnor_number(f, gens, backend="auto") # (mu, route)
tjurina_number(f, gens)                # (tau, route)
singularity_report(f, gens).print_report()
```

- Quasi-homogeneity is decided **exactly by a linear system**, not by an LP
  approximation. This is the justification for the resolver's weight rule
- μ, τ use Singular's **local ordering `ds`** when available, giving the value
  at the origin only; otherwise sympy's global Gröbner basis gives `Σ_p μ_p`
  (e.g. `x⁵+y⁵+x²y²`: local μ = 11, global 16)
- Non-isolated ⟹ μ = τ = `oo`. μ = 0 means "smooth at the origin"

**Caveat**: μ and τ are invariants over an algebraically closed field. A real
isolatedness test is not enough: `(x²+y²)²` has only the origin as a real
singular point but is non-isolated over ℂ, so μ = ∞. For cross-checking the
real RLCT, use `1/m ≤ λ ≤ n/m` rather than μ.

### 4.5 `lean_export.py` — Lean 4 proof obligations

```python
exp = res.to_lean("lean_out", certificate=cert)
exp.print_summary()
exp.verify()        # compiles if lake/lean is present
```

Files written: `LeanCover.lean` (hand-written general lemmas),
`Resolution.lean` (generated), `lakefile.lean`, `README.md`.

| Theorem | Statement | Tactic |
|---|---|---|
| `chart_*` | `f∘φ = y^k · unit · f_rest` | `ring` |
| `unit_*` | `f_rest(0) ≠ 0` | `norm_num` |
| `jac_*` | Jacobian determinant `= y^h · unit` (n≤3) | `Matrix.det_fin_*` + `ring` |
| `inv_*` | coordinate change is bijective (explicit inverse) | `ring` / `field_simp` |
| `cover_*` | chart family is surjective (nothing missing) | `blowup_cover` |
| `image_*` | image stays in the box (nothing extra) | `blowup_image` |

**Every identity is verified in sympy before being emitted**, so the module
works as a bookkeeping checker even without Lean (the demo deliberately
corrupts `k` and shows the failure).

**Not verified**: no Lean toolchain was available here, so the generated Lean
has never been compiled. The hand-written `blowup_cover` depends on Mathlib
lemma names and may need adjustment. The covering lemma for weighted blow-ups is
unformalized; those nodes are skipped and recorded as such.

### 4.6 `nn_rlct.py` — general models and ReLU networks

```python
local_rlct_from_ideal(generators, variables, symmetry_vectors=[...])
relu_local_rlct(X, A_star, b_star, c_star, d_star)
```

Three stages:

1. **Fix the activation cell** — pre-activation signs at θ\* make the model
   polynomial. Exact when θ\* is in the cell interior; on a boundary, all
   incident cells are enumerated and the minimum taken (no cone restriction, so
   the value is a lower bound)
2. **Quotient** — linear reduction of generators → gauge-fixing the scaling
   symmetry (a free action leaves λ and m unchanged and drops the dimension) →
   elimination of regular directions (`λ += 1/2` per variable removed) →
   removal of free directions
3. **Resolve the core ideal**

Observation: with θ\* in a cell interior the core is usually empty, i.e. the
point is regular. **The essential singularities of ReLU networks concentrate on
cell boundaries.**

### 4.7 `torch_rlct.py` — importing PyTorch models

```python
rep = torch_local_rlct(net, X, params=["2.weight"], zero_tol=1e-2)
rep.print_report()
```

- Walks a sequential model, expanding each parameter symbolically as `θ*+u` or
  as a constant
- Supported layers: Linear / ReLU / LeakyReLU / Identity / Flatten, plus Tanh /
  Sigmoid / SiLU / Softplus / GELU (Taylor expansion around the pre-activation)
- Scaling symmetries are extracted automatically — but **if any parameter it
  moves is frozen it is no longer a symmetry** and is not quotiented
- `zero_tol` rounds small weights to zero; without it a trained point is almost
  always regular
- `targets` is only used to verify θ\* is realizable; the fibre ideal is always
  built treating the model at θ\* as the truth

### 4.7.5 `timing.py` — phase-level timing

A small thread-local profiler used to relate runtime to polynomial /
network size and degree.

```python
from timing import record, timed, last_record, timing_summary

with record("case-1", n_vars=4, degree=6):
    with timed("resolve"):
        res = resolve_singularities(f, gens)
    with timed("certify"):
        cert = certify(res)
rec = last_record()
rec.phases      # elapsed per phase (nested included)
rec.exclusive   # self time only (no double counting)
rec.counts      # call counts
rec.as_dict()   # JSON-serialisable
```

Phase names are fixed: `family`, `newton`, `newton_chart`, `ideal`,
`resolve`, `certify`, `eliminate_regular`, `nn_reduce+resolve`, `lp`, `smt`.

`certify.rlct_certified()` opens a record internally, so
`timing.last_record()` gives the breakdown after a call:

```
>>> rlct_certified((x*y + z**2)**2, (x, y, z))
(1/2, 'proved', 'blowup+certify')
>>> print(timing.last_record())
rlct_certified: wall=0.248s [resolve=0.157s, ideal=0.046s, certify=0.022s,
                             newton=0.008s, newton_chart=0.003s]
```

`bench/run.py` and `bench/nn_cases.py` store `time` (whole case) and
`timing` (phase breakdown plus covariates) on every record;
`bench/report.py`, `nn_cases.summarize()` and `bench/regression.py`
aggregate them with `timing_summary()` by variable count, degree,
architecture and activation.

### 4.8 `bench/` — the improvement loop

```sh
cd bench && export PYTHONPATH=..:.
python guards.py                          # are the judges still strict? (gate)
python run.py --max-vars 4 --random --out r.json
python report.py r.json
python run.py --failures                  # re-test archived failures
python run.py --hard --montecarlo --covering   # hard families + independent checks
python regression.py --all                     # all suites at once (for CI)
```

| File | Role |
|---|---|
| `cases.py` | structured families and random ones (n, degree, #terms, coefficient bits varied independently) |
| `oracles.py` | closed forms, cross-settings agreement, the `1/m ≤ λ ≤ n/m` check |
| `guards.py` | negative tests, mutation tests, independent checks, Aoyagi lemmas, Vandermonde ground truth, **coordinate invariance**, plus known unfixed bugs (reported outside the gate) |
| `run.py` | execution and failure classification, JSON output |
| `report.py` | aggregation: class distribution, routes, proved rate by n and degree, correlation with μ |
| `failures.py` | failure archive and automatic minimization |
| `nn_cases.py` | neural-network grid (MLP x {relu, tanh} x degeneracy type), torch-free |
| `regression.py` | one-shot runner for all suites (guards / structured / random / hard / nn / failures) |
| `independent.py` | checks that share nothing with the resolver (Monte-Carlo λ from volume asymptotics, forward covering sampling) |
| `known_families.py` | families that are both hard and have independent ground truth (generic hyperplane arrangements, homogeneous forms, Vandermonde type) |
| `aoyagi_lemmas.py` | Aoyagi's lemmas (Entropy 2019) turned into truth-free checks: ideal invariance, monotonicity, separation, deepest point |
| `vandermonde.py` | Ground truth for Vandermonde matrix-type singularities: Aoyagi (2019) Theorem 6 (H<=3) and the exact N=1 formula (both lambda and the order theta) |

#### Failure classes and where to patch

| Class | Meaning | Where to patch |
|---|---|---|
| `resolve_max_depth` | resolution does not terminate | centre selection, weights |
| `smt_unknown` | decision procedure undecided | face decomposition, fibre constraints, timeout |
| `nc_refuted` | normal crossing fails somewhere | localization, domain split |
| `identity_refuted` | bookkeeping identity fails | the resolver |
| `covering_unknown` | covering not established | add lemmas |
| `inconsistent` | disagrees with an independent value | **investigate first** |

#### Working rules

Raising the proved rate can be achieved by fixing the judges *or by weakening
them*. To keep the two apart:

1. Always minimize a failure before patching
2. Never accept a patch unless `guards.py` is fully green
3. The headline metric is not the proved rate but the number of proved results
   **that agree with a known value**
4. Archive failures, minimize them automatically, and re-test with
   `run.py --failures` after each fix

---

## 5. Current state

| Set | proved |
|---|---|
| structured, 51 cases (n ≤ 6, d ≤ 13) | 51/51 |
| random, 111 cases (n ≤ 3, d ≤ 6) | 111/111 |
| random, 36 cases (n = 4–5, d = 4–8) | 22/36 (all failures are `resolve_max_depth`) |
| guards, 17 checks | 17/17 green |
| disagreements with independent values | 0 |

At n = 4,5 **every** failure is on the resolution side; none are on the proof
side. Raising `max_depth` to 30 does not help, so this is a limitation of the
centre selection rule, not of search depth.

---

## 6. Bugs found during development

This code returned wrong values of λ several times; each time a new check was
added.

| Bug | Symptom | Fix |
|---|---|---|
| missed non-normal-crossing points on the exceptional divisor | `(x−y)²` gave λ=1 (true 1/2) | added recentring |
| recentring too aggressive | considered points not mapping to the origin | added the `φ(p)=0` test |
| dropped units were not recorded | identity failed | track `Chart.unit` |
| certify never checked the identity | corrupting `k` still gave proved | added exact identity check |
| pruning budget tied to `max_depth` | silently returned λ=∞ | detect contradiction with the upper bound |
| charts that were resolved *and* recentred were discarded | λ could be overestimated | add them to `done` |
| used a real isolatedness test for the μ formula | `(x²+y²)²` gave μ=9 (true ∞) | use complex zero-dimensionality |
| CBC precision | pruned every chart, λ=∞ | 1e-6 relative slack on the upper bound |

The hard part of this problem is that **a wrong answer does not look wrong**.
The four defences that actually worked: comparison against known closed forms,
computing the same quantity along two independent routes, internal invariant
checks, and mutation testing.

---

## 6.5 Known unfixed bug: coordinate dependence

Rewriting the same function by an invertible change of coordinates with
Jacobian 1 changes the answer. For the Vandermonde case (M=N=1, H=3, Q=1),
the local problem in the first blow-up chart

    P = v^4 (G_1^2 + v^2 G_2^2 + v^4 G_3^2),  G_k = a_1 + a_2 b_2^k + a_3 b_3^k

gives lambda = 3/2 in the chart coordinates and 7/6 after taking G_1 as a
coordinate.

The cause: the resolver carries the **polynomial** sum g_i^2, not the
**ideal** <g_i>. Since lambda depends only on the ideal (Aoyagi Lemma 1(2)),
collapsing to a polynomial loses the ability to take a generator as a
coordinate. `_smooth_factor_change` only inspects *factors* of f_rest, so a
*summand* like G_1 can never be used.

The fix is to carry the generators in `Chart` and, at every node, try
(1) linear reduction with constant coefficients, (2) **elimination of regular
directions**, (3) removal of unit factors, (4) blow-up. Step (2) currently
runs only once, as preprocessing in `nn_rlct`; it needs to run at every node.

`known_issues()` in `guards.py` reports this mismatch on every run (outside
the green gate). Once fixed it turns OK and should be promoted to a normal
guard.

## 7. Known limitations

- **(h) is unformalized**: Watanabe's theorem that the pole sits at
  `min (h_j+1)/k_j` is assumed
- **Lean never compiled**: no toolchain in this environment
- **Weighted covering lemma unformalized**: `cover_*` is skipped for those nodes
- **Some `x³+y⁴+z⁵`-type cases**: solved by weighted blow-ups, but
  non-quasi-homogeneous examples at n = 4,5 still fail to terminate. The right
  generalization is toric resolution (regular subdivision of the dual fan)
- **Automatic minimization struggles on large inputs**: each probe is expensive
  and signal-based timeouts are delayed inside C-level calls
- **No independent check of covering completeness**: the "drop one chart"
  mutation is undetectable when the dropped chart was not the minimizer

---

## 8. Selected references

- Watanabe, *Algebraic Geometry and Statistical Learning Theory*, Cambridge, 2009
- Lin, *Algebraic Methods for Evaluating Integrals in Bayesian Statistics*, PhD thesis, 2011
- Howald, "Multiplier ideals of monomial ideals", TAMS 353 (2001)
- Varchenko (1976); Arnold–Gusein-Zade–Varchenko
- Aoyagi & Watanabe, IEICE Trans. 88(10), 2005 (Vandermonde matrix type singularities)
- Abramovich, Temkin, Włodarczyk, "Functorial embedded resolution via weighted blowings up" (arXiv:1906.07106)
- Bierstone & Milman, Invent. Math. 128 (1997); Encinas–Villamayor, Acta Math. 181 (1998)
- Blanco & Frühbis-Krüger (comparison of desingularization implementations)
- Lau, Furman, Wang, Murfet, Wei, "The Local Learning Coefficient" (arXiv:2308.12108)
- Şimşek et al., "Geometry of the Loss Landscape in Overparameterized Neural Networks", ICML 2021
- Grigsby, Lindsey, Rolnick, "Hidden symmetries of ReLU networks", ICML 2023
