# Stage 1: exact (λ, m) per training stage versus numerical estimation (2026-10-13)

Code: `stage_dln.py`, `stage_mlp.py`, `stage_report.py`. Tables are in `summary.md`, the figure is `stage1.png`.

## Definition
A plateau point θ_k is a saddle, so the ordinary local λ is undefined there. We define
**λ_k := the local RLCT of the realizable problem that takes θ_k itself as the truth**
(K_k(w) = E‖f(x;w) − f(x;θ_k)‖²/2). This is the same quantity that `llc_estimate.realizable_targets` estimates.

## Results

### Deep linear networks (exact = Aoyagi, strata formula = independent implementation; the two agree at every stage)
| Network | Stages k | λ_k (exact) | m_k (exact) | HMC at t=3·10⁴ | m fit |
|---|---|---|---|---|---|
| (3,3,3) | 0,1,2,3 | 7/2, 4, 9/2, 9/2 | 2,1,2,1 | 3.37, 3.98, 4.41, 4.52 | 2.36, 1.37, 2.02, 1.0 |
| (3,3,3,3) | 0,1,2,3 | 3, 4, 9/2, 9/2 | 1,3,3,1 | 2.89, 3.77, 4.30, 4.51 | **2.42**, 3.67, 3.36, 1.0 |
| (3,3,3), data rank q=2 | 0,1,2 | 5/2, 3, 3 | 1,2,1 | 2.48, 2.91, 3.03 | 1.42, 2.06, 1.02 |

- Training passes through the predicted strata in order: all layer ranks equal k, and the distance to the stratum is ≤ 3.4·10⁻².
- **The final stage does not raise λ; only m drops** ((3,3,3): 9/2 with m 2 → 9/2 with m 1). HMC reproduces this difference too.
- **Data rank (L2) confirmed:** λ is obtained by replacing the input width with q. The number of stages goes from 4 to 3, and **the m pattern changes too** (2,1,2,1 → 1,2,1).
- In every stage except the last, λ_k > ρ_k/2 (Fisher rank / 2). For example, at k=1 of (3,3,3): 4 versus 5/2. **The core contribution is nonzero, and HMC measures it.**
- Reading m numerically: fit λ − λ̂(t) = (m−1)/(log t − r). This model was checked against exact integrals (Z ∝ t^{−λ} P(log t), deg P = m−1). **The fit is consistent in 10 of 11 DLN stages; it fails at the origin of (3,3,3,3) (m=1 → 2.42).** It is a supporting tool, not a decisive one.

### 2-layer tanh network (N=M=1, no bias)
- **Stage recursion (conjecture): λ_k = k + λ_V(H−k), m_k = m_V(H−k)**, where λ_V is Aoyagi's Vandermonde singularity at the origin. This has the same structure as the DLN identity λ(H,r) = ρ/2 + λ(H−r,0). An elimination argument supports it (after removing the learned units' 2k tangent directions, the core of the dead units is the origin singularity).
- Symbolic (RosolveSingular): λ for H−k = 1 matches the conjecture (H=2,k=1 → 3/2; H=3,k=2 → 5/2).
- **Solver bug:** m depends on the variable order and on the number of redundant generators (for the same ideal, (u0,u1,v0,v1) gives m=1 and (u0,v0,u1,v1) gives m=2; with K=3 generators m=1, with K=4–6 m=2). λ is stable. The solver's m must not be trusted until its status is 'proved'.
- Training: units activate one at a time (0 → 1 → 2), the b values stay distinct, and there is no condensation. Training does pass through the "k learned + rest dead" strata.
- HMC:
  - Regular points match after the closed-form γ correction (H=2: 2.01 for 2; H=3: 2.94 for 3).
  - Origin (Aoyagi's theorem): the **polynomial model with the same local ideal approaches from below** (0.615 for 2/3). The tanh network overshoots (0.80 for 2/3). The cause is saturation: with x~N(0,3²), the λ=1 planes of W₀ carry most of the prior mass and dominate the pre-asymptotic regime. (Ideals with more moments were checked and do not increase λ.)
  - Intermediate stages: the measured value falls short of the conjecture (H=2,k=1: 1.26 for 3/2). **The exact finite-t curve of the effective core, computed by quadrature, matches HMC**: 1.04/1.12/1.24 against HMC 1.07/1.11/1.26 at t=10³/10⁴/10⁵. The core's first-order coefficient c≈0.06 is small, so up to large t the core looks like a deeper singularity (≈⟨ab³⟩). That makes the convergence to 1/2 extremely slow (about half the asymptotic value at t=10⁵).
  - The H=3 polynomial model fails even at its regular point (1.83 for 3; mixing failure from stiffness). H=3 polynomial results are **unreliable**.

### SGLD (`llc_estimate.llc_sgld`)
On the DLNs it is biased upward by +0.2 to +2.0 and the chains drift heavily (step size c_eps=0.3, γ=1). Mixing along the flat gauge directions (the GL orbits) is the bottleneck. A small step size does not mix; a large one adds discretization bias. In the stage-1 comparisons we used it only as a reference value.

## What this establishes
1. In DLNs, the stagewise change of LLC (λ_k, m_k) **agrees exactly with three paths: the symmetry strata (the rank profiles of the GL orbits), Aoyagi's formula, and numerics (HMC)**, including m and including changes in data rank.
2. At the stages of nonlinear (tanh) networks, the exact λ fits "regular part + origin singularity of the dead subnetwork" (a conjecture), but **at finite t the measured LLC follows the effective normal form (the deeper singularity), not the asymptotic value**. This is quantitatively explained by an exact finite-t curve. Being able to predict an LLC estimate's shortfall from the singularity structure is itself a claim of this work.

## Next steps
- Prove the stage recursion for tanh (elimination plus unit factors) and fix the solver's m.
- Fix HMC stiffness in the H=3 polynomial model (preconditioning or Riemannian HMC).
- Stage 2: classification (logit-difference ideal) and small CNN / 1-layer Transformer.
