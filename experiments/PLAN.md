# Experiment: 1D SI smoke test for deformable bodies

First numerical test of the sequential-impulses retrofit for DBD. Pure Python, single file, NumPy + matplotlib only. Lives at `experiments/si_1d_stack.py`.

## Goal

Show that a stack of stiff-but-compliant 1D bodies settles under gravity to a physically correct compressed equilibrium, using a local-implicit SI solver, and that the inner Newton's energy leakage is measurable and bounded. This is the smallest test that exercises the core DBD idea (per-body deformation DoF coupled to contact via local Newton) without any of the 2D/3D complications (convex collision, polar decomposition, inversion).

Rigid 1D dynamics is trivial, so any difficulty that shows up here is specifically from the extension to stiff compliant elasticity — which is exactly the part we need to de-risk.

## Model

**Body.** `(x, s, v_x, v_s)` — center position, half-extent, and their rates. Mass `m` on `x`, generalized mass `μ` on `s` (treat as `m/3` as a placeholder — the 1D analogue of affine inertia; tune later). Rest half-extent `s₀`. Internal elastic energy `½ k (s − s₀)²`, so elastic force on `s` is `−k(s − s₀)`.

**Gravity.** Acts on `x` only (1D world is vertical; `y ≡ x`).

**Contact.** Unilateral between adjacent bodies: gap `g = x_{i+1} − s_{i+1} − x_i − s_i ≥ 0`. Jacobian has entries in all four DoFs of the pair: `∂g/∂x_i = −1`, `∂g/∂s_i = −1`, `∂g/∂x_{i+1} = +1`, `∂g/∂s_{i+1} = −1`. Floor is an infinite-mass, infinite-stiffness body at `x = 0` with `s = 0` fixed.

**Material assumption.** Isochoric — for display only, the transverse extent is `s₀² / s` so visible area is constant. No transverse physics.

## Solver

Outer loop: symplectic Euler on `(x, s)` with gravity and elastic force applied explicitly, then SI contact sweeps.

Inner per-contact local Newton: for each active contact between bodies `i, i+1`, solve the 5-DoF system `(Δx_i, Δs_i, Δx_{i+1}, Δs_{i+1}, λ)` that enforces the contact constraint at the end of the step while respecting the local mass + elastic Hessian. Gauss-Newton with PSD regularization (`εI` on the affine block), 1–2 inner iterations, warm-started multiplier from previous substep.

Floor contacts collapse to 3-DoF `(Δx_1, Δs_1, λ)`.

SI sweeps iterate until `max(|residual|, |Δλ|) < tol` or a hard cap (e.g. 50). Residual per sweep is logged.

Baseline for comparison inside the same file: **naive explicit-elastic SI** — same outer loop, but the contact impulse uses the rigid Jacobian (no `s` coupling), and `s` is updated purely by the explicit elastic force. This is the "obvious wrong thing" that the local-implicit version should beat.

## Tests

1. **Unit test — single body on floor.** Analytic equilibrium: `k(s₀ − s_eq) = mg`, so `s_eq = s₀ − mg/k`. Run to steady state, check final `s` matches within tolerance. This is the sanity check before trusting the stack.

2. **Main experiment — 5-body stack.** Equal bodies, resting on floor under gravity, initial configuration just-touching with zero velocity. Run to steady state.

3. **Stiffness sweep.** Grid over `k ∈ {very soft … near-rigid}` (probably 5–6 values spanning several orders of magnitude). For each `k`, record: final compression profile, energy drift curve over time, SI sweep count per step.

## Diagnostics

- **Energy.** Track KE, gravitational PE, elastic PE, total. Plot all four vs time. Expected: monotone drift of total (the inner Newton leaks).
- **Convergence.** Residual vs sweep index, per step. Plot as heatmap (step × sweep) or as overlaid curves for a few representative steps.
- **Equilibrium correctness.** Final `s_i` for each body vs analytic prediction from the static problem (each body carries the weight of everything above it; `k(s₀ − s_i) = (N − i + ½) mg` for body `i` counted from the bottom, accounting for the body's own weight).

## Visualization

Matplotlib. Two outputs:

- **Animation.** Stack drawn as ellipses, 1D axis vertical, transverse extent from the isochoric rule. Gravity pulls them down; watch them compress and settle.
- **Figures.** (a) Final stack snapshots across the stiffness sweep, side by side. (b) Energy curves overlaid, colored by `k`. (c) Convergence plot for a representative run. (d) Final-compression-vs-analytic-prediction plot.

## File structure

Single file `experiments/si_1d_stack.py`:

```
# constants, body dataclass
# energy functions (KE, PE_grav, PE_elastic)
# analytic equilibrium helpers (for unit test + main test checks)
# outer integrator (symplectic Euler on x, s)
# local Newton: 5-DoF body-body, 3-DoF body-floor
# SI sweep driver with residual logging
# naive-baseline solver for comparison
# test 1: single body on floor
# test 2: 5-body stack
# test 3: stiffness sweep
# visualization helpers (animate, plot_energy, plot_convergence, plot_sweep)
# __main__: CLI with flags to pick which test to run
```

Total target: ~500–700 lines. If it grows past that, split — but not before.

## Success criteria

- Unit test: single-body final `s` within 0.1% of analytic value.
- 5-stack: final `s_i` within a few percent of analytic prediction across the full stiffness sweep.
- Energy drift: bounded and monotone, scales gracefully with `k` (i.e. no blow-up at stiff end).
- Naive baseline visibly fails at stiff end (either oscillates, blows up, or settles to wrong equilibrium) — this is what motivates the local-implicit approach.
- SI sweep count stays bounded (doesn't explode as `k → rigid`).

## Out of scope

- Inversion / collapse (`s < 0`). Tune stiffness and mass so it doesn't happen.
- Friction (1D — no tangent).
- Restitution / Newton's cradle. Deferred per user request.
- Plasticity. Deferred.
- Substepping. Start with 1 substep; add if stability demands it.
- Mass ratio stress tests. All bodies identical in the first pass.

## Next steps after this works

- Cradle analogue (bodies separated, incoming velocity on one end). Tests in-sweep impulse propagation.
- 2D port using the same local-Newton structure, but with 6-DoF bodies and real contact Jacobians via GJK/SAT. That's the bridge into the C solver2d code.
