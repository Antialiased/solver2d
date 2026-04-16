# 1D solver/integrator validation plan

Five probes in `experiments/si_1d_stack.py` that close open design questions
for the 2D TGS-Soft-based deformable solver. Prioritised; #1 and #3 are the
highest leverage.

Context: the existing file already has `local_implicit` (true backward Euler
on the affine `s`-DoF), kinematic ceiling, per-body `k_vec`/`m_vec`, plastic
hardening, and a cubic-force (`beta3`) nonlinear material. See
`memory/project_dbd_backward_euler_robust.md` for the stability baseline and
the `robustness` test suite (`fast_impact`, `mixed_ratios`, `convergence`,
`mass_ratio`, `nonlinear`) for what's already in place.

---

## 1. Warm-start benefit on PGS convergence

**Closes:** the open finding from `test_convergence_scaling` — with
`warm_start=False` the sweep count saturates at `max_sweeps` for k ≥ 1e4.

**Setup:** re-run the existing `test_convergence_scaling` crush sweep
(k ∈ [1e2, 1e12], 5-body stack, kinematic ceiling crush) with
`warm_start=True`. Also repeat the single-step residual-decay panel.

**Metrics:**
- mean sweeps/step vs k (warm vs cold)
- per-step residual decay curves (warm vs cold) at k ∈ {1e3, 1e6, 1e9}

**Success:** warm-started sweep count stays bounded (say, ≤50) across the
full k range, or at worst degrades logarithmically.

**Why it matters for 2D:** if warm-starting is the decisive factor, the 2D
solver must carry per-contact multiplier state across steps (already Box2D
standard; just confirming). If not, something subtler is going on and we
need to investigate before production.

---

## 2. Substep vs inner-iteration trade-off

**Closes:** the core design bet of TGS-Soft — that substepping beats more
PGS sweeps at equal compute cost.

**Setup:** this is the most work because the prototype has no substep loop.
Add a thin outer loop that calls the existing per-dt update with `dt/K` for
`K` substeps. Then for a fixed total compute budget
`substeps × max_sweeps = const`, sweep the allocation:
`(K, max_sweeps) ∈ {(1, 200), (2, 100), (4, 50), (8, 25), (16, 12)}`.

**Scenarios:** the existing cradle impact test (high-frequency, stiff
regime) and the ceiling crush test (sustained contact, plasticity active).
These stress different parts of the pipeline.

**Metrics:**
- final-state error vs analytic/high-resolution reference
- max gap violation during impact
- energy drift
- wall time per scenario (rough — it's Python, but relative numbers hold)

**Success:** clear Pareto frontier showing that substepping dominates at
equal cost on at least one of the two scenarios. Characterize where the
optimum sits.

**Why it matters for 2D:** this is *the* TGS-Soft design decision. If
substepping doesn't help in 1D we should question why we're picking it as
the 2D base.

---

## 3. Long-chain PGS scaling

**Closes:** how deep a stack can the technique afford before PGS
convergence degrades unacceptably.

**Setup:** existing stack settle + ceiling crush with `N ∈ {5, 10, 20, 50,
100, 200}`. Fix `dt`, `k`, `damping`. Warm-start on (use #1's result).
Measure:

- mean sweeps/step over the settle-and-hold phase
- max gap violation anywhere in the chain
- time-to-equilibrium (settling time)

**Scaling hypothesis:** PGS is O(N) per sweep, and Gauss-Seidel takes ~O(N)
sweeps to propagate a pressure wave through the chain, so we expect O(N²)
work per step. Plot log-log; look for the exponent.

**Success:** clean empirical scaling law that extrapolates to an N we can
afford in 2D. If it's worse than N², we learn we need a better solver
(direct, multigrid, or domain decomposition) for deep stacks.

**Why it matters for 2D:** sets the practical budget for tall stacks
(towers, piles) and informs whether PGS alone is enough or whether we need
an NGS-style position pass.

---

## 4. dt refinement / temporal convergence order

**Closes:** sanity check that the pipeline is actually first-order in dt
(backward Euler's theoretical rate) and that contact projection doesn't
degrade that.

**Setup:** a deterministic non-trivial scenario — a 2-body cradle impact
is a good candidate — run at `dt ∈ {2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4}`.
Compare to a very-high-resolution reference (`dt = 1e-5`). Measure a
scalar observable (e.g. body-0 position at t=0.1 s, or peak contact
impulse).

**Metrics:** error vs dt on a log-log plot; fit slope.

**Success:** slope ≈ 1 (first-order) for backward Euler. Worse → bug or
contact-projection error dominating.

**Why it matters for 2D:** if 1D is second-order unexpectedly, we're
benefiting from a symmetry we might lose. If it's worse than first-order,
there's a structural issue to fix before scaling up.

---

## 5. Affine inertia (μ_frac) sensitivity

**Closes:** `mu_frac = 1/3` is inherited from the DBD literature without
strong justification. Is it load-bearing?

**Setup:** crush test with `mu_frac ∈ {0.01, 0.1, 0.33, 1.0, 3.0}`. Fix
everything else.

**Metrics:**
- stability (NaN check) across the full range
- mean sweeps/step
- final-state error vs analytic equilibrium
- max Newton iters (for nonlinear variant with `beta3 > 0`)

**Success:** either (a) results are insensitive over the range, meaning
`mu_frac` is a free parameter we can pick for convenience, or (b) there's
a clear optimum, in which case we learn what it is and why.

**Why it matters for 2D:** in 2D every affine DoF carries its own inertia
allocation. Knowing whether `mu_frac` is critical or arbitrary affects
whether we need to expose it per-material or hardcode it.

---

## Running order recommendation

Do #1 first (quick, closes an open loop from earlier tests). Then #3
(uses #1's warm-start finding). Then #4 (cheap sanity check). Then #5
(parameter sweep, no new machinery). Leave #2 for last — it's the biggest
build because of the substep loop, but it's also the highest-value
structural validation if the earlier tests don't surface anything
blocking.
