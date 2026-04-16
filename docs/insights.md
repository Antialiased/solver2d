# Deformable body dynamics — insights

Running catalogue of what we have actually learned from experiments, as
distinct from what we plan to try (`docs/plans/`) or how the code is
organised (`docs/code_map.md`). Each entry: setup, finding, "so what" for
the 2D TGS-Soft-based deformable solver.

Companions: `docs/research.md` (theory, hypotheses, open questions),
`docs/followups.md` (downstream applications of DBD material state),
`docs/code_map.md` (file-by-file index of the codebase),
`docs/plans/` (active implementation and experiment plans).

All entries prior to 2026-04-15 are from the 1D prototype in
`experiments/si_1d_stack.py` (DBD-style affine `s`-DoF per body, PGS
contacts, kinematic ceiling driver).

---

## Integrator choice

### Semi-implicit Euler is not backward Euler — and the difference is a stability cliff

**Setup:** kinematic ceiling crush on a single body, `dt=5e-4`,
`μ_frac=0.1`, `k` swept from 1e2 to 1e14.

**Finding:** the original `local_implicit` branch was actually
symplectic Euler (`v += (F(x_old)/m) dt; x += v dt`), stable only for
`ωdt < 2`. At `k=1e7` the post-lift elastic relaxation blew up to
`s=-0.65`; `k=1e9` went NaN. A one-line rewrite to true backward Euler
on `u = s − s_rest`,

```
alpha = 1 + dt² k / μ
u_new = (u + dt v) / alpha
v_new = (v − (dt k / μ) u) / alpha
```

is stable for **all** `k ∈ [1e2, 1e14]` — thirteen orders of magnitude —
and degrades gracefully: at extreme stiffness `apos_s → 0` so contacts
can no longer move the s-DoF and the body behaves rigidly.

**So what:** default free-flight integrator for the affine DoFs in 2D is
true backward Euler. Any paper that says "implicit" without writing the
update rule is suspect — check whether `F` is evaluated at the old or
new state. The graceful rigid-limit degradation is the robustness
contract we want at production stiffness ratios.

### Backward Euler silently damps well-resolved stiff modes (L-stability tax)

**Setup:** cradle-style impact chain, `damping=0`, compared
`local_implicit` vs `exponential` free-flight integrators.

**Finding:** backward Euler is A-stable *and* L-stable — it damps stiff
modes toward zero every step. Great for robustness, but it also removes
energy from modes we wanted to preserve (the stiff wave in the cradle).
The exponential integrator is symplectic on the linear oscillator and
preserved the wave, at the cost of a hard stability cliff at `ωdt = π`
where `sin(ωdt)` flips sign and the update explodes.

**So what:** hybrid rule is the pragmatic choice — exponential when
`ωdt < 1` (well-resolved: accuracy beats damping), backward Euler when
`ωdt > 1` (unresolved: kill it rather than ring on it). Matches the
Michels/Desbrun advice. The precise per-cycle damping budget of
backward Euler in the well-resolved regime is still to be measured
(plan #8 in `plans/1d_material_richness.md`).

---

## Plasticity

### Perfect plasticity in a chain localises — default to isotropic hardening

**Setup:** 5-body uniform chain, `k=1e3`, `σ_Y=20`, crushed by
kinematic ceiling. Return-mapping plasticity with `H=0`.

**Finding:** exactly one body absorbs essentially all the plastic flow
(`s_p = −0.94`) while the other four stay near zero. The identity of
the winner is picked by PGS sweep-order asymmetry, but the underlying
phenomenon — strain localisation in a non-hardening plastic chain — is
physical, not numerical. Once a body yields, its elastic stress is
capped, it becomes the weakest link, and there is nothing to push the
deformation elsewhere.

Adding linear isotropic hardening (`σ_Y_eff = σ_Y + H·|ε_p_accum|`,
consistency `Δε_p = (|σ_tr| − σ_Y_eff)/(k + H)`) with `H=500` restored
uniform compaction (`s_p ≈ −0.148` for every body).

**So what:** in 2D any mesh/chain will exhibit the same pathology.
Default plasticity model must include at least some hardening. Perfect
plasticity stays valid as a single-body/single-element option but
should not be the default.

---

## Nonlinear material law

### Quartic-energy / cubic-force is strictly convex — scalar Newton converges without line search

**Setup:** added `F(u) = −(k u + β u³)` to the free-flight backward
Euler, `W = ½ k u² + ¼ β u⁴`. The Newton residual is
`G(u_new) = A u_new + B u_new³ − C` with `A = 1 + dt²k/μ`,
`B = dt²β/μ`, `C = u_old + dt v`. `G'(u) = A + 3B u² ≥ A > 0`.

**Finding:** Newton converges in 3–4 iterations from a linear warm
start (`u_new = C/A`) across `β ∈ {1e3, 1e4, 1e5}`. No line search
needed in 1D. The effective stiffness `k_eff(u) = k + 3β u²` is
everywhere ≥ k > 0, so the contact-sweep linearisation never sees a
negative tangent — no force-inversion pathology even under compression.
Per-step frozen linearisation of `k_eff` between free-flight and
contact sweeps is sufficient (one relinearisation per step).

**Caveat:** terminology matters. "Cubic-in-energy" `(c/3) u³` gives a
*quadratic* force with a zero at `u = −k/c` and therefore an inversion
pathology. We deliberately chose quartic-energy / cubic-force to avoid
this. When we write "cubic force" in plans, we mean the strictly
monotone kind.

**So what:** 1D says nonlinear elasticity is cheap in this pipeline —
no new global solver machinery needed, just a scalar Newton inside the
free-flight update. In 2D we expect the same for a Yeoh-style
polynomial strain-energy (no barrier term), provided we stay on
strictly convex pieces.

### Yeoh collapses to a spring spline in 1D

**Theorycraft (not yet empirically tested).** In 1D the left Cauchy-Green
invariant reduces to `Ī₁ = λ² + 2/λ`, so a Yeoh polynomial in `Ī₁ − 3`
becomes a scalar function of stretch — essentially a spring spline over
extension/compression. Gent adds a log-barrier at a lock-up stretch;
Hencky is quadratic in `log(λ)`. All three collapse to 1D splines,
which is why the quartic-force probe above is an honest stand-in for
"Yeoh-like in 1D." The distinction between these models only re-emerges
in 2D/3D where the deviatoric/volumetric split matters.

**So what:** our deviatoric-hardening plan for 2D (harden under
compression via the deviatoric part, not volumetric) is consistent
with Yeoh and friends. The volumetric part can stay near-linear; we do
not need a barrier to prevent collapse if the deviatoric law is stiff
enough.

---

## PGS convergence and contact

### Warm-starting is near-mandatory for PGS at high stiffness

**Setup:** `test_convergence_scaling`, 5-body crush, `k` swept from
1e2 to 1e12, cold-start contact multipliers.

**Finding:** sweep count saturates at `max_sweeps` (the cap) for
`k ≥ 1e4` — i.e. PGS is not converging, just running out the budget.
Quantitative warm-start comparison is pending (plan #1 in
`plans/1d_solver_validation.md`), but the cold-start failure is already
unambiguous.

**So what:** the 2D solver must carry per-contact multiplier state
across steps. This is already Box2D standard and inherited from TGS
Soft; this experiment confirms the requirement rather than discovering
it. If warm-starting is not the decisive factor when we actually
measure it, something subtler is going on.

### Alternating *mass* ratios hurt PGS more than alternating stiffness

**Setup:** 4-config crush sweep — uniform, k-alternating, m-alternating,
k+m alternating.

**Finding:** the m-alternating chain takes ~3× the PGS sweeps of the
uniform chain. k-alternating alone is much milder. Intuitively,
Gauss-Seidel has to propagate pressure through the chain once per
sweep, and the effective wave speed is governed by mass more than
stiffness under our per-step linearisation.

**So what:** stress-tests for a 2D PGS pipeline should include
alternating-mass scenes specifically (not just stiffness sweeps). The
worst case for convergence in deformable stacks is likely where a
light layer sits between heavy layers, not where a stiff layer sits
between compliant ones.

### Single-contact mass ratio is well-behaved to 1:1e6

**Setup:** two-body cradle impact, mass ratio from 1:1 to 1:1e6.

**Finding:** stable across the entire range; no special treatment
needed for a single heavy-vs-light contact.

**So what:** the hard convergence problems live in *chains*, not in
*pairs*. A pairwise robustness test is not a useful proxy for stack
behaviour.

---

## Impacts and self-barrier

### The affine s-DoF has no intrinsic inversion barrier

**Setup:** `test_fast_impact`, `v ∈ {1, 10, 100, 1e3, 1e4}` into the
floor.

**Finding:** the system is numerically stable at all tested speeds,
but the minimum `s` went to roughly `−0.28` at the high end — i.e. the
body inverted through its own rest configuration. Nothing in the
linear-elastic affine model prevents this; the quartic term helps but
does not rule it out either.

**So what:** for 2D, we cannot rely on the elastic law alone to keep
bodies from inverting under impulsive loading. Either (a) the substep
loop has to be fine enough that no single step crosses `u = 0`, (b) we
accept a compliant-contact layer that takes most of the impulse before
it reaches the body's internal DoFs, or (c) we add a genuine barrier
term for the subset of materials where inversion must be ruled out.
This is probably the biggest open question to carry into 2D.

---

## 1D solver validation probes (2026-04-15)

From `docs/plans/1d_solver_validation.md`. See
`experiments/si_1d_stack.py::test_warm_start_benefit`,
`test_long_chain_scaling`, `test_dt_refinement`,
`test_mu_frac_sensitivity`, `test_substep_tradeoff`.

### Warm-starting rescues a mid-stiffness band, not the whole range

**Setup:** `test_convergence_scaling` with both `warm_start=False` and
`True`. 5-body stack, kinematic ceiling crush, `k ∈ [1e2, 1e12]`,
`dt=5e-4`, `max_sweeps=300`.

**Finding:** cold PGS saturates at the sweep cap for `k ≥ 1e6`. Warm
start dramatically helps in the mid-stiffness band: at `k=1e6..1e7` it
drops to **42–87** sweeps/step (meeting the plan's ≤50 target at the
sweet spot). But warm-started PGS still saturates at `k ≥ 1e9`, and —
unexpectedly — is **worse** than cold at `k ≤ 1e3` (122 vs 10 sweeps
at `k=1e2`). The low-stiffness regression looks like stale multipliers
carried from the descent phase biasing the post-crush equilibrium.

**So what:** warm-starting contact multipliers is necessary for
production stiffness in 2D but is not a universal fix. Stale state at
active-set transitions hurts; we probably want to decay or clamp warm
multipliers when the contact has been inactive. At very high stiffness
(`k ≥ 1e9`) warm start alone isn't enough — needs pairing with NGS-style
position projection or a direct factorisation fallback.

### Long-chain PGS is sub-linear in N up to ~40, then knees up

**Setup:** `test_long_chain_scaling`. Gravity settle → ceiling crush on
`N ∈ {5, 10, 20, 40, 80}`, `k=1e5`, `dt=5e-4`, warm start on, `mu_frac=0.1`.

**Finding:** mean sweeps/step during the crush hold are essentially
flat at **47–51** for `N = 5..40`, then jump to **77** at `N=80`. Max
gap violation is machine precision (`< 4e-14`) across all N. Plan's
O(N²) work hypothesis is decisively wrong in this range — warm start
plus the compressible `s`-DoF absorbs pressure-wave propagation so
locally that most sweeps terminate on the same tolerance regardless of
chain length.

**So what:** for the 2D solver, sustained-contact sweep budgets set on
a 5-body test generalise surprisingly well to moderate stacks. Don't
budget for O(N²) *a priori* — measure. The knee near `N=80` hints that
for towers beyond ~100 bodies we start paying for pressure-wave
propagation and likely need a position-projection (NGS) pass on top.
Finding this for real 2D should be the first thing to check once
coupled contacts are in.

### Temporal convergence is cleanly first-order

**Setup:** `test_dt_refinement`. 2-body cradle impact, `k=1e4`,
`gap=0.05`, `v0=2.0`, reference `dt=1e-5`, observable `x_0(t=0.1)`.

**Finding:** log-log slope **1.047** on both bodies. Errors:
`dt=2e-3 → 1.1e-3`, `1e-3 → 5.4e-4`, `5e-4 → 2.6e-4`, `2.5e-4 → 1.3e-4`,
`1.25e-4 → 6.0e-5`.

**So what:** contact projection does not degrade the backward-Euler
temporal order through a frictionless impact. The pipeline is
first-order as advertised — good sanity check before moving to 2D.
Anything better than slope 1 in 2D would imply a symmetry we shouldn't
count on; anything worse is a bug to hunt down.

### `mu_frac` has a broad sweet-spot plateau, not a unique optimum

**Setup:** `test_mu_frac_sensitivity`. 5-body ceiling crush with
`mu_frac ∈ {0.01, 0.1, 0.33, 1.0, 3.0}`, linear and cubic (`β₃=1e5`)
force laws.

**Finding:** mean sweeps/step is U-shaped in `mu_frac`:

| mu_frac | sweeps (β=0) | sweeps (β=1e5) |
| ---     | ---          | ---            |
| 0.01    | 207          | 208            |
| 0.10    | 83           | 82             |
| 0.33    | 83           | 83             |
| 1.00    | 87           | 89             |
| 3.00    | 191          | 192            |

Extreme values (very light or very heavy affine inertia) **double** PGS
work. `[0.1, 1.0]` is a flat plateau. Nonlinearity is essentially free —
Newton converges in ≤3 iterations everywhere, max sweep cost unaffected.

**So what:** the literature's `1/3` sits in the middle of a plateau,
not at a unique minimum. We can pick `mu_frac` per material for
convenience (modelling anisotropic response, tuning internal-mode
frequency) as long as we stay inside `[0.1, 1.0]`. Outside that band we
pay 2× in solver iterations, so exposing the knob is fine but the
default should live in the plateau.

### Substep dominates inner sweeps at fixed compute on cradle impact

**Setup:** `test_substep_tradeoff`. Budget `K · max_sweeps = 200` held
fixed, `K ∈ {1, 2, 4, 8, 16}`. Scenario A: 2-body cradle impact, 0.1 s,
observable `x_0` vs `dt=1e-5` reference.

**Finding (cradle):** error at `x_0` halves every time `K` doubles —
clean first-order scaling in `dt`.

| K  | ms   | dx₀      | wall   |
| -- | ---- | -------- | ------ |
| 1  | 200  | 5.4e-4   | 40 ms  |
| 2  | 100  | 2.6e-4   | 49 ms  |
| 4  | 50   | 1.3e-4   | 58 ms  |
| 8  | 25   | 6.0e-5   | 79 ms  |
| 16 | 12   | 2.7e-5   | 124 ms |

Energy drift also decreases monotonically with `K`. K=16 is ~20× more
accurate than K=1 for ~3× the wall time — Pareto-dominant.

**Finding (crush, sustained contact):** richer picture. Scenario B is
the 5-body ceiling crush (`k=1e5`, `dt_macro=5e-4`, 1.5 s). The
substepping Pareto has an **interior optimum**, not a corner:

| K  | ms   | max|Δs|  | max gap | wall |
| -- | ---- | -------- | ------- | ---- |
| 1  | 200  | 2.6e-3   | 3e-15   | 4.7 s |
| 2  | 100  | 2.1e-3   | 1e-15   | 10.7 s |
| 4  | 50   | 1.2e-3   | 4e-13   | 13.0 s |
| 8  | 25   | **2.9e-4** | 7e-9 | 13.9 s |
| 16 | 12   | 5.5e-4   | 6e-7    | 15.2 s |

Accuracy improves monotonically through `K=8`, then **regresses** at
`K=16`: 12 inner sweeps are not enough to drive the gap residual to
convergence, penetration climbs from 7e-9 to 6e-7, and the `s`-error
bounces back. At the other end, `K=1,ms=200` is limited by dt truncation
(not contact resolution — penetration is machine precision).

**So what:** TGS-Soft's substep bet holds — but for sustained contact
there is a sweet spot past which substepping hurts because each inner
step starves on sweeps. Practical 2D rule: substep until the sweep
budget per step reaches the point where residuals still decay past the
active-set transitions (in this 1D probe that was around 25 sweeps).
Budgeting the inner PGS too tight trades accurate bulk kinematics for
penetration creep, which is visually much worse. Validate the 2D
version by watching both an `|Δ|`-against-reference error *and* the
max gap violation — the failure mode is the penetration going up before
the position error goes up.

### Frozen vs relinearized contact tangent at coarse dt (nonlinear materials)

**Setup:** `test_dt_refinement_nonlinear` in `experiments/si_1d_stack.py`.
2-body cradle impact, `k=1e4`, `v0=3`, `β₃ ∈ {0, 1e6, 1e7, 1e8}`,
`dt ∈ [2e-3, 6.25e-5]`, reference `dt=2e-5`. Two solver modes:
`local_implicit` (frozen tangent, `k_eff` computed once per step at
`u_post`) and a new `local_implicit_relin` that maintains a per-body
accumulator `f_s[i]` and locally Newton-solves the cubic BE
`A u + B u³ = C + (dt²/μ) f_s` on every contact impulse update,
refreshing `state.s`, `state.vs = (u − u_init)/dt`, and the live
`apos_s(u)` used for the next K computation. Prototype disables
contact warm-starting in relin mode (tangent-inconsistent with carried
multipliers).

**Finding — stale-linearization floor is real and relin removes it.**
For frozen at `β₃=1e8` (cubic term ≳ linear), halving `dt` from 2 ms
to 1 ms only drops the error from 6.6e-3 to 5.7e-3 — the start-of-step
`k_eff` is so wrong for the post-impact trajectory that refining `dt`
barely helps. Coarse-half log-log slope is `+0.40`; it only climbs
back toward 1.0 once `dt` is fine enough that integrator error catches
up with linearization error. Relin has `slope_coarse ≈ 0.88–0.96`
across all β — clean first-order throughout.

**Finding — energy-leak tax, consistent with the RBD ball-and-socket
analogue.** Relin has a ~2–3× larger error *prefactor* at every `dt`
for every `β`, including `β=0` where the two modes are analytically
equivalent (both collapse to the same linear closed form). The gap is
the per-iteration Newton reconstruction introducing a constant per-step
drift. Concretely at `dt=2ms`:

| β₃   | frozen   | relin    | ratio |
| ---- | -------- | -------- | ----- |
| 0    | 5.07e-3  | 1.27e-2  | 2.5×  |
| 1e6  | 5.28e-3  | 1.29e-2  | 2.4×  |
| 1e7  | 6.28e-3  | 1.44e-2  | 2.3×  |
| 1e8  | 6.59e-3  | 2.17e-2  | 3.3×  |

**So what:** per-iteration relinearization is a targeted fix for the
stale-tangent error at coarse `dt` with strong nonlinearity, exactly
matching a known RBD pattern (re-solving ball/socket joints per
iteration at large dt). The crossover is sharp: for frozen at `β₃=1e8,
dt=2ms` the error is 16× larger than relin, but for gentle
nonlinearity or well-resolved `dt` the constant-factor leak makes
frozen strictly better. In 2D, reserve a relin-style path for
materials and time steps where the frozen tangent is known to be in
the saturated regime (probably: large-strain hyperelastic + big
substep). For the default TGS-Soft pipeline, frozen is still the right
choice. The prototype also showed a subtler cost — warm-starting
contact multipliers becomes tangent-inconsistent once the Jacobian
moves between steps, which needs thinking through before relin is
production-ready.

**Also noted:** the `slope_fine ≈ 1.20` seen in both modes is
reference contamination — `dt_ref=2e-5` is only 3.2× finer than the
finest test `dt=6.25e-5`, so the reference carries its own
first-order error and the fine-end of the sweep can't separate from
it. True fine-regime slope is 1.0 for both.

---

## 1D material-richness probes (2026-04-15)

From `docs/plans/1d_material_richness.md`. See
`experiments/si_1d_stack.py::test_free_oscillation_drift`,
`test_asymmetric_cubic`, `test_cor_soft_contact`.

### Backward-Euler damping on a free oscillator is exactly log(1+(ωdt)²)/(2 dt)

**Setup:** single body, no contacts, `g=0`, `damping=0`, `s(0)=s0+0.1`,
`vs(0)=0`. Run `local_implicit` for ~30 natural periods (or 4000 steps,
whichever is longer). Sweep `ωdt ∈ {0.1, 0.3, 1.0, 3.0, 10.0}` by
varying `k` with `μ=1/3`. Measure amplitude decay by linear fit of
`log(KE + PE_e)` versus time. Exponential integrator runs alongside
as the energy-preserving baseline.

**Finding — theory holds to machine precision.** The measured decay
constant matches `λ = log(1 + ω² dt²)/(2 dt)` with `rel_err = 0.00%`
at every ωdt. The exponential integrator preserves energy to
~5×10⁻¹⁴ across 10000 steps. No surprises — BE's L-stability tax is
predictable and cleanly characterised.

**So what — the plan's "well under 1% per cycle for ωdt<1" target is
wildly optimistic.** Per-cycle amplitude loss = `1 − exp(−2π·λ/ω)`.
At `ωdt=0.1` that's 27% per cycle, at `ωdt=0.3` it's 63%, at `ωdt=1.0`
it's 89%. To get *under* 1% per cycle you need `ωdt < 0.003` — two
orders of magnitude finer than the plan expected. Consequence for the
2D solver: "well-resolved" for energy-conserving purposes means
`ωdt ≲ 3×10⁻³`. Anything coarser and BE is silently killing the modes
we want to preserve. This is the concrete basis for the hybrid
integrator threshold (fall back to the exponential/symplectic path
when the user wants to keep a stiff mode alive, e.g. cradle impact
chains), and the practical reason why substepping TGS-Soft is
load-bearing — it's not just about stability, it's about keeping ωdt
in the regime where BE damping is negligible.

### Piecewise cubic law (tension-soft / compression-hard) works cleanly

**Setup:** Added `Params.beta_comp` and `Params.beta_tens` fields
(None → fall back to `beta3`). Force law becomes
`F(u) = -(k u + β(u) u³)` with `β(u) = beta_comp` on `u<0` and
`beta_tens` on `u≥0`. C¹ at `u=0` because both pieces and their
derivatives vanish there. Inside the backward-Euler Newton loop,
`B = dt²·β(u_new)/μ` is re-evaluated each iteration on the current
sign of `u_new`. Same pattern in the `local_implicit_relin` closures.

**Finding — crossings of u=0 are not a numerical problem.** The
monotone cubic `A u + B(u) u³ = C` with `B ≥ 0` on both sides keeps
`G'(u) = A + 3 B(u) u² > 0` everywhere, including at the kink. Newton
converges in ≤5 iterations on a free oscillation starting at u=+0.15
through several crossings in both directions.

**Finding — asymmetric response is as expected in the free oscillator.**
Free oscillation from `u(0)=0.15, v(0)=0` (scenario A):

| case              | u_min reached | Newton iters |
| ----              | ------------- | ------------ |
| sym β=1e6         | −0.109        | 4            |
| comp-hard 1e7/0   | −0.058        | 4            |
| tens-hard 0/1e7   | −0.251        | 5            |

The comp-hard body squashes half as far as symmetric; the tens-hard
body squashes twice as far (the stiff tension branch drains energy
more slowly, leaving more to spend on the compression side).

**Finding — the drop test (scenario B) is diagnostic but degenerate.**
A single body dropped onto the floor from `h=0.5` under gravity stays
on the compression branch throughout (`u_min ≈ −0.023`), so comp-hard
and symmetric are indistinguishable. This is *correct* — the
asymmetry only matters when both branches are exercised. For the 2D
solver this means asymmetric materials are a ~free upgrade for
compression-dominated scenarios (piles, squashing) since the tension
branch is inert until the body actually stretches.

**So what:** the asymmetric law is cheap, robust, and semantically
what users want (real materials are much stiffer in compression).
Default to exposing (k, β_comp) with `β_tens = 0` for the 2D
materials; symmetric cubic becomes the special case where both are
set.

### Soft contact (ωₙ, ζ) gives predictable tunable COR with a dt-biased offset

**Setup:** Added `Params.contact_wn` (rad/s) and `Params.contact_zeta`.
Floor contact in `si_step` drives the gap toward a per-step target
`g_target = (g_free + 2Ωζ g_old)/α` where `Ω=dt ωₙ`,
`α=1+Ω²+2Ωζ`, derived by exact backward-Euler on the damped oscillator
ODE `μẍ + cẋ + kx = 0` with `k=μωₙ²`, `c=2μζωₙ`. `g_old` is snapshotted
before any free-flight work; `g_free` is the post-free-flight gap.
PGS update: `dλ = (g_target − g)/K`, clamped unilaterally to `λ ≥ 0`.
All `K_hard` and `M_eff` factors cancel in the target (the soft
formulation is purely geometric once you write it this way). Test:
rigid body (`k=1e10`, `μ_frac=1`) dropped with various `v_in` onto
the soft floor, no gravity, COR measured as `|v_out/v_in|` at release.

**Finding — predictability is good, bias is real.** At well-resolved
`ωₙ·dt=0.1`, measured COR matches theory `exp(−πζ/√(1−ζ²))` within
**14–19% relative error** across `ζ ∈ {0, 0.1, 0.3, 0.5, 0.7}`. COR
is exactly independent of `v_in ∈ {1, 5, 20}` — confirms the
soft-contact response is linear in the initial velocity, as the
spring-damper model predicts. Example (`ωₙ=200`, `ωₙdt=0.1`):

| ζ    | COR_meas | COR_theory |
| ---- | -------- | ---------- |
| 0.0  | 0.856    | 1.000      |
| 0.1  | 0.623    | 0.729      |
| 0.3  | 0.316    | 0.372      |
| 0.5  | 0.137    | 0.163      |
| 0.7  | 0.037    | 0.046      |

The measured COR is always *lower* than theory — BE discretisation
adds its own implicit damping on top of the physical ζ. The bias is
consistent across ζ (roughly a multiplicative factor), so users can
still treat `(ωₙ, ζ)` as monotone tuning knobs.

**Finding — bias vanishes with dt refinement.** Fixing `ωₙ=500`,
`ζ=0.3` (theory 0.372) and sweeping `dt`:

| dt        | ωₙ·dt   | COR_meas | |err|   |
| --------- | ------- | -------- | ------ |
| 2.0e-3    | 1.000   | 0.0933   | 0.279  |
| 1.0e-3    | 0.500   | 0.1737   | 0.199  |
| 5.0e-4    | 0.250   | 0.2485   | 0.124  |
| 2.5e-4    | 0.125   | 0.3044   | 0.068  |
| 1.25e-4   | 0.0625  | 0.3374   | 0.035  |

Error is roughly first-order in `dt`, approaching 3.5% absolute at
`ωₙ·dt≈0.06`. The plan's 5% target is reachable if `ωₙ·dt ≲ 0.05`.
At coarser `ωₙ·dt` the measured COR drops sharply — at `ωₙ·dt=2.5`
the body barely rebounds (COR ≈ 0.1 even at ζ=0).

**Finding — coarse ωₙ·dt destroys the damping distinction.** At
`ωₙ·dt=2.5` every ζ from 0 to 0.7 gives nearly the same COR (0.09 →
0.001), because the contact resolves in ~1 BE step and the physical
damping has no time to act. At `ωₙ·dt=0.5` the distinction is partial
(0.48 → 0.017). Only at `ωₙ·dt=0.1` does the full ζ sweep span the
theoretical range.

**So what:**

1. **The soft-contact parameterisation is usable for the 2D solver's
   user-facing COR knob.** Monotone in ζ, flat in v_in, deterministic
   — these are the qualitative properties that matter for gameplay
   tuning. The 10–20% offset from theory at `ωₙ·dt≈0.1` is a
   predictable empirical bias the user compensates by lowering ζ.

2. **The hybrid-threshold budget from probe 8 bites here too.** To
   match theory within 5%, the contact must be resolved with
   `ωₙ·dt ≲ 0.05`. Combined with probe 8's finding that BE damping
   on internal modes is non-negligible at `ωdt > 0.003`, the safe
   zone for TGS-Soft is extremely fine for the high-frequency
   constraints. The practical recipe for 2D is: pick `ωₙ` low enough
   that `ωₙ·dt < 0.1` at the default dt, then let ζ be the user
   bouncy-knob. If you need stiffer contact, substep.

3. **The pure-geometric target formulation is simpler than the
   Box2D CFM/ERP convention** and drops out of one line of BE algebra
   on the spring-damper ODE. Worth porting to the 2D code as
   `s2PrepareContacts_Soft`'s position bias, with the caveat that
   body-body contacts need both sides' gap contributions — but the
   structure is identical.

**Open gap:** body-body and ceiling contacts are still hard in the
prototype. For a full 2D port we need the same soft formulation on
every contact, and a consistent `g_old` snapshot for each. The 1D
probe only exercised floor contact because that's all the COR test
needed.

### Bouncy-ball sweep — elastic contact × integrator choice (2026-04-15)

**Setup:** `experiments/si_1d_stack.py::test_bouncy_balls`. Single body
dropped from `h0 = 1` under gravity onto a hard floor, no velocity
damping, coefficient of restitution `e = 1`. Sweep internal stiffness
`k ∈ {1e3, 1e4, 1e5, 1e6}` at `dt = 5e-4`, `μ=1/3`. Three solver
combinations:

- **BE + coupled** (`local_implicit`) — backward Euler on `s`, contact
  applies a velocity-reflection impulse coupled to both `x` and `s`.
- **Exp + coupled** (`exponential`) — harmonic-oscillator free-flight
  on `s`, same velocity reflection applied against the exp integrator's
  position/velocity response.
- **BE + rigid** (`naive`) — `s` decoupled from contact, velocity
  reflection acts only on `x`.

Restitution is implemented as a one-shot velocity-level impulse
`dλ = −(1+e)·v_rel / (avel_x + avel_s)` applied once per fresh impact
(detected via `lam_prev = 0` at step start). The PGS sweep for that
contact is *skipped* on the reflection step — attempting to run a
plastic position-level sweep afterward cancels the reflection because
the body sits above the floor post-impulse and the zero-gap target
pulls it back down.

**Finding 1 — BE + rigid is perfect.** Across all four stiffnesses,
`E_end / E_0 = 1.005` and `h_peak_last / h0 = 1.009` (tiny numerical
gain from BE's gravity integration). The ball bounces indefinitely at
unit height. This is the "ideal bouncy ball" reference.

**Finding 2 — BE + coupled loses energy, rate grows with ωdt.** The
internal mode takes part of the contact impulse; BE then damps the
resulting oscillation, draining mechanical energy. Rates:

| k     | ω_int·dt | E_end/E₀ | h_peak_last/h₀ |
| ----- | -------- | -------- | -------------- |
| 1e3   | 0.027    | 0.496    | 0.348          |
| 1e4   | 0.087    | 0.347    | 0.057          |
| 1e5   | 0.274    | 0.333    | 0.011          |
| 1e6   | 0.866    | 0.333    | 0.000          |

At `k=1e6` the ball has fully settled within 4 s (E is just the static
gravity PE of the body sitting on the floor, = 1/3 of drop PE). At
`k=1e3` half the energy is still alive. The softer internal mode
absorbs less impulse from each collision, so less energy gets routed
through the damped BE integrator.

**Finding 3 — Exp + coupled with my velocity-reflection is unstable
(energy pumping), but only because of the reflection, not the
integrator.** At `k=1e5` with `e=1` reflection, total energy after 4 s
is **4× the initial energy** and the ball climbs to `h≈4.9`. The
mechanism: my velocity reflection is applied to the post-free-flight
`(u, vs)` state, but the exponential integrator's rotation in phase
space means reflecting `vs` alone lands on a higher-energy orbit than
the pre-impact state, and each impact ratchets up.

**Counterpoint from the no-restitution run** (second figure,
`bouncy_balls_no_restitution.png`): when `restitution=0` and the ball
rebounds *only* via internal elastic release, Exp + coupled is
perfectly well-behaved and actually gives *more* bounces than BE
(14–18 vs 5–11). The exp integrator preserves the internal vibration
between impacts instead of letting BE damp it, so the "stored spring"
leaks back into bulk motion on subsequent contacts. Both coupled
modes still settle to `E = mg·s₀` (ball resting) within a few
seconds because the position-level plastic PGS zeros relative
velocity at every impact. So: internal elasticity alone gives
*approximate* bouncing via compress-release, but the plastic
position-PGS contact drains the bulk velocity every impact regardless
of whether the integrator is dissipative, and the ball settles.

**So what:**

1. **The "elastic contact" coupling is the load-bearing distinction**
   for bouncy-ball behaviour. Decoupling `s` from the contact
   (`naive`) gives perfect bouncing regardless of which integrator
   runs on the internal DoF. As soon as you let the contact impulse
   modify `s`, any damping on `s` bleeds energy (BE) and any
   phase-preserving rotation (exp) can pump energy.

2. **Exp-integrator + coupled contact is a dangerous combination** in
   the unilateral-bounce regime. The one-shot velocity reflection we
   used here makes it unstable, but even a multi-sweep PGS is not
   going to fix it: the integrator's rotation means that contact
   "impulse" and "position" are out of phase in a way that hard
   constraints don't handle correctly. Possible fixes: (a) apply the
   reflection before the exp free-flight rather than after, (b) treat
   the contact as soft with a matched-phase damping term, (c) use
   exp only for free flight and swap back to BE during contact.

3. **BE + coupled loses energy predictably** as a function of
   `ω_int · dt`. For 2D deformable bodies, the "bouncy" user knob
   will have to be either an explicit restitution with careful
   velocity-level bookkeeping, or soft-contact `(ωₙ, ζ)` as in probe 6
   with the `dt`-proportional bias understood. Do **not** try to use
   internal elastic rebound as the source of bounciness — BE damping
   kills it and exp pumping blows it up.

4. **For visualising "how stiff a material is" in 2D demos**, the
   BE+coupled result is the faithful one: a stiffer material
   (higher `k`) bounces *less*, not more, under backward Euler. This
   is counterintuitive but physically what a TGS-Soft 2D pile will
   do out of the box. Users will expect stiff → bouncy and instead
   get stiff → mushy. Planning note: the "stiffness" knob in the 2D
   API should probably be exposed as `(ωₙ, ζ)` for bounciness and a
   separate `(k, β)` for internal deformability, rather than
   conflating them.

### Two balls attracting in free space — internal elasticity DOES give restitution (once you turn off warm start) (2026-04-15)

**Setup:** `experiments/si_1d_stack.py::test_two_ball_attract`. Two
deformable bodies in free space at `x = ±1`, each pulled toward the
origin by a constant attractive gravity (per-body `g_vec = [-g, +g]`,
added as a new `Params` field). No floor, no explicit restitution, no
velocity hacks. Expected physical behaviour: bodies meet, compress,
separate via spring release, fly apart, gravity pulls them back,
repeat. `dt=5e-4`, `mu_frac=1/3`, sweep `k ∈ {1e3..1e6}` × three
solver modes × also a diagnostic variant with no gravity launched at
each other at ±2 m/s, spanning `k ∈ {1e4..1e8}` with BE, Exp, and
`local_implicit_relin`.

**First pass — bodies never separated, I blamed geometry.** With
`warm_start=True` (the default for body-body contacts), every
combination of mode × stiffness × gravity/no-gravity I tried gave
zero separations. The bodies met at the origin and stuck there,
oscillating internally. I initially diagnosed this as a geometric
obstruction — claim: the internal spring acts on `s` not `x`, so
it can't push COMs apart, blah blah. **That diagnosis was wrong.**

**Correct diagnosis — warm-started λ hides the separation
mechanism.** Dense single-step tracing around one oscillation cycle
shows exactly what's happening:

- Compression phase (steps 500–573, one half of a `Σ` oscillation):
  `gap_free = (β/A_BE)·(vs_Σ·dt + U)` is slightly negative. PGS
  accumulates λ from ~1000 to ~6800.
- Release phase (573–645): `gap_free` flips **positive** (my
  analytical prediction at `U = 0, vs_Σ > 0` was right). Each step
  PGS computes `dλ < 0` (tries to drain the accumulator) and the
  clamp `λ_new = max(0, λ + dλ)` does decrement λ. Over the
  ~72-step release phase, λ drops from 6801 down to 1381 — a
  discharge of 5420.

But the release phase ends before λ reaches zero. Next compression
cycle starts, λ builds back up. Over many cycles, λ oscillates
between ~1300 and ~6800, *pinned permanently above zero*, and the
unilateral constraint stays active the entire time. The bodies
never get their "free flight" window because the accumulated λ
from the previous half-cycle hasn't been fully unwound.

**Fix — disable warm starting on the body-body contact.** With
`warm_start=False`, λ resets to 0 at every step. The moment
`gap_free > 0` (first step of the release phase), PGS computes
`dλ < 0`, clamps at 0 → **the constraint instantly releases**, the
bodies fly apart under their current velocity, and gravity pulls
them back for the next collision.

| mode                     | separations (warm on) | separations (warm off) |
| ---                      | ---                   | ---                    |
| BE + coupled, k=1e3..1e6 | 0                     | 2–6                    |
| Exp + coupled, k=1e3..1e6| 0                     | 5–7                    |
| BE + rigid (naive)       | 0                     | 0                      |

The visualisation (`experiments/out/two_ball_attract.png`, top and
middle rows) clearly shows body 0 and body 1 oscillating about
`x=±0.5` with visible amplitude, and the inter-body gap spiking
positive (up to ~0.8) at each separation event. Exactly the
"meet, compress, push back, separate, gravity pulls back, repeat"
dynamics the user predicted.

**Why BE + rigid still doesn't bounce** (the negative control): with
`naive` the `s` DoF is decoupled from the contact Jacobian, so the
contact impulse only affects `x`. There's no internal elastic
storage path, so the first collision plastically zeros relative
velocity and there's nowhere for energy to go. Confirms the
s-coupling in the body-body Jacobian is load-bearing — this is the
correct 2D takeaway, just for a different reason than I thought.

**Energy decay is real but orthogonal** (bottom-row plot): each
collision still bleeds a fraction of the macroscopic KE to the
plastic position-level PGS (zeros relative velocity at impact).
After ~5–7 bounces the system has lost enough energy to rest at
the origin with `E = 0.5·E₀`. To get indefinite bouncing you'd
need either a velocity-level restitution impulse (explicit COR)
or a soft contact path — see the earlier floor-bouncy insight.
The key point here is that the rebound *mechanism* works; the
decay rate is just set by how plastic the contact projection is.

**Why warm-start pins λ, in words:** position-level PGS is an
accumulator, not a stateless solver. Across steps, λ is a physical
"stored impulse" that keeps a persistent contact active. In a
steadily oscillating constrained system, that accumulator reaches
a dynamic equilibrium that's mostly balanced: each cycle's
compression phase adds ≈ what the release phase removes. The net
drift is second-order and dominated by BE's small damping on the
internal oscillator, which turns out to be in the wrong direction
to push λ to zero. So λ gets stuck in a basin around some positive
mean, and the constraint never releases even though the
**instantaneous** `gap_free > 0` happens on every cycle. Turning
warm-start off forces a stateless evaluation each step: at the
first positive-`gap_free` step of the release phase, λ goes to 0
and stays 0 until the next penetration. That's what unlocks the
behaviour.

**Planning implications for 2D:**

1. **Warm-starting contact multipliers needs a "release on positive
   free-flight gap" check**, not just the `λ ≥ 0` clamp. In 2D
   terms: a contact whose accumulator would produce a separation
   signal over the next step should zero its λ rather than ramp it
   down slowly. This is a one-line change in Box2D-style
   persistent-manifold code and is worth prototyping in `solve_tgs_soft.c`.

2. **The s-coupling in the body-body Jacobian is the load-bearing
   piece** for natural rebound from deformable materials (confirmed
   by the naive control). Keep this structure in 2D — don't
   "optimise" by treating contacts as rigid-body-only for
   deformable bodies.

3. **My earlier claim that "internal elasticity can't provide
   restitution in the 1D model" was wrong.** It can, and it works
   as physically expected. The debugging detour is itself a
   useful insight: `warm_start` interacts with the oscillation
   dynamics of constrained systems in a way that hides mechanisms
   that would otherwise be visible. Watch for this when debugging
   2D deformable contact behaviour — if something "should" work
   and doesn't, try disabling warm-start as a diagnostic even if
   the production answer should keep it on.

---

### Two-ball bouncy (on a floor) — sustained-contact COM bouncing (2026-04-15)

**Setup:** `experiments/si_1d_stack.py::test_two_ball_bouncy`. Body 0
resting on the hard floor at `x=s0`, body 1 released from height
`h0=1.0` above body 0 under gravity `g=9.81`, both bodies
deformable with the same internal `(k, μ)`. **No explicit
restitution** — the hard floor is plastic, body-body contact is the
standard PGS hard contact, rebound (if any) comes entirely from the
internal elastic compress-release cycle. `dt=5e-4`, `mu_frac=1/3`,
sweep `k ∈ {1e3, 1e4, 1e5, 1e6}` × three solver modes.

**Finding — the bodies never free-flight separate, but body 1's
centre bounces via the constraint coupling.** The body-body gap
stays at machine precision throughout (they remain in sustained
contact), but `x1 = const + s1` under the active constraint means
that as body 1's internal `s1` oscillates, its centre of mass `x1`
translates with it. The bouncing is *real* in the COM sense, just
not in the "bodies come apart" sense. Measured COM oscillation
amplitude with Exp+coupled at k=1e4 reaches ±0.08 m about the rest
height of `3·s0 = 1.5`, sustained for the full 4 s run.

**Finding — integrator choice fully determines decay.** Across
all k:

| mode           | E_end/E₀ | oscillation decays? | cycles counted |
| ---            | ---      | ---                 | ---            |
| BE + coupled   | 0.67–0.86| yes, rapidly        | 7–72           |
| Exp + coupled  | 0.95–0.97| no, sustained       | 7–235          |
| BE + rigid     | 0.667    | never starts        | 0              |

At `k=1e6` Exp+coupled counts 235 distinct oscillation cycles in 4 s
with no visible amplitude decay. At `k=1e3` both BE and Exp count
only 7 cycles because the internal period is long (~60 ms) so there
aren't many cycles in 4 s at all, but the Exp amplitude holds while
BE's shrinks.

**Finding — rigid contact cannot bounce even with gravity coupling
through two deformable bodies.** `BE + rigid` (the `naive` mode with
`s` decoupled from the contact Jacobian) gives body 1 falling from
2.5 to 1.5 then sitting there flat for 4 s, exactly like the
single-ball no-restitution case. The decoupling removes any pathway
for internal spring energy to lift the COM. This is the negative
control — if we'd left `s` out of the body-body Jacobian there would
be no bouncing at all, which means the 2D solver's `s2Contact_Soft`
preparation must carry through the s-coupling in the Jacobian for
this mechanism to exist.

**So what:**

1. **The theory holds.** A deformable body with internal elastic
   energy and a coupled-to-s contact Jacobian produces natural
   rebound via compress-release, with no explicit restitution term
   anywhere. No velocity-level hack, no `(1+e)·g/K` trick, no soft
   contact parameters. Just the standard plastic PGS acting on a
   contact whose Jacobian touches both `x` and `s`.

2. **The rebound is sustained-contact COM oscillation, not
   separation.** In 1D it looks like body 1 bobbing on body 0 with
   the internal spring providing the restoring force. For 2D this
   generalises naturally (internal modes excite COM motion along
   the contact normal, constrained bodies exchange energy through
   coupled Jacobian entries), and users get "bouncy" as a natural
   consequence of the deformable-body model without tuning a
   coefficient of restitution.

3. **Integrator choice is the knob for how long the bounce lasts.**
   BE damps macroscopic rebound in 2–4 seconds at realistic
   stiffness. Exp keeps it going indefinitely. For a game-feel demo
   of a squishy bouncy ball, the exp path on the internal DoF is
   the right default; for a quickly-settling pile, BE is what we
   want. This matches the hybrid-integrator planning from probe 8.

4. **The `naive` / rigid-contact mode is the negative control** —
   it proves the s-coupling in the contact Jacobian is what
   makes this work. Anything that tries to "simplify" by treating
   contacts as rigid-body-only will lose the bouncing-for-free
   property entirely.

---

### Warm-start compressive-phase gate fixes the two-ball-attract bounce (2026-04-15)

**Setup:** Ported solver2d's warm-start gate from `src/solve_common.c:133`
(`s2PrepareContacts_PGS`: `if (warmStart && cp->separation <= 0.0f)`) into
the 1D SI prototype. In `si_step`, after free-flight prediction and before
the PGS sweep, any contact whose predicted gap `g > 0` has its carried λ
zeroed; contacts with `g <= 0` keep their warm-start as before.

**Result:** With the gate in place, `test_two_ball_attract` with
`warm_start=True` produces trajectories **bit-identical** to the
`warm_start=False` baseline across `k ∈ {1e3, 1e4, 1e5, 1e6}` and all
three solver modes — same energy ratio (~0.5), same rebound amplitude
(`|x0|_last ≈ 0.5`), same separation-flip counts. Stack equilibrium
(settled contacts have `g ≈ 0`, pass the gate) and Newton's cradle
(chained compressive impacts) both unchanged — no regression.

**Mechanism (confirming the earlier hypothesis at line 785):** PGS with
`lam_new = max(0, lam_carry + dlam)` silently uses `lam_carry` as an
attractive bias when the contact is separated — `dlam` has to overcome
the full carried impulse before projection bites. Dropping `lam_carry`
on separated contacts turns the projection into a pure fresh evaluation,
recovering the stateless behaviour that previously required
`warm_start=False` as a workaround.

**So what:** the blanket "disable warm start when debugging deformable
contact" advice from the earlier two-ball-attract entry is obsolete — the
gate lets us keep warm-start on for stiffness/convergence benefits
without hiding separation events. When this moves to 2D TGS Soft, the
gate is equally a one-liner inside `s2PrepareContacts_Soft`
(`src/solve_common.c:232`) — the Soft path currently warm-starts
unconditionally, and while the compliance damping partially masks the
issue, the same failure mode will surface on anything that needs clean
bilateral-style rebound (cloth, rope, soft body self-contact).

### Two-pass (velocity-then-position) SI is a wash vs. single-pass BE on the 1D prototype (2026-04-15)

Implemented a Box2D-style two-pass mode in `experiments/si1d/solver.py` (`mode="two_pass"`):
pass 1 solves the velocity-level constraint `J·v ≥ 0` with `K_v = avel_x + avel_s`
(the BE-consistent velocity response `avel_s = apos_s/dt`); pass 2 is a split-impulse
position correction that only touches positions. Both passes use PGS with `λ ≥ 0`.

A/B on the full suite (stack equilibrium, two-ball-attract across
k∈{1e3,1e4,1e5,1e6}, Newton's cradle, single-ball floor bounce) vs. the existing
single-pass `local_implicit`:

| scenario          | single-pass      | two-pass         |
|-------------------|------------------|------------------|
| stack rel err x   | 1.29e-03         | 1.42e-03         |
| two-ball E_end/E0 | 0.4975 → 0.5000  | 0.4972 → 0.5000  |
| two-ball flips    | 5,6,4,2          | 6,6,4,2          |
| floor E_end/E0    | 0.5000           | 0.5000           |
| floor x_min       | 0.49220          | 0.49227          |

Nearly bit-identical. Not a regression, but not the velocity-quality improvement
the plan speculated about either. The reason is structural: single-pass BE already
produces BE-consistent velocities because an impulse `dλ` simultaneously updates
position by `apos·dλ` and velocity by `avel·dλ = (apos/dt)·dλ`. Decomposing into a
velocity pass + split position pass just reshuffles the same linear system.

**Gotcha worth saving: the velocity pass needs an explicit active-set gate.** The
first implementation froze the two-ball-attract scenario completely — the
velocity-level target `J·v = 0` was being enforced on the body-body contact even
when the bodies were separated by a gap of 1.0, because `−J·v/K_v` is positive for
any approach velocity regardless of distance, and `λ ≥ 0` does not clamp it (unlike
position-level single-pass, where `g > 0` directly gives `dλ < 0` that the clamp
kills). Fix: before the velocity sweep, mark each contact active iff the
post-free-flight gap is `≤ 0`, and only enforce `J·v ≥ 0` on active contacts. This
mirrors Box2D's manifold model (velocity constraints only exist where a contact
manifold exists). **Carry to 2D:** if we ever split TGS Soft's unified solve into
explicit velocity/position passes, the velocity pass MUST be guarded by the
broadphase/near-phase contact existence check — a contact in the solver array is
not automatically "active" for velocity purposes. Without this gate, bodies under
any attractive force (joint pull, gravity, magnetism) will freeze before touching.

**So what:** the two-pass formulation is available as `mode="two_pass"` for future
experiments (particularly restitution-biased velocity solves, where decoupling a
velocity-level COR bias from position drift correction may have a cleaner shape
than bolting restitution onto a position-level solve), but there is no reason to
switch the default off single-pass. The interesting open door is: *restitution in
the velocity-pass bias*, which single-pass can't express as naturally.

---

### The deformable bounce ≈0.5 energy floor is the BE position projection — velocity-level restitution with the physical mass metric fixes it (2026-04-16)

**Setup:** Narrow single-bounce diagnostic added in
`experiments/si1d/diag_single_bounce.py`. One deformable body, **no gravity**,
dropped with `vx = −V` onto the floor; `mu_frac = 1/3`, k ∈ {1e3..1e6}. Energy
decomposed per step into `KE_x`, `KE_s`, `PE_e`; the contact window is bracketed
and losses attributed to entry step / spring phase / release step.

**Finding:** With the old position-level BE contact (`restitution=0`), **100%
of the energy loss happens at the single entry step**, losing exactly
`1 - 1/(1+mu_frac) = 1/4` of `E₀` (0.75 ratio for `mu_frac=1/3`), independent
of k. Spring phase and release are lossless to 5+ decimals. The two-ball-attract
0.5 baseline is just two bodies each paying this tax once.

Algebraic mechanism: BE enforces the post-step gap `g_next = 0`, which means
`J·v_new = vx_new − vs_new = 0`. The minimum-norm impulse in the mass metric
that does this is elastic-collision-style with **fully inelastic target**:

```
J = [+1, −1],  M = diag(m, μ),  K = 1/m + 1/μ
λ = (J·v_pre) / K                       # velocity-projection multiplier
v_new = v_pre − M⁻¹ Jᵀ λ
```

For `vx_pre = −V, vs_pre = 0, m=1, μ=1/3`: K=4, λ=−V/4, giving
`(vx_new, vs_new) = (−3V/4, −3V/4)` and `KE_new/KE_pre = 3/4` exactly.

That is: the contact velocity target of zero mass-projects away a chunk of
kinetic energy that rigid-body contacts hand back via restitution. Deformable
bodies under this solver have **no source of restitution at the moment of
first contact** — the internal oscillator can only re-expand what BE hasn't
already projected away.

**Fix:** Replace the contact velocity target with `J·v_new = −e · J·v_pre`
and apply the impulse **velocity-only, using the physical K = 1/m + 1/μ**, not
the dt-based `avel_x + avel_s` linearised response. Two distinct gotchas cost
a full iteration cycle to untangle:

1. **Position updates must be zero.** An impulsive reflection jumps `v` and
   leaves `x` alone — the *next* free-flight step advances position with the
   reflected velocities. Adding `Δx = (dt²/m)·dlam` alongside `Δv` pumps
   energy because it corresponds to a sustained force over dt, not an
   instantaneous impulse. This was injecting +0.19 `E₀` per re-impact.
2. **K must be the physical mass metric.** `avel_x + avel_s = dt/m + dt·cos(ωdt)/μ`
   is a BE/exponential per-step response that only matches `1/m + 1/μ` to
   O(dt). Using the dt-based K makes the reflection algebraically
   energy-drifty.

Under the corrected rule for `m=1, μ=1/3, vx_pre=−V, vs_pre=0`, K_phys=4,
`Δ(J·v) = 2V`, `(vx_new, vs_new) = (−V/2, −3V/2)`, `KE_new = V²/2 = E₀`
**exact**.

**Verification:** single-body no-gravity, `restitution=1.0`, mode
`two_pass_exp` (new mode — two-pass contact structure on top of exponential
free flight; needed because plain exponential single-pass couples position
correction back into velocity via the sweep, and plain two-pass uses BE free
flight which L-damps the internal oscillator over long runs):

| k | impacts | E_end/E₀ |
|---|---|---|
| 1e3 | 3 | **1.0000** |
| 1e4 | 3 | **1.0000** |
| 1e5 | 3 | **1.0000** |
| 1e6 | 3 | **1.0000** |

Each bounce event generates 3 re-impacts (the ringing internal oscillator's
bottom dips into the floor twice before escaping), and **each reflection
preserves energy individually**, not just on average. Gif:
`experiments/out/bounce_sweep_restitution.gif` shows 4 stiffnesses bouncing
under gravity at `restitution=1` — with gravity, ~0.6% is lost per 4 s run,
likely from reflections happening while the body is still in sustained contact.

Two-body version added the same velocity-only reflection to pairs (gated on
fresh impact `lam_prev=0`, approaching `J·v<0`, and penetrating `g<0`):

| k | hard BE baseline | restitution=1 |
|---|---|---|
| 1e3 | 0.500 | **0.985** |
| 1e4 | 0.500 | **0.985** |
| 1e5 | 0.500 | **0.985** |
| 1e6 | 0.500 | 1.179 (overshoot — open) |

30× improvement at reasonable k. Gif:
`experiments/out/two_body_attract_restitution.gif`.

**Why the `two_pass_exp` mode is necessary.** A straight single-pass sweep
still fires position-correction impulses on sub-penetrating gaps *after* the
reflection, each of which does work against the now-ringing internal
oscillator (this was the +0.1 `E₀` pump seen before the fix). Splitting
position correction into its own pass that doesn't touch velocity kills that
pump. Exponential free flight is required because the post-bounce body
undergoes thousands of steps of free internal oscillation — BE numerical
damping would eat amplitude between bounces.

**So what for 2D TGS Soft:**
- The hidden ~25% per-contact energy tax is not a property of "deformable
  bodies" or of numerical damping — it is the velocity-projection effect of a
  position-level non-penetration constraint on a compound (COM + shape) DoF
  space. Box2D's velocity-level restitution bias is already the right shape
  for this; we just need to make sure the bias uses the physical mass metric
  and applies only velocity, not position.
- The entry-step is the only place the energy tax lives. A once-per-contact
  "initial velocity bias" (Box2D's pattern) is sufficient; no sustained
  velocity-pass restitution is needed.
- Position correction must be split from velocity (two-pass), or restitution
  will be pumped out by drift correction on subsequent sub-steps. TGS Soft
  already does this — the split position pass is a free win here.
- The `restitution=1` limit chatters (3 reflections per bounce event). A
  production solver should switch from reflection to a clamped sustained
  contact once `|J·v|` drops below a threshold, mirroring Box2D's
  `s2_velocityThreshold`.

**Open:** `k=1e6` body-body overshoot (E_end/E₀ = 1.18). The reflection
conditions fire in a configuration where gap detection hasn't caught up with
the near-instantaneous penetration; likely needs an "approach-and-penetrating-
within-dt" test rather than separate `jv<0 ∧ g<0`.

---

## Open items carried into 2D plans

- Warm-start quantitative benefit (plan #1, solver validation) — **closed above, but revealed a low-k regression worth fixing**
- Substep vs inner-iteration allocation at equal compute (plan #2) — **closed; substep dominates cradle, interior optimum for sustained contact**
- Long-chain PGS scaling exponent (plan #3) — **closed above, better than hypothesised**
- Temporal convergence order of the full pipeline (plan #4) — **closed above (slope 1.05)**
- `μ_frac` sensitivity (plan #5) — **closed above (plateau `[0.1, 1.0]`)**
- COR tuning via soft contact `(ωₙ, ζ)` (plan #6, material richness) — **closed above; usable with a dt-proportional bias**
- Asymmetric tension/compression cubic law (plan #7) — **closed above; drop-in, cheap**
- Backward-Euler per-cycle damping budget in the well-resolved regime (plan #8) — **closed above; need ωdt < 3e-3 for <1%/cycle**
- Finite-mass pusher variant of the inversion test — the kinematic
  ceiling forces `s` to track `ceiling/2` regardless of β, so the
  current `test_nonlinear_inversion` demonstrates integrator stability
  but not hardening resistance. A finite-mass pusher would separate
  the two effects.

---

## 2D prototype (si2d) — affine ellipse bodies with SNH energy

Entries from 2026-04-16 onward are from the 2D prototype in
`experiments/si2d/`, which uses 6-DoF affine bodies (center + 2x2
deformation gradient F) with Stable Neo-Hookean energy.

### SNH energy verified in 2D (2026-04-16)

**Setup:** 2D specialisation of Kim & Eberle's SNH: `Psi = 0.5*[mu*(Ic-2) + lam*(J-alpha)^2]`
where `alpha = 1 + mu/lam`. F stored as flat 4-vector `[F11, F12, F21, F22]`.

**Finding:** Analytic Hessian `H = mu*I + lam*cof⊗cof + lam*(J-alpha)*d²J/dF²`
verified against sympy symbolic, finite differences (10 random F configs, grad
err < 1e-7, hess err < 2e-7), and 1D reduction (uniaxial stretch F=diag(1,s)
gives d²Psi/ds² = mu+lam = k exactly). SPD projection via 4×4 eigendecomposition
passes 100/100 random configs including near-singular and inverted F.

**Lame mapping:** `mu = k*(1-2*nu)`, `lam = 2*k*nu` gives uniaxial stiffness k.
Caveat: `nu=0` is singular (`alpha = 1 + mu/lam` diverges). Require `nu > 0`.

**So what:** SNH avoids the polar decomposition entirely — the Hessian is a
simple algebraic expression of F. This is a major simplification vs corotated
linear elasticity in 2D.

### Block-diagonal mass matrix (2026-04-16)

**Setup:** Uniform-density disk of radius r0, affine map x(X) = c + F*X.

**Finding:** Mass matrix is `diag(M, M, mu, mu, mu, mu)` where `mu = M*r0²/4`.
The ratio mu/M = r0²/4 controls the inertial coupling between translation and
deformation. For r0=0.5: mu = M/16 = 0.0625M — deformation modes have very
low inertia relative to translation.

**So what:** The low mu/M ratio means contact impulses distributed via the full
Jacobian pump most of their energy into F modes (see Moreau dissipation below).

### BE free-flight energy (M1, 2026-04-16)

**Setup:** One body, no gravity, no contacts. F=I, vF has off-diagonal spin +
diagonal breathing components. k=1000, nu=0.3, dt=1/240, 2400 steps (10s).

**Finding:** Energy drift = -7.4e-4 (0.07% damping over 10s). Monotonically
decreasing — BE damps but never adds energy. Rigid limit (k=1e6): F stays
within 1.2e-7 of a rotation matrix, det(F) = 1.000000.

**So what:** BE is stable and accurate for free flight. The energy damping is
acceptable as a baseline; exponential integrator would eliminate it.

### Moreau dissipation in 2D is severe (M2, 2026-04-16)

**Setup:** Single disk (r0=0.5, k=5000, nu=0.3) dropped from h0=5 onto floor.
Floor Jacobian J = [0, 1, 0, 0, -r0*F21/L, -r0*F22/L]. At F=I: J = [0, 1, 0, 0, 0, -0.5].

**Finding (theory):** Effective mass K = 1/M + r0²/mu = 5/M. Single Moreau
projection preserves KE (redistributes from vy to vF22) but gives vy_new = 4/5*vy,
so h_ratio = 0.64 per bounce. In 1D this was 0.75 (less DoFs coupled).

**Finding (simulation):** Full coupled mode gives h1/h0 = 0.23 — much worse than
the single-projection 0.64. The extra dissipation comes from multi-timestep position
correction pumping energy into F, which BE then damps. Restitution e=1 in full
coupled mode gives 0.22 (actually worse! — the restitution impulse also couples to F).

**Fix (decoupled mode):** Position pass corrects only center-of-mass (not F),
velocity pass acts only on translational DoFs with K_trans = 1/M. Result:
h1/h0 = 0.995 with e=1 — near-perfect energy preservation. Slow decay to 0.92
over 10 bounces from residual BE damping of F oscillations.

**So what:** This confirms the si1d finding in 2D: contact impulses must NOT
couple to deformation DoFs. The "decoupled" mode (c-only position correction +
c-only velocity restitution) is essential for energy preservation. The deformation
response to contact should come purely from the elastic restoring forces in the
integrator, not from the contact solver.

### SNH area-preservation weakens under large compression (2026-04-16)

**Setup:** Floor compression of a disk (k=50, nu=0.45) with relin mode (BE
re-solve inside SI loop). Examined equilibrium F11 as a function of F22.

**Finding:** SNH energy Ψ = ½[μ(Ic−2) + λ(J−α)²] couples F11 to F22 through
the cofactor: P11 = μ·F11 + λ·(J−α)·F22. Setting P11=0 gives
F11_eq = λ·α·F22 / (μ + λ·F22²). At moderate compression (F22=0.5) this
predicts F11_eq ≈ 1.54 (correct horizontal expansion). But at heavy compression
(F22→0), F11_eq → 0 — the model predicts horizontal *collapse*, not expansion.

**Root cause:** SNH deliberately replaces the log(J) barrier of standard
Neo-Hookean with the polynomial (J−α)² to remain smooth through J=0 (element
inversion). This removes the singularity that enforces incompressibility. Standard
NH gives F^{−T} forces that diverge as J→0, producing F11 = 1/F22 at equilibrium
(exact area preservation). SNH's polynomial penalty has no such divergence.

**So what:** SNH cannot produce realistic area-preserving squish (pancake spread)
under large compression. Options for future work: (1) hard det(F)=1 constraint in
the SI loop, (2) log-barrier penalty −κ·log(J) added to SNH, (3) standard NH
if inversion robustness is not needed. This is a material model limitation, not a
solver deficiency — no amount of relin or Newton iterations can fix it.

## Collision detection

### Alternating projection oscillates for overlapping ellipses → replaced with non-iterative (2026-04-16)

**Setup:** Two vertically stacked circles (r0=0.5), centers 0.9997 apart
(penetration ~0.0003). Alternating projection with initial direction = center-to-center.

**Finding:** The alternating projection oscillates between d=[0,−1] and d=[0,+1]
and never converges for overlapping bodies. The vector pB−pA reverses sign each
iteration. After 20 iterations, floating-point noise amplifies exponentially
(~1e-19 → ~0.18), producing a spurious tilted normal that launches bodies sideways.

**Fix:** Replaced entirely with a non-iterative algorithm (Mueller's oriented
particles approach): transform to B's frame (B becomes a circle via FB⁻¹),
evaluate A's support function in the center-to-center direction, project onto B's
circle, transform back. Zero iterations, no convergence issues. For circles it's
exact; for ellipses it's a one-step approximation that's accurate for moderate
deformations. Three-body stack now has exactly zero horizontal drift.

**So what:** Non-iterative collision is simpler, faster, and more robust than
alternating projection. The only limitation: for deeply overlapping bodies (gap
far below zero), the gap magnitude underestimates the full Minkowski penetration
depth. This doesn't matter in practice since the contact solver keeps penetrations
small.

### Body-body contact works with decoupled SI (M4/M5, 2026-04-16)

**Setup:** M4: head-on collision of two equal disks (v=±3, no gravity, e=1).
M5: three disks stacked vertically, dropped onto floor (e=0.2).

**Finding:** M4 gives perfect KE conservation (ratio=1.000000) and momentum
conservation. M5 settles to y=[0.500, 1.500, 2.500] with |x| < 0.001 over 10s.
The decoupled contact mode (position/velocity pass acts only on translational DoFs)
extends naturally from floor contacts to body-body contacts. The velocity pass
with Gauss-Seidel converges for stacked contacts (pair+floor) with 16 iterations.

**So what:** The ellipse-ellipse Jacobian and collision detection are working. The
prototype can now handle multi-body scenes. The main remaining gaps are friction
(needed for realistic oblique interactions) and the SNH area-preservation limitation
(no visible squish during body-body compression).
