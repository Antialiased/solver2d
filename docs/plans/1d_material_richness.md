# 1D material/physics richness plan

Three probes in `experiments/si_1d_stack.py` that extend the material model
beyond the current linear-elastic + linear-hardening + cubic-force setup.
These are secondary to the solver-validation plan
(`1d_solver_validation.md`) — they answer "what kinds of materials can
we represent" rather than "does the solver work at scale".

Context: the existing file has linear elastic, linear isotropic hardening
(`sigma_Y`, `H_hard`), and a symmetric cubic-force nonlinearity (`beta3`)
with scalar Newton in the backward-Euler free-flight update. See
`memory/project_dbd_plasticity_hardening.md` and
`memory/project_dbd_backward_euler_robust.md` for the baseline.

---

## 6. COR tuning via compliant contact

**Closes:** whether the soft-constraint `(frequency, damping ratio)`
parameterization gives predictable, tunable coefficient of restitution.
This is the user-facing knob for "how bouncy" a deformable body is, and
validating it in 1D is much cheaper than debugging it in 2D.

**Setup:** single body dropped onto the floor at a known velocity `v_in`,
no gravity during rebound. The current `si_step` uses a hard PGS projection
at every contact; to turn this into a compliant contact we need a
Baumgarte-style bias term or a soft-constraint reformulation following the
TGS Soft pipeline. Either:

- **(a)** add an `(omega_n, zeta)` pair per contact that sets a position
  bias `b = -gamma * g` and an effective compliance `alpha` per the
  `extras/soft_constraint.m` derivation, *or*
- **(b)** keep hard contact and instead tune the body's internal `k` /
  `damping` to control rebound.

Option (a) is the real test; (b) is a fallback if setting up soft-contact
in the prototype is too invasive.

**Sweep:** `(omega_n, zeta) ∈ grid`. For each combination, drop from
`v_in ∈ {1, 5, 20}`, measure the outgoing velocity `v_out` after one
rebound cycle. Compute empirical COR = `|v_out / v_in|`.

**Metrics:**
- COR vs `zeta` at fixed `omega_n` — should be monotone decreasing
- COR vs `v_in` at fixed `(omega_n, zeta)` — should be flat (linear model
  prediction); deviations indicate material nonlinearity bleeding in
- predicted vs measured COR using the soft-constraint formula from the
  Matlab derivation

**Success:** measured COR matches the analytic prediction within ~5% over
a two-decade `omega_n` range. That validates that TGS-Soft's
`(omega_n, zeta)` → COR map is usable in our pipeline.

**Why it matters for 2D:** once we have 2D demos, "how bouncy" is the
single parameter users will want to tune first. Making sure the mapping
is predictable is table stakes.

---

## 7. Asymmetric material law (tension-soft, compression-hard)

**Closes:** the symmetric cubic we prototyped in test 9 (`nonlinear`) is
convenient but not realistic — real solids are usually much stiffer in
compression than in tension, sometimes by an order of magnitude. Extend
`beta3` to piecewise behaviour.

**Setup:** replace the single `beta3` with two coefficients `beta_comp`
(active when `u < 0`) and `beta_tens` (active when `u > 0`). The force
law becomes `F(u) = -(k u + beta(u) u^3)` with `beta(u) = beta_comp` for
`u < 0` and `beta_tens` for `u > 0`. This stays C¹ at `u = 0` because
both pieces and their derivatives vanish there.

**Free-flight update:** the scalar Newton on the cubic still works —
`G'(u) = A + 3 B(u) u² > 0` everywhere because `B ≥ 0` on both sides.
The only subtlety is that `B` now depends on `sign(u_new)`, so a Newton
step across `u = 0` should re-evaluate `B`. In practice this is a single
branch inside the Newton loop.

**Test scenarios:**
- **stretch-to-fail:** ceiling lifted above rest, body pulled past rest
  length. With `beta_tens = 0` and `beta_comp >> 0`, body should stretch
  freely but resist compression.
- **rebound asymmetry:** drop onto floor, measure depth of penetration vs
  rebound height. A real-solid asymmetric model should penetrate less and
  rebound similarly to the symmetric case.

**Metrics:**
- max `|u|` reached in each direction
- Newton iters per step (check that crossing `u = 0` doesn't blow up
  convergence)
- force law plot (should show the kink at `u = 0`)

**Why it matters for 2D:** in 2D, squashing a body is the common case and
stretching is rare. A material law biased toward compressive hardening
is both more realistic and saves us from having to tune a conservative
symmetric stiffness.

---

## 8. Free-oscillation energy drift (cost of L-stability)

**Closes:** how much amplitude does true backward Euler kill per cycle
on an isolated oscillator? This is the "cost of L-stability" and we need
a number for it before deciding when to fall back to the exponential
integrator for stiff-wave fidelity.

**Setup:** single body, no contacts, `damping = 0`, `g = 0`, initial
condition `s(0) = s0 + 0.1`, `vs(0) = 0`. Let it oscillate. Compare
`local_implicit` (backward Euler, L-stable) against `exponential`
(symplectic, energy-preserving on the linear oscillator).

**Sweep:** `ω dt ∈ {0.1, 0.3, 1.0, 3.0, 10.0}` by varying `k` at fixed
`dt`. For each, run for 100 oscillation periods. Measure:

- amplitude decay rate (exponential fit `A(t) = A_0 exp(-lambda t)`)
- per-cycle energy ratio `E(t_n+1) / E(t_n)`

**Metrics:** plot `lambda / omega` vs `omega dt`. Theory says backward
Euler's numerical damping scales like `(omega dt)^2 / (1 + omega² dt²)`
per step. Verify that empirically.

**Success:** measured per-cycle damping matches the backward-Euler
theoretical damping within ~20%, and is *small* (well under 1% per cycle)
for `omega dt < 1`. That tells us the "well-resolved modes" regime is
the safe zone for the implicit integrator, and the hybrid threshold
discussed in `memory/project_dbd_backward_euler_robust.md` has a concrete
numerical basis.

**Why it matters for 2D:** once we build a cradle-style demo in 2D, we
need to know how much energy the integrator is stealing from modes we
*want* to preserve (the stiff wave) vs modes we *want* to kill (the
contact-induced high-frequency rattle). This experiment gives the
budget.

---

## Running order recommendation

Start with #8 — it's the cheapest (single body, no contacts), gives a
concrete number, and validates the theoretical prediction baked into our
memory. Then #7 (small extension of existing cubic-force machinery).
Leave #6 for last — it requires adding soft-constraint contact to the
prototype, which is the biggest build in this plan and the closest in
spirit to the 2D TGS-Soft pipeline. Getting the soft-contact prototype
working in 1D is itself a useful 2D prerequisite.
