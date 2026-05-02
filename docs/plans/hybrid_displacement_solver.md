# Hybrid Displacement-Based Solver: PGS + Barrier-Newton Coupling

## Motivation

Our current FQ2D deformable pipeline is pure VBD: per-body Newton with
frozen neighbours, smooth clamped-quadratic contact penalty. This works
well in isolation but does not naturally compose with the PGS-style
velocity-iteration loops that mainstream rigid-body solvers
(solver2d's `solve_tgs_soft.c`, Box2D, etc.) are built around. We need
a story for **mixed rigid + deformable** scenes that doesn't require
either:

- promoting every rigid body to FQ2D (cheating; expensive; loses
  rigid-body PGS machinery), or
- demoting every soft body to a velocity-impulse model (incorrect for
  stiff nonlinear elasticity).

We also have a known weakness in the current smooth-penalty contact:
penetration scales as `~1/κ` by construction, forcing brittle κ tuning
(see `body_body_drop_fq_soft.gif` history — needed `κ_mult=100` to look
clean, with no principled stopping rule).

This plan captures the design we want to explore next, and the small
in-VBD pilot that validates the foundation before any rigid-coupling
work.

## The core idea

**Everything interacts through displacements** — tentative positional
increments over the substep — but the *update rule* for a constraint
is chosen by the body types involved:

| Constraint type    | Update rule                                          |
|--------------------|------------------------------------------------------|
| rigid ↔ rigid      | Standard PGS velocity iteration (interpret Δx = v·h) |
| soft ↔ soft        | 2-body Newton block (expanded VBD edge)              |
| rigid ↔ soft       | 2-body Newton block (rigid is BE position-DoF, M/h²) |

This is structurally **VBD generalised from per-vertex blocks to
per-edge (2-body) blocks** for soft-touching constraints, while keeping
PGS for purely-rigid edges. Both are block-coordinate descent on a
single displacement-currency state vector; the block size just changes
with constraint type.

### Currency unification

A PGS impulse Δp converts to a position correction `Δx = M⁻¹·Δp·h`. In
displacement currency the rigid body's BE Hessian is `M/h²`, which
glues directly into the soft body's elastic Hessian inside the 2-body
Newton block. No accounting trick needed.

### 2-body block size (2D)

| Pair                | DoFs        | Hessian |
|---------------------|-------------|---------|
| FQ2D ↔ planar rigid | 12 + 3 = 15 | 15×15   |
| FQ2D ↔ FQ2D         | 12 + 12 = 24| 24×24   |

Both factor cheaply. The soft+soft block is a *strict upgrade* over
the current single-body VBD with frozen neighbour: no convergence
penalty for the larger block, just a constant-factor cost.

## Frozen-normal IPC barrier (the refinement that makes it tractable)

Inside each 2-body Newton block, replace the smooth clamped-quadratic
penalty with an **IPC log barrier** `b(g, dhat)` on the gap. Critically:
**freeze the contact normal `n` at the start of the substep**.

### What this buys

- **Penetration is zero by construction.** Barrier diverges at `g=0`;
  line-search backoff guarantees the iterate stays in the feasible
  region. The `~1/κ` penetration scaling — and its tuning headache —
  goes away.
- **No discrete active set inside Newton.** The barrier is a smooth
  active set with continuous gating in `[0, dhat]`. Newton stays C∞
  inside the block. (This was the central objection to true
  active-set logic: discrete switching wrecks Newton convergence. The
  barrier dodges it cleanly.)
- **CCD step is closed-form.** FQ2D point positions are linear in `q`,
  and `n` is fixed during the substep, so
  `gap(α) = g₀ + α·(nᵀ·J·δ)` is exactly linear in the Newton step
  length `α`. Maximum feasible step is
  `α_max = -g₀ / (nᵀ·J·δ)` if the denominator is negative, else `∞`.
  One dot product per contact. Cheaper than computing the gap.
- **Same Hessian skeleton as current code.** `Jn = nᵀ · J(ξᵢ)` is
  already what `_collect_body_body_contacts_fq` builds. Only the
  scalar penalty function changes:
    - gradient `b'(g)·Jn`
    - Hessian  `b''(g)·Jnᵀ·Jn`
  Single-line edit per contact direction.
- **Friction integrates cleanly.** IPC's mollified Coulomb cone is a
  known-good companion. The smooth-penalty path didn't have an
  obvious friction story.

### Costs of freezing the normal

- **Tangential drift** within a substep is bounded by substep size.
  TGS Soft already substeps; refresh `n` per substep, not per Newton
  iter. That's the whole point of the simplification.
- **Discontinuous normal at chord-polygon vertices** (sub-cell
  partition seams, corner contacts). When the active edge changes
  between substeps the normal jumps. Mitigation: averaged normal at
  shared edges, or weight contributions from both incident edges
  within a transition band. Cheap; preserves Newton smoothness within
  the substep.
- **Initial feasibility.** Barrier needs `g > 0` at substep entry.
  Standard IPC trick: continue the barrier as a quadratic below
  `g = 0.1·dhat` so warm-start overshoot still has a defined energy.
  Or one-shot untangling at substep entry.

## Coupling rule at the rigid–soft boundary

When a rigid body participates in **both** a PGS edge (rigid–rigid)
and a Newton-block edge (rigid–soft), its position update comes from
two sub-solvers with different rules in the same outer iteration. The
barrier framing gives a clean coupling rule:

> After each PGS sweep, for every rigid body that participates in a
> soft-touching contact, compute `α_max` from the frozen-normal
> feasibility check and clamp the PGS positional correction to it.

This is "PGS with a barrier-derived motion limit". One dot product per
cross-type edge. Preserves IPC's hard non-penetration guarantee at the
system level. It also resolves a coupling concern from earlier
discussion (PGS oscillation against a stiff Newton block) by making
the barrier the single source of feasibility truth.

Multiplier carry-over (ADMM/AL style; XPBD already does this) is the
standard fix for warm-starting the boundary across substeps. Worth
adopting from day one.

## Where this slots in

Inside TGS Soft's substep (`solve-with-bias` stage), one sweep becomes:

1. PGS sweep over rigid–rigid edges, with α_max clamps from
   soft-touching neighbours.
2. Newton-block sweep over soft-touching edges (rigid–soft and
   soft–soft), barrier-feasible by construction.
3. Color the constraint graph; process colors in parallel.

Per substep: refresh frozen normals, re-warm-start multipliers, repeat.

## Pilot: in-VBD barrier swap (no rigid coupling yet)

The user has explicitly directed: **continue to prove this out in the
existing pure-VBD setting first.** This isolates the barrier mechanism
from the harder PGS-coupling question.

### Smallest validating change

Swap *only* the contact penalty in `_collect_body_body_contacts_fq`
(and `_outer_floor_contacts_from_q`) from clamped-quadratic to IPC
barrier with frozen normal + Newton-block line search. Keep:

- the existing per-body Newton (don't expand to 2-body blocks yet),
- the owner-vertex GS pattern,
- the existing 8 outer-ξ × 4 sub-cell vertex-vs-sub-cell SDF.

What changes:

- penalty function: `½κ·max(-g,0)²` → `b(g, dhat)`
- line search inside `vbd_body_step_fq`: add α_max feasibility clamp
  before Armijo backtracking
- substep entry: optional untangling projection if any `g ≤ 0` warm
  start
- κ knob → dhat knob (barrier width); `barrier_strength` if we keep
  IPC's standard parameterisation

### Validation suite

Re-run the existing scenes that motivated this:

1. `test_floor_fq` — single-body and cantilever drop. Expect
   penetration to drop to numerical zero with no κ tuning.
2. `test_body_contact_fq` — stack drop, side collision, stiffness
   sweep. The "stiffness sweep" should now show penetration **≈ 0**
   across `dhat ∈ {0.001, 0.01, 0.1}·h` rather than the `1/κ` scaling
   we currently see.
3. `render_floor_fq`, `render_body_contact_fq` — visual comparison
   against the existing GIFs. Specifically the soft-k=200 variants
   that needed κ_mult=100 — they should look clean at default barrier
   settings.

### Pass criteria

- Penetration max < `0.01·h` across all scenes, regardless of `k`.
- Newton iteration counts within 2× of current smooth-penalty path.
- No tuning step required to move between `k=200` (soft) and
  `k=2000` (current default) — same dhat works.
- min det J ≥ current values (no inversion regression).

If all four pass, the barrier foundation is validated and we can move
on to the 2-body-block expansion (still in pure VBD, still no rigid
coupling), then to the PGS rigid-side, then to the boundary clamp.

### Estimated effort

~2–3 days for the pilot, mostly tuning `dhat` and the line-search
tolerance. The Jacobian/Hessian projection structure is already in
place.

## Staged rollout (after pilot validates)

1. **Pilot (this plan, soon)**: barrier swap in pure VBD, single-body
   Newton blocks. Validates the barrier mechanism in isolation.
2. **2-body Newton blocks** in pure VBD: expand the soft–soft edge
   block from 12 to 24 DoF. Compare against owner-vertex GS pattern
   on stack-drop convergence rate.
3. **Planar rigid body type** in the experiments harness: 3-DoF
   `BodyRigid2D` with the same `q`-style API. Pure-rigid stack-drop
   demo using only the Newton-block path (rigid as a degenerate
   "1-element FQ2D"-like block).
4. **PGS for rigid–rigid** edges: import or port the standard 2D PGS
   contact + joint solver into the experiments harness. Same currency
   (displacement). Rigid-only stack-drop matches solver2d baseline.
5. **Barrier-derived α_max clamp** on PGS at rigid–soft boundaries.
   Pilot scene: FQ2D body resting on a kinematic rigid plank that
   swings on a hinge. Compare against the cheating "rigid-as-stiff-
   FQ2D" baseline.
6. **TGS Soft port**: drop the validated hybrid into
   `solve_tgs_soft.c`'s substep `solve-with-bias` stage. End state.

Each stage is independently demoable and independently regression-
testable. The user can stop at any stage and ship; later stages are
strict capability extensions.

## Open questions

- **Barrier parameterisation.** Stick with IPC's `(dhat, κ_b)` knobs,
  or re-parameterise to `(ωₙ, ζ)` like TGS Soft? IPC default first;
  re-parameterise only if it composes badly with the rest of the
  TGS Soft compliance model.
- **Warm-start untangling cost.** How often does substep entry land
  in penetration (`g ≤ 0`) under realistic stacking? If it's rare,
  the quadratic-continuation extension below dhat is enough; if
  common, a one-shot projection step at substep entry pays for
  itself.
- **Vertex-vs-edge geometry.** Current contact is vertex-into-sub-cell
  SDF — vertex-on-edge contacts work, but edge-on-edge does not. The
  barrier doesn't fix this; it's an orthogonal upgrade. Edge-edge
  integrated barrier (the "follow-up edge-edge contact" item from the
  body-body plan) covers it naturally and composes with the barrier
  framework cleanly. Defer to the body-body plan's follow-up list.
- **Frozen-normal failure rate.** How often does an outer-ξ vertex
  drift more than ~`h/4` tangentially within a single substep, enough
  that a frozen normal becomes visibly wrong? Measure during the
  pilot; if rare, leave alone; if common, refresh per Newton iter
  (loses closed-form CCD but keeps the smooth barrier).
- **Friction integration.** When we add the IPC mollified Coulomb
  cone, does the friction tangent direction also need to be frozen
  per substep, or only the normal? IPC's literature has the standard
  answer; verify it matches our substep granularity.

## What this is NOT

- Not a replacement for VBD. The 2-body Newton block IS VBD with a
  larger block. Same convergence theory.
- Not a constraint-satisfaction-only method (PBD-style). The barrier
  has gradient and Hessian; the soft body's elastic energy is
  unmodified. Energy is well-defined throughout.
- Not a complete IPC port. We deliberately freeze the normal to avoid
  IPC's per-iteration global CCD cost. Accuracy loss is bounded by
  substep size; tractability gain is large.
- Not a TGS Soft replacement. The end state is hybrid IPC-Newton
  blocks **inside** TGS Soft's substep `solve-with-bias` stage, not
  instead of it.

## References

- Li et al., "Incremental Potential Contact: Intersection- and
  Inversion-free Large Deformation Dynamics", SIGGRAPH 2020.
- Lan et al., "Penetration-free Projective Dynamics on the GPU"
  (stencil-descent / 2-body Newton blocks).
- Macklin et al., "XPBD" (multiplier carry-over for warm-starting
  the boundary across substeps).
- Bouaziz et al., "Projective Dynamics" (positional Newton ancestor).
- `docs/insights.md` 2026-04-17 entry on banana modes / convex
  decomposition (motivates per-sub-cell chord polygon collision).
- `docs/plans/spicy-singing-sphinx.md` (the body-body smooth-penalty
  plan whose `1/κ` tuning headache motivates this upgrade).
