# Implementation Plan: Affine Bodies in solver2d

Code-side companion to `../research.md`. That doc holds the theory, hypotheses, and open questions; this doc holds the staged implementation plan, checkpoints, and code-level notes. See `../code_map.md` for the file-by-file index of the existing codebase, and `../insights.md` for findings from experiments as they accumulate.

## Ground rules

- Base solver: `s2Solve_TGS_Soft` in `src/solve_tgs_soft.c`. All affine work branches from the TGS Soft pipeline.
- Keep the rigid limit working at every step — as affine stiffness → ∞ behaviour must match rigid TGS Soft.
- New solver variant rather than replacing the existing one: add `s2_solverAffine` (or similar) to `s2SolverType` in `include/solver2d/types.h` and a dispatch case in `s2World_Step` (`src/world.c`, around line 208). This keeps baselines runnable for comparison.
- Stack-allocated per-step scratch only (`world->stackAllocator`). No per-step heap.

## Solver strategy

Start with a **sequential-impulses retrofit**: keep the PGS/TGS Soft outer structure, replace the per-contact velocity update with a small local implicit solve over the two contacting bodies' affine DoFs plus the contact multiplier (~13-dim dense Newton, Gauss-Newton with PSD regularization, 1–2 inner iters, warm-started). This slots into the existing solver as effectively a new contact type, handles mixed rigid/affine bodies uniformly (rigid = stiff limit of the affine block), and lets impulses propagate along chains within a single sweep — important for Newton's Cradle.

Rationale for SI over AVBD as the starting point: solver2d only ships PGS variants, so SI is a retrofit rather than a new solver. AVBD is theoretically a cleaner fit for DBD's block structure (stiff intra-body, soft inter-body coupling) and makes nonlinear material energies first-class, but community consensus on AVBD is unsettled — parameter tuning ("magic numbers") is a known sore point — and committing to it up front bakes in a bet we don't need to make yet. SI also keeps the result more broadly applicable: if DBD works on top of PGS, it works on top of the solver most engines actually ship.

**Fallback:** if SI convergence is unsatisfactory (slow on stacks, cradle fails, stiff-limit blowups we can't regularize away), revisit AVBD as a second-phase solver. The physics work — material model, collapse handling, plasticity, restitution parameterization — lives in the energy and transfers directly; only the outer solver changes.

**AVBD empirical note (2026-04-14).** A toy AVBD implementation on the 1D stack (`experiments/si_1d_stack.py`, since removed) reproduced the magic-number problem cleanly: with a fixed penalty `β = α · max(k, m/h²)`, the solver plateaued at ~5% equilibrium error at soft k and became *less* accurate as `α` was increased past ~10, independent of outer-iteration count. The pairwise local-implicit SI had no such bias on the same problem. This is exactly the AVBD pathology documented in the literature, and confirms the plan: do not adopt AVBD without an adaptive-stiffness scheme.

**Possible AVBD rescue for later: Ando's cubic-barrier dynamic stiffness.** Ryoichi Ando's cubic-barrier contact paper derives a contact stiffness that is a closed-form function of the current elastic configuration — the stiffness adapts to the state of the body rather than being a user-chosen penalty. Folding this into AVBD's `β` would replace the hand-tuned `avbd_beta_factor` with a physically motivated per-contact stiffness that scales correctly with material compliance. This is the most promising direction if we ever revisit AVBD, because it attacks the magic-number problem at its root rather than papering over it with heuristics or adaptive ramps. File under "if we return to AVBD in phase 2".

## Stages

Each stage is a checkpoint — stop, evaluate, update `research.md` running log, then continue.

### 1. Baseline & instrumentation
Pick reference scenes in `samples/collection/`: stable stack, pile, high mass ratio, thin wall, mouse-drag stress. Some may need to be added. Record rigid TGS Soft wall-clock per step, iteration counts, and failure modes. This is the number H1/H2 will be measured against.

### 2. Affine state on the body
Extend `s2Body` (`src/body.h`) with affine fields: `A`, `A0` (substep start), affine velocity (conjugate to `A`), and affine "mass" / generalized inertia. Initialize to identity. Solver still rigid — just carry the state and verify nothing breaks. Serialization/debug draw should visualize `A`.

### 3. Affine integrator
Generalize `s2IntegrateVelocities` / `s2IntegratePositions` / `s2FinalizePositions` in `src/solve_common.c` (or fork into an affine-specific file) to advance `(A, t)` and their velocities. Validate free-flight of a single affine body against rigid (should be identical when no elastic force is active).

### 4. Elastic restoring constraint
Add a per-body soft constraint pulling `A` toward its polar-decomposition rotation `R(A)` (or plastic rest `A_p`) at a chosen Hertz/damping. Plug into the TGS Soft substep loop alongside joint/contact solves. Demo scene: jelly square that settles. Verify rigid limit as Hertz → max.

### 5. Contact with affine bodies
Update contact prepare/solve in the affine TGS path so anchors use `A * localAnchor`, support points go through `A`. Audit `s2PrepareContacts_Soft` and the inner loop in `solve_tgs_soft.c` (the "updated anchors" path is already close to what we need). Re-run stage-1 reference scenes; stacking must hold.

### 6. Plasticity
Add a yield model updating `A_p` when stress on the elastic part exceeds threshold. Verify permanent dents, no energy drift, rigid-limit preserved when yield is ∞.

### 7. Joints
Extend mouse + revolute joints (`src/mouse_joint.c`, `src/revolute_joint.c`) to affine bodies. First pass: project onto the rigid part `R(A)` for the Jacobian. Refine if it causes artifacts.

### 8. Stress tests & numbers
Re-run the stage-1 reference scenes on the affine solver. Record wall-clock ratio vs. rigid, iteration counts, failure modes. Compare to H1/H2 targets.

### 9. Write-up
Collect results, scenes, failure modes. Update `research.md` with verdict on H1–H4.

## Checkpoint log

Progress entries per stage. Date each entry.

### 2026-04-14 — plan created
No code yet. Next action: set up stage-1 reference scenes and baseline measurements.
