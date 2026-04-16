# Solver2D Code Map

A quick index of the codebase for navigating during deformable-body work. Paths are relative to repo root.

**Companions:** `research.md` (theory, hypotheses, open questions for the DBD project), `insights.md` (running catalogue of experimental findings), `followups.md` (downstream applications of DBD material state), `plans/` (active implementation and experiment plans).

## Layout

- `include/solver2d/` — public C API headers.
- `src/` — library implementation (C).
- `samples/` — GLFW + ImGui sample app, scenes under `samples/collection/`.
- `extern/` — vendored deps (glad, jsmn).
- `extras/` — misc (logo, `soft_constraint.m` Mathematica/Octave notes on soft constraints).

## Public API (`include/solver2d/`)

- `solver2d.h` — world/body/shape/joint entry points. Key call: `s2World_Step(world, dt, velIters, posIters, warmStart)`.
- `types.h` — `s2WorldDef`, `s2BodyDef`, `s2ShapeDef`, `s2SolverType` enum (selects which solver `s2World_Step` dispatches to).
- `math.h` / `aabb.h` / `distance.h` / `geometry.h` / `hull.h` / `manifold.h` — math, shapes, GJK/SAT, contact manifolds.
- `joint_types.h` — mouse + revolute joint defs.
- `id.h` — opaque handles (`s2WorldId`, `s2BodyId`, …).
- `constants.h` — tunables (e.g. `s2_contactHertz`, `s2_jointHertz`, Baumgarte limits).
- `callbacks.h`, `debug_draw.h`, `timer.h`, `color.h` — misc.

## Core runtime (`src/`)

### World / objects
- `world.{h,c}` — `s2World` struct, `s2CreateWorld`, `s2World_Step` dispatcher (switch on `solverType` around line 208 in `world.c`).
- `body.{h,c}` — `s2Body` (note soft-solver fields: `deltaPosition`, `deltaPosition0`, `rot0`, `linearVelocity0`, Jacobi scratch `dv`/`dw`).
- `shape.{h,c}` — shape storage + queries.
- `contact.{h,c}` — contact graph edges, manifold caching, contact lifecycle.
- `joint.{h,c}`, `mouse_joint.c`, `revolute_joint.c` — joint storage + per-joint solver variants (`s2PrepareJoint_Soft`, `s2SolveJoint_Soft`, `s2WarmStartJoint`).
- `broad_phase.{h,c}`, `dynamic_tree.c`, `aabb.c` — broadphase AABB tree.
- `distance.c`, `manifold.c`, `geometry.c`, `hull.c`, `math.c` — collision + math primitives.

### Memory / containers
- `allocate.{h,c}`, `block_allocator.{h,c}`, `stack_allocator.{h,c}` — per-step scratch, pooled blocks.
- `pool.{h,c}` — generational object pools backing bodies/contacts/joints/shapes.
- `array.{h,c}`, `table.{h,c}` — small dynamic arrays and hash table.

### Solvers (`solvers.h` declares them all)
- `solve_common.c` — shared helpers: `s2IntegrateVelocities`, `s2IntegratePositions`, `s2FinalizePositions`, `s2PrepareContacts_PGS`, `s2PrepareContacts_Soft`, `s2WarmStartContacts`, `s2SolveContact_NGS`, `s2StoreContactImpulses`. Defines `s2ContactConstraint` / `s2ContactConstraintPoint`.
- `solve_pgs.c` — baseline sequential-impulse PGS.
- `solve_pgs_ngs.c` — PGS + non-linear Gauss-Seidel position solve.
- `solve_pgs_ngs_block.c` — block (2-contact) PGS + NGS.
- `solve_pgs_soft.c` — PGS with soft (compliant) contact constraints.
- `solve_jacobi.c` — Jacobi-style parallelizable variant.
- `solve_xpbd.c` — Extended Position-Based Dynamics.
- `solve_tgs_ngs.c` — TGS (substepped) + NGS position correction.
- **`solve_tgs_soft.c`** — TGS with soft/compliant contacts. **Primary reference for our deformable work.** Structure: prepare contacts/joints once, then a substep loop doing `IntegrateVelocities → warm start → solve joints+contacts (with bias) → IntegratePositions → optional relaxation pass (no bias)`, then `FinalizePositions` + store impulses. Uses updated anchors each substep.
- `solve_tgs_sticky.c` — TGS variant with sticky contact behavior.
- `solve_soft_step.c` — experimental soft-step integrator.

## Samples (`samples/`)
- `main.cpp`, `sample.{h,cpp}`, `settings.{h,cpp}`, `draw.{h,cpp}` — framework.
- `collection/` — scenes: `sample_contact.cpp`, `sample_far.cpp`, `sample_joints.cpp`, plus `human.{h,cpp}` ragdoll. This is where to add new deformable-body demos.

## Step pipeline (TGS Soft)

Selected in `s2WorldDef.solverType = s2_solverTGS_Soft`. Per `s2World_Step`:
1. Broadphase update → contact creation/destruction → manifold update.
2. Build `s2ContactConstraint` array on the stack allocator.
3. `s2PrepareContacts_Soft` computes per-point mass/bias/impulse coefficients from `contactHertz` (clamped to `0.25 * inv_h`).
4. `s2PrepareJoint_Soft` per joint.
5. Substep loop × `iterations`:
   - `s2IntegrateVelocities(h)` — gravity, forces, damping.
   - Optional warm start (joints + contacts).
   - Solve joints + contacts with `useBias = true` (position correction folded into velocity via Baumgarte-like bias).
   - `s2IntegratePositions(h)` — advances `deltaPosition` and `rot`.
   - Optional relaxation pass with `useBias = false` when `extraIterations > 0`.
6. `s2FinalizePositions` commits `origin`/`position` from accumulated `deltaPosition`.
7. `s2StoreContactImpulses` writes warm-start impulses back to contact manifolds.

## Where to extend for deformables

- New body representation: extend `s2Body` or add a parallel particle/node struct in `body.{h,c}`; register a new pool in `s2World`.
- New constraint type (distance, bending, volume): model after `joint.{h,c}` + a `solve_*_soft` routine, or add directly into the TGS Soft substep loop in `solve_tgs_soft.c`.
- Compliance parameters: follow the `contactHertz`/`jointHertz` pattern in `s2PrepareContacts_Soft` (see `extras/soft_constraint.m` for the math).
- New solver variant: add prototype to `solvers.h`, implement `solve_<name>.c`, add enum entry in `types.h` (`s2SolverType`), add case in the `s2World_Step` dispatch switch in `world.c`.
- Demo scene: add a file under `samples/collection/` and register it in the samples CMake list.
