# CLAUDE.md

## Project goals

This repo starts from Erin Catto's **solver2d** (a testbed comparing rigid-body constraint solvers) and is being extended into an experimental **real-time 2D deformable body simulator**.

We build on the **TGS Soft** solver variant: substepped Truncated Gauss-Seidel with compliant (soft) constraints. That pipeline already gives us the two properties deformables need — small effective timesteps via substepping, and frequency/damping-parameterized compliance — so it's our base for bending, stretching, and volumetric constraints.

Upstream solver2d is a comparison harness; we treat the other solver variants (`PGS*`, `XPBD`, `Jacobi`, `TGS_NGS`, `TGS_Sticky`, `SoftStep`) as reference implementations and benchmarking baselines, not production targets.

## Primary reference solver

`src/solve_tgs_soft.c` — study this first. The substep loop structure (prepare once → per-substep integrate/warm-start/solve-with-bias/integrate-positions/optional-relax → finalize) is the template for any new deformable constraint we add.

Supporting math lives in `src/solve_common.c` (`s2PrepareContacts_Soft`, integrators) and `extras/soft_constraint.m`.

## Code organization (summary)

- `include/solver2d/` — public C API (`solver2d.h` entry, `types.h` has `s2SolverType` enum).
- `src/` — library implementation.
  - Runtime core: `world.{h,c}`, `body.{h,c}`, `contact.{h,c}`, `joint.{h,c}`, `shape.{h,c}`, broadphase.
  - Solvers: `solvers.h` declares them; each `solve_*.c` is one variant. TGS Soft is ours.
  - Memory/containers: block + stack allocators, pools, arrays, tables.
- `samples/` — GLFW/ImGui app; scenes under `samples/collection/`. New deformable demos go here.
- `extern/` — vendored glad, jsmn.
- `extras/` — logo and `soft_constraint.m` derivation notes.

## Docs

**Read `docs/code_map.md` first** for a file-by-file index, the full TGS Soft step pipeline walkthrough, and guidance on where to hook in new body types, constraint types, and solver variants.

**Maintain `docs/insights.md`** as a running catalogue of what we have learned about deformable body dynamics through experiments (1D prototype probes, 2D demos, solver comparisons). Every time an experiment produces a non-obvious finding — a stability boundary, a convergence pathology, a material-model caveat, a validated design choice — append a short entry with the setup, the result, and the "so what" for the 2D solver. Prefer updating an existing entry over duplicating. This is the long-lived physics/numerics knowledge base; plan files in `docs/plans/` describe what to try, `insights.md` records what we learned.

**Docs convention.** `docs/` holds **permanent** knowledge docs (`research.md`, `code_map.md`, `followups.md`, `insights.md`) at the top level. Every permanent doc should backlink the others in a "Companions" header so each one is a discovery entry point into the rest — when you add a new permanent doc, cross-reference it from the existing ones. **Plans go in `docs/plans/`** (implementation plans, experiment plans, anything describing "what we intend to try"). Plans are transient and do *not* need to be backlinked from permanent docs — their findings flow into `insights.md` once the experiment runs, and the plan itself eventually gets deleted or archived.

## Conventions

- C99, Box2D-style naming (`s2` prefix, lowerCamelCase functions, `s2Body`-style structs).
- No heap allocations per step — use `world->stackAllocator` for scratch constraint arrays.
- Object lifetime via generational pools (`s2Pool`); always guard iteration with `s2IsFree(&obj->object)`.
- When adding a solver variant: declare in `solvers.h`, implement in `solve_<name>.c`, extend `s2SolverType` in `include/solver2d/types.h`, and add a dispatch case in `s2World_Step` (`src/world.c`).
