# Plans

Active implementation and experiment plans live here. Plans describe
**what we intend to try**; once an experiment produces a finding,
record it in `../insights.md` rather than editing the plan in place.

## Convention

Plans do **not** need to be backlinked from the permanent docs
(`research.md`, `code_map.md`, `followups.md`, `insights.md`). They are
transient: they get executed, their findings flow into `insights.md`,
and eventually they are deleted or archived. Permanent docs, by
contrast, should cross-reference each other so any one of them is a
discovery entry point into the rest.

New plans go in this folder. Name them by scope
(e.g. `1d_solver_validation.md`, not `plan_foo.md`).

## Current plans

- [plan.md](plan.md) — staged implementation plan for affine bodies in
  solver2d (the main 2D build-out).
- [1d_solver_validation.md](1d_solver_validation.md) — 1D probes that
  close open solver/integrator design questions.
- [1d_material_richness.md](1d_material_richness.md) — 1D probes that
  extend the material model (COR tuning, asymmetric laws, L-stability
  damping budget).
- [hybrid_displacement_solver.md](hybrid_displacement_solver.md) —
  hybrid PGS + 2-body-Newton-block solver with frozen-normal IPC
  barrier contact. Pilot: in-VBD barrier swap, no rigid coupling yet.
