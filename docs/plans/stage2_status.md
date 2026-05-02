# Stage 2 (2-body Newton IPC blocks) — work-in-progress status

Snapshot date: 2026-04-18.  Plan file: `docs/plans/hybrid_displacement_solver.md` (stage 2).
Detailed plan we were executing from: `C:\Users\chlewin\.claude\plans\spicy-singing-sphinx.md`.

## What was done (committable)

All in `experiments/biq2d/`:

1. **`solver.py` — extended `_subcell_polygons_world`** to include a
   `corner_xi (4, 2)` field per sub-cell, so the active-edge picker can
   return the ξ of the chosen edge-origin vertex.  Needed downstream to
   build `J_B(ξ_e)` for the 2-body cross-block.

2. **`solver.py` — refactored the inside/outside edge-selection logic**
   out of `_body_body_active_contacts` into a shared helper
   `_pick_active_edge(P, sc_poly, dhat) -> (n_e, V_e, xi_e) | None`.
   The seam-handling rules (skip interior in OUTSIDE branch, argmax
   over EXTERIOR-only in INSIDE branch) are preserved.  Stage-1
   per-body builder now calls the helper; behaviour unchanged
   (regressions pass).

3. **`solver.py` — added `_body_body_active_pair(q_a, h_a, q_b, h_b,
   polys_a, polys_b, dhat)`**.  Returns
   `[(Jn_a (12,), Jn_b (12,), g_offset=0)]` covering BOTH directions
   (A-vertex → B-sub-cell and B-vertex → A-sub-cell).  Sign convention:
   - direction 1: `Jn_a = +nᵀJ_A(ξ_A)`, `Jn_b = -nᵀJ_B(ξ_e)`.
   - direction 2: `Jn_b = +nᵀJ_B(ξ_B)`, `Jn_a = -nᵀJ_A(ξ_e)`.

   `g_offset` is identically zero because both terms are q-linear with
   no constant once both bodies' DoFs are in the active set.

4. **`solver.py` — added `_active_edge_graph(state, dt, dhat,
   kappa_body)`**.  Per-substep builder; iterates ordered pairs `i<j`
   over dynamic bodies; for each pair calls
   `_body_body_active_pair(q_init_i, q_init_j, …)` from the
   `q_pre + dt·v_pre` snapshot.  Returns `(edges, isolated)`.  Static
   bodies are excluded from `isolated` and from edge enumeration on
   the outer index — pairs (static, dynamic) end up with the dynamic
   body in `isolated`, where Stage-1 per-body GS handles the contact
   against the frozen static partner.  This is the `vbd_edge_step_fq`
   "static partner ⇒ fall back to per-body" path, which is correct.

5. **`solver.py` — added `vbd_edge_step_fq(body_a, body_b, idx_a,
   idx_b, state, contacts_pair, …)`**.  24-D Newton over `[v_a; v_b]`:
   - block-diagonal solo terms (inertia, gravity, elastic, joints,
     floor barrier, non-partner body contacts);
   - cross-block coupling ONLY from `contacts_pair` (rank-1 outer
     `bpp · outer([Jn_a; Jn_b], [Jn_a; Jn_b])` per contact);
   - α_max feasibility clamp aggregated over all contact families;
   - Armijo line search on the joint IP energy.

   Static-partner cases short-circuit to `vbd_body_step_fq` for the
   non-static side.

6. **`solver.py` — extended `Params` with `block_mode: str =
   "per_body"`** and added the dispatch in `step_vbd_fq`.  Default
   is the validated Stage-1 path; opt-in `"per_edge"` builds the edge
   graph once per substep, then per outer sweep iterates edges
   forward + backward and isolated bodies forward + backward.

7. **`experiments/biq2d/test_2body_block_fq.py` — new file** with six
   scenarios: static-A stack drop, dynamic stack, side collision,
   3-body stack, per_body↔per_edge parity, convergence advantage.
   File is in place but is currently failing (see below).

## Regression status (Stage-1 path)

All Stage-1 regressions still pass with the default `block_mode`:

- `tests_fq` ✓
- `test_floor_fq` ✓
- `test_barrier_contact_fq` ✓
- `test_body_contact_fq` ✓
- `test_cantilever_fq` ✓

So the refactor + new dispatch did not regress anything on the
validated path.  Safe to commit.

## What is broken

The new `block_mode="per_edge"` path **diverges on the both-dynamic
stack drop** (`test_two_body_stack_dynamic_per_edge`):

```
step 285  cy_A=0.480 cy_B=1.463  vy_A=-0.013 vy_B=-1.11   (just touching)
step 286  cy_A=0.481 cy_B=1.459  vy_A=+0.092 vy_B=-1.00   ← A reverses upward
step 290  cy_A=0.485 cy_B=1.448  vy_A=+0.31  vy_B=-0.44
step 320  cy_A=0.595 cy_B=1.631  vy_A=+1.18  vy_B=+2.12   ← bodies launched
…
final     cy_A ≈ cy_B ≈ 0.51 (fully merged) for κ_body ∈ {10, 100, 1000}
```

The static-A scene (`test_two_body_stack_drop_per_edge`) passes,
because `vbd_edge_step_fq` short-circuits to per-body when one side is
static — no cross-coupling ever runs.

## Diagnosis

The 24-D Hessian has a near-soft "common mode" — translate both
bodies up by the same amount.  In that mode:
- `body-body barrier` Hessian contribution is 0 (the gap is
  invariant under common translation, so `outer([Jn_a; Jn_b], …)`
  has 0 projection on the (1,1,…,1,1,…) common direction);
- `floor barrier` Hessian acts only on body A's c_y (≈ 53);
- `inertia` contributes `M_a + M_b` ≈ 2.

So the common-mode eigenvalue ≈ 55 — versus the diff-mode (bodies
move oppositely along `[Jn_a; Jn_b]`) eigenvalue ≈ 4273 dominated by
the 12 active body-body contacts at κ_body=1000.  Condition number
≈ 78.

Worked example at first contact (g≈0, A at floor with light penetration):
- `grad_a_cy` ≈ +0.04 (gravity) − 210 (floor pushing A up)
                + 130 (body dir 1 pushing A down) + 130 (body dir 2)
                = +50.8
- `grad_b_cy` ≈ +0.04 (gravity) − 130 (body dir 1 pushing B up)
                − 130 (body dir 2)  = −261

Newton solve of the (c_a_y, c_b_y) sub-system:
```
H_sub = [[2178, -2124], [-2124, 2125]]
grad_sub = [+51, -261]
dv_sub = -H_sub^{-1} · grad_sub ≈ [+1.27, +1.39]   ← BOTH UP
```

The Newton step is mostly in the soft common mode: it lifts A away
from the floor (correct response to floor barrier alone) and drags
B with it (because the cross-block makes B's c_y tightly linked to
A's c_y in the soft mode).  A position change of ~8 mm in one
substep is enough to pop the bodies apart, then gravity drags them
back, and the cycle repeats — visible as the upward bounce in the
trace.

## Why per-body GS doesn't see this

Stage 1's `vbd_body_step_fq` only sees one body's 12-D block at a
time.  Each body's solve has NO common mode — the 12-D Hessian is
strictly the SUM of inertia + (its own) elastic + (its own) floor +
(its own) body barriers.  Body A's solve correctly drives c_a_y
DOWN under B's contact pressure (the partner B is frozen, so the
contact gradient is unidirectional).  GS reciprocity through B's
own visit closes the loop.  No 24-D soft mode, no pathology.

## Why standard IPC handles this

IPC and similar 2-body / global Newton methods solve in **position
(q) space**, where the inertia Hessian is `M / dt²` ≈ `M·240²` ≈
57600 — large enough to dominate the common mode regardless of
contact stiffness.  We solve in **velocity (v) space**, where
inertia is just `M` ≈ 1.  This is fine for per-body GS (no soft
mode) but exposes the soft-mode pathology in 2-body blocks.

This is the root cause.  The pilot Stage-1 derivation chose v-space
specifically to keep the gradient/Hessian assembly simple and to
match the per-body Newton signature; the consequence — that
2-body blocks need different treatment — wasn't anticipated in the
plan.

## Options to resolve (next-context decision point)

A. **Move to q-space (positional Newton) inside `vbd_edge_step_fq`
   only.**  Per-body keeps the v-space formulation; per-edge solves
   `min_x ½(x − x̃)ᵀ M (x − x̃) / dt² + Φ(x)`.  This is the IPC
   standard.  Solo per-body terms also need re-derivation in q-space
   for the edge solver (gravity contributes a constant gradient,
   elastic energy is unchanged, barriers unchanged).  Cleanest, but
   touches many lines.

B. **Add a regulariser to the soft mode.**  Damps the common-mode
   eigenvalue artificially.  Crude; would need careful tuning to not
   regress diff-mode convergence.  Not principled.

C. **Accept slower convergence in v-space and use more Newton
   iters per edge visit.**  Possibly works (the soft-mode step is
   not WRONG, just slow); needs measurement.  Was about to test
   `(sweeps=20, newton=20)` vs current `(5, 5)` when this status
   doc was started.

D. **Drop the cross-block term (back to per-body equivalent) and
   abandon Stage 2.**  But then there's no point to "per_edge" mode.

Recommended: **try (C) first** as a cheap measurement; if even
generous iter budgets don't converge, **commit to (A)** as the
proper fix and refactor `vbd_edge_step_fq` into q-space.

## Restart instructions

1. Re-read this file, `docs/plans/hybrid_displacement_solver.md`,
   and `C:\Users\chlewin\.claude\plans\spicy-singing-sphinx.md`.
2. Working tree state:
   - `experiments/biq2d/solver.py` — modified (committable; Stage-1
     regressions all pass).
   - `experiments/biq2d/test_2body_block_fq.py` — new file (tests
     fail on `test_two_body_stack_dynamic_per_edge`; do NOT commit
     until the per-edge bug is resolved).
3. The first decision is option (C) vs (A).  Quick test for (C):
   ```
   PYTHONIOENCODING=utf-8 PYTHONPATH=. python /tmp/debug_per_edge3.py
   ```
   (script is gone — recreate from this doc's "What is broken"
   section; sweeps × newton grid).  If even `(20, 50)` diverges,
   commit to (A).
4. If (A) is taken, the q-space refactor is local to
   `vbd_edge_step_fq` (about 200 lines).  Keep `vbd_body_step_fq`
   in v-space — no need to touch the validated Stage-1 path.

## Files produced this session (for git)

```
modified:   experiments/biq2d/solver.py
new file:   experiments/biq2d/test_2body_block_fq.py
new file:   docs/plans/stage2_status.md   (this file)
```

`solver.py` is safe to commit independently if Stage 2 is paused —
it adds Stage-2 plumbing (helpers + `vbd_edge_step_fq` + dispatch)
behind an opt-in `Params.block_mode` that defaults to the validated
Stage-1 path.  The test file should NOT be committed yet (failing
tests).
