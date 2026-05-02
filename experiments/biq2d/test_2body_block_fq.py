"""2-body Newton block (Stage 2) tests for FQ2D bodies.

Six scenarios — same physical setups as `test_body_contact_fq.py` plus a
3-body stack and a per-body-vs-per-edge parity / convergence comparison.
The 2-body block route is selected via `Params.block_mode = "per_edge"`.

The pair-contact gap remains q-linear in BOTH bodies' DoFs once the
normal is frozen, so the line-search feasibility clamp is still one
dot product per active contact.  What's new vs Stage 1 is the
cross-coupled rank-1 Hessian update and the 24-D solve.

Pass criteria mirror the Stage-1 body-body suite:
  - max penetration stays ~0 across all scenes (barrier guarantee),
  - settle position matches per-body within ~dhat (no end-state regression),
  - convergence on a 3-body stack must not be worse than per-body,
  - no inversion (sampled min det J > 0.3).
"""
import time
import numpy as np
from .body import BodyFQ2D
from .solver import (State, Params, step_vbd_fq, _OUTER_XI,
                     _outer_world_pts_from_q, _subcell_polygons_world,
                     _vertex_in_subcell)


# ── helpers (copied from test_body_contact_fq.py) ────────────────────

def _q_of(b):
    return np.concatenate([b.c, b.F, b.G, b.Hx, b.Hy])


def _outer_world(b):
    return _outer_world_pts_from_q(_q_of(b), b.h)


def _max_speed(state):
    out = 0.0
    for b in state.bodies:
        if b.static:
            continue
        v = np.concatenate([b.vc, b.vF, b.vG, b.vHx, b.vHy])
        out = max(out, float(np.linalg.norm(v)))
    return out


def _min_det(state):
    return min(b.min_det_J_sampled(grid_n=9)
               for b in state.bodies if not b.static)


def _max_pair_penetration(state):
    polys_by = {i: _subcell_polygons_world(b) for i, b in enumerate(state.bodies)}
    pmax = 0.0
    for i, ba in enumerate(state.bodies):
        if ba.static:
            continue
        pts = _outer_world(ba)
        for j, bb in enumerate(state.bodies):
            if j == i:
                continue
            for sc in polys_by[j]:
                for P in pts:
                    active, depth, _ = _vertex_in_subcell(P, sc)
                    if active and depth > pmax:
                        pmax = depth
    return pmax


# ── 1. static-A stack drop, per-edge ─────────────────────────────────

def test_two_body_stack_drop_per_edge():
    """Per-edge mode parity check: static A on floor, B drops onto A.

    Static A → no DoF cross-coupling possible; per-edge degenerates to
    per-body for the dynamic B.  This test guards that the static-side
    fallback in `vbd_edge_step_fq` works at all.
    """
    print("\n=== body-body stack drop, per-edge (A static) ===")
    h = 0.5
    k = 2000.0
    A = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35, static=True)
    A.c = np.array([0.0, h])
    B = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
    B.c = np.array([0.0, 2.5])
    state = State(bodies=[A, B], joints=[])
    params = Params(gravity=np.array([0.0, -9.81]), dt=1.0 / 240.0,
                    vbd_sweeps=5, vbd_newton=5, block_mode="per_edge")
    alpha = 10.0 * k * h * h
    barrier_kappa_floor = 100.0
    barrier_kappa_body  = 1000.0
    barrier_dhat  = 0.05 * h
    n_steps = 1500

    pen_max_ever = 0.0
    min_det_ever = float("inf")
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha,
                    barrier_kappa_floor=barrier_kappa_floor, y_floor=0.0,
                    barrier_kappa_body=barrier_kappa_body,
                    barrier_dhat=barrier_dhat)
        pen_max_ever = max(pen_max_ever, _max_pair_penetration(state))
        min_det_ever = min(min_det_ever, _min_det(state))

    sep = B.c[1] - A.c[1]
    print(f"  cy_B={B.c[1]:.4f}  sep={sep:.4f}  (target ~ 2h = {2*h})")
    print(f"  max pen ever   = {pen_max_ever:.4e}  (dhat={barrier_dhat:.4f})")
    print(f"  min_det ever   = {min_det_ever:.4f}")
    print(f"  final |v|      = {_max_speed(state):.4e}")

    assert abs(sep - 2.0 * h) < 1.5 * barrier_dhat, f"sep {sep:.4f}"
    assert pen_max_ever < 0.5 * barrier_dhat, f"max pen {pen_max_ever:.4e}"
    assert min_det_ever > 0.3, f"min_det={min_det_ever:.3f}"
    assert _max_speed(state) < 0.1, "B failed to settle"
    print("  PASS")


# ── 2. dynamic stack, per-edge ───────────────────────────────────────

def test_two_body_stack_dynamic_per_edge():
    """Both bodies dynamic — exercises the 24-D cross-coupled Newton.

    A on the floor compresses under B's weight; B settles 2h above
    A's centre (modulo dhat).  This is the smallest scene where the
    per-edge cross-block actually has DoFs on both sides.
    """
    print("\n=== body-body stack drop, per-edge (both dynamic) ===")
    h = 0.5
    k = 2000.0
    A = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
    A.c = np.array([0.0, h])
    B = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
    B.c = np.array([0.0, 2.5])
    state = State(bodies=[A, B], joints=[])
    params = Params(gravity=np.array([0.0, -9.81]), dt=1.0 / 240.0,
                    vbd_sweeps=5, vbd_newton=5, block_mode="per_edge")
    alpha = 10.0 * k * h * h
    barrier_kappa_floor = 100.0
    barrier_kappa_body  = 1000.0
    barrier_dhat  = 0.05 * h
    n_steps = 1500

    pen_max_ever = 0.0
    min_det_ever = float("inf")
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha,
                    barrier_kappa_floor=barrier_kappa_floor, y_floor=0.0,
                    barrier_kappa_body=barrier_kappa_body,
                    barrier_dhat=barrier_dhat)
        pen_max_ever = max(pen_max_ever, _max_pair_penetration(state))
        min_det_ever = min(min_det_ever, _min_det(state))

    sep = B.c[1] - A.c[1]
    print(f"  cy_A={A.c[1]:.4f}  cy_B={B.c[1]:.4f}  sep={sep:.4f}")
    print(f"  max pen ever   = {pen_max_ever:.4e}  (dhat={barrier_dhat:.4f})")
    print(f"  min_det ever   = {min_det_ever:.4f}")
    print(f"  final |v|      = {_max_speed(state):.4e}")

    assert A.c[1] < h + barrier_dhat, f"A did not compress: cy_A={A.c[1]:.4f}"
    assert sep < 2.0 * h + 2.0 * barrier_dhat, f"sep {sep:.4f} too large"
    assert sep > 1.5 * h, f"sep {sep:.4f} too small (over-compression)"
    assert pen_max_ever < 0.5 * barrier_dhat, f"max pen {pen_max_ever:.4e}"
    assert min_det_ever > 0.3, f"min_det={min_det_ever:.3f}"
    assert _max_speed(state) < 0.2, "failed to settle"
    print("  PASS")


# ── 3. side collision, per-edge ──────────────────────────────────────

def test_two_body_side_collision_per_edge():
    """Two bodies launched along x; no floor; both dynamic.

    Pure cross-block exercise: contact normal is horizontal, both
    bodies have full DoFs.  The 2-body block transports the impact
    through a single coupled descent direction rather than two
    alternating frozen-neighbour solves.
    """
    print("\n=== body-body side collision, per-edge ===")
    h = 0.5
    k = 2000.0
    m = 1.0
    v0 = 2.0
    A = BodyFQ2D(mass=m, half_extent=h, k=k, nu=0.35)
    A.c = np.array([-1.5, 1.0]); A.vc = np.array([+v0, 0.0])
    B = BodyFQ2D(mass=m, half_extent=h, k=k, nu=0.35)
    B.c = np.array([+1.5, 1.0]); B.vc = np.array([-v0, 0.0])
    state = State(bodies=[A, B], joints=[])
    params = Params(gravity=np.array([0.0, 0.0]), dt=1.0 / 240.0,
                    vbd_sweeps=5, vbd_newton=5, block_mode="per_edge")
    alpha = 10.0 * k * h * h
    barrier_kappa = 1000.0
    barrier_dhat  = 0.05 * h
    n_steps = 800

    p0_x = m * (A.vc[0] + B.vc[0])
    p0_abs = m * (abs(A.vc[0]) + abs(B.vc[0]))
    closest_dx = float("inf")
    crossed = False
    pen_max_ever = 0.0
    min_det_ever = float("inf")
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha,
                    barrier_kappa_body=barrier_kappa,
                    barrier_dhat=barrier_dhat)
        if not np.all(np.isfinite(_q_of(A))) or not np.all(np.isfinite(_q_of(B))):
            assert False, f"diverged at step {i}"
        dx = B.c[0] - A.c[0]
        closest_dx = min(closest_dx, dx)
        if dx < 0.0:
            crossed = True
        pen_max_ever = max(pen_max_ever, _max_pair_penetration(state))
        min_det_ever = min(min_det_ever, _min_det(state))

    p_final_x = m * (A.vc[0] + B.vc[0])
    print(f"  closest |dcx|  = {closest_dx:.4f}  (target > 0)")
    print(f"  max pen ever   = {pen_max_ever:.4e}")
    print(f"  min_det ever   = {min_det_ever:.4f}")
    print(f"  px_final       = {p_final_x:+.4e}  (target ~ 0)")
    print(f"  vc_A={A.vc}, vc_B={B.vc}")

    assert not crossed, "centers crossed - tunneling!"
    assert pen_max_ever < 0.5 * barrier_dhat, f"max pen {pen_max_ever:.4e}"
    assert min_det_ever > 0.2, f"min_det={min_det_ever:.3f}"
    assert abs(p_final_x) < 0.05 * p0_abs, f"net momentum drift {p_final_x:.3e}"
    assert A.vc[0] < 0.0 and B.vc[0] > 0.0, "bodies did not rebound"
    print("  PASS")


# ── 4. three-body stack drop, per-edge ───────────────────────────────

def _three_body_stack(block_mode, n_steps, vbd_sweeps=5, vbd_newton=5,
                       record_pen=True):
    """Run a 3-body stack: static A on floor, B above A, C above B,
    drops one onto the next.  Returns (state, max_pen_ever, min_det_ever,
    settle_step, wall_time)."""
    h = 0.5
    k = 2000.0
    A = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35, static=True)
    A.c = np.array([0.0, h])
    B = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
    B.c = np.array([0.0, 3.0 * h])         # touching A from above
    C = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
    C.c = np.array([0.0, 5.5 * h])         # drops onto B
    state = State(bodies=[A, B, C], joints=[])
    params = Params(gravity=np.array([0.0, -9.81]), dt=1.0 / 240.0,
                    vbd_sweeps=vbd_sweeps, vbd_newton=vbd_newton,
                    block_mode=block_mode)
    alpha = 10.0 * k * h * h
    barrier_kappa_floor = 100.0
    barrier_kappa_body  = 1000.0
    barrier_dhat  = 0.05 * h

    pen_max_ever = 0.0
    min_det_ever = float("inf")
    settle_step = None
    t0 = time.perf_counter()
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha,
                    barrier_kappa_floor=barrier_kappa_floor, y_floor=0.0,
                    barrier_kappa_body=barrier_kappa_body,
                    barrier_dhat=barrier_dhat)
        if record_pen:
            pen_max_ever = max(pen_max_ever, _max_pair_penetration(state))
            min_det_ever = min(min_det_ever, _min_det(state))
        if settle_step is None and _max_speed(state) < 5e-3:
            settle_step = i
    wall = time.perf_counter() - t0
    return state, pen_max_ever, min_det_ever, settle_step, wall, barrier_dhat


def test_three_body_stack_drop():
    """A static, B and C dynamic; B already at A's contact, C drops.

    Middle body B lives in TWO active edges (A,B) and (B,C) — the
    benchmark scenario for 2-body block convergence.  Per-edge mode
    sees A as a static partner (degenerates to per-body for that edge)
    and B-C as a true 24-D coupled solve.  The cross-block coupling
    propagates C's impact into B's deformation in one Newton step,
    rather than waiting for the next GS sweep.
    """
    print("\n=== 3-body stack drop, per-edge ===")
    h = 0.5
    state, pen_max, min_det, settle_step, wall, dhat = _three_body_stack(
        "per_edge", n_steps=1800)

    A, B, C = state.bodies
    sep_AB = B.c[1] - A.c[1]
    sep_BC = C.c[1] - B.c[1]
    print(f"  cy_A={A.c[1]:.4f}  cy_B={B.c[1]:.4f}  cy_C={C.c[1]:.4f}")
    print(f"  sep AB = {sep_AB:.4f}  sep BC = {sep_BC:.4f}  (~ 2h = {2*h})")
    print(f"  max pen ever   = {pen_max:.4e}  (dhat={dhat:.4f})")
    print(f"  min_det ever   = {min_det:.4f}")
    print(f"  settle step    = {settle_step}")
    print(f"  final |v|      = {_max_speed(state):.4e}")
    print(f"  wall time      = {wall:.2f}s for 1800 steps")

    assert pen_max < 0.5 * dhat, f"max pen {pen_max:.4e}"
    assert abs(sep_AB - 2.0 * h) < 2.0 * dhat, f"sep AB {sep_AB:.4f}"
    assert abs(sep_BC - 2.0 * h) < 2.0 * dhat, f"sep BC {sep_BC:.4f}"
    assert min_det > 0.3, f"min_det={min_det:.3f}"
    assert _max_speed(state) < 0.05, "failed to settle"
    print("  PASS")


# ── 5. parity (per-body vs per-edge) ─────────────────────────────────

def test_per_body_vs_per_edge_parity():
    """Static-A drop: per-body and per-edge must reach the same end-state
    (within ~½·dhat).  Per-edge changes convergence rate, not behaviour.
    """
    print("\n=== parity: per-body vs per-edge end-state ===")
    h = 0.5
    k = 2000.0
    barrier_dhat = 0.05 * h

    def run(mode):
        A = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35, static=True)
        A.c = np.array([0.0, h])
        B = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
        B.c = np.array([0.0, 2.5])
        state = State(bodies=[A, B], joints=[])
        params = Params(gravity=np.array([0.0, -9.81]), dt=1.0 / 240.0,
                        vbd_sweeps=5, vbd_newton=5, block_mode=mode)
        alpha = 10.0 * k * h * h
        pen_max = 0.0
        for _ in range(1500):
            step_vbd_fq(state, params, alpha=alpha,
                        barrier_kappa_floor=100.0, y_floor=0.0,
                        barrier_kappa_body=1000.0,
                        barrier_dhat=barrier_dhat)
            pen_max = max(pen_max, _max_pair_penetration(state))
        return state, pen_max

    sb, pb = run("per_body")
    se, pe = run("per_edge")
    cy_pb = sb.bodies[1].c[1]
    cy_pe = se.bodies[1].c[1]
    print(f"  per_body cy_B={cy_pb:.6f}  pen_max={pb:.4e}")
    print(f"  per_edge cy_B={cy_pe:.6f}  pen_max={pe:.4e}")
    print(f"  |cy diff| = {abs(cy_pb - cy_pe):.4e}  (tol = {0.5 * barrier_dhat:.4e})")
    print(f"  |pen diff| = {abs(pb - pe):.4e}")

    assert abs(cy_pb - cy_pe) < 0.5 * barrier_dhat, (
        f"end-state diverges: cy_pb={cy_pb:.4e} cy_pe={cy_pe:.4e}")
    assert abs(pb - pe) < 0.2 * barrier_dhat, (
        f"max pen disagrees: pb={pb:.4e} pe={pe:.4e}")
    print("  PASS")


# ── 6. convergence advantage on 3-body stack ─────────────────────────

def test_per_edge_convergence_advantage():
    """3-body stack run with REDUCED sweep budget: does per-edge settle
    sooner than per-body?

    Pass condition is conservative — per-edge mode must NOT regress
    (settle step <= per-body's, within +20 step tolerance).  A clear
    advantage gets recorded in `docs/insights.md`; a tie/loss is also
    a valid finding.
    """
    print("\n=== convergence: per-body vs per-edge on 3-body stack ===")
    n_steps = 2400
    sweeps = 1
    newton = 3

    state_pb, pen_pb, det_pb, settle_pb, wall_pb, dhat = _three_body_stack(
        "per_body", n_steps=n_steps, vbd_sweeps=sweeps, vbd_newton=newton)
    state_pe, pen_pe, det_pe, settle_pe, wall_pe, _ = _three_body_stack(
        "per_edge", n_steps=n_steps, vbd_sweeps=sweeps, vbd_newton=newton)

    A_pb, B_pb, C_pb = state_pb.bodies
    A_pe, B_pe, C_pe = state_pe.bodies
    print(f"  per_body  settle={settle_pb}  cy_B={B_pb.c[1]:.4f}  cy_C={C_pb.c[1]:.4f}  "
          f"|v|={_max_speed(state_pb):.3e}  pen={pen_pb:.3e}  wall={wall_pb:.2f}s")
    print(f"  per_edge  settle={settle_pe}  cy_B={B_pe.c[1]:.4f}  cy_C={C_pe.c[1]:.4f}  "
          f"|v|={_max_speed(state_pe):.3e}  pen={pen_pe:.3e}  wall={wall_pe:.2f}s")

    if settle_pb is not None and settle_pe is not None:
        delta = settle_pe - settle_pb
        print(f"  per_edge settle delta = {delta:+d} steps "
              f"({'faster' if delta < 0 else 'slower' if delta > 0 else 'tie'})")
    print(f"  wall ratio per_edge/per_body = {wall_pe / wall_pb:.2f}x "
          f"(expected > 1 due to 24x24 solve)")

    assert pen_pe < dhat, f"per_edge max pen {pen_pe:.4e} above dhat"
    assert det_pe > 0.2, f"per_edge inversion: min_det={det_pe:.3f}"
    if settle_pe is not None and settle_pb is not None:
        assert settle_pe <= settle_pb + 20, (
            f"per_edge convergence regression: settle pe={settle_pe} "
            f"vs pb={settle_pb}")
    print("  PASS")


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_two_body_stack_drop_per_edge()
    test_two_body_stack_dynamic_per_edge()
    test_two_body_side_collision_per_edge()
    test_three_body_stack_drop()
    test_per_body_vs_per_edge_parity()
    test_per_edge_convergence_advantage()
    print("\nAll FQ2D 2-body block tests passed.")
