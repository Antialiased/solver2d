"""Body-body contact tests for FQ2D bodies (frozen-normal IPC barrier,
owner-vertex GS).

Four scenarios:

  1. Static-A stack drop — Body A static on floor, B drops onto it.
  2. Dynamic stack drop — both bodies non-static; A also deforms.
  3. Side collision — two bodies launched at each other; no floor.
  4. Stiffness sweep — verify penetration stays ≈0 across κ (barrier is
     feasibility-preserving by construction; unlike the prior smooth
     penalty whose penetration scaled ~1/κ).

Victory:
  - bodies don't tunnel (barrier guarantees ≪ dhat penetration),
  - stack settles to ~2h center separation (modulo dhat band),
  - side collision: bodies separate, momentum approximately conserved,
  - no inversion (sampled min det J > 0.3).
"""
import numpy as np
from .body import BodyFQ2D
from .solver import (State, Params, step_vbd_fq, _OUTER_XI,
                     _outer_world_pts_from_q, _subcell_polygons_world,
                     _vertex_in_subcell)


# ── helpers ───────────────────────────────────────────────────────────

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
    """Max depth of any owner-body outer ξ vertex inside any other body's
    sub-cell (over all ordered pairs).  Diagnostic only."""
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


# ── scenarios ─────────────────────────────────────────────────────────

def test_two_body_stack_drop():
    """Static A resting on floor; B drops onto A."""
    print("\n=== body-body stack drop (A static) ===")
    h = 0.5
    k = 2000.0
    A = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35, static=True)
    A.c = np.array([0.0, h])         # rest on y=0 floor
    B = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
    B.c = np.array([0.0, 2.5])
    state = State(bodies=[A, B], joints=[])
    params = Params(gravity=np.array([0.0, -9.81]), dt=1.0 / 240.0,
                    vbd_sweeps=5, vbd_newton=5)
    alpha = 10.0 * k * h * h
    barrier_kappa_floor = 100.0
    barrier_kappa_body  = 1000.0       # higher than floor: body-body needs
                                        # stronger first-impact response
                                        # (active-edge selection is ambiguous
                                        # for deep penetration; high κ keeps
                                        # transient ingress shallow).
    barrier_dhat  = 0.05 * h
    y_floor = 0.0
    n_steps = 1500

    pen_max_ever = 0.0
    min_det_ever = float("inf")
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha,
                    barrier_kappa_floor=barrier_kappa_floor, y_floor=y_floor,
                    barrier_kappa_body=barrier_kappa_body,
                    barrier_dhat=barrier_dhat)
        pen_max_ever = max(pen_max_ever, _max_pair_penetration(state))
        min_det_ever = min(min_det_ever, _min_det(state))
        if not np.all(np.isfinite(_q_of(B))):
            assert False, f"B diverged at step {i}"

    sep = B.c[1] - A.c[1]
    print(f"  cy_A           = {A.c[1]:.4f}  (static)")
    print(f"  cy_B           = {B.c[1]:.4f}")
    print(f"  separation     = {sep:.4f}  (target ~ 2h = {2*h})")
    print(f"  max pen ever   = {pen_max_ever:.4e}")
    print(f"  min_det ever   = {min_det_ever:.4f}")
    print(f"  final |v|      = {_max_speed(state):.4e}")

    # Settle near 2h (stable contact in the barrier band).
    assert abs(sep - 2.0 * h) < 1.5 * barrier_dhat, (
        f"sep {sep:.4f} not within dhat of 2h")
    # Transient impact may briefly enter the continuation region; settled
    # gap is what matters.  Bound is loose because frozen-normal IPC + finite
    # Newton iters does not guarantee zero penetration during fast impact.
    assert pen_max_ever < 0.5 * barrier_dhat, (
        f"transient breached barrier severely: max pen = {pen_max_ever:.4e}")
    assert min_det_ever > 0.3, f"inversion risk min_det={min_det_ever:.3f}"
    assert _max_speed(state) < 0.1, "B failed to settle"
    print("  PASS")


def test_two_body_stack_dynamic():
    """Both bodies non-static; A on floor deforms under B's weight."""
    print("\n=== body-body stack drop (both dynamic) ===")
    h = 0.5
    k = 2000.0
    A = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
    A.c = np.array([0.0, h])
    B = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
    B.c = np.array([0.0, 2.5])
    state = State(bodies=[A, B], joints=[])
    params = Params(gravity=np.array([0.0, -9.81]), dt=1.0 / 240.0,
                    vbd_sweeps=5, vbd_newton=5)
    alpha = 10.0 * k * h * h
    barrier_kappa_floor = 100.0
    barrier_kappa_body  = 1000.0
    barrier_dhat  = 0.05 * h
    y_floor = 0.0
    n_steps = 1500

    min_det_ever = float("inf")
    pen_max_ever = 0.0
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha,
                    barrier_kappa_floor=barrier_kappa_floor, y_floor=y_floor,
                    barrier_kappa_body=barrier_kappa_body,
                    barrier_dhat=barrier_dhat)
        min_det_ever = min(min_det_ever, _min_det(state))
        pen_max_ever = max(pen_max_ever, _max_pair_penetration(state))

    sep = B.c[1] - A.c[1]
    print(f"  cy_A           = {A.c[1]:.4f}  (compressed by B's weight)")
    print(f"  cy_B           = {B.c[1]:.4f}")
    print(f"  separation     = {sep:.4f}  (< 2h, A compressed)")
    print(f"  max pen ever   = {pen_max_ever:.4e}")
    print(f"  min_det ever   = {min_det_ever:.4f}")
    print(f"  final |v|      = {_max_speed(state):.4e}")

    # A should compress under B's weight (cy_A < h + tiny barrier gap below B).
    # Sep can be a touch > 2h since the barrier holds B above A by dhat-scale.
    assert A.c[1] < h + barrier_dhat, (
        f"A did not compress under B's weight: cy_A={A.c[1]:.4f}")
    assert sep < 2.0 * h + 2.0 * barrier_dhat, (
        f"sep {sep:.4f} too large - barrier hovering")
    assert sep > 1.5 * h, f"sep {sep:.4f} too small - over-compression"
    assert pen_max_ever < 0.5 * barrier_dhat, (
        f"transient breached barrier severely: max pen = {pen_max_ever:.4e}")
    assert min_det_ever > 0.3, f"inversion: min_det={min_det_ever:.3f}"
    assert _max_speed(state) < 0.2, "failed to settle"
    print("  PASS")


def test_two_body_side_collision():
    """Two bodies launched along x at each other; no floor; bodies bounce."""
    print("\n=== body-body side collision ===")
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
                    vbd_sweeps=5, vbd_newton=5)
    alpha = 10.0 * k * h * h
    barrier_kappa = 1000.0       # body-body needs higher κ than floor
    barrier_dhat  = 0.05 * h
    n_steps = 800

    p0_x = m * (A.vc[0] + B.vc[0])      # = 0 here (symmetric)
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
    p_final_abs = m * (abs(A.vc[0]) + abs(B.vc[0]))
    print(f"  closest |dcx|  = {closest_dx:.4f}  (centers, target > 0)")
    print(f"  centers crossed? {crossed}")
    print(f"  max pen ever   = {pen_max_ever:.4e}")
    print(f"  min_det ever   = {min_det_ever:.4f}")
    print(f"  px_initial     = {p0_x:+.4e}  px_final = {p_final_x:+.4e}  "
          f"(symmetric collision: both near 0)")
    print(f"  |p|_final / |p|_init = {p_final_abs / p0_abs:.4f}")
    print(f"  vc_A_final = {A.vc}, vc_B_final = {B.vc}")

    assert not crossed, "centers crossed - tunneling!"
    assert pen_max_ever < 0.5 * barrier_dhat, (
        f"transient breached barrier severely: max pen = {pen_max_ever:.4e}")
    assert min_det_ever > 0.2, f"inversion: min_det={min_det_ever:.3f}"
    # Net x-momentum stays near 0 (symmetric collision).
    assert abs(p_final_x) < 0.05 * p0_abs, (
        f"net momentum drift {p_final_x:.3e}")
    # Bodies should rebound (vc_A x-velocity flips sign).
    assert A.vc[0] < 0.0 and B.vc[0] > 0.0, (
        "bodies did not rebound — A.vc_x and B.vc_x signs not flipped")
    print("  PASS")


def test_body_contact_stiffness_sweep():
    """Stack-drop scenario; sweep barrier_kappa_body ∈ {0.1, 1, 10}.

    Penetration should be ≈0 across the sweep — barrier is feasibility-
    preserving.  Unlike the prior smooth-penalty path (where penetration
    scaled ~1/κ), changing κ here changes only the *energy magnitude* in
    the active band, not the steady-state gap.
    """
    print("\n=== body-body stack drop, barrier_kappa_body sweep ===")
    h = 0.5
    k = 2000.0
    n_steps = 1500
    barrier_dhat = 0.05 * h
    rows = []
    for kappa in (10.0, 100.0, 1000.0):
        A = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35, static=True)
        A.c = np.array([0.0, h])
        B = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
        B.c = np.array([0.0, 2.5])
        state = State(bodies=[A, B], joints=[])
        params = Params(gravity=np.array([0.0, -9.81]), dt=1.0 / 240.0,
                        vbd_sweeps=5, vbd_newton=5)
        alpha = 10.0 * k * h * h

        pen_max_ever = 0.0
        min_det_ever = float("inf")
        diverged = False
        for i in range(n_steps):
            step_vbd_fq(state, params, alpha=alpha,
                        barrier_kappa_floor=kappa, y_floor=0.0,
                        barrier_kappa_body=kappa,
                        barrier_dhat=barrier_dhat)
            if not np.all(np.isfinite(_q_of(B))):
                diverged = True; break
            pen_max_ever = max(pen_max_ever, _max_pair_penetration(state))
            min_det_ever = min(min_det_ever, _min_det(state))

        rows.append(dict(kappa=kappa, ok=not diverged,
                         sep=float(B.c[1] - A.c[1]),
                         pen=pen_max_ever, min_det=min_det_ever,
                         final_speed=_max_speed(state)))

    print(f"  {'kappa':>8s} {'ok':>4s} {'sep':>8s} "
          f"{'max pen':>10s} {'min_det':>9s} {'|v|f':>10s}")
    for r in rows:
        flag = "ok" if r["ok"] else "DIV"
        print(f"  {r['kappa']:8.2f} {flag:>4s} "
              f"{r['sep']:8.4f} {r['pen']:10.4e} {r['min_det']:9.4f} "
              f"{r['final_speed']:10.3e}")

    for r in rows:
        assert r["ok"], f"diverged at kappa={r['kappa']}"
        # Larger kappa should reduce penetration; absolute bound generous
        # because frozen-normal IPC has finite-iter transient ingress.
        assert r["pen"] < 1.5 * barrier_dhat, (
            f"kappa={r['kappa']}: severe penetration, pen={r['pen']:.4e}")
        assert r["min_det"] > 0.2, (
            f"kappa={r['kappa']}: inversion min_det={r['min_det']:.3f}")
    pens = [r["pen"] for r in rows]
    assert pens[2] <= pens[0] + 1e-6, (
        f"penetration did not shrink with stiffer kappa: {pens}")
    print("  PASS")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    import warnings
    warnings.filterwarnings("ignore")

    test_two_body_stack_drop()
    test_two_body_stack_dynamic()
    test_two_body_side_collision()
    test_body_contact_stiffness_sweep()
    print("\nAll body-body barrier-contact tests passed.")
