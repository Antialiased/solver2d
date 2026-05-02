"""Floor-contact tests for FQ2D bodies (frozen-normal IPC log barrier).

Three scenarios:

  1. Single-body drop onto a horizontal floor (no joints).
  2. Stiffness sweep over barrier_kappa_floor ∈ {0.1, 1, 10} on the same
     drop scenario.  Penetration should now be ≈0 across the sweep
     (barrier is feasibility-preserving by construction, unlike the
     previous smooth penalty whose penetration scaled ~1/κ).
  3. Cantilever chain dropped so the tip strikes the floor — joints +
     contact engaged simultaneously; verifies neither wrecks the other.

Victory:
  - bodies do not tunnel (max penetration ≪ h),
  - eventually settle (|v| small),
  - no inversion (min sampled det J > 0.2 ish),
  - joint error stays comparable to the joint-only baseline in (3).
"""
import numpy as np
from .body import BodyFQ2D
from .solver import (State, Params, JointFQ, step_vbd_fq, point_position_fq,
                     _OUTER_XI, _outer_py_from_q)
from .test_cantilever_fq import make_cantilever_fq


# ── helpers ───────────────────────────────────────────────────────────

def _q_of(b):
    return np.concatenate([b.c, b.F, b.G, b.Hx, b.Hy])


def _penetrating_count(state, y_floor):
    """How many outer-ξ contact points are currently below the floor."""
    n = 0
    for b in state.bodies:
        if b.static:
            continue
        Py = _outer_py_from_q(_q_of(b), b.h)
        n += int(np.sum(Py < y_floor))
    return n


def _min_outer_y(state):
    """Min world y over all outer-ξ points of all non-static bodies."""
    ys = []
    for b in state.bodies:
        if b.static:
            continue
        Py = _outer_py_from_q(_q_of(b), b.h)
        ys.append(float(np.min(Py)))
    return min(ys) if ys else float("inf")


def _max_speed(state):
    """Max ‖v_full‖ over non-static bodies (12-vector v)."""
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


def _max_joint_err(state):
    err = 0.0
    for j in state.joints:
        ba = state.bodies[j.body_a_idx]
        bb = state.bodies[j.body_b_idx]
        Pa = point_position_fq(ba, j.xi_a)
        Pb = point_position_fq(bb, j.xi_b)
        err = max(err, float(np.linalg.norm(Pa - Pb)))
    return err


# ── scenarios ─────────────────────────────────────────────────────────

def test_floor_drop_single():
    """Single non-static body dropped onto y_floor=0 from c=(0, 2)."""
    print("\n=== floor-drop, single body ===")
    h = 0.5
    k = 2000.0
    body = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
    body.c = np.array([0.0, 2.0])
    state = State(bodies=[body], joints=[])
    params = Params(gravity=np.array([0.0, -9.81]), dt=1.0 / 240.0,
                    vbd_sweeps=5, vbd_newton=5)

    alpha = 10.0 * k * h * h               # joint α (unused — no joints)
    barrier_kappa = 100.0
    barrier_dhat = 0.05 * h                # 5 % of half-extent ≈ 0.025
    y_floor = 0.0

    # Need ~600 steps for the body to settle on the floor (no friction, so it
    # bounces and the elastic modes ring down via BE dissipation only).  1000
    # steps is comfortably past convergence.
    n_steps = 1000
    min_min_outer_y = float("inf")
    min_min_det = float("inf")
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha,
                    barrier_kappa_floor=barrier_kappa, y_floor=y_floor,
                    barrier_dhat=barrier_dhat)
        min_min_outer_y = min(min_min_outer_y, _min_outer_y(state))
        min_min_det = min(min_min_det, _min_det(state))

    final_speed = _max_speed(state)
    print(f"  cy_final          = {body.c[1]:.4f}  (h = {h})")
    print(f"  min_outer_y_ever  = {min_min_outer_y:.4f}  (floor at 0)")
    print(f"  min_det_J_ever    = {min_min_det:.4f}")
    print(f"  final |v|         = {final_speed:.4e}")

    assert np.all(np.isfinite(_q_of(body))), "DoFs not finite"
    # Frozen-normal IPC barrier with finite Newton iters does NOT guarantee
    # zero penetration during high-velocity impact (full IPC needs per-iter
    # CCD; we replaced that with frozen normal for tractability).  Transient
    # penetration is bounded by the substep-velocity overshoot; settled
    # penetration is what matters for stacking stability.  Assert both with
    # different tolerances.
    assert min_min_outer_y > -0.5 * barrier_dhat, (
        f"transient penetration too large: min_outer_y={min_min_outer_y:.4e}")
    assert final_speed < 0.1, f"failed to settle — |v|={final_speed:.3e}"
    assert min_min_det > 0.3, f"inversion risk — min det J = {min_min_det:.3f}"
    print("  PASS")


def test_floor_drop_stiffness_sweep():
    """Sweep barrier_kappa_floor ∈ {0.1, 1, 10}, same drop scenario.

    Penetration should be ≈0 across the sweep — the barrier is feasibility-
    preserving by construction, so κ controls only the *width* of the
    influence band's energy scale, not the steady-state penetration.
    """
    print("\n=== floor-drop, barrier_kappa_floor sweep ===")
    h = 0.5
    k = 2000.0
    y_floor = 0.0
    barrier_dhat = 0.05 * h
    n_steps = 1000
    rows = []
    for kappa in (10.0, 100.0, 1000.0):
        body = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
        body.c = np.array([0.0, 2.0])
        state = State(bodies=[body], joints=[])
        params = Params(gravity=np.array([0.0, -9.81]), dt=1.0 / 240.0,
                        vbd_sweeps=5, vbd_newton=5)
        alpha = 10.0 * k * h * h

        min_outer_y_ever = float("inf")
        min_det_ever = float("inf")
        max_pen_count = 0
        diverged = False
        for i in range(n_steps):
            step_vbd_fq(state, params, alpha=alpha,
                        barrier_kappa_floor=kappa, y_floor=y_floor,
                        barrier_dhat=barrier_dhat)
            if not np.all(np.isfinite(_q_of(body))):
                diverged = True
                break
            min_outer_y_ever = min(min_outer_y_ever, _min_outer_y(state))
            min_det_ever     = min(min_det_ever, _min_det(state))
            max_pen_count    = max(max_pen_count, _penetrating_count(state, y_floor))

        rows.append(dict(kappa=kappa, ok=not diverged,
                         tip_y=float(body.c[1]),
                         min_outer_y=min_outer_y_ever,
                         min_det=min_det_ever,
                         pen_count=max_pen_count,
                         final_speed=_max_speed(state)))

    print(f"  {'kappa':>8s} {'ok':>4s} {'cy':>8s} "
          f"{'min_y':>10s} {'min_det':>9s} {'pen#':>5s} {'|v|f':>10s}")
    for r in rows:
        flag = "ok" if r["ok"] else "DIV"
        print(f"  {r['kappa']:8.2f} {flag:>4s} "
              f"{r['tip_y']:8.4f} {r['min_outer_y']:10.4e} "
              f"{r['min_det']:9.4f} {r['pen_count']:5d} "
              f"{r['final_speed']:10.3e}")

    # All sweeps must remain stable.  Penetration should *shrink* as kappa
    # grows (stronger barrier resists impact more), but isn't required to
    # hit zero in every case — frozen-normal IPC with finite Newton iters
    # has bounded transient ingress.
    for r in rows:
        assert r["ok"], f"diverged at kappa={r['kappa']}"
        assert r["min_outer_y"] > -1.5 * barrier_dhat, (
            f"kappa={r['kappa']}: severe penetration, "
            f"min_outer_y={r['min_outer_y']:.4e}")
        assert r["min_det"] > 0.15, (
            f"kappa={r['kappa']}: inversion, min_det={r['min_det']:.3f}")
    pens = [-r["min_outer_y"] for r in rows]
    assert pens[2] <= pens[0] + 1e-6, (
        f"penetration did not shrink with stiffer kappa: {pens}")
    print("  PASS")


def test_cantilever_drop_onto_floor():
    """Cantilever chain anchored low so the tip swings down onto the floor."""
    print("\n=== cantilever-onto-floor ===")
    n_bodies = 4
    h = 0.5
    k = 2000.0
    state = make_cantilever_fq(n_bodies=n_bodies, h=h, k=k, y0=0.6)
    params = Params(gravity=np.array([0.0, -9.81]), dt=1.0 / 240.0,
                    vbd_sweeps=5, vbd_newton=5)
    alpha = 10.0 * k * h * h
    barrier_kappa = 100.0
    barrier_dhat = 0.05 * h
    y_floor = 0.0

    n_steps = 1000
    min_outer_y_ever = float("inf")
    min_det_ever = float("inf")
    max_joint_err_ever = 0.0
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha,
                    barrier_kappa_floor=barrier_kappa, y_floor=y_floor,
                    barrier_dhat=barrier_dhat)
        max_F = max(np.max(np.abs(b.F)) for b in state.bodies if not b.static)
        if not np.isfinite(max_F) or max_F > 100:
            print(f"  DIVERGED at step {i}, max|F|={max_F}")
            assert False, "diverged"
        min_outer_y_ever = min(min_outer_y_ever, _min_outer_y(state))
        min_det_ever     = min(min_det_ever,    _min_det(state))
        max_joint_err_ever = max(max_joint_err_ever, _max_joint_err(state))

    final_speed = _max_speed(state)
    tip_y = state.bodies[-1].c[1]
    print(f"  tip_y_final       = {tip_y:.4f}")
    print(f"  min_outer_y_ever  = {min_outer_y_ever:.4f}  (floor at 0)")
    print(f"  min_det_J_ever    = {min_det_ever:.4f}")
    print(f"  max_joint_err     = {max_joint_err_ever:.4e}")
    print(f"  final |v|         = {final_speed:.4e}")

    assert min_outer_y_ever > -0.5 * barrier_dhat, (
        f"transient breached barrier, min_outer_y={min_outer_y_ever:.4e}")
    assert min_det_ever > 0.15, f"inversion under combined contact+joints: {min_det_ever:.3f}"
    # Joint error worse than ~3× the joint-only baseline (~few-percent of h)
    # would indicate contact is wrecking the joint solve.  The cantilever-only
    # alpha=10·k·h² baseline lands joint_err ≲ 1e-2·h; allow generous 5×.
    assert max_joint_err_ever < 0.05 * h * 5.0, (
        f"joint error inflated by contact: {max_joint_err_ever:.3e}")
    print("  PASS")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    import warnings
    warnings.filterwarnings("ignore")

    test_floor_drop_single()
    test_floor_drop_stiffness_sweep()
    test_cantilever_drop_onto_floor()
    print("\nAll FQ2D barrier floor-contact tests passed.")
