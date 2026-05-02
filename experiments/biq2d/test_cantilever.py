"""M1 milestone: BQ2D cantilever bending test using pure VBD with smooth penalty.

Chain of n_bodies bodies, first one static.  Adjacent bodies connected by an
"edge connection" = two corner-pair point joints (top + bottom corners shared).

We sweep alpha_mult ∈ {1, 10, 100} to characterise stiffness sensitivity.

Victory:  tip_y < 2.5 (visible bending) and stable.
Refute :  tip_y ≈ 3.0 (rigid bar — just like the affine cantilever).
"""
import numpy as np
from .body import BodyBQ2D
from .solver import State, Params, Joint, step_vbd, corner_position
from . import energy


def make_cantilever(n_bodies=4, h=0.5, k=2000.0, nu=0.35, mass=0.5, y0=3.0):
    """Horizontal chain along +x at y=y0, first body static (anchor)."""
    bodies = []
    for i in range(n_bodies):
        b = BodyBQ2D(mass=mass, half_extent=h, k=k, nu=nu, static=(i == 0))
        b.c = np.array([2.0 * h * i, y0])
        bodies.append(b)

    joints = []
    for i in range(n_bodies - 1):
        # Bottom corner pair: right-bottom of i ↔ left-bottom of i+1
        joints.append(Joint(body_a_idx=i, body_b_idx=i + 1,
                            corner_a=(+1, -1), corner_b=(-1, -1)))
        # Top corner pair
        joints.append(Joint(body_a_idx=i, body_b_idx=i + 1,
                            corner_a=(+1, +1), corner_b=(-1, +1)))

    return State(bodies=bodies, joints=joints)


def _diagnostics(state, label):
    """Return dict of stability + bending diagnostics over non-static bodies."""
    nonstatic = [b for b in state.bodies if not b.static]
    I_flat = np.array([1.0, 0.0, 0.0, 1.0])

    min_det = min(b.min_det_J() for b in nonstatic)
    max_F_dev = max(np.linalg.norm(b.F - I_flat) for b in nonstatic)
    max_G = max(np.linalg.norm(b.G) for b in nonstatic)
    max_v = max(np.linalg.norm(b.v) for b in nonstatic)
    tip_y = state.bodies[-1].c[1]

    # Joint position error
    max_joint_err = 0.0
    for j in state.joints:
        ba = state.bodies[j.body_a_idx]
        bb = state.bodies[j.body_b_idx]
        Pa = corner_position(ba, j.corner_a)
        Pb = corner_position(bb, j.corner_b)
        err = np.linalg.norm(Pa - Pb)
        max_joint_err = max(max_joint_err, err)

    return dict(label=label, min_det=min_det, max_F_dev=max_F_dev, max_G=max_G,
                max_v=max_v, tip_y=tip_y, max_joint_err=max_joint_err)


def _print_diag(d):
    print(f"  {d['label']:>12s}  min_det={d['min_det']:.4f}  "
          f"|F-I|={d['max_F_dev']:.4f}  |G|={d['max_G']:.4f}  "
          f"tip_y={d['tip_y']:.4f}  joint_err={d['max_joint_err']:.4e}  "
          f"|v|={d['max_v']:.4e}")


def run_cantilever(alpha_mult=1.0, n_bodies=4, n_steps=500, dt=1.0/240.0,
                   k=2000.0, vbd_sweeps=5, vbd_newton=5, verbose=True):
    state = make_cantilever(n_bodies=n_bodies, k=k)
    params = Params(dt=dt, vbd_sweeps=vbd_sweeps, vbd_newton=vbd_newton)

    b_ref = next(b for b in state.bodies if not b.static)
    alpha = alpha_mult * b_ref.k * (b_ref.h ** 2)

    if verbose:
        print(f"\n=== alpha_mult={alpha_mult}, n={n_bodies}, k={k}, "
              f"alpha={alpha:.2f}, sweeps={vbd_sweeps}x newton={vbd_newton} ===")
        _print_diag(_diagnostics(state, "init"))

    for step_i in range(n_steps):
        step_vbd(state, params, alpha=alpha)

        # Cheap divergence check
        max_F = max(np.max(np.abs(b.F)) for b in state.bodies if not b.static)
        if not np.isfinite(max_F) or max_F > 100:
            print(f"  DIVERGED at step {step_i}, max|F|={max_F}")
            return state, False

        if verbose and (step_i + 1) % 100 == 0:
            _print_diag(_diagnostics(state, f"step {step_i+1}"))

    if verbose:
        _print_diag(_diagnostics(state, "FINAL"))

    return state, True


def test_vbd_cantilever_sweep():
    print("=== M1: BQ2D cantilever, pure-VBD with smooth penalty ===")
    np.set_printoptions(precision=4, suppress=True)
    import warnings
    warnings.filterwarnings("ignore")

    results = []
    for alpha_mult in [1.0, 10.0, 100.0]:
        state, ok = run_cantilever(alpha_mult=alpha_mult)
        d = _diagnostics(state, f"a={alpha_mult}")
        results.append((alpha_mult, ok, d))

    print("\n=== summary ===")
    print("  alpha_mult  stable  tip_y    |G|max   |F-I|max  joint_err  min_det")
    for am, ok, d in results:
        flag = "ok" if ok else "DIV"
        print(f"  {am:9.1f}   {flag:6s}  {d['tip_y']:.4f}   {d['max_G']:.4f}   "
              f"{d['max_F_dev']:.4f}    {d['max_joint_err']:.2e}   {d['min_det']:.4f}")


if __name__ == "__main__":
    test_vbd_cantilever_sweep()
