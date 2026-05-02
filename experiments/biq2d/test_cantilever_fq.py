"""M2 milestone: FQ2D cantilever bending test using pure VBD with smooth penalty.

Chain of n_bodies full-quadratic bodies, first one static.  Adjacent bodies
are connected by a 3-point edge joint (bottom corner + edge midpoint + top
corner).  The midpoint constraint is what engages the banana (Hx, Hy) modes
— corner-only joints cannot distinguish bananas from a global offset.

We sweep alpha_mult ∈ {1, 10, 100} to characterise stiffness sensitivity.

Victory:  tip_y < 2.5 (visible bending), max|Hx| or max|Hy| > 0.01
          (banana activation), stable across the α sweep.
"""
import numpy as np
from .body import BodyFQ2D
from .solver import State, Params, JointFQ, step_vbd_fq, point_position_fq


def make_cantilever_fq(n_bodies=4, h=0.5, k=2000.0, nu=0.35, mass=0.5, y0=3.0):
    """Horizontal chain along +x at y=y0, first body static (anchor).

    Each adjacent pair gets 3 JointFQ constraints (bottom corner, edge
    midpoint, top corner).  Three points pin the degree-2 edge polynomial,
    giving exact edge continuity for the full-quadratic basis.
    """
    bodies = []
    for i in range(n_bodies):
        b = BodyFQ2D(mass=mass, half_extent=h, k=k, nu=nu, static=(i == 0))
        b.c = np.array([2.0 * h * i, y0])
        bodies.append(b)

    joints = []
    for i in range(n_bodies - 1):
        # bottom corner
        joints.append(JointFQ(body_a_idx=i, body_b_idx=i + 1,
                              xi_a=(+1.0, -1.0), xi_b=(-1.0, -1.0)))
        # edge midpoint — engages bananas
        joints.append(JointFQ(body_a_idx=i, body_b_idx=i + 1,
                              xi_a=(+1.0,  0.0), xi_b=(-1.0,  0.0)))
        # top corner
        joints.append(JointFQ(body_a_idx=i, body_b_idx=i + 1,
                              xi_a=(+1.0, +1.0), xi_b=(-1.0, +1.0)))

    return State(bodies=bodies, joints=joints)


def _diagnostics(state, label):
    """Return dict of stability + bending diagnostics over non-static bodies."""
    nonstatic = [b for b in state.bodies if not b.static]
    I_flat = np.array([1.0, 0.0, 0.0, 1.0])

    min_det = min(b.min_det_J_sampled(grid_n=9) for b in nonstatic)
    max_F_dev = max(np.linalg.norm(b.F - I_flat) for b in nonstatic)
    max_G  = max(np.linalg.norm(b.G)  for b in nonstatic)
    max_Hx = max(np.linalg.norm(b.Hx) for b in nonstatic)
    max_Hy = max(np.linalg.norm(b.Hy) for b in nonstatic)
    max_v = max(np.linalg.norm(np.concatenate([b.vc, b.vF, b.vG, b.vHx, b.vHy]))
                for b in nonstatic)
    tip_y = state.bodies[-1].c[1]

    # Joint position error across all 3 point pairs per edge
    max_joint_err = 0.0
    for j in state.joints:
        ba = state.bodies[j.body_a_idx]
        bb = state.bodies[j.body_b_idx]
        Pa = point_position_fq(ba, j.xi_a)
        Pb = point_position_fq(bb, j.xi_b)
        err = np.linalg.norm(Pa - Pb)
        max_joint_err = max(max_joint_err, err)

    return dict(label=label, min_det=min_det, max_F_dev=max_F_dev,
                max_G=max_G, max_Hx=max_Hx, max_Hy=max_Hy,
                max_v=max_v, tip_y=tip_y, max_joint_err=max_joint_err)


def _print_diag(d):
    print(f"  {d['label']:>12s}  min_det={d['min_det']:.4f}  "
          f"|F-I|={d['max_F_dev']:.4f}  |G|={d['max_G']:.4f}  "
          f"|Hx|={d['max_Hx']:.4f}  |Hy|={d['max_Hy']:.4f}  "
          f"tip_y={d['tip_y']:.4f}  joint_err={d['max_joint_err']:.4e}  "
          f"|v|={d['max_v']:.4e}")


def run_cantilever_fq(alpha_mult=1.0, n_bodies=4, n_steps=500, dt=1.0/240.0,
                      k=2000.0, vbd_sweeps=5, vbd_newton=5, verbose=True):
    state = make_cantilever_fq(n_bodies=n_bodies, k=k)
    params = Params(dt=dt, vbd_sweeps=vbd_sweeps, vbd_newton=vbd_newton)

    b_ref = next(b for b in state.bodies if not b.static)
    alpha = alpha_mult * b_ref.k * (b_ref.h ** 2)

    if verbose:
        print(f"\n=== FQ2D alpha_mult={alpha_mult}, n={n_bodies}, k={k}, "
              f"alpha={alpha:.2f}, sweeps={vbd_sweeps}x newton={vbd_newton} ===")
        _print_diag(_diagnostics(state, "init"))

    for step_i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha)

        max_F = max(np.max(np.abs(b.F)) for b in state.bodies if not b.static)
        if not np.isfinite(max_F) or max_F > 100:
            print(f"  DIVERGED at step {step_i}, max|F|={max_F}")
            return state, False

        if verbose and (step_i + 1) % 100 == 0:
            _print_diag(_diagnostics(state, f"step {step_i+1}"))

    if verbose:
        _print_diag(_diagnostics(state, "FINAL"))

    return state, True


def test_vbd_cantilever_fq_sweep():
    print("=== M2: FQ2D cantilever, pure-VBD with smooth penalty ===")
    np.set_printoptions(precision=4, suppress=True)
    import warnings
    warnings.filterwarnings("ignore")

    results = []
    for alpha_mult in [1.0, 10.0, 100.0]:
        state, ok = run_cantilever_fq(alpha_mult=alpha_mult)
        d = _diagnostics(state, f"a={alpha_mult}")
        results.append((alpha_mult, ok, d))

    print("\n=== summary ===")
    print("  alpha_mult  stable  tip_y    |G|max   |Hx|max   |Hy|max   "
          "|F-I|max  joint_err  min_det")
    for am, ok, d in results:
        flag = "ok" if ok else "DIV"
        print(f"  {am:9.1f}   {flag:6s}  {d['tip_y']:.4f}   {d['max_G']:.4f}   "
              f"{d['max_Hx']:.4f}   {d['max_Hy']:.4f}   "
              f"{d['max_F_dev']:.4f}    {d['max_joint_err']:.2e}   "
              f"{d['min_det']:.4f}")


if __name__ == "__main__":
    test_vbd_cantilever_fq_sweep()
