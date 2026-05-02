"""Test different VBD architectures for stability."""
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from .body import Body2D, be_elastic_F
from .solver import (State, Params, Joint, detect_contacts,
    _prepare_joints_tgs, _prepare_contacts_tgs, _warm_start_tgs,
    _joint_position_error, _joint_angle_error, _joint_pos_jacobians,
    _joint_angle_jacobians, _solve_contact_tgs)
from . import body as body_mod
from . import energy as energy_mod


def solve_joint_c_only(joint, bodies, params, h, use_bias):
    ba = bodies[joint.body_a_idx]
    bb = bodies[joint.body_b_idx]

    C_pos = _joint_position_error(joint, bodies)
    C_angle = _joint_angle_error(joint, bodies)
    Jx_a, Jx_b, Jy_a, Jy_b = _joint_pos_jacobians(joint)
    Ja_angle, Jb_angle = _joint_angle_jacobians(joint, bodies)

    constraints = [
        (C_pos[0], Jx_a, Jx_b, '_lam_x'),
        (C_pos[1], Jy_a, Jy_b, '_lam_y'),
        (C_angle, Ja_angle, Jb_angle, None),
    ]

    for C_val, J_a, J_b, lam_attr in constraints:
        K = 0.0
        for b, J_full in [(ba, J_a), (bb, J_b)]:
            if not b.static:
                J_c = J_full[:2]
                K += float(np.dot(J_c, J_c)) * b.inv_mass
        if K < 1e-15:
            continue

        Cdot = 0.0
        for b, J_full in [(ba, J_a), (bb, J_b)]:
            if not b.static:
                Cdot += float(np.dot(J_full[:2], b.vc))

        lam_accum = getattr(joint, lam_attr) if lam_attr else 0.0
        if use_bias:
            bias = joint._bias_coeff * C_val
            impulse = (-joint._mass_coeff * (Cdot + bias) / K
                       - joint._impulse_coeff * lam_accum)
        else:
            impulse = -Cdot / K

        if lam_attr:
            setattr(joint, lam_attr, getattr(joint, lam_attr) + impulse)

        for b, J_full in [(ba, J_a), (bb, J_b)]:
            if not b.static:
                b.vc += impulse * J_full[:2] * b.inv_mass


def vbd_sweep_F(state, params, dt):
    """Per-body VBD sweep: find F that minimizes IP subject to incident constraints."""
    bodies = state.bodies

    # Build incident constraint list per body
    body_constraints = {i: [] for i in range(len(bodies))}
    for j in state.joints:
        ba = bodies[j.body_a_idx]
        bb = bodies[j.body_b_idx]
        la, lb = j.local_a, j.local_b
        aa, ab = j.axis_a, j.axis_b

        # Position constraints (from the perspective of each body)
        # For body a: anchor_a = c_a + F_a*la should equal anchor_b = c_b + F_b*lb
        # C_pos = anchor_a - anchor_b
        # dC/dF_a: [la[0], la[1], 0, 0] for x, [0, 0, la[0], la[1]] for y

        pa_target_x = bb.c[0] + bb.F[0]*lb[0] + bb.F[1]*lb[1]
        pa_target_y = bb.c[1] + bb.F[2]*lb[0] + bb.F[3]*lb[1]
        pb_target_x = ba.c[0] + ba.F[0]*la[0] + ba.F[1]*la[1]
        pb_target_y = ba.c[1] + ba.F[2]*la[0] + ba.F[3]*la[1]

        if not ba.static:
            JF_x_a = np.array([la[0], la[1], 0.0, 0.0])
            JF_y_a = np.array([0.0, 0.0, la[0], la[1]])
            C_x = ba.c[0] + ba.F[0]*la[0] + ba.F[1]*la[1] - pa_target_x
            C_y = ba.c[1] + ba.F[2]*la[0] + ba.F[3]*la[1] - pa_target_y
            body_constraints[j.body_a_idx].append((C_x, JF_x_a))
            body_constraints[j.body_a_idx].append((C_y, JF_y_a))

        if not bb.static:
            JF_x_b = np.array([-lb[0], -lb[1], 0.0, 0.0])
            JF_y_b = np.array([0.0, 0.0, -lb[0], -lb[1]])
            C_x_b = bb.c[0] + bb.F[0]*lb[0] + bb.F[1]*lb[1] - pb_target_x
            C_y_b = bb.c[1] + bb.F[2]*lb[0] + bb.F[3]*lb[1] - pb_target_y
            body_constraints[j.body_b_idx].append((C_x_b, JF_x_b))
            body_constraints[j.body_b_idx].append((C_y_b, JF_y_b))

        # Angle constraint
        fa = np.array([ba.F[0]*aa[0] + ba.F[1]*aa[1],
                        ba.F[2]*aa[0] + ba.F[3]*aa[1]])
        fb = np.array([bb.F[0]*ab[0] + bb.F[1]*ab[1],
                        bb.F[2]*ab[0] + bb.F[3]*ab[1]])
        C_angle = fa[0]*fb[1] - fa[1]*fb[0]

        if not ba.static:
            Ja = np.array([aa[0]*fb[1], aa[1]*fb[1], -aa[0]*fb[0], -aa[1]*fb[0]])
            body_constraints[j.body_a_idx].append((C_angle, Ja))
        if not bb.static:
            Jb = np.array([-fa[1]*ab[0], -fa[1]*ab[1], fa[0]*ab[0], fa[0]*ab[1]])
            body_constraints[j.body_b_idx].append((C_angle, Jb))

    # VBD solve per body
    for i, b in enumerate(bodies):
        if b.static:
            continue
        constraints = body_constraints[i]
        if not constraints:
            continue

        mu_i = b.mu_inertia
        mu_l, lam_l = b.lame
        scale = b._energy_scale

        # IP Hessian and residual at current F
        H_spd = energy_mod.hessian_spd(b.F, mu_l, lam_l) * scale
        A = mu_i * np.eye(4) + dt**2 * H_spd
        pk1_val = energy_mod.pk1(b.F, mu_l, lam_l) * scale
        r = mu_i * (b.vF - b._vF_pre) + dt * pk1_val

        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(r)):
            continue

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue

        # Build constraint matrix J (n_c x 4) and residual vector C (n_c)
        n_c = len(constraints)
        J = np.zeros((n_c, 4))
        C = np.zeros(n_c)
        for ci, (c_val, jf) in enumerate(constraints):
            C[ci] = c_val
            J[ci] = jf

        # Penalty formulation: balance elastic energy vs constraint satisfaction
        # min  0.5*dvF^T A dvF + r^T dvF  +  (alpha/2)*||C + dt*J*dvF||^2
        # => (A + alpha*dt^2*J^T*J) dvF = -(r + alpha*dt*J^T*C)
        alpha = np.trace(A) / 4.0  # match elastic stiffness scale
        H = A + alpha * dt**2 * (J.T @ J)
        g = r + alpha * dt * (J.T @ C)

        try:
            dvF = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            continue

        b.vF += dvF
        b.F = b._F_pre + dt * b.vF


def step_tgs_v3(state, params, vbd_iters=3):
    bodies = state.bodies
    dt = params.dt
    n_sub = params.substeps
    h = dt / n_sub

    for b in bodies:
        if not b.static:
            b._F_pre = b.F.copy()
            b._vF_pre = b.vF.copy()

    # BE for everything (gravity + elastic)
    for b in bodies:
        body_mod.integrate_backward_euler(b, dt, params.gravity)
    for b in bodies:
        if not b.static:
            b._F_pre = b.F.copy()
            b._vF_pre = b.vF.copy()

    state.contacts = detect_contacts(state)
    _prepare_joints_tgs(state, params, h)
    _prepare_contacts_tgs(state, params, h)

    # c-only TGS substep loop
    for substep in range(n_sub):
        for b in bodies:
            if not b.static:
                b.vc += h * params.gravity
        _warm_start_tgs(state)
        for j in state.joints:
            solve_joint_c_only(j, bodies, params, h, use_bias=True)
        for j in reversed(state.joints):
            solve_joint_c_only(j, bodies, params, h, use_bias=True)
        for ct in state.contacts:
            _solve_contact_tgs(ct, bodies, h, use_bias=True)
        for b in bodies:
            if not b.static:
                b.c += h * b.vc
        for _ in range(params.relax_iters):
            for j in state.joints:
                solve_joint_c_only(j, bodies, params, h, use_bias=False)
            for j in reversed(state.joints):
                solve_joint_c_only(j, bodies, params, h, use_bias=False)

    # VBD sweep for F (after TGS c-solve)
    for _ in range(vbd_iters):
        vbd_sweep_F(state, params, dt)

    state.time += dt
    state.step_count += 1


def make_cantilever(n_bodies=8, r0=0.3, k=2000.0, nu=0.35):
    bodies = []
    for i in range(n_bodies):
        b = Body2D(mass=0.5, r0=r0, k=k, nu=nu, static=(i == 0))
        b.c = np.array([2 * r0 * i, 3.0])
        bodies.append(b)
    joints = [Joint(body_a_idx=i, body_b_idx=i + 1,
                    local_a=np.array([r0, 0.0]),
                    local_b=np.array([-r0, 0.0])) for i in range(n_bodies - 1)]
    return State(bodies=bodies, joints=joints)


def run_test(n_bodies=8, substeps=4, n_steps=500, vbd_iters=3):
    dt = 1.0 / 240.0
    state = make_cantilever(n_bodies)
    params = Params(dt=dt, substeps=substeps)

    for step_i in range(n_steps):
        step_tgs_v3(state, params, vbd_iters=vbd_iters)
        maxF = max(np.max(np.abs(b.F)) for b in state.bodies[1:])
        if maxF > 10:
            print(f"n={n_bodies} sub={substeps} vbd={vbd_iters}: DIVERGED step {step_i}")
            return
        if step_i % 50 == 0:
            min_det = min(energy_mod._det2(b.F) for b in state.bodies[1:])
            tip_y = state.bodies[-1].c[1]
            pos_errs = [np.linalg.norm(_joint_position_error(j, state.bodies))
                        for j in state.joints]
            print(f"  step {step_i:3d}: min_det={min_det:.4f} max|F|={maxF:.4f} "
                  f"tip_y={tip_y:.3f} pos_err={max(pos_errs):.6f}")

    min_det = min(energy_mod._det2(b.F) for b in state.bodies[1:])
    tip_y = state.bodies[-1].c[1]
    maxF = max(np.max(np.abs(b.F)) for b in state.bodies[1:])
    print(f"n={n_bodies} sub={substeps} vbd={vbd_iters}: STABLE {n_steps} steps, "
          f"min_det={min_det:.4f} max|F|={maxF:.4f} tip_y={tip_y:.3f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    run_test(n_bodies=8, substeps=4, vbd_iters=3)
