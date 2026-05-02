"""Test: constrained BE solver where constraints are penalty terms in the IP."""
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from .body import Body2D
from .solver import (State, Params, Joint, detect_contacts,
    _prepare_joints_tgs, _prepare_contacts_tgs, _warm_start_tgs,
    _joint_position_error, _joint_angle_error,
    _solve_contact_tgs)
from . import body as body_mod
from . import energy as energy_mod


def _collect_constraints_for_body(body_idx, state):
    """Collect all constraint equations involving body_idx, expressed as
    functions of that body's F (with other bodies' states fixed).

    Returns list of (C_val, J_F) where:
      C_val: current constraint value
      J_F: (4,) Jacobian dC/dF for this body
    """
    bodies = state.bodies
    b = bodies[body_idx]
    constraints = []

    for j in state.joints:
        if j.body_a_idx != body_idx and j.body_b_idx != body_idx:
            continue

        ba = bodies[j.body_a_idx]
        bb = bodies[j.body_b_idx]
        la, lb = j.local_a, j.local_b
        aa, ab = j.axis_a, j.axis_b

        # Position error: pA - pB
        pa = ba.c + np.array([ba.F[0]*la[0]+ba.F[1]*la[1],
                               ba.F[2]*la[0]+ba.F[3]*la[1]])
        pb = bb.c + np.array([bb.F[0]*lb[0]+bb.F[1]*lb[1],
                               bb.F[2]*lb[0]+bb.F[3]*lb[1]])
        C_pos = pa - pb

        if j.body_a_idx == body_idx and not b.static:
            # dCx/dF_a = [la[0], la[1], 0, 0]
            constraints.append((C_pos[0], np.array([la[0], la[1], 0.0, 0.0])))
            constraints.append((C_pos[1], np.array([0.0, 0.0, la[0], la[1]])))
        elif j.body_b_idx == body_idx and not b.static:
            constraints.append((-C_pos[0], np.array([-lb[0], -lb[1], 0.0, 0.0])))
            constraints.append((-C_pos[1], np.array([0.0, 0.0, -lb[0], -lb[1]])))

        # Angle error
        fa = np.array([ba.F[0]*aa[0]+ba.F[1]*aa[1],
                        ba.F[2]*aa[0]+ba.F[3]*aa[1]])
        fb = np.array([bb.F[0]*ab[0]+bb.F[1]*ab[1],
                        bb.F[2]*ab[0]+bb.F[3]*ab[1]])
        C_angle = fa[0]*fb[1] - fa[1]*fb[0]

        if j.body_a_idx == body_idx and not b.static:
            Ja = np.array([aa[0]*fb[1], aa[1]*fb[1], -aa[0]*fb[0], -aa[1]*fb[0]])
            constraints.append((C_angle, Ja))
        elif j.body_b_idx == body_idx and not b.static:
            Jb = np.array([-fa[1]*ab[0], -fa[1]*ab[1], fa[0]*ab[0], fa[0]*ab[1]])
            constraints.append((C_angle, Jb))

    return constraints


def be_elastic_F_constrained(body, body_idx, state, dt, alpha,
                              max_newton=15, ls_max=20):
    """Backward Euler for F-DoFs with constraint penalty in the IP.

    Minimizes:
        IP(vF) = 0.5*mu_i*||vF - vF_pre||^2
                 + psi(F_pre + dt*vF) * scale
                 + (alpha/2) * sum_j ||C_j(F_pre + dt*vF)||^2

    The constraint penalties couple F to the joint constraints,
    letting elastic energy naturally regulate how much F deforms.
    """
    if body.static:
        return

    mu_i = body.mu_inertia
    mu_l, lam_l = body.lame
    scale = body._energy_scale
    F_pre = body._F_pre
    vF_pre = body._vF_pre

    def ip_energy(vF):
        F_trial = F_pre + dt * vF
        # Save/restore body F temporarily for constraint eval
        F_save = body.F.copy()
        body.F = F_trial
        constraints = _collect_constraints_for_body(body_idx, state)
        body.F = F_save

        E = (0.5 * mu_i * float(np.dot(vF - vF_pre, vF - vF_pre))
             + energy_mod.psi(F_trial, mu_l, lam_l) * scale)

        for C_val, J_F in constraints:
            # C_val is already computed at F_trial (we set body.F = F_trial above)
            E += 0.5 * alpha * C_val ** 2

        return E

    vF = body.vF.copy()

    for newton_iter in range(max_newton):
        F_cur = F_pre + dt * vF

        # Elastic part
        f_el = -energy_mod.pk1(F_cur, mu_l, lam_l) * scale
        H_el = energy_mod.hessian_spd(F_cur, mu_l, lam_l) * scale

        # Constraint penalty part
        F_save = body.F.copy()
        body.F = F_cur
        constraints = _collect_constraints_for_body(body_idx, state)
        body.F = F_save

        # Constraint gradient and Hessian contributions
        g_c = np.zeros(4)
        H_c = np.zeros((4, 4))
        for C_val, J_F in constraints:
            # The constraint at F_cur needs to be recomputed
            # Since we set body.F = F_cur for the collection, C_val is correct
            g_c += alpha * C_val * J_F * dt  # d/dvF of (alpha/2)*C^2, using chain rule dC/dvF = J_F*dt
            H_c += alpha * dt**2 * np.outer(J_F, J_F)

        # IP gradient
        residual = mu_i * (vF - vF_pre) - dt * f_el + g_c
        # IP Hessian (SPD)
        A = mu_i * np.eye(4) + dt**2 * H_el + H_c

        try:
            dvF = np.linalg.solve(A, -residual)
        except np.linalg.LinAlgError:
            break

        # Backtracking line search
        E_cur = ip_energy(vF)
        directional = float(np.dot(residual, dvF))
        alpha_ls = 1.0
        for _ in range(ls_max):
            vF_trial = vF + alpha_ls * dvF
            E_trial = ip_energy(vF_trial)
            if E_trial <= E_cur + 1e-4 * alpha_ls * directional:
                break
            alpha_ls *= 0.5

        vF = vF + alpha_ls * dvF
        if np.max(np.abs(alpha_ls * dvF)) < 1e-12:
            break

    body.vF = vF
    body.F = F_pre + dt * body.vF


def solve_joint_c_only(joint, bodies, params, h, use_bias):
    ba = bodies[joint.body_a_idx]
    bb = bodies[joint.body_b_idx]
    from .solver import _joint_pos_jacobians, _joint_angle_jacobians

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


def step_constrained(state, params, alpha=None, vbd_sweeps=3):
    """TGS c-solve + constrained BE for F.

    Alpha is the constraint penalty weight in the BE. If None, uses k*scale.
    """
    bodies = state.bodies
    dt = params.dt
    n_sub = params.substeps
    h = dt / n_sub

    # Compute default alpha from material stiffness
    if alpha is None:
        # Match constraint stiffness to elastic stiffness
        b_ref = [b for b in bodies if not b.static][0]
        alpha = b_ref.k * b_ref._energy_scale

    # Save state
    for b in bodies:
        if not b.static:
            b._F_pre = b.F.copy()
            b._vF_pre = b.vF.copy()

    # BE for gravity on c, elastic on F (no constraints yet)
    for b in bodies:
        if not b.static:
            b.vc += dt * params.gravity
            b.c += dt * b.vc

    state.contacts = detect_contacts(state)
    _prepare_joints_tgs(state, params, h)
    _prepare_contacts_tgs(state, params, h)

    # c-only TGS substep loop
    for substep in range(n_sub):
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

    # Constrained BE for F (GS over bodies)
    for sweep in range(vbd_sweeps):
        for i, b in enumerate(bodies):
            be_elastic_F_constrained(b, i, state, dt, alpha)

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


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    for alpha_mult in [1.0, 10.0, 100.0]:
        state = make_cantilever(8)
        params = Params(dt=1.0/240.0, substeps=4)
        b_ref = [b for b in state.bodies if not b.static][0]
        alpha = alpha_mult * b_ref.k * b_ref._energy_scale

        diverged = False
        for step_i in range(500):
            step_constrained(state, params, alpha=alpha, vbd_sweeps=3)
            maxF = max(np.max(np.abs(b.F)) for b in state.bodies[1:])
            if maxF > 10:
                print(f"alpha_mult={alpha_mult}: DIVERGED step {step_i}")
                diverged = True
                break
            if step_i % 100 == 0:
                min_det = min(energy_mod._det2(b.F) for b in state.bodies[1:])
                tip_y = state.bodies[-1].c[1]
                pos_errs = [np.linalg.norm(_joint_position_error(j, state.bodies))
                            for j in state.joints]
                angle_errs = [abs(_joint_angle_error(j, state.bodies))
                              for j in state.joints]
                print(f"  alpha={alpha_mult} step {step_i:3d}: min_det={min_det:.4f} "
                      f"max|F|={maxF:.4f} tip={tip_y:.3f} "
                      f"pos_err={max(pos_errs):.4f} ang_err={max(angle_errs):.4f}")

        if not diverged:
            min_det = min(energy_mod._det2(b.F) for b in state.bodies[1:])
            maxF = max(np.max(np.abs(b.F)) for b in state.bodies[1:])
            tip_y = state.bodies[-1].c[1]
            print(f"  alpha={alpha_mult}: STABLE, min_det={min_det:.4f} "
                  f"max|F|={maxF:.4f} tip={tip_y:.3f}")
