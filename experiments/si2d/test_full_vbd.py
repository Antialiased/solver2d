"""Test: Full VBD approach - per-body optimization of all 6 DoFs (c + F)
with constraint penalties. No TGS for joints; TGS only for contacts."""
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from .body import Body2D
from .solver import (State, Params, Joint, detect_contacts,
    _joint_position_error, _joint_angle_error)
from . import energy as energy_mod


def _collect_body_constraints(body_idx, state):
    """Collect constraint equations involving body_idx.

    Returns list of (C_val, J_q) where J_q is (6,) Jacobian dC/dq.
    q = [cx, cy, F11, F12, F21, F22].
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
            # dCx/dq_a = [1, 0, la[0], la[1], 0, 0]
            # dCy/dq_a = [0, 1, 0, 0, la[0], la[1]]
            constraints.append((C_pos[0],
                np.array([1.0, 0.0, la[0], la[1], 0.0, 0.0])))
            constraints.append((C_pos[1],
                np.array([0.0, 1.0, 0.0, 0.0, la[0], la[1]])))
        elif j.body_b_idx == body_idx and not b.static:
            constraints.append((-C_pos[0],
                np.array([-1.0, 0.0, -lb[0], -lb[1], 0.0, 0.0])))
            constraints.append((-C_pos[1],
                np.array([0.0, -1.0, 0.0, 0.0, -lb[0], -lb[1]])))

        # Angle constraint
        fa = np.array([ba.F[0]*aa[0]+ba.F[1]*aa[1],
                        ba.F[2]*aa[0]+ba.F[3]*aa[1]])
        fb = np.array([bb.F[0]*ab[0]+bb.F[1]*ab[1],
                        bb.F[2]*ab[0]+bb.F[3]*ab[1]])
        C_angle = fa[0]*fb[1] - fa[1]*fb[0]

        if j.body_a_idx == body_idx and not b.static:
            Ja = np.array([0.0, 0.0, aa[0]*fb[1], aa[1]*fb[1],
                           -aa[0]*fb[0], -aa[1]*fb[0]])
            constraints.append((C_angle, Ja))
        elif j.body_b_idx == body_idx and not b.static:
            Jb = np.array([0.0, 0.0, -fa[1]*ab[0], -fa[1]*ab[1],
                           fa[0]*ab[0], fa[0]*ab[1]])
            constraints.append((C_angle, Jb))

    return constraints


def vbd_body_step(body, body_idx, state, dt, gravity, alpha,
                   max_newton=10, ls_max=20):
    """One VBD Newton step for a single body's 6 DoFs.

    Minimizes the incremental potential:
        IP(v) = 0.5*(v-v_pre)^T M (v-v_pre)
                - M*g . c_trial        (gravity potential)
                + psi(F_trial)*scale    (elastic)
                + (alpha/2)*sum||C_j||^2 (constraint penalty)

    where c_trial = c_pre + dt*vc, F_trial = F_pre + dt*vF.
    """
    if body.static:
        return

    M = body.mass
    mu_i = body.mu_inertia
    mu_l, lam_l = body.lame
    scale = body._energy_scale
    mass_vec = body.mass_vec  # [M, M, mu_i, mu_i, mu_i, mu_i]

    c_pre = body._c_pre
    F_pre = body._F_pre
    v_pre = np.concatenate([body._vc_pre, body._vF_pre])

    def ip_energy(v):
        vc = v[:2]
        vF = v[2:]
        c_trial = c_pre + dt * vc
        F_trial = F_pre + dt * vF

        # Inertia
        dv = v - v_pre
        E = 0.5 * float(np.dot(dv * mass_vec, dv))
        # Gravity
        E -= M * float(np.dot(gravity, c_trial))
        # Elastic
        E += energy_mod.psi(F_trial, mu_l, lam_l) * scale

        # Constraint penalty (recompute at trial state)
        c_save, F_save = body.c.copy(), body.F.copy()
        body.c = c_trial
        body.F = F_trial
        constraints = _collect_body_constraints(body_idx, state)
        body.c = c_save
        body.F = F_save

        for C_val, J_q in constraints:
            E += 0.5 * alpha * C_val ** 2

        return E

    v = np.concatenate([body.vc, body.vF])

    for _ in range(max_newton):
        vc = v[:2]
        vF = v[2:]
        c_cur = c_pre + dt * vc
        F_cur = F_pre + dt * vF

        # Elastic gradient and Hessian (F part only)
        pk1_val = energy_mod.pk1(F_cur, mu_l, lam_l) * scale
        H_spd = energy_mod.hessian_spd(F_cur, mu_l, lam_l) * scale

        # Collect constraints at current state
        c_save, F_save = body.c.copy(), body.F.copy()
        body.c = c_cur
        body.F = F_cur
        constraints = _collect_body_constraints(body_idx, state)
        body.c = c_save
        body.F = F_save

        # IP gradient (6-vector)
        dv = v - v_pre
        grad = dv * mass_vec  # inertia
        grad[:2] -= dt * M * gravity  # gravity force
        grad[2:] += dt * pk1_val  # elastic force

        # Constraint gradient
        for C_val, J_q in constraints:
            grad += alpha * C_val * dt * J_q  # chain rule: dC/dv = J_q * dt

        # IP Hessian (6x6, block diagonal for inertia + elastic, dense for constraints)
        H = np.diag(mass_vec.copy())  # inertia
        H[2:, 2:] += dt**2 * H_spd  # elastic

        # Constraint Hessian (Gauss-Newton: alpha * dt^2 * J^T J)
        for C_val, J_q in constraints:
            H += alpha * dt**2 * np.outer(J_q, J_q)

        try:
            dv_step = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            break

        # Line search
        E_cur = ip_energy(v)
        directional = float(np.dot(grad, dv_step))
        alpha_ls = 1.0
        for _ in range(ls_max):
            v_trial = v + alpha_ls * dv_step
            E_trial = ip_energy(v_trial)
            if E_trial <= E_cur + 1e-4 * alpha_ls * directional:
                break
            alpha_ls *= 0.5

        v = v + alpha_ls * dv_step
        if np.max(np.abs(alpha_ls * dv_step)) < 1e-12:
            break

    body.vc = v[:2]
    body.vF = v[2:]
    body.c = c_pre + dt * body.vc
    body.F = F_pre + dt * body.vF


def step_full_vbd(state, params, alpha=None, n_sweeps=5):
    """Full VBD step: per-body optimization of all 6 DoFs."""
    bodies = state.bodies
    dt = params.dt
    gravity = params.gravity

    if alpha is None:
        # High penalty to enforce constraints
        b_ref = [b for b in bodies if not b.static][0]
        alpha = 100.0 * b_ref.k * b_ref._energy_scale

    # Save state
    for b in bodies:
        if not b.static:
            b._c_pre = b.c.copy()
            b._F_pre = b.F.copy()
            b._vc_pre = b.vc.copy()
            b._vF_pre = b.vF.copy()

    # VBD sweeps (forward + backward)
    for sweep in range(n_sweeps):
        for i in range(len(bodies)):
            vbd_body_step(bodies[i], i, state, dt, gravity, alpha)
        for i in range(len(bodies) - 1, -1, -1):
            vbd_body_step(bodies[i], i, state, dt, gravity, alpha)

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

    state = make_cantilever(8)
    params = Params(dt=1.0/240.0)

    for step_i in range(500):
        step_full_vbd(state, params, n_sweeps=5)
        maxF = max(np.max(np.abs(b.F)) for b in state.bodies[1:])
        if maxF > 10:
            print(f"DIVERGED step {step_i}")
            break
        if step_i % 50 == 0:
            min_det = min(energy_mod._det2(b.F) for b in state.bodies[1:])
            tip_y = state.bodies[-1].c[1]
            pos_errs = [np.linalg.norm(_joint_position_error(j, state.bodies))
                        for j in state.joints]
            angle_errs = [abs(_joint_angle_error(j, state.bodies))
                          for j in state.joints]
            print(f"step {step_i:3d}: min_det={min_det:.4f} max|F|={maxF:.4f} "
                  f"tip={tip_y:.3f} pos_err={max(pos_errs):.4f} "
                  f"ang_err={max(angle_errs):.4f}")
    else:
        min_det = min(energy_mod._det2(b.F) for b in state.bodies[1:])
        maxF = max(np.max(np.abs(b.F)) for b in state.bodies[1:])
        tip_y = state.bodies[-1].c[1]
        pos_errs = [np.linalg.norm(_joint_position_error(j, state.bodies))
                    for j in state.joints]
        print(f"STABLE 500 steps: min_det={min_det:.4f} max|F|={maxF:.4f} "
              f"tip={tip_y:.3f} pos_err={max(pos_errs):.4f}")
