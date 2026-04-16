"""Sequential-impulse solver for 2D affine bodies.

Step pipeline:
1. Integrate (backward Euler for elastic + gravity)
2. Detect contacts (floor + ellipse-ellipse)
3. Position pass: correct penetration
4. Velocity pass: apply non-penetration impulse (with optional restitution)
"""
import numpy as np
from dataclasses import dataclass, field
from . import body as body_mod
from . import collision


@dataclass
class Params:
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -10.0]))
    dt: float = 1.0 / 60.0
    position_iters: int = 4
    velocity_iters: int = 4
    restitution: float = 0.0
    friction: float = 0.0
    bias_factor: float = 0.2
    position_correct_F: bool = True  # if False, position pass only corrects center-of-mass
    velocity_couple_F: bool = True  # if False, velocity pass only acts on translational DoFs
    relin: bool = False  # if True, re-solve BE for F after each position impulse


@dataclass
class Contact:
    """Active contact constraint."""
    type: str  # "floor" or "pair"
    body_a_idx: int
    body_b_idx: int  # -1 for floor
    gap: float
    normal: np.ndarray
    J_a: np.ndarray  # (6,)
    J_b: np.ndarray  # (6,) or None for floor
    K_inv: float  # 1 / effective mass
    lam: float = 0.0  # accumulated normal impulse
    Jt_a: np.ndarray = None  # tangential Jacobian body A (6,)
    Jt_b: np.ndarray = None  # tangential Jacobian body B (6,)
    lam_t: float = 0.0  # accumulated tangential (friction) impulse
    restitution_bias: float = 0.0


@dataclass
class Joint:
    """Weld joint between two affine bodies.

    Constrains:
    1. Position match at anchor points (2 bilateral constraints)
    2. Relative orientation via cross-product of F*axis (1 bilateral constraint)

    Total: 3 constraints — same as a rigid-body weld in 2D.
    Each body keeps 3 independent deformation DoFs (stretch + shear).
    """
    body_a_idx: int
    body_b_idx: int
    local_a: np.ndarray  # (2,) anchor on A in reference space
    local_b: np.ndarray  # (2,) anchor on B in reference space
    axis_a: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    axis_b: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    lam_pos_x: float = 0.0
    lam_pos_y: float = 0.0
    lam_angle: float = 0.0


@dataclass
class State:
    bodies: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    contacts: list = field(default_factory=list)
    time: float = 0.0
    step_count: int = 0


def _effective_mass(J_a, body_a, J_b=None, body_b=None):
    """K = J_a M_a^{-1} J_a^T + J_b M_b^{-1} J_b^T (scalar)."""
    inv_m_a = body_a.inv_mass_vec
    K = float(np.dot(J_a * inv_m_a, J_a))
    if J_b is not None and body_b is not None:
        inv_m_b = body_b.inv_mass_vec
        K += float(np.dot(J_b * inv_m_b, J_b))
    return K


def detect_contacts(state):
    """Build contact list from current positions."""
    contacts = []
    bodies = state.bodies

    # Floor contacts
    for i, b in enumerate(bodies):
        gap = collision.floor_gap(b)
        if gap < 0.01 * b.r0:
            J_a = collision.floor_jacobian(b)
            K = _effective_mass(J_a, b)
            if K > 1e-15:
                contacts.append(Contact(
                    type="floor", body_a_idx=i, body_b_idx=-1,
                    gap=gap, normal=np.array([0.0, 1.0]),
                    J_a=J_a, J_b=None, K_inv=1.0 / K,
                ))

    # Ellipse-ellipse contacts
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            result = collision.ellipse_ellipse(bodies[i], bodies[j])
            if result is None:
                continue
            gap, normal, pA, pB = result
            if gap < 0.01 * min(bodies[i].r0, bodies[j].r0):
                J_a, J_b = collision.ellipse_ellipse_jacobian(
                    bodies[i], bodies[j], normal, pA, pB)
                K = _effective_mass(J_a, bodies[i], J_b, bodies[j])
                if K > 1e-15:
                    contacts.append(Contact(
                        type="pair", body_a_idx=i, body_b_idx=j,
                        gap=gap, normal=normal,
                        J_a=J_a, J_b=J_b, K_inv=1.0 / K,
                    ))

    return contacts


def _apply_velocity_impulse(contact, bodies, dlam):
    """Apply impulse dlam along the contact Jacobian (velocity only)."""
    b_a = bodies[contact.body_a_idx]
    inv_m_a = b_a.inv_mass_vec
    b_a.vc += dlam * contact.J_a[:2] * inv_m_a[:2]
    b_a.vF += dlam * contact.J_a[2:] * inv_m_a[2:]

    if contact.J_b is not None and contact.body_b_idx >= 0:
        b_b = bodies[contact.body_b_idx]
        inv_m_b = b_b.inv_mass_vec
        b_b.vc += dlam * contact.J_b[:2] * inv_m_b[:2]
        b_b.vF += dlam * contact.J_b[2:] * inv_m_b[2:]


def _apply_position_impulse(contact, bodies, dlam, correct_F=True):
    """Apply position correction impulse (modifies positions only)."""
    b_a = bodies[contact.body_a_idx]
    inv_m_a = b_a.inv_mass_vec
    b_a.c += dlam * contact.J_a[:2] * inv_m_a[:2]
    if correct_F:
        b_a.F += dlam * contact.J_a[2:] * inv_m_a[2:]

    if contact.J_b is not None and contact.body_b_idx >= 0:
        b_b = bodies[contact.body_b_idx]
        inv_m_b = b_b.inv_mass_vec
        b_b.c += dlam * contact.J_b[:2] * inv_m_b[:2]
        if correct_F:
            b_b.F += dlam * contact.J_b[2:] * inv_m_b[2:]


def _relin_be_F(body, dt, impulse_F, newton_iters=10, ls_max=20):
    """Re-solve backward Euler for F given accumulated constraint impulse on F-DoFs.

    Minimises the modified incremental potential:
        E(vF) = (mu_i/2)||vF - vF_pre||^2  +  Psi(F_pre + dt*vF)*scale
                - impulse_F . vF

    The stationarity condition is the BE momentum balance:
        mu_i*(vF - vF_pre) - dt*f_elastic(F_pre + dt*vF) - impulse_F = 0

    Uses Newton iterations with backtracking line search to handle barrier
    energies (Bower) where the energy surface is steep near det(F) = 0.
    """
    from . import energy as energy_mod

    mu_i = body.mu_inertia
    mu_l, lam_l = body.lame
    scale = body._energy_scale
    psi_fn, pk1_fn, _, hess_spd_fn = body._energy_funcs()
    need_det_guard = (body.energy_model == "bower")

    F_pre = body._F_pre
    vF_pre = body._vF_pre

    def ip_energy(vF):
        F_trial = F_pre + dt * vF
        return (0.5 * mu_i * float(np.dot(vF - vF_pre, vF - vF_pre))
                + psi_fn(F_trial, mu_l, lam_l) * scale
                - float(np.dot(impulse_F, vF)))

    # Warm-start from current vF (previous re-solve result) rather than
    # vF_pre (pre-integration).  Each iteration only adds incremental
    # impulse, so warm-starting converges much faster for large chains.
    vF = body.vF.copy()

    for _ in range(newton_iters):
        F_cur = F_pre + dt * vF
        f_el = -pk1_fn(F_cur, mu_l, lam_l) * scale
        H_el = hess_spd_fn(F_cur, mu_l, lam_l) * scale

        residual = mu_i * (vF - vF_pre) - dt * f_el - impulse_F
        A = mu_i * np.eye(4) + dt ** 2 * H_el

        try:
            dvF = np.linalg.solve(A, -residual)
        except np.linalg.LinAlgError:
            break

        # Backtracking line search (Armijo)
        E_cur = ip_energy(vF)
        directional = float(np.dot(residual, dvF))
        alpha = 1.0
        for _ in range(ls_max):
            vF_trial = vF + alpha * dvF
            F_trial = F_pre + dt * vF_trial
            if need_det_guard and energy_mod._det2(F_trial) < energy_mod._BOWER_J_FLOOR:
                alpha *= 0.5
                continue
            E_trial = ip_energy(vF_trial)
            if E_trial <= E_cur + 1e-4 * alpha * directional:
                break
            alpha *= 0.5

        vF = vF + alpha * dvF
        if np.max(np.abs(alpha * dvF)) < 1e-12:
            break

    body.vF = vF
    body.F = F_pre + dt * body.vF


def _contact_velocity(contact, bodies):
    """Compute J * v for this contact."""
    b_a = bodies[contact.body_a_idx]
    Jv = float(np.dot(contact.J_a, b_a.v))
    if contact.J_b is not None and contact.body_b_idx >= 0:
        b_b = bodies[contact.body_b_idx]
        Jv += float(np.dot(contact.J_b, b_b.v))
    return Jv


def _contact_velocity_trans(contact, bodies):
    """Compute J_trans * v (translational DoFs only)."""
    b_a = bodies[contact.body_a_idx]
    Jv = float(np.dot(contact.J_a[:2], b_a.vc))
    if contact.J_b is not None and contact.body_b_idx >= 0:
        b_b = bodies[contact.body_b_idx]
        Jv += float(np.dot(contact.J_b[:2], b_b.vc))
    return Jv


# ── Joint constraint helpers ─────────────────────────────────────────────

def _apply_impulse_pos(body, J, dlam):
    """Apply position impulse dlam along Jacobian J to a single body."""
    inv_m = body.inv_mass_vec
    body.c += dlam * J[:2] * inv_m[:2]
    body.F += dlam * J[2:] * inv_m[2:]


def _apply_impulse_pos_relin(body, J, dlam, dt):
    """Apply position impulse in relin mode: c directly, F accumulated.

    The F-part of the impulse is converted to a momentum impulse (÷ dt)
    and accumulated into body._impulse_F.  The caller must call _relin_be_F
    once per body after all impulses in a pass are accumulated.

    Direct mode:  ΔF = dlam * J / mu_i  →  ΔvF = ΔF / dt
    Momentum:     P = mu_i * ΔvF = dlam * J / dt
    """
    inv_m = body.inv_mass_vec
    body.c += dlam * J[:2] * inv_m[:2]
    if body.static:
        return
    body._impulse_F += dlam * J[2:] / dt


def _apply_impulse_vel(body, J, dlam):
    """Apply velocity impulse dlam along Jacobian J to a single body."""
    inv_m = body.inv_mass_vec
    body.vc += dlam * J[:2] * inv_m[:2]
    body.vF += dlam * J[2:] * inv_m[2:]


def _joint_position_error(joint, bodies):
    """Position constraint violation: pA - pB (2-vector)."""
    ba = bodies[joint.body_a_idx]
    bb = bodies[joint.body_b_idx]
    la, lb = joint.local_a, joint.local_b
    pa = ba.c + np.array([ba.F[0]*la[0] + ba.F[1]*la[1],
                           ba.F[2]*la[0] + ba.F[3]*la[1]])
    pb = bb.c + np.array([bb.F[0]*lb[0] + bb.F[1]*lb[1],
                           bb.F[2]*lb[0] + bb.F[3]*lb[1]])
    return pa - pb


def _joint_angle_error(joint, bodies):
    """Angle constraint: (F_a * axis_a) x (F_b * axis_b) (scalar).

    Zero when the axes are parallel (same orientation).
    """
    ba = bodies[joint.body_a_idx]
    bb = bodies[joint.body_b_idx]
    aa, ab = joint.axis_a, joint.axis_b
    fa = np.array([ba.F[0]*aa[0] + ba.F[1]*aa[1],
                    ba.F[2]*aa[0] + ba.F[3]*aa[1]])
    fb = np.array([bb.F[0]*ab[0] + bb.F[1]*ab[1],
                    bb.F[2]*ab[0] + bb.F[3]*ab[1]])
    return fa[0] * fb[1] - fa[1] * fb[0]


def _joint_pos_jacobians(joint):
    """Position Jacobians: x-row and y-row for bodies A, B.

    C_pos = c_a + F_a * local_a - c_b - F_b * local_b
    Rows decouple: x touches (cx, F11, F12), y touches (cy, F21, F22).
    """
    la, lb = joint.local_a, joint.local_b
    Jx_a = np.array([1.0, 0.0, la[0], la[1], 0.0, 0.0])
    Jx_b = np.array([-1.0, 0.0, -lb[0], -lb[1], 0.0, 0.0])
    Jy_a = np.array([0.0, 1.0, 0.0, 0.0, la[0], la[1]])
    Jy_b = np.array([0.0, -1.0, 0.0, 0.0, -lb[0], -lb[1]])
    return Jx_a, Jx_b, Jy_a, Jy_b


def _joint_angle_jacobians(joint, bodies):
    """Angle constraint Jacobians.

    C_angle = (F_a*axis_a) x (F_b*axis_b)
    Only involves F-DoFs (not center-of-mass).
    """
    ba = bodies[joint.body_a_idx]
    bb = bodies[joint.body_b_idx]
    aa, ab = joint.axis_a, joint.axis_b
    fa = np.array([ba.F[0]*aa[0] + ba.F[1]*aa[1],
                    ba.F[2]*aa[0] + ba.F[3]*aa[1]])
    fb = np.array([bb.F[0]*ab[0] + bb.F[1]*ab[1],
                    bb.F[2]*ab[0] + bb.F[3]*ab[1]])

    Ja = np.array([0.0, 0.0,
                    aa[0]*fb[1],  aa[1]*fb[1],
                   -aa[0]*fb[0], -aa[1]*fb[0]])
    Jb = np.array([0.0, 0.0,
                   -fa[1]*ab[0], -fa[1]*ab[1],
                    fa[0]*ab[0],  fa[0]*ab[1]])
    return Ja, Jb


def _solve_joint_position(joint, bodies, bias=0.8, relin=False, dt=None):
    """One position-correction iteration for a weld joint.

    Uses higher bias than contacts (bilateral constraints need tighter
    enforcement). Forward-backward sweeps in the caller handle chain
    convergence.

    With relin=True, F-DoF impulses are accumulated and mediated through
    backward Euler re-solve, letting the elastic energy (including barriers)
    control deformation.
    """
    ba = bodies[joint.body_a_idx]
    bb = bodies[joint.body_b_idx]
    apply = (_apply_impulse_pos_relin if relin
             else lambda body, J, dlam, dt=None: _apply_impulse_pos(body, J, dlam))

    # Position constraints (x, y)
    C = _joint_position_error(joint, bodies)
    Jx_a, Jx_b, Jy_a, Jy_b = _joint_pos_jacobians(joint)

    K_x = _effective_mass(Jx_a, ba, Jx_b, bb)
    if K_x > 1e-15:
        dlam = -bias * C[0] / K_x
        apply(ba, Jx_a, dlam, dt)
        apply(bb, Jx_b, dlam, dt)

    K_y = _effective_mass(Jy_a, ba, Jy_b, bb)
    if K_y > 1e-15:
        dlam = -bias * C[1] / K_y
        apply(ba, Jy_a, dlam, dt)
        apply(bb, Jy_b, dlam, dt)

    # Angle constraint
    C_angle = _joint_angle_error(joint, bodies)
    Ja, Jb = _joint_angle_jacobians(joint, bodies)
    K_angle = _effective_mass(Ja, ba, Jb, bb)
    if K_angle > 1e-15:
        dlam = -bias * C_angle / K_angle
        apply(ba, Ja, dlam, dt)
        apply(bb, Jb, dlam, dt)


def _solve_joint_velocity(joint, bodies):
    """One velocity-correction iteration for a weld joint (bilateral)."""
    ba = bodies[joint.body_a_idx]
    bb = bodies[joint.body_b_idx]

    Jx_a, Jx_b, Jy_a, Jy_b = _joint_pos_jacobians(joint)

    # X velocity
    Jv_x = float(np.dot(Jx_a, ba.v) + np.dot(Jx_b, bb.v))
    K_x = _effective_mass(Jx_a, ba, Jx_b, bb)
    if K_x > 1e-15:
        dlam = -Jv_x / K_x
        _apply_impulse_vel(ba, Jx_a, dlam)
        _apply_impulse_vel(bb, Jx_b, dlam)

    # Y velocity
    Jv_y = float(np.dot(Jy_a, ba.v) + np.dot(Jy_b, bb.v))
    K_y = _effective_mass(Jy_a, ba, Jy_b, bb)
    if K_y > 1e-15:
        dlam = -Jv_y / K_y
        _apply_impulse_vel(ba, Jy_a, dlam)
        _apply_impulse_vel(bb, Jy_b, dlam)

    # Angle velocity
    Ja, Jb = _joint_angle_jacobians(joint, bodies)
    Jv_a = float(np.dot(Ja, ba.v) + np.dot(Jb, bb.v))
    K_angle = _effective_mass(Ja, ba, Jb, bb)
    if K_angle > 1e-15:
        dlam = -Jv_a / K_angle
        _apply_impulse_vel(ba, Ja, dlam)
        _apply_impulse_vel(bb, Jb, dlam)


# ── Contact velocity helpers ────────────────────────────────────────────

def _contact_velocity_tangent(contact, bodies):
    """Compute Jt * v for this contact (tangential relative velocity)."""
    b_a = bodies[contact.body_a_idx]
    Jv = float(np.dot(contact.Jt_a, b_a.v))
    if contact.Jt_b is not None and contact.body_b_idx >= 0:
        b_b = bodies[contact.body_b_idx]
        Jv += float(np.dot(contact.Jt_b, b_b.v))
    return Jv


def _contact_velocity_tangent_trans(contact, bodies):
    """Compute Jt_trans * v (translational DoFs only)."""
    b_a = bodies[contact.body_a_idx]
    Jv = float(np.dot(contact.Jt_a[:2], b_a.vc))
    if contact.Jt_b is not None and contact.body_b_idx >= 0:
        b_b = bodies[contact.body_b_idx]
        Jv += float(np.dot(contact.Jt_b[:2], b_b.vc))
    return Jv


def step(state, params):
    """One simulation step."""
    bodies = state.bodies
    dt = params.dt

    # 1. Save pre-integration state for relin mode
    use_relin = params.relin
    if use_relin:
        for b in bodies:
            b._F_pre = b.F.copy()
            b._vF_pre = b.vF.copy()

    # 2. Integrate (backward Euler: gravity + elastic)
    for b in bodies:
        body_mod.integrate_backward_euler(b, dt, params.gravity)

    # 3. Detect contacts
    state.contacts = detect_contacts(state)

    # 4. Position pass: correct penetration
    correct_F = params.position_correct_F

    if use_relin:
        for b in bodies:
            b._impulse_F = np.zeros(4)

    for _ in range(params.position_iters):
        # Joint position constraints (bilateral, higher bias than contacts)
        # Forward-backward sweep for fast convergence in chains
        for j in state.joints:
            _solve_joint_position(j, bodies, relin=use_relin, dt=dt)

        # Relin re-solve between sweeps: update F so the backward sweep
        # sees correct angle Jacobians (which depend on F*axis).
        if use_relin and state.joints:
            for b in bodies:
                if not b.static and np.any(b._impulse_F != 0):
                    _relin_be_F(b, dt, b._impulse_F)

        for j in reversed(state.joints):
            _solve_joint_position(j, bodies, relin=use_relin, dt=dt)

        # Second relin re-solve after backward sweep
        if use_relin and state.joints:
            for b in bodies:
                if not b.static and np.any(b._impulse_F != 0):
                    _relin_be_F(b, dt, b._impulse_F)

        for c in state.contacts:
            if c.gap >= 0:
                continue

            if use_relin:
                # Use full K (including F) for impulse magnitude
                dlam = -params.bias_factor * c.gap * c.K_inv

                # Apply position impulse to c only (not F directly)
                b_a = bodies[c.body_a_idx]
                inv_m_a = b_a.inv_mass_vec
                b_a.c += dlam * c.J_a[:2] * inv_m_a[:2]

                # Accumulate contact impulse on F (convert to momentum: ÷ dt)
                b_a._impulse_F += dlam * c.J_a[2:] / dt

                if c.J_b is not None and c.body_b_idx >= 0:
                    b_b = bodies[c.body_b_idx]
                    inv_m_b = b_b.inv_mass_vec
                    b_b.c += dlam * c.J_b[:2] * inv_m_b[:2]
                    b_b._impulse_F += dlam * c.J_b[2:] / dt

                # Re-solve BE for contact bodies
                _relin_be_F(b_a, dt, b_a._impulse_F)
                if c.J_b is not None and c.body_b_idx >= 0:
                    _relin_be_F(b_b, dt, b_b._impulse_F)
            else:
                if correct_F:
                    K_pos_inv = c.K_inv
                else:
                    b_a = bodies[c.body_a_idx]
                    K_pos = float(np.dot(c.J_a[:2] * b_a.inv_mass_vec[:2], c.J_a[:2]))
                    if c.J_b is not None and c.body_b_idx >= 0:
                        b_b = bodies[c.body_b_idx]
                        K_pos += float(np.dot(c.J_b[:2] * b_b.inv_mass_vec[:2], c.J_b[:2]))
                    K_pos_inv = 1.0 / K_pos if K_pos > 1e-15 else 0.0
                dlam = -params.bias_factor * c.gap * K_pos_inv
                _apply_position_impulse(c, bodies, dlam, correct_F=correct_F)

            # Re-evaluate gap (accounts for both c and F changes)
            if c.type == "floor":
                c.gap = collision.floor_gap(bodies[c.body_a_idx])
            elif c.type == "pair":
                result = collision.ellipse_ellipse(
                    bodies[c.body_a_idx], bodies[c.body_b_idx])
                if result is not None:
                    c.gap, c.normal, pA, pB = result
                    c.J_a, c.J_b = collision.ellipse_ellipse_jacobian(
                        bodies[c.body_a_idx], bodies[c.body_b_idx],
                        c.normal, pA, pB)
                    K = _effective_mass(c.J_a, bodies[c.body_a_idx],
                                        c.J_b, bodies[c.body_b_idx])
                    c.K_inv = 1.0 / K if K > 1e-15 else 0.0

    # 4. Velocity pass: non-penetration + friction constraints
    couple_F = params.velocity_couple_F
    use_friction = params.friction > 0.0

    # Compute per-contact velocity K and tangent Jacobians for the velocity pass
    for c in state.contacts:
        if couple_F:
            c._K_vel_inv = c.K_inv
        else:
            b_a = bodies[c.body_a_idx]
            K_vel = float(np.dot(c.J_a[:2] * b_a.inv_mass_vec[:2], c.J_a[:2]))
            if c.J_b is not None and c.body_b_idx >= 0:
                b_b = bodies[c.body_b_idx]
                K_vel += float(np.dot(c.J_b[:2] * b_b.inv_mass_vec[:2], c.J_b[:2]))
            c._K_vel_inv = 1.0 / K_vel if K_vel > 1e-15 else 0.0

        # Tangent Jacobians for friction
        if use_friction:
            if c.type == "floor":
                c.Jt_a = collision.tangent_jacobian(
                    bodies[c.body_a_idx], c.normal, +1.0)
                c.Jt_b = None
            else:
                c.Jt_a = collision.tangent_jacobian(
                    bodies[c.body_a_idx], c.normal, -1.0)
                c.Jt_b = collision.tangent_jacobian(
                    bodies[c.body_b_idx], c.normal, +1.0)

            Kt = _effective_mass(
                c.Jt_a, bodies[c.body_a_idx],
                c.Jt_b if c.body_b_idx >= 0 else None,
                bodies[c.body_b_idx] if c.body_b_idx >= 0 else None)
            c._Kt_vel_inv = 1.0 / Kt if Kt > 1e-15 else 0.0

            if not couple_F:
                b_a = bodies[c.body_a_idx]
                Kt_t = float(np.dot(c.Jt_a[:2] * b_a.inv_mass_vec[:2], c.Jt_a[:2]))
                if c.Jt_b is not None and c.body_b_idx >= 0:
                    b_b = bodies[c.body_b_idx]
                    Kt_t += float(np.dot(c.Jt_b[:2] * b_b.inv_mass_vec[:2], c.Jt_b[:2]))
                c._Kt_vel_inv = 1.0 / Kt_t if Kt_t > 1e-15 else 0.0

    # Store pre-impulse Jv for restitution target (computed once, not per iteration)
    for c in state.contacts:
        if couple_F:
            Jv_pre = _contact_velocity(c, bodies)
        else:
            Jv_pre = _contact_velocity_trans(c, bodies)
        c.restitution_bias = -params.restitution * min(Jv_pre, 0.0)

    for _ in range(params.velocity_iters):
        # Joint velocity constraints (forward-backward sweep)
        for j in state.joints:
            _solve_joint_velocity(j, bodies)
        for j in reversed(state.joints):
            _solve_joint_velocity(j, bodies)

        for c in state.contacts:
            # --- Normal impulse ---
            if couple_F:
                Jv = _contact_velocity(c, bodies)
            else:
                Jv = _contact_velocity_trans(c, bodies)
            dlam = -(Jv - c.restitution_bias) * c._K_vel_inv
            new_lam = c.lam + dlam
            new_lam = max(new_lam, 0.0)
            dlam = new_lam - c.lam
            c.lam = new_lam
            b_a = bodies[c.body_a_idx]
            inv_m_a = b_a.inv_mass_vec
            b_a.vc += dlam * c.J_a[:2] * inv_m_a[:2]
            if couple_F:
                b_a.vF += dlam * c.J_a[2:] * inv_m_a[2:]
            if c.J_b is not None and c.body_b_idx >= 0:
                b_b = bodies[c.body_b_idx]
                inv_m_b = b_b.inv_mass_vec
                b_b.vc += dlam * c.J_b[:2] * inv_m_b[:2]
                if couple_F:
                    b_b.vF += dlam * c.J_b[2:] * inv_m_b[2:]

            # --- Friction impulse (Coulomb) ---
            if use_friction and c.lam > 0:
                if couple_F:
                    Jt_v = _contact_velocity_tangent(c, bodies)
                else:
                    Jt_v = _contact_velocity_tangent_trans(c, bodies)
                dlam_t = -Jt_v * c._Kt_vel_inv
                max_friction = params.friction * c.lam
                new_lam_t = c.lam_t + dlam_t
                new_lam_t = max(-max_friction, min(new_lam_t, max_friction))
                dlam_t = new_lam_t - c.lam_t
                c.lam_t = new_lam_t
                b_a = bodies[c.body_a_idx]
                inv_m_a = b_a.inv_mass_vec
                b_a.vc += dlam_t * c.Jt_a[:2] * inv_m_a[:2]
                if couple_F:
                    b_a.vF += dlam_t * c.Jt_a[2:] * inv_m_a[2:]
                if c.Jt_b is not None and c.body_b_idx >= 0:
                    b_b = bodies[c.body_b_idx]
                    inv_m_b = b_b.inv_mass_vec
                    b_b.vc += dlam_t * c.Jt_b[:2] * inv_m_b[:2]
                    if couple_F:
                        b_b.vF += dlam_t * c.Jt_b[2:] * inv_m_b[2:]

    state.time += dt
    state.step_count += 1
