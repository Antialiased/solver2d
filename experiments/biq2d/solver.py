"""Solver pipeline for biq2d.

M0 milestone:  `step()` — single backward-Euler step, no joints/contacts.
M1 (cantilever) milestone:  `step_vbd()` — pure VBD per-body block coordinate
descent with smooth finite-α joint penalties; supports edge corner-pair joints.
M2 (FQ cantilever):  `step_vbd_fq()` — same VBD recipe extended to the
12-DoF full-quadratic body, with 3-point-per-edge `JointFQ` constraints that
can sample the edge polynomial (not just its endpoints).
"""
import numpy as np
from dataclasses import dataclass, field
from .body import BodyBQ2D, BodyFQ2D, integrate_backward_euler, _HX_OFFSET, _HY_OFFSET
from . import energy


# ── State and params ──────────────────────────────────────────────────

@dataclass
class State:
    bodies: list
    joints: list = field(default_factory=list)
    time: float = 0.0
    step_count: int = 0


@dataclass
class Params:
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -10.0]))
    dt: float = 1.0 / 240.0
    # VBD knobs
    vbd_sweeps: int = 5      # GS sweeps per step (each = forward + backward)
    vbd_newton: int = 5      # max Newton iters per body per sweep visit
    # Block mode for body-body contact:
    #   "per_body" — owner-vertex GS, 12-DoF Newton per body visit (Stage 1).
    #   "per_edge" — 24-DoF Newton per active contact pair (Stage 2).
    block_mode: str = "per_body"


# ── Joint dataclass and corner geometry ───────────────────────────────

@dataclass
class Joint:
    """Point joint between two bodies' specified reference corners.

    corner_a, corner_b are (σ1, σ2) ∈ {-1, +1}² indicating which of the four
    reference-square corners participates from each body.  The constraint is
    P_a(corner_a) = P_b(corner_b) (2 scalar equations).

    For an "edge connection" between two bodies, create *two* joints — one per
    shared corner — to obtain rigid edge-matching.
    """
    body_a_idx: int
    body_b_idx: int
    corner_a: tuple
    corner_b: tuple


def corner_position(body, corner):
    """World position of body's corner ξ = (σ1, σ2)."""
    s1, s2 = corner
    h = body.h
    # F flat row-major [F11, F12, F21, F22]; column 0 = (F11, F21), column 1 = (F12, F22)
    col0 = np.array([body.F[0], body.F[2]])
    col1 = np.array([body.F[1], body.F[3]])
    return body.c + h * (s1 * col0 + s2 * col1 + s1 * s2 * body.G)


def corner_jacobian_q(corner, h):
    """∂P / ∂q  where  q = (cx, cy, F11, F12, F21, F22, Gx, Gy).  Shape (2, 8).

    P is *linear* in q for fixed corner — this Jacobian is constant.
    """
    s1, s2 = corner
    J = np.zeros((2, 8))
    J[0, 0] = 1.0;  J[0, 2] = h * s1;  J[0, 3] = h * s2;  J[0, 6] = h * s1 * s2
    J[1, 1] = 1.0;  J[1, 4] = h * s1;  J[1, 5] = h * s2;  J[1, 7] = h * s1 * s2
    return J


# ── M0: simple BE step ────────────────────────────────────────────────

def step(state, params):
    """Single BE step.  No joints, no contacts.  Dispatches per body type."""
    for body in state.bodies:
        if not body.static:
            body.integrate_be(params.dt, params.gravity)
    state.time += params.dt
    state.step_count += 1


# ── M1: pure VBD with smooth penalty ──────────────────────────────────

def _collect_incident_constraints(body_idx, state):
    """Per-body view of all incident corner constraints.

    Returns a list of (J_q, target) where J_q is the (2, 8) self-Jacobian and
    target is the world position of the OTHER body's matching corner at its
    *current* state (held fixed during this body's solve).
    """
    bodies = state.bodies
    self_b = bodies[body_idx]
    h = self_b.h
    out = []
    for j in state.joints:
        if j.body_a_idx == body_idx:
            other = bodies[j.body_b_idx]
            J_q = corner_jacobian_q(j.corner_a, h)
            target = corner_position(other, j.corner_b)
        elif j.body_b_idx == body_idx:
            other = bodies[j.body_a_idx]
            J_q = corner_jacobian_q(j.corner_b, h)
            target = corner_position(other, j.corner_a)
        else:
            continue
        out.append((J_q, target.copy()))
    return out


def vbd_body_step(body, body_idx, state, dt, gravity, alpha,
                  max_newton=5, ls_max=20):
    """One VBD visit to a single body — Newton + Armijo on its 8-D IP.

    Minimises (over v = (vc, vF, vG)):

        IP(v) = ½(v - v_pre)ᵀ M (v - v_pre)
              − M_c · g · (c_pre + dt·vc)              # gravity (c only)
              + ∫ Ψ(J(F + dt·vF, G + dt·vG; ξ)) dX   # elastic
              + (α/2) Σ_corners ‖J_q (q_pre + dt v) − target‖²
              + (optional) det-J barrier on (F, G)
    """
    if body.static:
        return

    M = body.mass_vec.copy()           # (8,)
    M_c = body.mass
    mu_l, lam_l = body.lame
    h = body.h

    q_pre = np.concatenate([body._c_pre, body._F_pre, body._G_pre])
    v_pre = np.concatenate([body._vc_pre, body._vF_pre, body._vG_pre])

    incident = _collect_incident_constraints(body_idx, state)

    kappa = body.barrier_kappa
    b_eps = body.barrier_eps

    def ip_energy(v):
        q = q_pre + dt * v
        c_t = q[:2]
        F_t = q[2:6]
        G_t = q[6:]
        dv = v - v_pre
        E = 0.5 * float(np.dot(dv * M, dv))
        E -= M_c * float(np.dot(gravity, c_t))
        E += energy.integrated_energy(F_t, G_t, mu_l, lam_l, h)
        for J_q, target in incident:
            r = J_q @ q - target
            E += 0.5 * alpha * float(np.dot(r, r))
        if kappa > 0.0:
            Eb, _, _ = energy.barrier_energy_grad_hess(F_t, G_t, kappa, b_eps)
            E += Eb
        return E

    v = v_pre.copy()

    for _ in range(max_newton):
        q = q_pre + dt * v
        F_t = q[2:6]
        G_t = q[6:]

        # Gradient w.r.t. v
        grad = M * (v - v_pre)
        grad[:2] -= dt * M_c * gravity
        g_el_FG = energy.integrated_grad(F_t, G_t, mu_l, lam_l, h)
        grad[2:] += dt * g_el_FG
        for J_q, target in incident:
            r = J_q @ q - target
            grad += alpha * dt * (J_q.T @ r)
        if kappa > 0.0:
            _, g_b_FG, _ = energy.barrier_energy_grad_hess(F_t, G_t, kappa, b_eps)
            grad[2:] += dt * g_b_FG

        # SPD Hessian (8×8)
        H = np.diag(M)
        H_el_FG = energy.integrated_hessian_spd(F_t, G_t, mu_l, lam_l, h)
        H[2:, 2:] += dt * dt * H_el_FG
        for J_q, _ in incident:
            H += alpha * dt * dt * (J_q.T @ J_q)
        if kappa > 0.0:
            _, _, H_b_FG = energy.barrier_energy_grad_hess(F_t, G_t, kappa, b_eps)
            H[2:, 2:] += dt * dt * H_b_FG

        try:
            dv = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            break

        # Armijo line search
        E_cur = ip_energy(v)
        directional = float(np.dot(grad, dv))
        a = 1.0
        for _ in range(ls_max):
            v_trial = v + a * dv
            if ip_energy(v_trial) <= E_cur + 1e-4 * a * directional:
                break
            a *= 0.5

        v = v + a * dv
        if np.max(np.abs(a * dv)) < 1e-12:
            break

    body.vc = v[:2]
    body.vF = v[2:6]
    body.vG = v[6:]
    body.c = q_pre[:2]  + dt * body.vc
    body.F = q_pre[2:6] + dt * body.vF
    body.G = q_pre[6:]  + dt * body.vG


def step_vbd(state, params, alpha):
    """Pure-VBD step: GS sweeps over bodies with finite-α corner penalties.

    No TGS substeps, no warm-start, no impulse accumulation.  Each body's
    visit is a Newton minimisation of its local IP (see `vbd_body_step`).
    """
    # Save initial state per non-static body
    for b in state.bodies:
        if not b.static:
            b._c_pre  = b.c.copy()
            b._F_pre  = b.F.copy()
            b._G_pre  = b.G.copy()
            b._vc_pre = b.vc.copy()
            b._vF_pre = b.vF.copy()
            b._vG_pre = b.vG.copy()

    n = len(state.bodies)
    for _ in range(params.vbd_sweeps):
        for i in range(n):
            vbd_body_step(state.bodies[i], i, state, params.dt, params.gravity,
                          alpha, max_newton=params.vbd_newton)
        for i in range(n - 1, -1, -1):
            vbd_body_step(state.bodies[i], i, state, params.dt, params.gravity,
                          alpha, max_newton=params.vbd_newton)

    state.time += params.dt
    state.step_count += 1


# ── M2: FQ2D pure VBD with smooth penalty ─────────────────────────────

@dataclass
class JointFQ:
    """Point joint between two FQ2D bodies at arbitrary reference points.

    xi_a, xi_b ∈ [-1, 1]² are reference-space coordinates (not restricted to
    corners).  The constraint is P_a(xi_a) = P_b(xi_b) (2 scalar equations).

    A rigid edge connection between two FQ2D bodies is realised by *three*
    JointFQ instances — one per matched point along the shared edge (e.g.
    the two corners and the midpoint).  Three points uniquely pin the
    degree-2 edge polynomial, giving exact edge continuity for the
    full-quadratic basis.
    """
    body_a_idx: int
    body_b_idx: int
    xi_a: tuple
    xi_b: tuple


def point_position_fq(body, xi):
    """World position of BodyFQ2D at reference point ξ ∈ [-1, 1]²."""
    return body.map_points(np.array([xi], dtype=float))[0]


def point_jacobian_q_fq(xi, h):
    """∂P / ∂q for BodyFQ2D, q = (cx, cy, F11, F12, F21, F22, Gx, Gy,
    Hx0, Hx1, Hy0, Hy1).  Shape (2, 12).

    With the mean-zero banana basis φx = ξ₁² − 1/3, φy = ξ₂² − 1/3, the
    map is linear in q for fixed ξ, so this Jacobian is constant.
    """
    x1, x2 = xi
    phi_x = x1 * x1 - _HX_OFFSET
    phi_y = x2 * x2 - _HY_OFFSET
    J = np.zeros((2, 12))
    # x-row: cx, F11·ξ₁, F12·ξ₂, Gx·ξ₁ξ₂, Hx0·φx, Hy0·φy
    J[0, 0]  = 1.0
    J[0, 2]  = h * x1
    J[0, 3]  = h * x2
    J[0, 6]  = h * x1 * x2
    J[0, 8]  = h * phi_x
    J[0, 10] = h * phi_y
    # y-row: cy, F21·ξ₁, F22·ξ₂, Gy·ξ₁ξ₂, Hx1·φx, Hy1·φy
    J[1, 1]  = 1.0
    J[1, 4]  = h * x1
    J[1, 5]  = h * x2
    J[1, 7]  = h * x1 * x2
    J[1, 9]  = h * phi_x
    J[1, 11] = h * phi_y
    return J


_OUTER_XI = np.array([(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0), (1.0, 0.0),
                      (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)],
                     dtype=float)


def _outer_py_from_q(q, h):
    """(8,) world y-coord at each outer ξ point given full-quadratic q."""
    x1 = _OUTER_XI[:, 0]
    x2 = _OUTER_XI[:, 1]
    phi_hx = x1 * x1 - _HX_OFFSET
    phi_hy = x2 * x2 - _HY_OFFSET
    return q[1] + h * (q[4] * x1 + q[5] * x2 + q[7] * x1 * x2
                       + q[9] * phi_hx + q[11] * phi_hy)


# ── IPC log barrier with C² quadratic continuation below ε ───────────
#
# b(d; dhat, κ) = -κ · (d - dhat)² · ln(d / dhat)        for ε ≤ d < dhat
#               = κ · [ b̃(ε) + b̃'(ε)·(d-ε) + ½ b̃''(ε)·(d-ε)² ]  for d < ε
#               = 0                                       for d ≥ dhat
#
# where b̃ is the barrier with κ=1.  ε = 0.01·dhat keeps the function C²
# everywhere — including for warm starts that overshoot into d ≤ 0
# (penetrating).  Inside (ε, dhat) the gradient and hessian come from
# direct differentiation:
#   b̃'(d)  = -2u·L - u²/d
#   b̃''(d) = -2L  - 4u/d + u²/d²
# where u = d - dhat (negative for d < dhat) and L = ln(d/dhat) (negative
# for d < dhat).  At d = dhat: u = L = 0 → b, b', b'' all = 0 (smooth
# attachment to the inactive band).  At d → 0+: b'' → +∞ → quadratic
# continuation pulls the iterate back fast.
_BARRIER_EPS_FRAC = 0.01


def _barrier_value(d, dhat, kappa):
    if d >= dhat:
        return 0.0
    eps = _BARRIER_EPS_FRAC * dhat
    if d <= eps:
        u_e = eps - dhat
        L_e = np.log(eps / dhat)
        b_e   = -(u_e * u_e) * L_e
        bp_e  = -2.0 * u_e * L_e - (u_e * u_e) / eps
        bpp_e = -2.0 * L_e - 4.0 * u_e / eps + (u_e * u_e) / (eps * eps)
        delta = d - eps
        return kappa * (b_e + bp_e * delta + 0.5 * bpp_e * delta * delta)
    u = d - dhat
    L = np.log(d / dhat)
    return kappa * (-(u * u) * L)


def _barrier_grad(d, dhat, kappa):
    if d >= dhat:
        return 0.0
    eps = _BARRIER_EPS_FRAC * dhat
    if d <= eps:
        u_e = eps - dhat
        L_e = np.log(eps / dhat)
        bp_e  = -2.0 * u_e * L_e - (u_e * u_e) / eps
        bpp_e = -2.0 * L_e - 4.0 * u_e / eps + (u_e * u_e) / (eps * eps)
        delta = d - eps
        return kappa * (bp_e + bpp_e * delta)
    u = d - dhat
    L = np.log(d / dhat)
    return kappa * (-2.0 * u * L - (u * u) / d)


def _barrier_hess(d, dhat, kappa):
    if d >= dhat:
        return 0.0
    eps = _BARRIER_EPS_FRAC * dhat
    if d <= eps:
        u_e = eps - dhat
        L_e = np.log(eps / dhat)
        bpp_e = -2.0 * L_e - 4.0 * u_e / eps + (u_e * u_e) / (eps * eps)
        return kappa * bpp_e
    u = d - dhat
    L = np.log(d / dhat)
    return kappa * (-2.0 * L - 4.0 * u / d + (u * u) / (d * d))


def _floor_active_contacts(q, h, y_floor, dhat):
    """Active-set list of (Jn (12,), g_offset) for outer ξ vertices within
    `dhat` of the floor (or below it).  Frozen normal n = (0, 1).

    Each entry encodes a 1D linear functional g(q) = Jn·q + g_offset where
    g = py - y_floor (positive = separated, negative = penetrating).  Jn
    is the y-row of point_jacobian_q_fq for that ξ point — constant in q
    for a full-quadratic body.
    """
    Py = _outer_py_from_q(q, h)
    out = []
    for i, py in enumerate(Py):
        g0 = float(py - y_floor)
        if g0 >= dhat:
            continue
        Jn = point_jacobian_q_fq(_OUTER_XI[i], h)[1, :].copy()
        out.append((Jn, -float(y_floor)))
    return out


# ── Body-body contact: per-sub-cell chord polygons ───────────────────

# 4 sub-cell ξ-corners (CCW in ξ-space; CCW in world iff det J > 0).
# Edge i = from vertex i to vertex (i+1) % 4.
_SUBCELL_XI = (
    np.array([(-1.0, -1.0), ( 0.0, -1.0), ( 0.0,  0.0), (-1.0,  0.0)]),  # SW
    np.array([( 0.0, -1.0), ( 1.0, -1.0), ( 1.0,  0.0), ( 0.0,  0.0)]),  # SE
    np.array([( 0.0,  0.0), ( 1.0,  0.0), ( 1.0,  1.0), ( 0.0,  1.0)]),  # NE
    np.array([(-1.0,  0.0), ( 0.0,  0.0), ( 0.0,  1.0), (-1.0,  1.0)]),  # NW
)
# Exterior-edge mask per sub-cell: edges that lie on the body's outer hull
# (not on the interior cross at ξ₁=0 or ξ₂=0).
_SUBCELL_EXTERIOR = (
    np.array([True,  False, False, True ]),   # SW: bottom (0), left (3)
    np.array([True,  True,  False, False]),   # SE: bottom (0), right (1)
    np.array([False, True,  True,  False]),   # NE: right (1), top (2)
    np.array([False, False, True,  True ]),   # NW: top (2), left (3)
)


def _subcell_polygons_world(body):
    """Return list of 4 dicts (one per sub-cell), each with:

      - verts:         (4, 2) world positions, CCW order
      - normals:       (4, 2) outward unit normals per edge i = (Vᵢ → Vᵢ₊₁)
      - exterior_mask: (4,) bool — True where edge lies on body's outer hull
      - corner_xi:     (4, 2) ξ-coordinates of each vertex (constant per
                       sub-cell; needed to assemble the edge-origin Jacobian
                       Jn_B for 2-body Newton blocks).
    """
    polys = []
    for sc_xi, ext_mask in zip(_SUBCELL_XI, _SUBCELL_EXTERIOR):
        verts = body.map_points(sc_xi)                       # (4, 2)
        edges = np.roll(verts, -1, axis=0) - verts            # (4, 2)
        # Outward normal for CCW polygon: rotate edge CW: (ey, -ex), then normalise.
        nx =  edges[:, 1]
        ny = -edges[:, 0]
        norms = np.sqrt(nx * nx + ny * ny)
        # Guard against zero-length edges (shouldn't happen for a valid body).
        norms = np.where(norms > 1e-15, norms, 1.0)
        normals = np.column_stack([nx / norms, ny / norms])
        polys.append({"verts": verts, "normals": normals,
                      "exterior_mask": ext_mask, "corner_xi": sc_xi})
    return polys


def _vertex_in_subcell(P, polygon):
    """Test if world point P is inside this sub-cell polygon (weak inclusion)
    and, if so, return depth + outward normal of the deepest *exterior* edge.

    Weak inclusion (max sₑ ≤ 0) is required: a vertex sliding along a
    sub-cell partition (max sₑ = 0 exactly) is on the boundary of two
    adjacent sub-cells; both should return active so the contact pair
    isn't lost in degenerately-aligned configurations.  Edges marked
    interior to the body never contribute to the depth, so on-partition
    points get pushed only along genuine exterior normals.

    Returns (active: bool, depth: float, n_e: (2,) ndarray).
    """
    verts   = polygon["verts"]
    normals = polygon["normals"]
    ext     = polygon["exterior_mask"]
    rel = P[None, :] - verts                                  # (4, 2)
    s = np.einsum("ei,ei->e", normals, rel)                   # (4,)
    # Weak interior test: outside iff strictly positive on any edge.
    if np.max(s) > 0.0:
        return False, 0.0, normals[0]
    # Pick deepest *exterior* edge: argmax of sₑ over exterior edges
    # (sₑ ≤ 0 inside; "deepest" = least negative = max sₑ).
    s_ext = np.where(ext, s, -np.inf)
    if not np.any(np.isfinite(s_ext)):
        return False, 0.0, normals[0]
    e_idx = int(np.argmax(s_ext))
    return True, float(-s_ext[e_idx]), normals[e_idx]


def _outer_world_pts_from_q(q, h):
    """(8, 2) world positions of the 8 outer ξ vertices given q and h.

    Mirrors `_outer_py_from_q` but returns (x, y) for both rows.
    """
    x1 = _OUTER_XI[:, 0]
    x2 = _OUTER_XI[:, 1]
    phi_hx = x1 * x1 - _HX_OFFSET
    phi_hy = x2 * x2 - _HY_OFFSET
    Px = q[0] + h * (q[2] * x1 + q[3] * x2 + q[6] * x1 * x2
                     + q[8]  * phi_hx + q[10] * phi_hy)
    Py = q[1] + h * (q[4] * x1 + q[5] * x2 + q[7] * x1 * x2
                     + q[9]  * phi_hx + q[11] * phi_hy)
    return np.column_stack([Px, Py])


def _pick_active_edge(P, sc_poly, dhat):
    """Active-edge selection for a vertex P against a sub-cell polygon.

    Branches on whether the vertex is OUTSIDE or INSIDE the convex sub-cell:

    - **Outside** (max sₑ > 0):  the most-violated edge is the contact
      face.  If that edge is on an interior partition seam, skip — the
      adjacent sub-cell will handle the same approach via its own
      exterior edge.  Active iff g₀ = sₑ < dhat.
    - **Inside** (max sₑ ≤ 0):  vertex is penetrating somewhere.  Pick
      the **deepest exterior edge** (argmax of sₑ over exterior edges
      only) — the closest exit face.  Always active (penetration is by
      definition within barrier domain).  This branch must NOT use
      "argmax over all edges" because vertices on partition seams
      (sₑ = 0 on an interior edge) would be skipped, leaving the
      contact silent.

    Returns (n_e (2,), V_e (2,), xi_e (2,)) for the chosen edge, or None
    if the contact is not active.  xi_e is the ξ-coordinate of the edge-
    origin corner V_e (one of the four sub-cell ξ corners).
    """
    verts   = sc_poly["verts"]
    normals = sc_poly["normals"]
    ext     = sc_poly["exterior_mask"]
    cxi     = sc_poly["corner_xi"]
    rel = P[None, :] - verts                                 # (4, 2)
    s = np.einsum("ei,ei->e", normals, rel)                  # (4,)
    if np.max(s) > 0.0:
        e_idx = int(np.argmax(s))
        if not ext[e_idx]:
            return None                                      # adjacent sub-cell handles it
        if float(s[e_idx]) >= dhat:
            return None                                      # outside barrier band
    else:
        s_ext = np.where(ext, s, -np.inf)
        if not np.any(np.isfinite(s_ext)):
            return None                                      # no exterior edges (shouldn't happen)
        e_idx = int(np.argmax(s_ext))
    return normals[e_idx], verts[e_idx], cxi[e_idx]


def _body_body_active_contacts(q_self, h_self, other_polys_list, dhat):
    """Active-set list of (Jn (12,), g_offset) for self's outer ξ vertices
    near or penetrating any other body's sub-cell polygon.

    Each entry encodes a 1D linear functional g(q) = Jn·q + g_offset
    where g = nᵀ·(P_i − V_e) (positive = separated, negative =
    penetrating) along the FROZEN outward normal n_e of the active edge.
    Jn = nᵀ·J_q(ξᵢ).
    """
    pts = _outer_world_pts_from_q(q_self, h_self)
    out = []
    for i in range(len(_OUTER_XI)):
        P = pts[i]
        J_q = point_jacobian_q_fq(_OUTER_XI[i], h_self)         # (2, 12)
        for polys in other_polys_list:
            for sc_poly in polys:
                pick = _pick_active_edge(P, sc_poly, dhat)
                if pick is None:
                    continue
                n_e, V_e, _xi_e = pick
                Jn = J_q.T @ n_e                                  # (12,)
                g_offset = -float(np.dot(n_e, V_e))
                out.append((Jn, g_offset))
    return out


def _body_body_active_pair(q_a, h_a, q_b, h_b, polys_a, polys_b, dhat):
    """Active-set list of (Jn_a (12,), Jn_b (12,), g_offset) for the contact
    pair (A, B).  Collects BOTH directions:

      - A vertex → B sub-cell:  Jn_a = +nᵀJ_A(ξ_A),  Jn_b = -nᵀJ_B(ξ_e^B)
      - B vertex → A sub-cell:  Jn_b = +nᵀJ_B(ξ_B),  Jn_a = -nᵀJ_A(ξ_e^A)

    Both packed into the same list with `g_offset = 0` (gap is purely
    q-linear in (q_a, q_b) once the normal is frozen, so there is no
    constant term).  Frozen normal per contact; the edge-origin corner
    vertex moves with q of its owning body, contributing the cross-block
    Hessian term that distinguishes the 2-body block from per-body GS.
    """
    out = []
    # Direction 1: A vertex → B sub-cell.
    pts_a = _outer_world_pts_from_q(q_a, h_a)
    for i in range(len(_OUTER_XI)):
        P = pts_a[i]
        J_A = point_jacobian_q_fq(_OUTER_XI[i], h_a)              # (2, 12)
        for sc_poly in polys_b:
            pick = _pick_active_edge(P, sc_poly, dhat)
            if pick is None:
                continue
            n_e, _V_e, xi_e = pick
            J_B = point_jacobian_q_fq(xi_e, h_b)                  # (2, 12)
            Jn_a =  J_A.T @ n_e                                    # (12,)
            Jn_b = -J_B.T @ n_e                                    # (12,)
            out.append((Jn_a, Jn_b, 0.0))
    # Direction 2: B vertex → A sub-cell.
    pts_b = _outer_world_pts_from_q(q_b, h_b)
    for i in range(len(_OUTER_XI)):
        P = pts_b[i]
        J_B = point_jacobian_q_fq(_OUTER_XI[i], h_b)              # (2, 12)
        for sc_poly in polys_a:
            pick = _pick_active_edge(P, sc_poly, dhat)
            if pick is None:
                continue
            n_e, _V_e, xi_e = pick
            J_A = point_jacobian_q_fq(xi_e, h_a)                  # (2, 12)
            Jn_b =  J_B.T @ n_e                                    # (12,)
            Jn_a = -J_A.T @ n_e                                    # (12,)
            out.append((Jn_a, Jn_b, 0.0))
    return out


def _collect_incident_constraints_fq(body_idx, state):
    """Per-body view of all incident JointFQ constraints.

    Returns a list of (J_q, target) where J_q is the (2, 12) self-Jacobian
    and target is the world position of the OTHER body's matched point at
    its *current* state (held fixed during this body's solve).
    """
    bodies = state.bodies
    self_b = bodies[body_idx]
    h = self_b.h
    out = []
    for j in state.joints:
        if not isinstance(j, JointFQ):
            continue
        if j.body_a_idx == body_idx:
            other = bodies[j.body_b_idx]
            J_q = point_jacobian_q_fq(j.xi_a, h)
            target = point_position_fq(other, j.xi_b)
        elif j.body_b_idx == body_idx:
            other = bodies[j.body_a_idx]
            J_q = point_jacobian_q_fq(j.xi_b, h)
            target = point_position_fq(other, j.xi_a)
        else:
            continue
        out.append((J_q, target.copy()))
    return out


def vbd_body_step_fq(body, body_idx, state, dt, gravity, alpha,
                     max_newton=5, ls_max=20,
                     barrier_kappa_floor=0.0, y_floor=None,
                     barrier_kappa_body=0.0, barrier_dhat=0.05):
    """One VBD visit to a single FQ2D body — Newton + Armijo on its 12-D IP.

    Minimises (over v = (vc, vF, vG, vHx, vHy)):

        IP(v) = ½(v - v_pre)ᵀ M (v - v_pre)
              − M_c · g · (c_pre + dt·vc)                       # gravity (c only)
              + ∫ Ψ(J(F_t, G_t, Hx_t, Hy_t; ξ)) dX              # elastic (full basis)
              + (α/2) Σ_points ‖J_q (q_pre + dt v) − target‖²   # joints
              + Σ_floor_contacts b(g; dhat, κ_floor)             # frozen-normal IPC barrier
              + Σ_body_contacts  b(g; dhat, κ_body)              # frozen-normal IPC barrier

    Floor and body-body contact use a frozen-normal IPC log-barrier.  The
    active set and contact normals are determined ONCE at body-visit entry
    (from the predicted iterate q_pre + dt·v_pre and the other bodies'
    current world state); inside the Newton loop the gap g(q) = Jn·q +
    g_offset is exactly q-linear, so the line-search feasibility clamp
    (α_max) reduces to one dot product per active contact.  See the
    `_barrier_*` and `_*_active_contacts` helpers above.
    """
    if body.static:
        return

    M = body.mass_vec.copy()           # (12,)
    M_c = body.mass
    mu_l, lam_l = body.lame
    h = body.h

    q_pre = np.concatenate([body._c_pre, body._F_pre, body._G_pre,
                            body._Hx_pre, body._Hy_pre])
    v_pre = np.concatenate([body._vc_pre, body._vF_pre, body._vG_pre,
                            body._vHx_pre, body._vHy_pre])

    incident = _collect_incident_constraints_fq(body_idx, state)
    floor_active = barrier_kappa_floor > 0.0 and y_floor is not None
    body_active  = barrier_kappa_body > 0.0 and len(state.bodies) > 1

    # Snapshot every other body's collision geometry once (held fixed for
    # the duration of this body's solve — owner-vertex GS pattern).
    other_polys_list = []
    if body_active:
        for j, other in enumerate(state.bodies):
            if j == body_idx:
                continue
            other_polys_list.append(_subcell_polygons_world(other))
        if not other_polys_list:
            body_active = False

    # Build the FROZEN-NORMAL active-set lists once, from the predicted
    # iterate q_init = q_pre + dt·v_pre (catches fast-moving vertices that
    # would penetrate this step).  Each entry is (Jn (12,), g_offset)
    # encoding g(q) = Jn·q + g_offset for the chosen frozen normal.
    q_init = q_pre + dt * v_pre
    contacts_floor = (_floor_active_contacts(q_init, h, y_floor, barrier_dhat)
                      if floor_active else [])
    contacts_body  = (_body_body_active_contacts(q_init, h, other_polys_list,
                                                 barrier_dhat)
                      if body_active else [])

    def ip_energy(v):
        q = q_pre + dt * v
        c_t  = q[:2]
        F_t  = q[2:6]
        G_t  = q[6:8]
        Hx_t = q[8:10]
        Hy_t = q[10:12]
        dv = v - v_pre
        E = 0.5 * float(np.dot(dv * M, dv))
        E -= M_c * float(np.dot(gravity, c_t))
        E += energy.integrated_energy_full(F_t, G_t, Hx_t, Hy_t, mu_l, lam_l, h)
        for J_q, target in incident:
            r = J_q @ q - target
            E += 0.5 * alpha * float(np.dot(r, r))
        for Jn, g_offset in contacts_floor:
            g_cur = float(np.dot(Jn, q) + g_offset)
            E += _barrier_value(g_cur, barrier_dhat, barrier_kappa_floor)
        for Jn, g_offset in contacts_body:
            g_cur = float(np.dot(Jn, q) + g_offset)
            E += _barrier_value(g_cur, barrier_dhat, barrier_kappa_body)
        return E

    v = v_pre.copy()

    for _ in range(max_newton):
        q = q_pre + dt * v
        F_t  = q[2:6]
        G_t  = q[6:8]
        Hx_t = q[8:10]
        Hy_t = q[10:12]

        # Gradient w.r.t. v  (12,)
        grad = M * (v - v_pre)
        grad[:2] -= dt * M_c * gravity
        g_el = energy.integrated_grad_full(F_t, G_t, Hx_t, Hy_t, mu_l, lam_l, h)  # (10,)
        grad[2:] += dt * g_el
        for J_q, target in incident:
            r = J_q @ q - target
            grad += alpha * dt * (J_q.T @ r)

        # SPD Hessian (12×12)
        H = np.diag(M)
        H_el = energy.integrated_hessian_spd_full(F_t, G_t, Hx_t, Hy_t,
                                                   mu_l, lam_l, h)            # (10,10)
        H[2:, 2:] += dt * dt * H_el
        for J_q, _ in incident:
            H += alpha * dt * dt * (J_q.T @ J_q)

        # Barrier contacts (frozen normal; g(q) = Jn·q + g_offset is linear).
        # ∇_q b = b'(g)·Jn → ∇_v b = dt·b'(g)·Jn.
        # ∇²_q b = b''(g)·Jn·Jnᵀ → ∇²_v b = dt²·b''(g)·Jn·Jnᵀ.
        for Jn, g_offset in contacts_floor:
            g_cur = float(np.dot(Jn, q) + g_offset)
            bp  = _barrier_grad(g_cur, barrier_dhat, barrier_kappa_floor)
            bpp = _barrier_hess(g_cur, barrier_dhat, barrier_kappa_floor)
            grad += dt * bp * Jn
            H    += dt * dt * bpp * np.outer(Jn, Jn)
        for Jn, g_offset in contacts_body:
            g_cur = float(np.dot(Jn, q) + g_offset)
            bp  = _barrier_grad(g_cur, barrier_dhat, barrier_kappa_body)
            bpp = _barrier_hess(g_cur, barrier_dhat, barrier_kappa_body)
            grad += dt * bp * Jn
            H    += dt * dt * bpp * np.outer(Jn, Jn)

        try:
            dv = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            break

        # α_max feasibility clamp (frozen-normal closed-form CCD).
        # In v-space, g(v + a·dv) = g(v) + a·dt·(Jn·dv).  If the step closes
        # the gap (Jn·dv < 0) and we're currently feasible (g_cur > 0), bound
        # a so the new gap stays ≥ ½·g_cur (standard IPC half-step).  When
        # g_cur ≤ 0 we're in the quadratic continuation — no singularity, no
        # clamp needed; the barrier itself drives healing.
        alpha_max = 1.0
        for Jn, _g_offset in (contacts_floor + contacts_body):
            g_cur = float(np.dot(Jn, q) + _g_offset)
            jdv   = float(dt * np.dot(Jn, dv))
            if g_cur > 0.0 and jdv < -1e-15:
                a_i = 0.5 * g_cur / (-jdv)
                if a_i < alpha_max:
                    alpha_max = a_i

        # Armijo line search (start from alpha_max so initial trial is feasible).
        E_cur = ip_energy(v)
        directional = float(np.dot(grad, dv))
        a = alpha_max
        for _ in range(ls_max):
            v_trial = v + a * dv
            if ip_energy(v_trial) <= E_cur + 1e-4 * a * directional:
                break
            a *= 0.5

        v = v + a * dv
        if np.max(np.abs(a * dv)) < 1e-12:
            break

    body.vc  = v[:2]
    body.vF  = v[2:6]
    body.vG  = v[6:8]
    body.vHx = v[8:10]
    body.vHy = v[10:12]
    body.c   = q_pre[:2]    + dt * body.vc
    body.F   = q_pre[2:6]   + dt * body.vF
    body.G   = q_pre[6:8]   + dt * body.vG
    body.Hx  = q_pre[8:10]  + dt * body.vHx
    body.Hy  = q_pre[10:12] + dt * body.vHy


def vbd_edge_step_fq(body_a, body_b, idx_a, idx_b, state, contacts_pair,
                     dt, gravity, alpha,
                     max_newton=5, ls_max=20,
                     barrier_kappa_floor=0.0, y_floor=None,
                     barrier_kappa_body=0.0, barrier_dhat=0.05):
    """One VBD visit to a contact PAIR of FQ2D bodies — Newton + Armijo
    on the joint 24-D inertial potential.

    Stage-2 generalisation of `vbd_body_step_fq`: instead of treating
    body B as frozen geometry while A solves (owner-vertex GS), both
    bodies' DoFs are stacked into a single 24-D Newton block and the
    body-body barrier between A and B contributes a CROSS-coupled
    rank-1 Hessian update per active contact.

    Energy minimised over (v_a, v_b) ∈ R²⁴:

        IP = E_A_solo(v_a) + E_B_solo(v_b)
           + Σ_pair_contacts b(g(q_a, q_b); dhat, κ_body)

    where `E_X_solo` is the per-body Stage-1 energy (inertia + gravity
    + elastic + joints + floor barrier + non-partner body contacts;
    these are block-diagonal in the 24×24 Hessian) and the pair-
    contact term is the only off-diagonal coupling.  Frozen normal per
    contact (snapshot at substep start), so g is q-linear in BOTH q_a
    and q_b: g = Jn_a · q_a + Jn_b · q_b (no constant term).

    Joints across the pair (A↔B) keep frozen-target behaviour from
    `_collect_incident_constraints_fq` — acknowledged simplification;
    no test scenario exercises an A-B joint together with body-body
    contact.

    `contacts_pair` is the precomputed list of (Jn_a, Jn_b, g_offset)
    triples from `_body_body_active_pair`, snapshotted from
    q_pre + dt·v_pre at the substep boundary.  Non-partner body-body
    contacts (A vs other dynamic bodies != B; B vs other dynamic
    bodies != A) are gathered here as 12-D Stage-1-style entries and
    contribute only to the corresponding diagonal block.
    """
    if body_a.static and body_b.static:
        return
    # Both bodies must be non-static for the 24-D coupled solve.  If
    # exactly one is static, we can still run with that body's DoF
    # block frozen out (mass→∞ effectively).  Easiest: route to the
    # single-body solver for the non-static side and let the contact
    # be handled via the per-body builder against the frozen partner.
    if body_a.static:
        polys_a = _subcell_polygons_world(body_a)
        # B's contacts vs A become per-body floor-style entries; reuse
        # the per-body solver path (it already handles "other bodies'
        # frozen polys" via `_body_body_active_contacts`).
        vbd_body_step_fq(body_b, idx_b, state, dt, gravity, alpha,
                         max_newton=max_newton, ls_max=ls_max,
                         barrier_kappa_floor=barrier_kappa_floor,
                         y_floor=y_floor,
                         barrier_kappa_body=barrier_kappa_body,
                         barrier_dhat=barrier_dhat)
        return
    if body_b.static:
        vbd_body_step_fq(body_a, idx_a, state, dt, gravity, alpha,
                         max_newton=max_newton, ls_max=ls_max,
                         barrier_kappa_floor=barrier_kappa_floor,
                         y_floor=y_floor,
                         barrier_kappa_body=barrier_kappa_body,
                         barrier_dhat=barrier_dhat)
        return

    h_a = body_a.h
    h_b = body_b.h
    M_a = body_a.mass_vec.copy()
    M_b = body_b.mass_vec.copy()
    Mc_a = body_a.mass
    Mc_b = body_b.mass
    mu_a, lam_a = body_a.lame
    mu_b, lam_b = body_b.lame

    q_pre_a = np.concatenate([body_a._c_pre, body_a._F_pre, body_a._G_pre,
                              body_a._Hx_pre, body_a._Hy_pre])
    v_pre_a = np.concatenate([body_a._vc_pre, body_a._vF_pre, body_a._vG_pre,
                              body_a._vHx_pre, body_a._vHy_pre])
    q_pre_b = np.concatenate([body_b._c_pre, body_b._F_pre, body_b._G_pre,
                              body_b._Hx_pre, body_b._Hy_pre])
    v_pre_b = np.concatenate([body_b._vc_pre, body_b._vF_pre, body_b._vG_pre,
                              body_b._vHx_pre, body_b._vHy_pre])

    incident_a = _collect_incident_constraints_fq(idx_a, state)
    incident_b = _collect_incident_constraints_fq(idx_b, state)

    floor_active = barrier_kappa_floor > 0.0 and y_floor is not None
    body_active  = barrier_kappa_body > 0.0

    # Snapshot non-partner geometry once for each body (frozen for the
    # whole edge solve — same per-visit freeze rule as Stage 1).
    other_polys_a = []
    other_polys_b = []
    if body_active:
        for j, other in enumerate(state.bodies):
            if j == idx_a or j == idx_b:
                continue
            other_polys_a.append(_subcell_polygons_world(other))
            other_polys_b.append(_subcell_polygons_world(other))

    # Active sets that depend only on a single body's q.  These are
    # block-diagonal contributions; the cross-coupled `contacts_pair`
    # is the only term that bridges the two 12-D blocks.
    q_init_a = q_pre_a + dt * v_pre_a
    q_init_b = q_pre_b + dt * v_pre_b
    contacts_floor_a = (_floor_active_contacts(q_init_a, h_a, y_floor, barrier_dhat)
                        if floor_active else [])
    contacts_floor_b = (_floor_active_contacts(q_init_b, h_b, y_floor, barrier_dhat)
                        if floor_active else [])
    contacts_other_a = (_body_body_active_contacts(q_init_a, h_a, other_polys_a, barrier_dhat)
                        if body_active and other_polys_a else [])
    contacts_other_b = (_body_body_active_contacts(q_init_b, h_b, other_polys_b, barrier_dhat)
                        if body_active and other_polys_b else [])

    def ip_energy(v):
        v_a = v[:12]
        v_b = v[12:]
        q_a = q_pre_a + dt * v_a
        q_b = q_pre_b + dt * v_b
        c_a, F_a, G_a, Hx_a, Hy_a = q_a[:2], q_a[2:6], q_a[6:8], q_a[8:10], q_a[10:12]
        c_b, F_b, G_b, Hx_b, Hy_b = q_b[:2], q_b[2:6], q_b[6:8], q_b[8:10], q_b[10:12]
        dv_a = v_a - v_pre_a
        dv_b = v_b - v_pre_b
        E = 0.5 * float(np.dot(dv_a * M_a, dv_a))
        E += 0.5 * float(np.dot(dv_b * M_b, dv_b))
        E -= Mc_a * float(np.dot(gravity, c_a))
        E -= Mc_b * float(np.dot(gravity, c_b))
        E += energy.integrated_energy_full(F_a, G_a, Hx_a, Hy_a, mu_a, lam_a, h_a)
        E += energy.integrated_energy_full(F_b, G_b, Hx_b, Hy_b, mu_b, lam_b, h_b)
        for J_q, target in incident_a:
            r = J_q @ q_a - target
            E += 0.5 * alpha * float(np.dot(r, r))
        for J_q, target in incident_b:
            r = J_q @ q_b - target
            E += 0.5 * alpha * float(np.dot(r, r))
        for Jn, g_offset in contacts_floor_a:
            E += _barrier_value(float(np.dot(Jn, q_a) + g_offset),
                                barrier_dhat, barrier_kappa_floor)
        for Jn, g_offset in contacts_floor_b:
            E += _barrier_value(float(np.dot(Jn, q_b) + g_offset),
                                barrier_dhat, barrier_kappa_floor)
        for Jn, g_offset in contacts_other_a:
            E += _barrier_value(float(np.dot(Jn, q_a) + g_offset),
                                barrier_dhat, barrier_kappa_body)
        for Jn, g_offset in contacts_other_b:
            E += _barrier_value(float(np.dot(Jn, q_b) + g_offset),
                                barrier_dhat, barrier_kappa_body)
        for Jn_a, Jn_b, g_offset in contacts_pair:
            g_cur = float(np.dot(Jn_a, q_a) + np.dot(Jn_b, q_b) + g_offset)
            E += _barrier_value(g_cur, barrier_dhat, barrier_kappa_body)
        return E

    v = np.concatenate([v_pre_a, v_pre_b])

    for _ in range(max_newton):
        v_a = v[:12]
        v_b = v[12:]
        q_a = q_pre_a + dt * v_a
        q_b = q_pre_b + dt * v_b
        F_a, G_a, Hx_a, Hy_a = q_a[2:6], q_a[6:8], q_a[8:10], q_a[10:12]
        F_b, G_b, Hx_b, Hy_b = q_b[2:6], q_b[6:8], q_b[8:10], q_b[10:12]

        # Block-diagonal solo gradient for each body.
        grad_a = M_a * (v_a - v_pre_a)
        grad_a[:2] -= dt * Mc_a * gravity
        grad_a[2:] += dt * energy.integrated_grad_full(F_a, G_a, Hx_a, Hy_a, mu_a, lam_a, h_a)
        for J_q, target in incident_a:
            grad_a += alpha * dt * (J_q.T @ (J_q @ q_a - target))

        grad_b = M_b * (v_b - v_pre_b)
        grad_b[:2] -= dt * Mc_b * gravity
        grad_b[2:] += dt * energy.integrated_grad_full(F_b, G_b, Hx_b, Hy_b, mu_b, lam_b, h_b)
        for J_q, target in incident_b:
            grad_b += alpha * dt * (J_q.T @ (J_q @ q_b - target))

        # Block-diagonal solo Hessian for each body.
        H_aa = np.diag(M_a)
        H_aa[2:, 2:] += dt * dt * energy.integrated_hessian_spd_full(
            F_a, G_a, Hx_a, Hy_a, mu_a, lam_a, h_a)
        for J_q, _ in incident_a:
            H_aa += alpha * dt * dt * (J_q.T @ J_q)

        H_bb = np.diag(M_b)
        H_bb[2:, 2:] += dt * dt * energy.integrated_hessian_spd_full(
            F_b, G_b, Hx_b, Hy_b, mu_b, lam_b, h_b)
        for J_q, _ in incident_b:
            H_bb += alpha * dt * dt * (J_q.T @ J_q)

        # Floor barrier on each body — diagonal block contributions.
        for Jn, g_offset in contacts_floor_a:
            g_cur = float(np.dot(Jn, q_a) + g_offset)
            bp  = _barrier_grad(g_cur, barrier_dhat, barrier_kappa_floor)
            bpp = _barrier_hess(g_cur, barrier_dhat, barrier_kappa_floor)
            grad_a += dt * bp * Jn
            H_aa   += dt * dt * bpp * np.outer(Jn, Jn)
        for Jn, g_offset in contacts_floor_b:
            g_cur = float(np.dot(Jn, q_b) + g_offset)
            bp  = _barrier_grad(g_cur, barrier_dhat, barrier_kappa_floor)
            bpp = _barrier_hess(g_cur, barrier_dhat, barrier_kappa_floor)
            grad_b += dt * bp * Jn
            H_bb   += dt * dt * bpp * np.outer(Jn, Jn)

        # Non-partner body-body contacts on each body (frozen partner;
        # diagonal-block contributions only).
        for Jn, g_offset in contacts_other_a:
            g_cur = float(np.dot(Jn, q_a) + g_offset)
            bp  = _barrier_grad(g_cur, barrier_dhat, barrier_kappa_body)
            bpp = _barrier_hess(g_cur, barrier_dhat, barrier_kappa_body)
            grad_a += dt * bp * Jn
            H_aa   += dt * dt * bpp * np.outer(Jn, Jn)
        for Jn, g_offset in contacts_other_b:
            g_cur = float(np.dot(Jn, q_b) + g_offset)
            bp  = _barrier_grad(g_cur, barrier_dhat, barrier_kappa_body)
            bpp = _barrier_hess(g_cur, barrier_dhat, barrier_kappa_body)
            grad_b += dt * bp * Jn
            H_bb   += dt * dt * bpp * np.outer(Jn, Jn)

        # Cross-coupled pair contacts: contribute to ALL FOUR sub-blocks.
        H_ab = np.zeros((12, 12))
        for Jn_a, Jn_b, g_offset in contacts_pair:
            g_cur = float(np.dot(Jn_a, q_a) + np.dot(Jn_b, q_b) + g_offset)
            bp  = _barrier_grad(g_cur, barrier_dhat, barrier_kappa_body)
            bpp = _barrier_hess(g_cur, barrier_dhat, barrier_kappa_body)
            grad_a += dt * bp * Jn_a
            grad_b += dt * bp * Jn_b
            H_aa   += dt * dt * bpp * np.outer(Jn_a, Jn_a)
            H_bb   += dt * dt * bpp * np.outer(Jn_b, Jn_b)
            H_ab   += dt * dt * bpp * np.outer(Jn_a, Jn_b)

        # Assemble 24×24 system.
        H = np.zeros((24, 24))
        H[:12, :12] = H_aa
        H[12:, 12:] = H_bb
        H[:12, 12:] = H_ab
        H[12:, :12] = H_ab.T
        grad = np.concatenate([grad_a, grad_b])

        try:
            dv = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            break

        dv_a = dv[:12]
        dv_b = dv[12:]

        # α_max feasibility clamp across all active contact families.
        alpha_max = 1.0
        for Jn, _g_offset in contacts_floor_a:
            g_cur = float(np.dot(Jn, q_a) + _g_offset)
            jdv   = float(dt * np.dot(Jn, dv_a))
            if g_cur > 0.0 and jdv < -1e-15:
                a_i = 0.5 * g_cur / (-jdv)
                if a_i < alpha_max:
                    alpha_max = a_i
        for Jn, _g_offset in contacts_floor_b:
            g_cur = float(np.dot(Jn, q_b) + _g_offset)
            jdv   = float(dt * np.dot(Jn, dv_b))
            if g_cur > 0.0 and jdv < -1e-15:
                a_i = 0.5 * g_cur / (-jdv)
                if a_i < alpha_max:
                    alpha_max = a_i
        for Jn, _g_offset in contacts_other_a:
            g_cur = float(np.dot(Jn, q_a) + _g_offset)
            jdv   = float(dt * np.dot(Jn, dv_a))
            if g_cur > 0.0 and jdv < -1e-15:
                a_i = 0.5 * g_cur / (-jdv)
                if a_i < alpha_max:
                    alpha_max = a_i
        for Jn, _g_offset in contacts_other_b:
            g_cur = float(np.dot(Jn, q_b) + _g_offset)
            jdv   = float(dt * np.dot(Jn, dv_b))
            if g_cur > 0.0 and jdv < -1e-15:
                a_i = 0.5 * g_cur / (-jdv)
                if a_i < alpha_max:
                    alpha_max = a_i
        for Jn_a, Jn_b, _g_offset in contacts_pair:
            g_cur = float(np.dot(Jn_a, q_a) + np.dot(Jn_b, q_b) + _g_offset)
            jdv   = float(dt * (np.dot(Jn_a, dv_a) + np.dot(Jn_b, dv_b)))
            if g_cur > 0.0 and jdv < -1e-15:
                a_i = 0.5 * g_cur / (-jdv)
                if a_i < alpha_max:
                    alpha_max = a_i

        E_cur = ip_energy(v)
        directional = float(np.dot(grad, dv))
        a = alpha_max
        for _ in range(ls_max):
            if ip_energy(v + a * dv) <= E_cur + 1e-4 * a * directional:
                break
            a *= 0.5

        v = v + a * dv
        if np.max(np.abs(a * dv)) < 1e-12:
            break

    v_a = v[:12]
    v_b = v[12:]
    body_a.vc, body_a.vF, body_a.vG = v_a[:2], v_a[2:6], v_a[6:8]
    body_a.vHx, body_a.vHy         = v_a[8:10], v_a[10:12]
    body_a.c   = q_pre_a[:2]    + dt * body_a.vc
    body_a.F   = q_pre_a[2:6]   + dt * body_a.vF
    body_a.G   = q_pre_a[6:8]   + dt * body_a.vG
    body_a.Hx  = q_pre_a[8:10]  + dt * body_a.vHx
    body_a.Hy  = q_pre_a[10:12] + dt * body_a.vHy
    body_b.vc, body_b.vF, body_b.vG = v_b[:2], v_b[2:6], v_b[6:8]
    body_b.vHx, body_b.vHy         = v_b[8:10], v_b[10:12]
    body_b.c   = q_pre_b[:2]    + dt * body_b.vc
    body_b.F   = q_pre_b[2:6]   + dt * body_b.vF
    body_b.G   = q_pre_b[6:8]   + dt * body_b.vG
    body_b.Hx  = q_pre_b[8:10]  + dt * body_b.vHx
    body_b.Hy  = q_pre_b[10:12] + dt * body_b.vHy


def _active_edge_graph(state, dt, barrier_dhat, barrier_kappa_body):
    """Build the per-substep contact graph from q_pre + dt·v_pre.

    Returns (edges, isolated_indices):
      edges            = list of (i, j, contacts_pair) with i < j and
                          at least one active contact between bodies i and j
      isolated_indices = sorted list of dynamic body indices not in any edge

    Static bodies are excluded from `isolated_indices` (they have no DoF
    to update); they CAN appear in edges (an edge with a static body is
    routed back to per-body Stage-1 inside `vbd_edge_step_fq`).
    """
    n = len(state.bodies)
    if barrier_kappa_body <= 0.0 or n < 2:
        return [], [i for i, b in enumerate(state.bodies) if not b.static]

    # Predicted iterates (use _pre snapshots, NOT current state — the
    # graph is fixed for the whole substep).
    q_init = []
    polys  = []
    hs     = []
    for b in state.bodies:
        if b.static:
            q = np.concatenate([b.c, b.F, b.G, b.Hx, b.Hy])
        else:
            q_pre = np.concatenate([b._c_pre, b._F_pre, b._G_pre, b._Hx_pre, b._Hy_pre])
            v_pre = np.concatenate([b._vc_pre, b._vF_pre, b._vG_pre, b._vHx_pre, b._vHy_pre])
            q = q_pre + dt * v_pre
        q_init.append(q)
        polys.append(_subcell_polygons_world(b))
        hs.append(b.h)

    edges = []
    in_edge = [False] * n
    for i in range(n):
        if state.bodies[i].static:
            continue
        for j in range(i + 1, n):
            # Skip pairs of two statics (no DoF at all).
            if state.bodies[i].static and state.bodies[j].static:
                continue
            contacts = _body_body_active_pair(q_init[i], hs[i], q_init[j], hs[j],
                                              polys[i], polys[j], barrier_dhat)
            if contacts:
                edges.append((i, j, contacts))
                in_edge[i] = True
                in_edge[j] = True

    isolated = [i for i, b in enumerate(state.bodies)
                if (not b.static) and (not in_edge[i])]
    return edges, isolated


def step_vbd_fq(state, params, alpha, barrier_kappa_floor=0.0, y_floor=None,
                barrier_kappa_body=0.0, barrier_dhat=0.05):
    """Pure-VBD step for FQ2D bodies: GS sweeps with frozen-normal IPC barrier
    contact.

    Floor contact is enabled when both `barrier_kappa_floor > 0` and
    `y_floor is not None`.  Body-body contact is enabled when
    `barrier_kappa_body > 0`.

    Both contacts share a single `barrier_dhat` (activation distance), with
    standard IPC defaults: dhat ≈ 5–10% of feature size, κ_b ≈ 1.  Inside
    each per-body Newton solve the active set + contact normals are frozen;
    the line search clamps step length to the barrier feasibility cone
    before Armijo backtracking, so the iterate never crosses the barrier.

    Two block modes (selected by `params.block_mode`):

      - "per_body" (default, Stage 1) — owner-vertex GS, 12-D Newton per
        body visit.  Reciprocity comes from the other body's own visit
        in the sweep.
      - "per_edge" (Stage 2) — 24-D Newton per active contact pair, with
        the body-body barrier contributing a cross-coupled rank-1
        Hessian update.  Bodies not present in any active edge fall back
        to per-body Stage-1 sweep.
    """
    for b in state.bodies:
        if not b.static:
            b._c_pre   = b.c.copy()
            b._F_pre   = b.F.copy()
            b._G_pre   = b.G.copy()
            b._Hx_pre  = b.Hx.copy()
            b._Hy_pre  = b.Hy.copy()
            b._vc_pre  = b.vc.copy()
            b._vF_pre  = b.vF.copy()
            b._vG_pre  = b.vG.copy()
            b._vHx_pre = b.vHx.copy()
            b._vHy_pre = b.vHy.copy()

    n = len(state.bodies)
    if getattr(params, "block_mode", "per_body") == "per_edge":
        edges, isolated = _active_edge_graph(state, params.dt, barrier_dhat,
                                             barrier_kappa_body)
        for _ in range(params.vbd_sweeps):
            # Forward sweep over edges, then over isolated bodies.
            for (i, j, contacts) in edges:
                vbd_edge_step_fq(state.bodies[i], state.bodies[j], i, j, state,
                                 contacts, params.dt, params.gravity, alpha,
                                 max_newton=params.vbd_newton,
                                 barrier_kappa_floor=barrier_kappa_floor,
                                 y_floor=y_floor,
                                 barrier_kappa_body=barrier_kappa_body,
                                 barrier_dhat=barrier_dhat)
            for i in isolated:
                vbd_body_step_fq(state.bodies[i], i, state, params.dt, params.gravity,
                                 alpha, max_newton=params.vbd_newton,
                                 barrier_kappa_floor=barrier_kappa_floor,
                                 y_floor=y_floor,
                                 barrier_kappa_body=barrier_kappa_body,
                                 barrier_dhat=barrier_dhat)
            # Backward sweep over edges, then over isolated bodies.
            for (i, j, contacts) in reversed(edges):
                vbd_edge_step_fq(state.bodies[i], state.bodies[j], i, j, state,
                                 contacts, params.dt, params.gravity, alpha,
                                 max_newton=params.vbd_newton,
                                 barrier_kappa_floor=barrier_kappa_floor,
                                 y_floor=y_floor,
                                 barrier_kappa_body=barrier_kappa_body,
                                 barrier_dhat=barrier_dhat)
            for i in reversed(isolated):
                vbd_body_step_fq(state.bodies[i], i, state, params.dt, params.gravity,
                                 alpha, max_newton=params.vbd_newton,
                                 barrier_kappa_floor=barrier_kappa_floor,
                                 y_floor=y_floor,
                                 barrier_kappa_body=barrier_kappa_body,
                                 barrier_dhat=barrier_dhat)
    else:
        for _ in range(params.vbd_sweeps):
            for i in range(n):
                vbd_body_step_fq(state.bodies[i], i, state, params.dt, params.gravity,
                                 alpha, max_newton=params.vbd_newton,
                                 barrier_kappa_floor=barrier_kappa_floor,
                                 y_floor=y_floor,
                                 barrier_kappa_body=barrier_kappa_body,
                                 barrier_dhat=barrier_dhat)
            for i in range(n - 1, -1, -1):
                vbd_body_step_fq(state.bodies[i], i, state, params.dt, params.gravity,
                                 alpha, max_newton=params.vbd_newton,
                                 barrier_kappa_floor=barrier_kappa_floor,
                                 y_floor=y_floor,
                                 barrier_kappa_body=barrier_kappa_body,
                                 barrier_dhat=barrier_dhat)

    state.time += params.dt
    state.step_count += 1
