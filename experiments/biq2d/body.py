"""2D bilinear-quadrilateral (Q4-analog) deformable body.

Per-body DoFs (8 total):
    c  ∈ ℝ²   centre of mass
    F  ∈ ℝ⁴  deformation gradient, flat [F11, F12, F21, F22]
    G  ∈ ℝ²  bilinear cross-coefficients (Gx, Gy)

Reference shape is ξ ∈ [-1, 1]² with half_extent h; world map is
    x(ξ) = c + h·F·ξ + h·G·(ξ₁ξ₂).

Mass matrix (derived by integrating ½ρ|ẋ|² over the physical body,
convention "B" — G dimensionless, X scaled by h):
    M_c = m · I₂
    M_F = (m·h²/3) · I₄
    M_G = (m·h²/9) · I₂

F and G DoFs are conjugated to the body polar moment m·h² — the factor
distinguishes shear/stretch (F) from the bilinear mode (G) by the
second vs fourth moment of ξ.
"""
import numpy as np
from dataclasses import dataclass, field
from . import energy


@dataclass
class BodyBQ2D:
    mass: float
    half_extent: float = 1.0
    k: float = 1000.0
    nu: float = 0.3

    static: bool = False

    c: np.ndarray = field(default_factory=lambda: np.zeros(2))
    F: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 1.0]))
    G: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vc: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vF: np.ndarray = field(default_factory=lambda: np.zeros(4))
    vG: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # Barrier strength and activation threshold for det J at corners.
    barrier_kappa: float = 0.0     # set > 0 to activate (scaled like elastic energy)
    barrier_eps:   float = 0.1

    def __post_init__(self):
        self.c  = np.array(self.c,  dtype=float)
        self.F  = np.array(self.F,  dtype=float)
        self.G  = np.array(self.G,  dtype=float)
        self.vc = np.array(self.vc, dtype=float)
        self.vF = np.array(self.vF, dtype=float)
        self.vG = np.array(self.vG, dtype=float)

    # ── geometry ───────────────────────────────────────────────────

    @property
    def h(self):
        return self.half_extent

    def corners(self):
        """(4, 2) world-space corners at ξ ∈ {(±1, ±1)}.  Order matches CORNERS."""
        h = self.h
        out = np.zeros((4, 2))
        for i, xi in enumerate(energy.CORNERS):
            x1, x2 = xi
            out[i, 0] = self.c[0] + h * (x1 * self.F[0] + x2 * self.F[1] + x1 * x2 * self.G[0])
            out[i, 1] = self.c[1] + h * (x1 * self.F[2] + x2 * self.F[3] + x1 * x2 * self.G[1])
        return out

    def min_det_J(self):
        """min det J across the 4 reference-square corners (det J is linear in ξ)."""
        return float(np.min(energy.det_J_at_corners(self.F, self.G)))

    # ── mass matrix (convention B) ─────────────────────────────────

    @property
    def _m_F_scalar(self):
        return self.mass * self.h * self.h / 3.0

    @property
    def _m_G_scalar(self):
        return self.mass * self.h * self.h / 9.0

    @property
    def mass_vec(self):
        """(8,) diagonal of full mass matrix for q = (c, F, G)."""
        mF = self._m_F_scalar
        mG = self._m_G_scalar
        return np.array([self.mass, self.mass, mF, mF, mF, mF, mG, mG])

    @property
    def inv_mass_vec(self):
        if self.static:
            return np.zeros(8)
        return 1.0 / self.mass_vec

    @property
    def mass_vec_FG(self):
        """(6,) diagonal mass for the (F, G) block used in the BE F+G solve."""
        mF = self._m_F_scalar
        mG = self._m_G_scalar
        return np.array([mF, mF, mF, mF, mG, mG])

    # ── material ───────────────────────────────────────────────────

    @property
    def lame(self):
        return energy.lame_from_k(self.k, self.nu)

    # ── state vectors ──────────────────────────────────────────────

    @property
    def q(self):
        return np.concatenate([self.c, self.F, self.G])

    @q.setter
    def q(self, val):
        self.c[:] = val[:2]
        self.F[:] = val[2:6]
        self.G[:] = val[6:]

    @property
    def v(self):
        return np.concatenate([self.vc, self.vF, self.vG])

    @v.setter
    def v(self, val):
        self.vc[:] = val[:2]
        self.vF[:] = val[2:6]
        self.vG[:] = val[6:]

    @property
    def q_FG(self):
        """6-vector (F, G)."""
        return np.concatenate([self.F, self.G])

    @property
    def v_FG(self):
        """6-vector (vF, vG)."""
        return np.concatenate([self.vF, self.vG])

    # ── energies ───────────────────────────────────────────────────

    def kinetic_energy(self):
        m = self.mass_vec
        v = self.v
        return 0.5 * float(np.dot(v * m, v))

    def potential_energy(self, gravity=np.array([0.0, -10.0])):
        return -self.mass * float(np.dot(gravity, self.c))

    def elastic_energy(self):
        mu_l, lam_l = self.lame
        return energy.integrated_energy(self.F, self.G, mu_l, lam_l, self.h)

    def barrier_energy(self):
        if self.barrier_kappa <= 0.0:
            return 0.0
        E, _, _ = energy.barrier_energy_grad_hess(
            self.F, self.G, self.barrier_kappa, self.barrier_eps)
        return E

    def total_energy(self, gravity=np.array([0.0, -10.0])):
        return (self.kinetic_energy() + self.potential_energy(gravity)
                + self.elastic_energy() + self.barrier_energy())

    def integrate_be(self, dt, gravity):
        integrate_backward_euler(self, dt, gravity)


def integrate_backward_euler(body, dt, gravity=np.array([0.0, -10.0]),
                              max_newton=10, ls_max=20):
    """One step of backward Euler on all 8 DoFs.

    COM: gravity is constant → exact Euler.
    (F, G):  Newton + Armijo line search on the IP

        IP(v) = ½ (v-v_pre)ᵀ M_FG (v-v_pre)
              + ∫ Ψ(J(F + dt vF, G + dt vG; ξ)) dX
              + barrier on det J at corners
    """
    if body.static:
        return

    # COM ------------------------------------------------------------
    body.vc = body.vc + dt * gravity
    body.c  = body.c  + dt * body.vc

    # (F, G) IP minimisation ----------------------------------------
    M = body.mass_vec_FG                  # (6,) diagonal
    mu_l, lam_l = body.lame
    h = body.h
    kappa = body.barrier_kappa
    b_eps = body.barrier_eps

    q0 = np.concatenate([body.F, body.G]).copy()     # (F, G) at step start
    v_pre = np.concatenate([body.vF, body.vG]).copy()

    def q_from_v(v):
        return q0 + dt * v

    def ip_energy(v):
        q = q_from_v(v)
        F_t, G_t = q[:4], q[4:]
        E_el = energy.integrated_energy(F_t, G_t, mu_l, lam_l, h)
        E_ba = 0.0
        if kappa > 0.0:
            E_ba, _, _ = energy.barrier_energy_grad_hess(F_t, G_t, kappa, b_eps)
        dv = v - v_pre
        return 0.5 * float(np.dot(dv * M, dv)) + E_el + E_ba

    v = v_pre.copy()

    for _ in range(max_newton):
        q = q_from_v(v)
        F_t, G_t = q[:4], q[4:]

        g_el = energy.integrated_grad(F_t, G_t, mu_l, lam_l, h)
        H_el = energy.integrated_hessian_spd(F_t, G_t, mu_l, lam_l, h)

        if kappa > 0.0:
            _, g_ba, H_ba = energy.barrier_energy_grad_hess(F_t, G_t, kappa, b_eps)
        else:
            g_ba = np.zeros(6)
            H_ba = np.zeros((6, 6))

        # IP gradient:  M(v - v_pre) + dt · (g_el + g_ba)
        residual = M * (v - v_pre) + dt * (g_el + g_ba)
        # IP Hessian:  diag(M) + dt² · (H_el + H_ba)
        A = np.diag(M) + dt * dt * (H_el + H_ba)

        try:
            dv = np.linalg.solve(A, -residual)
        except np.linalg.LinAlgError:
            break

        # Armijo line search
        E_cur = ip_energy(v)
        directional = float(np.dot(residual, dv))
        alpha = 1.0
        for _ in range(ls_max):
            v_trial = v + alpha * dv
            E_trial = ip_energy(v_trial)
            if E_trial <= E_cur + 1e-4 * alpha * directional:
                break
            alpha *= 0.5

        v = v + alpha * dv
        if np.max(np.abs(alpha * dv)) < 1e-12:
            break

    body.vF = v[:4]
    body.vG = v[4:]
    body.F  = q0[:4] + dt * body.vF
    body.G  = q0[4:] + dt * body.vG


# ═══════════════════════════════════════════════════════════════════════
# Full-quadratic body (12 DoFs): c, F, G, Hx, Hy
#
# Map uses mean-zero banana basis (φ_Hx = ξ₁² − 1/3, φ_Hy = ξ₂² − 1/3) so
# c stays the centre of mass and the mass matrix stays strictly diagonal.
# Jacobian ∂x/∂ξ is unchanged by the constant shift, so the elastic energy
# integral matches the standard ξ² basis exactly.
#
# Mass blocks (convention B):
#     M_c  = m · I_2
#     M_F  = (m·h²/3) · I_4
#     M_G  = (m·h²/9) · I_2
#     M_Hx = (m·h² · 4/45) · I_2
#     M_Hy = (m·h² · 4/45) · I_2
# ═══════════════════════════════════════════════════════════════════════


_HX_OFFSET = 1.0 / 3.0  # ⟨ξ₁²⟩ on [-1,1]², used by the mean-zero shift.
_HY_OFFSET = 1.0 / 3.0


@dataclass
class BodyFQ2D:
    mass: float
    half_extent: float = 1.0
    k: float = 1000.0
    nu: float = 0.3

    static: bool = False

    c:  np.ndarray = field(default_factory=lambda: np.zeros(2))
    F:  np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 1.0]))
    G:  np.ndarray = field(default_factory=lambda: np.zeros(2))
    Hx: np.ndarray = field(default_factory=lambda: np.zeros(2))
    Hy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vc:  np.ndarray = field(default_factory=lambda: np.zeros(2))
    vF:  np.ndarray = field(default_factory=lambda: np.zeros(4))
    vG:  np.ndarray = field(default_factory=lambda: np.zeros(2))
    vHx: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vHy: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self):
        self.c   = np.array(self.c,   dtype=float)
        self.F   = np.array(self.F,   dtype=float)
        self.G   = np.array(self.G,   dtype=float)
        self.Hx  = np.array(self.Hx,  dtype=float)
        self.Hy  = np.array(self.Hy,  dtype=float)
        self.vc  = np.array(self.vc,  dtype=float)
        self.vF  = np.array(self.vF,  dtype=float)
        self.vG  = np.array(self.vG,  dtype=float)
        self.vHx = np.array(self.vHx, dtype=float)
        self.vHy = np.array(self.vHy, dtype=float)

    @property
    def h(self):
        return self.half_extent

    # ── geometry (full-quadratic map with mean-zero banana shift) ──

    def map_points(self, xi_arr):
        """(N, 2) ξ points → (N, 2) world points via full-quadratic map."""
        xi = np.asarray(xi_arr, dtype=float)
        x1 = xi[:, 0]
        x2 = xi[:, 1]
        h = self.h
        phi_hx = x1 * x1 - _HX_OFFSET
        phi_hy = x2 * x2 - _HY_OFFSET
        wx = (self.c[0]
              + h * (self.F[0] * x1 + self.F[1] * x2
                     + self.G[0] * x1 * x2
                     + self.Hx[0] * phi_hx + self.Hy[0] * phi_hy))
        wy = (self.c[1]
              + h * (self.F[2] * x1 + self.F[3] * x2
                     + self.G[1] * x1 * x2
                     + self.Hx[1] * phi_hx + self.Hy[1] * phi_hy))
        return np.column_stack([wx, wy])

    def corners(self):
        """(4, 2) world-space corners at ξ ∈ {(±1, ±1)}.  Order matches CORNERS."""
        return self.map_points(energy.CORNERS)

    def sub_cell_corners(self, num_samples=16):
        """Return list of 4 closed polylines, one per sub-cell.

        Each sub-cell is a ξ-space rectangle bounded by {−1,0,+1} lines; its
        perimeter is sampled at `num_samples` points per edge (four edges).
        Under the full-quadratic map the parabolic edges become curved in
        world space.  The returned arrays close on themselves (first ==
        last) so they can be handed straight to matplotlib.patches.Polygon
        or plotted as polylines.
        """
        polys = []
        for (a, b), (c, d) in (((-1.0, 0.0), (-1.0, 0.0)),   # lower-left
                               (( 0.0, 1.0), (-1.0, 0.0)),   # lower-right
                               (( 0.0, 1.0), ( 0.0, 1.0)),   # upper-right
                               ((-1.0, 0.0), ( 0.0, 1.0))):  # upper-left
            t = np.linspace(0.0, 1.0, num_samples, endpoint=False)
            bot   = np.column_stack([a + (b - a) * t, np.full_like(t, c)])
            right = np.column_stack([np.full_like(t, b), c + (d - c) * t])
            top   = np.column_stack([b + (a - b) * t, np.full_like(t, d)])
            left  = np.column_stack([np.full_like(t, a), d + (c - d) * t])
            loop  = np.concatenate([bot, right, top, left], axis=0)
            # close the loop
            loop  = np.concatenate([loop, loop[:1]], axis=0)
            polys.append(self.map_points(loop))
        return polys

    def split_line_xi(self):
        """Return the two ξ-space split lines (ξ₁=0 and ξ₂=0) sampled densely,
        for drawing the internal sub-cell boundaries in world space."""
        t = np.linspace(-1.0, 1.0, 33)
        vert = np.column_stack([np.zeros_like(t), t])        # ξ₁ = 0
        horz = np.column_stack([t, np.zeros_like(t)])        # ξ₂ = 0
        return self.map_points(vert), self.map_points(horz)

    def sample_grid_3x3(self):
        """(9, 2) world positions at ξ ∈ {−1, 0, +1}² — the sub-cell corners."""
        return self.map_points(energy.SAMPLE_3X3)

    def min_det_J_sampled(self, grid_n=9):
        """Sampled min det J over an n×n ξ-grid."""
        return energy.min_det_J_sampled_full(self.F, self.G, self.Hx, self.Hy, grid_n)

    # ── mass matrix (all diagonal — mean-zero bananas decouple) ────

    @property
    def _m_F_scalar(self):
        return self.mass * self.h * self.h / 3.0

    @property
    def _m_G_scalar(self):
        return self.mass * self.h * self.h / 9.0

    @property
    def _m_H_scalar(self):
        # ∫ (ξ² − 1/3)² dξ / ∫ 1 dξ = (16/45) / 4 = 4/45
        return self.mass * self.h * self.h * 4.0 / 45.0

    @property
    def mass_vec(self):
        """(12,) diagonal of full mass matrix for q = (c, F, G, Hx, Hy)."""
        mF = self._m_F_scalar
        mG = self._m_G_scalar
        mH = self._m_H_scalar
        return np.array([self.mass, self.mass,
                         mF, mF, mF, mF,
                         mG, mG,
                         mH, mH,
                         mH, mH])

    @property
    def mass_vec_FGH(self):
        """(10,) diagonal mass for the (F, G, Hx, Hy) block used in the BE solve."""
        mF = self._m_F_scalar
        mG = self._m_G_scalar
        mH = self._m_H_scalar
        return np.array([mF, mF, mF, mF, mG, mG, mH, mH, mH, mH])

    # ── material ───────────────────────────────────────────────────

    @property
    def lame(self):
        return energy.lame_from_k(self.k, self.nu)

    # ── energies ───────────────────────────────────────────────────

    def kinetic_energy(self):
        v = np.concatenate([self.vc, self.vF, self.vG, self.vHx, self.vHy])
        return 0.5 * float(np.dot(v * self.mass_vec, v))

    def potential_energy(self, gravity=np.array([0.0, -10.0])):
        return -self.mass * float(np.dot(gravity, self.c))

    def elastic_energy(self):
        mu_l, lam_l = self.lame
        return energy.integrated_energy_full(
            self.F, self.G, self.Hx, self.Hy, mu_l, lam_l, self.h)

    def total_energy(self, gravity=np.array([0.0, -10.0])):
        return (self.kinetic_energy() + self.potential_energy(gravity)
                + self.elastic_energy())

    def integrate_be(self, dt, gravity):
        integrate_backward_euler_fq(self, dt, gravity)


def integrate_backward_euler_fq(body, dt, gravity=np.array([0.0, -10.0]),
                                max_newton=10, ls_max=20):
    """One step of backward Euler on all 12 DoFs of a BodyFQ2D.

    COM: gravity is constant → exact Euler.
    (F, G, Hx, Hy):  Newton + Armijo line search on the 10-D IP

        IP(v) = ½ (v-v_pre)ᵀ M_FGH (v-v_pre)
              + ∫ Ψ(J(F + dt vF, G + dt vG, Hx + dt vHx, Hy + dt vHy; ξ)) dX
    """
    if body.static:
        return

    # COM ------------------------------------------------------------
    body.vc = body.vc + dt * gravity
    body.c  = body.c  + dt * body.vc

    # (F, G, Hx, Hy) IP minimisation --------------------------------
    M = body.mass_vec_FGH                   # (10,) diagonal
    mu_l, lam_l = body.lame
    h = body.h

    q0 = np.concatenate([body.F, body.G, body.Hx, body.Hy]).copy()   # (10,)
    v_pre = np.concatenate([body.vF, body.vG, body.vHx, body.vHy]).copy()

    def unpack(q):
        return q[:4], q[4:6], q[6:8], q[8:10]

    def q_from_v(v):
        return q0 + dt * v

    def ip_energy(v):
        q = q_from_v(v)
        F_t, G_t, Hx_t, Hy_t = unpack(q)
        E_el = energy.integrated_energy_full(F_t, G_t, Hx_t, Hy_t, mu_l, lam_l, h)
        dv = v - v_pre
        return 0.5 * float(np.dot(dv * M, dv)) + E_el

    v = v_pre.copy()

    for _ in range(max_newton):
        q = q_from_v(v)
        F_t, G_t, Hx_t, Hy_t = unpack(q)

        g_el = energy.integrated_grad_full(F_t, G_t, Hx_t, Hy_t, mu_l, lam_l, h)
        H_el = energy.integrated_hessian_spd_full(F_t, G_t, Hx_t, Hy_t, mu_l, lam_l, h)

        residual = M * (v - v_pre) + dt * g_el
        A = np.diag(M) + dt * dt * H_el

        try:
            dv = np.linalg.solve(A, -residual)
        except np.linalg.LinAlgError:
            break

        E_cur = ip_energy(v)
        directional = float(np.dot(residual, dv))
        alpha = 1.0
        for _ in range(ls_max):
            v_trial = v + alpha * dv
            E_trial = ip_energy(v_trial)
            if E_trial <= E_cur + 1e-4 * alpha * directional:
                break
            alpha *= 0.5

        v = v + alpha * dv
        if np.max(np.abs(alpha * dv)) < 1e-12:
            break

    body.vF  = v[0:4]
    body.vG  = v[4:6]
    body.vHx = v[6:8]
    body.vHy = v[8:10]
    body.F   = q0[0:4] + dt * body.vF
    body.G   = q0[4:6] + dt * body.vG
    body.Hx  = q0[6:8] + dt * body.vHx
    body.Hy  = q0[8:10] + dt * body.vHy
