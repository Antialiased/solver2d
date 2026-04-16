"""
1D sequential-impulses prototype solver for a stack of deformable bodies.

Each body has DoFs (x, s) — centre position and half-extent — with an
internal elastic potential 0.5 k (s - s_rest)^2 (plus an optional cubic
beta_3 term and optional piecewise tension/compression asymmetry).

The solver supports four integrator modes on the internal (s, vs) DoFs:

    local_implicit         backward Euler on the internal oscillator;
                           contact sees apos_s from BE.
    exponential            harmonic-oscillator exact solution on (s, vs);
                           energy-preserving, non-damping.
    naive                  rigid-body contact Jacobian (s decoupled from
                           contact); internal s still BE-integrated.
    local_implicit_relin   like local_implicit, but re-solves the per-body
                           nonlinear backward-Euler cubic on every contact
                           impulse update so the tangent is always fresh.

Contact supports:
    - hard unilateral PGS (default)
    - soft contact parametrised by (contact_wn, contact_zeta)
    - hard contact with a one-shot velocity-level restitution impulse
    - plasticity on the s-DoF (linear isotropic hardening)

This module contains only the physics. Tests and animations live in the
sibling `tests_*.py` files and are dispatched from `run.py`.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Parameters & state
# ---------------------------------------------------------------------------

@dataclass
class Params:
    N: int = 5
    m: float = 1.0
    s0: float = 0.5
    k: float = 1.0e3
    g: float = 9.81
    dt: float = 1.0e-3
    steps: int = 4000
    max_sweeps: int = 50
    tol: float = 1.0e-8
    eps_reg: float = 1.0e-12
    mu_frac: float = 1.0 / 3.0  # mu = m * mu_frac ("affine inertia" on s)
    damping: float = 1.0        # linear velocity damping (1/s), drains kinetic energy so the stack settles
    floor: bool = True          # enable floor contact at x=0 for body 0
    warm_start: bool = True     # warm-start contact multipliers from previous step
    sigma_Y: float = float("inf")  # initial yield stress on elastic s-DoF
    H_hard: float = 0.0            # linear isotropic hardening modulus (0 = perfect plasticity)
    beta3: float = 0.0             # cubic-force coefficient: F_el = -(k u + beta3 u^3), W += (beta3/4) u^4
    beta_comp: Optional[float] = None  # piecewise cubic coeff for u<0 (compression). None => use beta3.
    beta_tens: Optional[float] = None  # piecewise cubic coeff for u>0 (tension). None => use beta3.
    contact_wn: Optional[float] = None  # soft-contact natural frequency (rad/s). None => hard.
    contact_zeta: float = 1.0           # soft-contact damping ratio (only used when contact_wn set)
    restitution: float = 0.0            # hard-contact coefficient of restitution (0=plastic, 1=elastic)
    k_vec: Optional[np.ndarray] = None  # per-body stiffness override (shape (N,))
    m_vec: Optional[np.ndarray] = None  # per-body mass override (shape (N,))
    g_vec: Optional[np.ndarray] = None  # per-body gravity override (shape (N,)); None => scalar g

    @property
    def mu(self) -> float:
        return self.m * self.mu_frac

    def k_arr(self) -> np.ndarray:
        return np.full(self.N, self.k, dtype=float) if self.k_vec is None else np.asarray(self.k_vec, dtype=float)

    def m_arr(self) -> np.ndarray:
        return np.full(self.N, self.m, dtype=float) if self.m_vec is None else np.asarray(self.m_vec, dtype=float)

    def mu_arr(self) -> np.ndarray:
        return self.m_arr() * self.mu_frac

    def g_arr(self) -> np.ndarray:
        return np.full(self.N, self.g, dtype=float) if self.g_vec is None else np.asarray(self.g_vec, dtype=float)

    def beta_of(self, u):
        """Piecewise cubic coefficient as a function of u.

        Returns a scalar (if u is scalar) or array matching u, with
        beta_comp on u<0 and beta_tens on u>=0. Falls back to beta3 for
        any side that wasn't explicitly set. This keeps legacy callers
        that set only beta3 working.
        """
        b_c = self.beta_comp if self.beta_comp is not None else self.beta3
        b_t = self.beta_tens if self.beta_tens is not None else self.beta3
        if np.isscalar(u):
            return b_c if u < 0.0 else b_t
        return np.where(u < 0.0, b_c, b_t)

    def has_piecewise_beta(self) -> bool:
        return (self.beta_comp is not None) or (self.beta_tens is not None)


@dataclass
class State:
    x: np.ndarray   # center positions (N,)
    s: np.ndarray   # half extents (N,)
    vx: np.ndarray  # d/dt x
    vs: np.ndarray  # d/dt s
    lam: np.ndarray # accumulated contact impulses, size N (contact 0 = floor-body0, c = body_{c-1}-body_c)
    s_p: Optional[np.ndarray] = None  # plastic strain offset (rest s_rest = s0 + s_p)
    ceiling_x: Optional[float] = None  # kinematic ceiling position (None = no ceiling)
    lam_ceiling: float = 0.0           # accumulated ceiling contact impulse

    def __post_init__(self):
        if self.s_p is None:
            self.s_p = np.zeros_like(self.s)

    def copy(self) -> "State":
        return State(self.x.copy(), self.s.copy(), self.vx.copy(), self.vs.copy(),
                     self.lam.copy(), self.s_p.copy(),
                     self.ceiling_x, self.lam_ceiling)


def make_cradle_state(p: Params, gap: float = 0.3, v0: float = 1.0,
                      inner_gap: float = 0.0) -> State:
    """
    Newton's cradle initial state.

    gap       : space between the incoming body (0) and the row (1..N-1).
    inner_gap : space between adjacent bodies in the row. Non-zero values
                temporally separate the pairwise collisions so the chain
                contact manifold is processed as a sequence rather than a
                simultaneous LCP — this is essential for near-ideal cradle
                transfer in the low-mu regime.
    """
    N = p.N
    s = np.full(N, p.s0)
    x = np.empty(N)
    row_left = 1.0
    x[1] = row_left + p.s0
    for i in range(2, N):
        x[i] = x[i - 1] + 2.0 * p.s0 + inner_gap
    x[0] = x[1] - 2.0 * p.s0 - gap
    vx = np.zeros(N); vx[0] = v0
    return State(x=x, s=s, vx=vx, vs=np.zeros(N), lam=np.zeros(N))


def make_initial_state(p: Params) -> State:
    N = p.N
    s = np.full(N, p.s0)
    x = np.empty(N)
    x[0] = p.s0
    for i in range(1, N):
        x[i] = x[i - 1] + 2.0 * p.s0
    return State(x=x, s=s, vx=np.zeros(N), vs=np.zeros(N), lam=np.zeros(N))


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def energies(state: State, p: Params):
    m = p.m_arr(); mu = p.mu_arr(); k = p.k_arr(); g_ = p.g_arr()
    KE = 0.5 * np.sum(m * state.vx ** 2) + 0.5 * np.sum(mu * state.vs ** 2)
    PE_g = np.sum(m * g_ * state.x)
    s_rest = p.s0 + state.s_p
    u = state.s - s_rest
    beta = p.beta_of(u) if p.has_piecewise_beta() else p.beta3
    PE_e = 0.5 * np.sum(k * u ** 2) + 0.25 * np.sum(beta * u ** 4)
    return KE, PE_g, PE_e


# ---------------------------------------------------------------------------
# Analytic equilibrium
# ---------------------------------------------------------------------------
#
# Free-body analysis of body i (counted 0 = bottom) in a stack of N equal bodies:
#   - gravity on x:   -m g
#   - lower contact impulse (floor or body below): upward, magnitude F_lo
#   - upper contact impulse (body above), downward, magnitude F_hi = (N - i - 1) m g
#     (supports the weight of everything above)
#   - static x-balance gives F_lo = (N - i) m g.
#
# s-equilibrium: the contact Jacobians have coefficient -1 on the s DoF of the
# owning body for *both* the lower and upper contact (bottom of body i is
# x_i - s_i, top is x_i + s_i, so |J_s| = 1 on both sides). Force on s from
# contacts is -F_lo - F_hi, elastic force is -k (s_i - s0). Setting to zero:
#
#     k (s0 - s_i) = F_lo + F_hi = (2(N - i) - 1) m g
#
# and positions stack as x_i = sum_{j<i} 2 s_j + s_i.

def analytic_equilibrium(p: Params):
    N = p.N
    s_eq = np.array([p.s0 - (2 * (N - i) - 1) * p.m * p.g / p.k for i in range(N)])
    x_eq = np.empty(N)
    x_eq[0] = s_eq[0]
    for i in range(1, N):
        x_eq[i] = x_eq[i - 1] + s_eq[i - 1] + s_eq[i]
    return x_eq, s_eq


def analytic_single(p: Params):
    return p.s0 - p.m * p.g / p.k


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _half_drift_exp(state: State, p: Params, k_arr, mu_arr, half_dt: float):
    """Symplectic half-drift: gravity half-kick, x drift, exponential (s,vs)
    update over half_dt. Damping is applied proportionally."""
    if p.damping != 0.0:
        damp_half = np.exp(-p.damping * half_dt)
        state.vx *= damp_half
        state.vs *= damp_half
    # Gravity half-kick on vx.
    state.vx -= p.g_arr() * half_dt
    # x drift.
    state.x = state.x + state.vx * half_dt
    # Exact harmonic oscillator step on (u=s-s_rest, vs) over half_dt.
    s_rest = p.s0 + state.s_p
    omega = np.sqrt(k_arr / mu_arr)
    cw = np.cos(omega * half_dt)
    sw = np.sin(omega * half_dt)
    u = state.s - s_rest
    u_new = u * cw + (state.vs / omega) * sw
    vs_new = -u * omega * sw + state.vs * cw
    state.s = s_rest + u_new
    state.vs = vs_new


def _si_step_moreau(state: State, p: Params, k_arr, m_arr, mu_arr):
    """Moreau midpoint time-stepping.

    Structure (symmetric, symplectic for smooth part):
        1. Half-drift dt/2 (exact oscillator + gravity half-kick + x drift).
        2. Velocity-level contact solve at the midpoint configuration,
           using inverse-effective-mass K_v and pure velocity impulses
           (NO position response — positions evolve via the second drift).
        3. Half-drift dt/2 with the updated velocities.

    Active-set gate: velocity constraints are only enforced on contacts
    whose midpoint gap is <= 0 (mirrors Box2D manifold creation; without
    this, approach velocity across an open gap gets killed).

    Contact warm-start is disabled — lam here is a momentum impulse, not a
    force, and the position/velocity semantics differ from the single-pass
    BE path, so carrying lam across modes would be incorrect.
    """
    dt = p.dt
    N = p.N
    residuals_dlam = []
    residuals_gap = []

    state.lam[:] = 0.0
    state.lam_ceiling = 0.0

    # -- first half-drift -------------------------------------------------
    _half_drift_exp(state, p, k_arr, mu_arr, 0.5 * dt)

    # -- active-set detection at the midpoint -----------------------------
    active_floor = p.floor and (state.x[0] - state.s[0]) <= 0.0
    active_cc = np.zeros(N, dtype=bool)
    for c in range(1, N):
        i, j = c - 1, c
        if (state.x[j] - state.s[j] - state.x[i] - state.s[i]) <= 0.0:
            active_cc[c] = True
    active_ceiling = (state.ceiling_x is not None
                      and (state.ceiling_x - state.x[N - 1] - state.s[N - 1]) <= 0.0)

    # -- velocity-level PGS ------------------------------------------------
    # Pure velocity impulses: K_v uses inverse effective masses only.
    # Impulse units: momentum (force*time); here we solve for lam such
    # that v_new = v + W * lam / M with J v_new >= 0 ⟂ lam >= 0.
    inv_m = 1.0 / (m_arr + p.eps_reg)
    inv_mu = 1.0 / (mu_arr + p.eps_reg)
    sweep_count = 0
    for sweep in range(p.max_sweeps):
        max_dlam = 0.0
        max_vviol = 0.0

        if active_floor:
            jv = state.vx[0] - state.vs[0]
            Kv = inv_m[0] + inv_mu[0]
            dlam = -jv / Kv
            lam_new = max(0.0, state.lam[0] + dlam)
            dlam = lam_new - state.lam[0]
            state.lam[0] = lam_new
            state.vx[0] += (+1.0) * inv_m[0] * dlam
            state.vs[0] += (-1.0) * inv_mu[0] * dlam
            max_dlam = max(max_dlam, abs(dlam))
            if jv < 0:
                max_vviol = max(max_vviol, -jv)

        for c in range(1, N):
            if not active_cc[c]:
                continue
            i, j = c - 1, c
            jv = state.vx[j] - state.vs[j] - state.vx[i] - state.vs[i]
            Kv = inv_m[i] + inv_m[j] + inv_mu[i] + inv_mu[j]
            dlam = -jv / Kv
            lam_new = max(0.0, state.lam[c] + dlam)
            dlam = lam_new - state.lam[c]
            state.lam[c] = lam_new
            state.vx[i] += (-1.0) * inv_m[i] * dlam
            state.vx[j] += (+1.0) * inv_m[j] * dlam
            state.vs[i] += (-1.0) * inv_mu[i] * dlam
            state.vs[j] += (-1.0) * inv_mu[j] * dlam
            max_dlam = max(max_dlam, abs(dlam))
            if jv < 0:
                max_vviol = max(max_vviol, -jv)

        if active_ceiling:
            top = N - 1
            jv = -state.vx[top] - state.vs[top]
            Kv = inv_m[top] + inv_mu[top]
            dlam = -jv / Kv
            lam_new = max(0.0, state.lam_ceiling + dlam)
            dlam = lam_new - state.lam_ceiling
            state.lam_ceiling = lam_new
            state.vx[top] += (-1.0) * inv_m[top] * dlam
            state.vs[top] += (-1.0) * inv_mu[top] * dlam
            max_dlam = max(max_dlam, abs(dlam))
            if jv < 0:
                max_vviol = max(max_vviol, -jv)

        residuals_dlam.append(max_dlam)
        residuals_gap.append(max_vviol)
        sweep_count = sweep + 1
        if max_dlam < p.tol and max_vviol < p.tol:
            break

    # -- second half-drift -------------------------------------------------
    _half_drift_exp(state, p, k_arr, mu_arr, 0.5 * dt)

    # -- plasticity return mapping (same as single-pass) ------------------
    if np.isfinite(p.sigma_Y):
        s_rest_post = p.s0 + state.s_p
        sigma = k_arr * (state.s - s_rest_post)
        sigma_Y_eff = p.sigma_Y + p.H_hard * np.abs(state.s_p)
        f = np.abs(sigma) - sigma_Y_eff
        over = f > 0.0
        if np.any(over):
            dp = f[over] / (k_arr[over] + p.H_hard)
            state.s_p[over] += np.sign(sigma[over]) * dp

    return residuals_dlam, residuals_gap, sweep_count, 0


def si_step(state: State, p: Params, mode: str = "local_implicit"):
    """
    One outer timestep: free-flight integration of the unconstrained dynamics,
    then SI contact sweeps that apply impulses against the integrator's
    per-DoF linearised response over one dt.

    See module docstring for the list of supported modes.
    """
    dt = p.dt
    N = p.N
    k_arr = p.k_arr()
    m_arr = p.m_arr()
    mu_arr = p.mu_arr()
    use_el = (mode == "local_implicit")
    use_exp = (mode == "exponential")
    use_exp_sust = (mode == "exp_sustained")
    use_relin = (mode == "local_implicit_relin")
    use_twopass = (mode == "two_pass" or mode == "two_pass_exp")
    use_twopass_exp = (mode == "two_pass_exp")
    use_moreau = (mode == "moreau")
    if use_exp_sust:
        use_exp = True  # share exact free-flight; only per-impulse apos/avel differ
    if use_relin:
        use_el = True  # share the local_implicit free-flight path
    if use_twopass and not use_twopass_exp:
        use_el = True  # two-pass default: BE-coupled free flight
    if use_twopass_exp:
        use_exp = True  # two-pass + exact analytic free flight for the s-DoF
    if use_moreau:
        # Moreau midpoint: half-drift + velocity-level contact jump + half-drift.
        # Dispatched entirely in a dedicated branch below — bail out of the
        # shared free-flight/PGS path.
        return _si_step_moreau(state, p, k_arr, m_arr, mu_arr)

    # Snapshot pre-step floor gap for the soft-contact target (needs the
    # gap BEFORE damping, gravity, or free-flight update).
    g_old_floor = state.x[0] - state.s[0] if p.floor else 0.0
    # Same snapshot for body-body pairs (index c holds gap between c-1 and c).
    g_old_cc = np.zeros(N)
    if p.contact_wn is not None:
        for _c in range(1, N):
            _i, _j = _c - 1, _c
            g_old_cc[_c] = state.x[_j] - state.s[_j] - state.x[_i] - state.s[_i]
    # Pre-step lam for detecting a fresh contact impact (restitution
    # applies only on the first impulse sweep of a new collision).
    lam_floor_prev = float(state.lam[0]) if p.floor else 0.0
    lam_cc_prev = state.lam.copy()  # per-pair (index c holds prev lam for pair c)

    # --- damping & gravity (all modes share these on x) ----------------
    damp = np.exp(-p.damping * dt)
    state.vx *= damp
    state.vs *= damp
    state.vx -= p.g_arr() * dt

    # --- free flight of the internal (s, vs) oscillator ----------------
    # Oscillator is around the plastically-shifted rest length s_rest = s0 + s_p.
    s_rest = p.s0 + state.s_p
    max_newton = 0
    # Snapshot pre-free-flight (u, v) for the relin mode, which needs to
    # reconstruct the full backward-Euler state from the original (u, v)
    # plus an accumulated contact force f_s on every Newton resolve.
    if use_relin:
        u_init_snap = (state.s - s_rest).copy()
        v_init_snap = state.vs.copy()
        A_body = 1.0 + (dt ** 2) * k_arr / mu_arr
        B_body = (dt ** 2) * p.beta3 / mu_arr
        C_body = u_init_snap + dt * v_init_snap
    if use_exp:
        omega = np.sqrt(k_arr / mu_arr)
        cw = np.cos(omega * dt)
        sw = np.sin(omega * dt)
        u = state.s - s_rest
        u_new = u * cw + (state.vs / omega) * sw
        vs_free = -u * omega * sw + state.vs * cw
        state.s = (s_rest + u_new).copy()
        state.vs = vs_free.copy()
    else:
        # True backward Euler for the harmonic oscillator u = s - s_rest,
        # μ ü = -k u. A-stable and L-stable; stiff modes are damped to rest.
        # Backward Euler on u = s - s_rest with force F(u) = -(k u + beta3 u^3).
        # G(u_new) = A u_new + B u_new^3 - C = 0,
        # where A = 1 + dt^2 k / mu, B = dt^2 beta3 / mu, C = u + dt v.
        # G is strictly monotone (G' = A + 3 B u_new^2 > 0), so Newton
        # from the linear warm start converges in a handful of iterations.
        A = 1.0 + (dt ** 2) * k_arr / mu_arr
        piecewise = p.has_piecewise_beta()
        u_old = state.s - s_rest
        C = u_old + dt * state.vs
        u_new = C / A  # linear warm start
        if piecewise:
            for it in range(20):
                beta_u = p.beta_of(u_new)
                B = (dt ** 2) * beta_u / mu_arr
                G = A * u_new + B * u_new ** 3 - C
                Gp = A + 3.0 * B * u_new ** 2
                du = G / Gp
                u_new = u_new - du
                max_newton = it + 1
                if np.max(np.abs(du)) < 1.0e-12:
                    break
            beta_final = p.beta_of(u_new)
            vs_new = state.vs - (dt / mu_arr) * (k_arr * u_new + beta_final * u_new ** 3)
        else:
            B = (dt ** 2) * p.beta3 / mu_arr
            if p.beta3 != 0.0:
                for it in range(20):
                    G = A * u_new + B * u_new ** 3 - C
                    Gp = A + 3.0 * B * u_new ** 2
                    du = G / Gp
                    u_new = u_new - du
                    max_newton = it + 1
                    if np.max(np.abs(du)) < 1.0e-12:
                        break
            vs_new = state.vs - (dt / mu_arr) * (k_arr * u_new + p.beta3 * u_new ** 3)
        state.s = (s_rest + u_new).copy()
        state.vs = vs_new.copy()

    x_pre = state.x + state.vx * dt
    s_pre = state.s.copy()
    state.x = x_pre.copy()

    if not p.warm_start:
        state.lam[:] = 0.0
        state.lam_ceiling = 0.0
    else:
        # Compressive-phase gate (mirrors solver2d s2PrepareContacts_PGS,
        # src/solve_common.c:133 — only warm-start when separation <= 0).
        # Without this, a carried-over lam on a now-separated contact acts
        # as an attractive impulse through the projection
        # lam_new = max(0, lam + dlam), silently pulling freshly-parted
        # bodies back together (breaks two-ball-attract bouncing).
        if p.floor and (state.x[0] - state.s[0]) > 0.0:
            state.lam[0] = 0.0
        for _c in range(1, N):
            _i, _j = _c - 1, _c
            if (state.x[_j] - state.s[_j] - state.x[_i] - state.s[_i]) > 0.0:
                state.lam[_c] = 0.0
        if state.ceiling_x is not None:
            if (state.ceiling_x - state.x[N - 1] - state.s[N - 1]) > 0.0:
                state.lam_ceiling = 0.0

    # --- per-impulse linearised responses over one dt -------------------
    # Convention: lambda carries units of force. dPos = apos * J * lambda,
    # dVel = avel * J * lambda, with J the +/-1 Jacobian entry.
    if use_exp_sust:
        # Sustained-force linearization: free particle under constant force F
        # over [0,dt] travels F*dt^2/(2m), not F*dt^2/m. BE's dt^2/m is the
        # "impulse applied instantly" limit, which silently injects factor-2
        # dissipation into the PGS response. See insights entry.
        apos_x = 0.5 * dt ** 2 / (m_arr + p.eps_reg)
    else:
        apos_x = dt ** 2 / (m_arr + p.eps_reg)
    avel_x = dt / (m_arr + p.eps_reg)
    if use_el:
        # k_eff is the linearisation of the cubic-force law at the post
        # free-flight u. One relinearisation per step — good enough to let
        # contact impulses see the stiffened response.
        u_post = state.s - s_rest
        beta_post = p.beta_of(u_post) if p.has_piecewise_beta() else p.beta3
        k_eff = k_arr + 3.0 * beta_post * u_post ** 2
        apos_s = 1.0 / (mu_arr / dt ** 2 + k_eff + p.eps_reg)
        avel_s = apos_s / dt  # implicit: dVs = dS / dt
    elif use_exp_sust:
        # Duhamel integral for constant force F over [0,dt] on the harmonic
        # oscillator μü + ku = F:
        #   u(dt) = (F/k)(1 - cos(ωdt))
        #   v(dt) = (F/(μω)) sin(ωdt)
        # Matches the "contact force is sustained across the step" semantics
        # that position-level PGS is actually simulating, and is
        # energy-exact for the linear oscillator.
        apos_s = (1.0 - cw) / (mu_arr * omega * omega + p.eps_reg)
        avel_s = sw / (mu_arr * omega + p.eps_reg)
    elif use_exp:
        apos_s = dt * sw / (mu_arr * omega + p.eps_reg)
        avel_s = dt * cw / (mu_arr + p.eps_reg)
    else:  # naive
        apos_s = np.zeros(N)
        avel_s = np.zeros(N)

    # --- s-DoF update closures (mode dispatch) --------------------------
    # get_apos_s(i): per-body effective position response (for K in PGS).
    # update_s(i, Js, dlam): apply the s-DoF half of a contact impulse.
    # Frozen modes use precomputed apos_s/avel_s. Relin mode maintains an
    # accumulator f_s and Newton-resolves the cubic BE each update so the
    # tangent is live at the current u_new.
    if use_relin:
        u_work = (state.s - s_rest).copy()  # per-body Newton warm start
        f_s = np.zeros(N)
        # Prototype: no contact warm starting — tangent-inconsistent with
        # carried-over multipliers.
        state.lam[:] = 0.0
        state.lam_ceiling = 0.0

        dt_over_mu = dt ** 2 / mu_arr
        _pw = p.has_piecewise_beta()
        _b_c = p.beta_comp if p.beta_comp is not None else p.beta3
        _b_t = p.beta_tens if p.beta_tens is not None else p.beta3

        def get_apos_s(i, _A=A_body, _u=u_work,
                       _mu=mu_arr, _dt=dt, _eps=p.eps_reg,
                       _beta3=p.beta3, _k=k_arr,
                       _pw=_pw, _bc=_b_c, _bt=_b_t):
            u = _u[i]
            beta = (_bc if u < 0.0 else _bt) if _pw else _beta3
            k_eff_i = _k[i] + 3.0 * beta * u * u
            return 1.0 / (_mu[i] / (_dt * _dt) + k_eff_i + _eps)

        def update_s(i, Js, dlam,
                     _A=A_body, _C=C_body, _dom=dt_over_mu,
                     _u=u_work, _fs=f_s, _st=state, _sr=s_rest,
                     _ui=u_init_snap, _dt=dt, _mu=mu_arr, _beta3=p.beta3,
                     _pw=_pw, _bc=_b_c, _bt=_b_t):
            _fs[i] += Js * dlam
            rhs = _C[i] + _dom[i] * _fs[i]
            u = _u[i]
            Ai = _A[i]
            for _ in range(10):
                beta = (_bc if u < 0.0 else _bt) if _pw else _beta3
                Bi = (_dt * _dt) * beta / _mu[i]
                G = Ai * u + Bi * u * u * u - rhs
                Gp = Ai + 3.0 * Bi * u * u
                du = G / Gp
                u -= du
                if abs(du) < 1.0e-14:
                    break
            _u[i] = u
            _st.s[i] = _sr[i] + u
            _st.vs[i] = (u - _ui[i]) / _dt
    else:
        _apos_s_ref = apos_s
        _avel_s_ref = avel_s

        def get_apos_s(i, _a=_apos_s_ref):
            return _a[i]

        def update_s(i, Js, dlam, _a=_apos_s_ref, _v=_avel_s_ref, _st=state):
            _st.s[i] += _a[i] * Js * dlam
            _st.vs[i] += _v[i] * Js * dlam

    # Soft-contact target gap for the floor (computed once per step from
    # the pre-step and post-free-flight gaps, held constant across PGS
    # sweeps). Derivation: exact BE on the mass-spring-damper gap ODE
    # μ ẍ + c ẋ + k x = 0 gives x_target = (x_free + 2Ωζ x_old)/α with
    # α = 1 + Ω² + 2Ωζ, Ω = dt·ωₙ. The PGS then drives g toward this
    # target rather than to zero.
    g_target_cc = np.zeros(N)
    if p.contact_wn is not None:
        Omega_soft = dt * p.contact_wn
        zeta_soft = p.contact_zeta
        alpha_soft = 1.0 + Omega_soft * Omega_soft + 2.0 * Omega_soft * zeta_soft
        if p.floor:
            g_free_floor = state.x[0] - state.s[0]
            g_target_floor = (g_free_floor + 2.0 * Omega_soft * zeta_soft * g_old_floor) / alpha_soft
        else:
            g_target_floor = 0.0
        for _c in range(1, N):
            _i, _j = _c - 1, _c
            g_free_cc = state.x[_j] - state.s[_j] - state.x[_i] - state.s[_i]
            g_target_cc[_c] = (g_free_cc + 2.0 * Omega_soft * zeta_soft * g_old_cc[_c]) / alpha_soft
    else:
        g_target_floor = 0.0

    # Hard-contact restitution for the floor: apply a velocity-level
    # reflection impulse ONCE before the PGS sweeps, sized so that the
    # relative velocity at the contact becomes -e·v_rel_old. Skip the
    # floor PGS entirely for this step — any residual gap is tolerated
    # to preserve energy. Unilateral constraint will kick in again on
    # the next fresh impact.
    skip_floor_pgs = False
    if (p.floor and p.restitution > 0.0 and lam_floor_prev == 0.0
            and not (p.contact_wn is not None)):
        v_rel_floor = float(state.vx[0] - state.vs[0])
        g_floor_post = state.x[0] - state.s[0]
        if v_rel_floor < 0.0 and g_floor_post < 0.0:
            # Pure velocity-level reflection: Δv only, no position update.
            # Using physical K = 1/m + 1/μ (mass metric), not dt-based
            # linearised response, so KE is preserved exactly for e=1.
            #   J·Δv = -(1+e)·J·v  =>  λ/K_phys  (λ in impulse units)
            K_phys = 1.0 / (m_arr[0] + p.eps_reg) + 1.0 / (mu_arr[0] + p.eps_reg)
            dJ = -(1.0 + p.restitution) * v_rel_floor  # desired jump in J·v
            lam_imp = dJ / K_phys                      # impulse (force·dt)
            state.vx[0] += lam_imp / (m_arr[0] + p.eps_reg)
            state.vs[0] -= lam_imp / (mu_arr[0] + p.eps_reg)
            # Leave positions alone — the already-completed free flight is the
            # trajectory the body was on; next step re-integrates with the
            # reflected velocities.
            state.lam[0] = 0.0
            skip_floor_pgs = True
    g_target_rest_floor = 0.0  # sweep loop uses plain zero target now

    # Body-body restitution: same velocity-only reflection for each pair
    # that is in fresh contact and approaching.
    skip_cc = np.zeros(N, dtype=bool)
    if (p.restitution > 0.0 and N > 1
            and not (p.contact_wn is not None)):
        for c in range(1, N):
            if lam_cc_prev[c] != 0.0:
                continue
            i, j = c - 1, c
            g_cc = state.x[j] - state.s[j] - state.x[i] - state.s[i]
            jv = state.vx[j] - state.vs[j] - state.vx[i] - state.vs[i]
            if jv >= 0.0 or g_cc >= 0.0:
                continue
            K_phys = (1.0 / (m_arr[i] + p.eps_reg)
                      + 1.0 / (mu_arr[i] + p.eps_reg)
                      + 1.0 / (m_arr[j] + p.eps_reg)
                      + 1.0 / (mu_arr[j] + p.eps_reg))
            dJ = -(1.0 + p.restitution) * jv
            lam_imp = dJ / K_phys
            # J on (x_i, s_i, x_j, s_j) = (-1, -1, +1, -1)
            state.vx[i] -= lam_imp / (m_arr[i] + p.eps_reg)
            state.vs[i] -= lam_imp / (mu_arr[i] + p.eps_reg)
            state.vx[j] += lam_imp / (m_arr[j] + p.eps_reg)
            state.vs[j] -= lam_imp / (mu_arr[j] + p.eps_reg)
            state.lam[c] = 0.0
            skip_cc[c] = True

    residuals_dlam = []
    residuals_gap = []
    sweep_count = 0

    if use_twopass:
        # --------------------------------------------------------------
        # Two-pass SI: velocity-level solve first (non-penetration on
        # J·v), then a split position correction. The velocity pass uses
        # K_v = avel_x + avel_s and updates BOTH positions and velocities
        # (BE-consistent — impulses of size dlam produce dv = avel·dlam
        # and dx = apos·dlam = dt·avel·dlam). The position pass is pure
        # drift correction with split impulses that do NOT touch velocity.
        #
        # lam_v (= state.lam) is warm-started across steps (with the
        # compressive-phase gate above). lam_p is step-local: split
        # impulses are drift correction only, not physical force.
        # --------------------------------------------------------------

        # Active-set gate: the velocity-level constraint J·v ≥ 0 is only
        # physically meaningful when the contact is actually in/near
        # penetration. Without this gate the velocity pass kills any
        # approach velocity across arbitrarily large gaps (two separated
        # balls attracted by gravity would freeze before ever touching).
        # Box2D avoids this by only creating manifolds on near-contact;
        # we replicate that by checking the post-free-flight gap.
        active_floor = (p.floor and not skip_floor_pgs
                        and (state.x[0] - state.s[0]) <= 0.0)
        active_cc = np.zeros(N, dtype=bool)
        for c in range(1, N):
            if skip_cc[c]:
                continue
            i, j = c - 1, c
            if (state.x[j] - state.s[j] - state.x[i] - state.s[i]) <= 0.0:
                active_cc[c] = True
        active_ceiling = (state.ceiling_x is not None
                          and (state.ceiling_x - state.x[N - 1] - state.s[N - 1]) <= 0.0)
        # Inactive contacts: force lam to zero (no stale warm-start bias).
        if p.floor and not active_floor:
            state.lam[0] = 0.0
        for c in range(1, N):
            if not active_cc[c]:
                state.lam[c] = 0.0
        if state.ceiling_x is not None and not active_ceiling:
            state.lam_ceiling = 0.0

        # Pass 1: velocity pass ----------------------------------------
        for sweep in range(p.max_sweeps):
            max_dlam = 0.0
            max_vviol = 0.0

            if active_floor:
                jv = state.vx[0] - state.vs[0]
                Kv = avel_x[0] + avel_s[0]
                dlam = -jv / Kv
                lam_new = max(0.0, state.lam[0] + dlam)
                dlam = lam_new - state.lam[0]
                state.lam[0] = lam_new
                state.x[0] += apos_x[0] * (+1.0) * dlam
                state.vx[0] += avel_x[0] * (+1.0) * dlam
                update_s(0, -1.0, dlam)
                max_dlam = max(max_dlam, abs(dlam))
                if jv < 0:
                    max_vviol = max(max_vviol, -jv)

            for c in range(1, N):
                if not active_cc[c]:
                    continue
                i, j = c - 1, c
                jv = state.vx[j] - state.vs[j] - state.vx[i] - state.vs[i]
                Kv = avel_x[i] + avel_x[j] + avel_s[i] + avel_s[j]
                dlam = -jv / Kv
                lam_new = max(0.0, state.lam[c] + dlam)
                dlam = lam_new - state.lam[c]
                state.lam[c] = lam_new
                state.x[i]  += apos_x[i] * (-1.0) * dlam
                state.vx[i] += avel_x[i] * (-1.0) * dlam
                state.x[j]  += apos_x[j] * (+1.0) * dlam
                state.vx[j] += avel_x[j] * (+1.0) * dlam
                update_s(i, -1.0, dlam)
                update_s(j, -1.0, dlam)
                max_dlam = max(max_dlam, abs(dlam))
                if jv < 0:
                    max_vviol = max(max_vviol, -jv)

            if active_ceiling:
                top = N - 1
                jv = -state.vx[top] - state.vs[top]
                Kv = avel_x[top] + avel_s[top]
                dlam = -jv / Kv
                lam_new = max(0.0, state.lam_ceiling + dlam)
                dlam = lam_new - state.lam_ceiling
                state.lam_ceiling = lam_new
                state.x[top]  += apos_x[top] * (-1.0) * dlam
                state.vx[top] += avel_x[top] * (-1.0) * dlam
                update_s(top, -1.0, dlam)
                max_dlam = max(max_dlam, abs(dlam))
                if jv < 0:
                    max_vviol = max(max_vviol, -jv)

            residuals_dlam.append(max_dlam)
            residuals_gap.append(max_vviol)
            sweep_count = sweep + 1
            if max_dlam < p.tol and max_vviol < p.tol:
                break

        # Pass 2: position pass (split impulses; positions only) -------
        lam_pos_floor = 0.0
        lam_pos_c = np.zeros(N)
        lam_pos_ceil = 0.0
        for sweep in range(p.max_sweeps):
            max_dlam_pos = 0.0
            max_gap_violation = 0.0

            if p.floor and not skip_floor_pgs:
                g = state.x[0] - state.s[0]
                K = apos_x[0] + apos_s[0]
                dlam = -g / K
                lam_new = max(0.0, lam_pos_floor + dlam)
                dlam = lam_new - lam_pos_floor
                lam_pos_floor = lam_new
                state.x[0] += apos_x[0] * (+1.0) * dlam
                state.s[0] += apos_s[0] * (-1.0) * dlam
                max_dlam_pos = max(max_dlam_pos, abs(dlam))
                if g < 0:
                    max_gap_violation = max(max_gap_violation, -g)

            for c in range(1, N):
                if skip_cc[c]:
                    continue
                i, j = c - 1, c
                g = state.x[j] - state.s[j] - state.x[i] - state.s[i]
                K = apos_x[i] + apos_x[j] + apos_s[i] + apos_s[j]
                dlam = -g / K
                lam_new = max(0.0, lam_pos_c[c] + dlam)
                dlam = lam_new - lam_pos_c[c]
                lam_pos_c[c] = lam_new
                state.x[i] += apos_x[i] * (-1.0) * dlam
                state.x[j] += apos_x[j] * (+1.0) * dlam
                state.s[i] += apos_s[i] * (-1.0) * dlam
                state.s[j] += apos_s[j] * (-1.0) * dlam
                max_dlam_pos = max(max_dlam_pos, abs(dlam))
                if g < 0:
                    max_gap_violation = max(max_gap_violation, -g)

            if state.ceiling_x is not None:
                top = N - 1
                g = state.ceiling_x - state.x[top] - state.s[top]
                K = apos_x[top] + apos_s[top]
                dlam = -g / K
                lam_new = max(0.0, lam_pos_ceil + dlam)
                dlam = lam_new - lam_pos_ceil
                lam_pos_ceil = lam_new
                state.x[top] += apos_x[top] * (-1.0) * dlam
                state.s[top] += apos_s[top] * (-1.0) * dlam
                max_dlam_pos = max(max_dlam_pos, abs(dlam))
                if g < 0:
                    max_gap_violation = max(max_gap_violation, -g)

            sweep_count += 1
            if max_dlam_pos < p.tol and max_gap_violation < p.tol:
                break

        # Skip the single-pass loop below.
        if np.isfinite(p.sigma_Y):
            s_rest_post = p.s0 + state.s_p
            sigma = k_arr * (state.s - s_rest_post)
            sigma_Y_eff = p.sigma_Y + p.H_hard * np.abs(state.s_p)
            f = np.abs(sigma) - sigma_Y_eff
            over = f > 0.0
            if np.any(over):
                dp = f[over] / (k_arr[over] + p.H_hard)
                state.s_p[over] += np.sign(sigma[over]) * dp
        return residuals_dlam, residuals_gap, sweep_count, max_newton

    for sweep in range(p.max_sweeps):
        max_dlam = 0.0
        max_gap_violation = 0.0

        # ---- floor contact: body 0 against x = 0 ----
        # J on (x_0, s_0) = (+1, -1)
        if p.floor and not skip_floor_pgs:
            g = state.x[0] - state.s[0]
            K = apos_x[0] + get_apos_s(0)
            if p.contact_wn is not None:
                dlam = (g_target_floor - g) / K
            else:
                dlam = (g_target_rest_floor - g) / K
            lam_new = max(0.0, state.lam[0] + dlam)
            dlam = lam_new - state.lam[0]
            state.lam[0] = lam_new
            state.x[0] += apos_x[0] * (+1.0) * dlam
            state.vx[0] += avel_x[0] * (+1.0) * dlam
            update_s(0, -1.0, dlam)
            max_dlam = max(max_dlam, abs(dlam))
            if g < 0:
                max_gap_violation = max(max_gap_violation, -g)

        # ---- body-body contacts ----
        # J on (x_i, s_i, x_j, s_j) = (-1, -1, +1, -1)
        for c in range(1, N):
            if skip_cc[c]:
                continue
            i, j = c - 1, c
            g = state.x[j] - state.s[j] - state.x[i] - state.s[i]
            K = apos_x[i] + apos_x[j] + get_apos_s(i) + get_apos_s(j)
            if p.contact_wn is not None:
                dlam = (g_target_cc[c] - g) / K
            else:
                dlam = -g / K
            lam_new = max(0.0, state.lam[c] + dlam)
            dlam = lam_new - state.lam[c]
            state.lam[c] = lam_new
            state.x[i]  += apos_x[i] * (-1.0) * dlam
            state.vx[i] += avel_x[i] * (-1.0) * dlam
            state.x[j]  += apos_x[j] * (+1.0) * dlam
            state.vx[j] += avel_x[j] * (+1.0) * dlam
            update_s(i, -1.0, dlam)
            update_s(j, -1.0, dlam)
            max_dlam = max(max_dlam, abs(dlam))
            if g < 0:
                max_gap_violation = max(max_gap_violation, -g)

        # ---- ceiling contact: body N-1 top against state.ceiling_x ----
        # J on (x_{N-1}, s_{N-1}) = (-1, -1). Kinematic side: zero inv mass.
        if state.ceiling_x is not None:
            top = N - 1
            g = state.ceiling_x - state.x[top] - state.s[top]
            K = apos_x[top] + get_apos_s(top)
            dlam = -g / K
            lam_new = max(0.0, state.lam_ceiling + dlam)
            dlam = lam_new - state.lam_ceiling
            state.lam_ceiling = lam_new
            state.x[top]  += apos_x[top] * (-1.0) * dlam
            state.vx[top] += avel_x[top] * (-1.0) * dlam
            update_s(top, -1.0, dlam)
            max_dlam = max(max_dlam, abs(dlam))
            if g < 0:
                max_gap_violation = max(max_gap_violation, -g)

        residuals_dlam.append(max_dlam)
        residuals_gap.append(max_gap_violation)
        sweep_count = sweep + 1
        if max_dlam < p.tol and max_gap_violation < p.tol:
            break

    # --- plastic return mapping (elastic predictor / plastic corrector) ---
    # Trial stress with current s_p. Yield surface f = |sigma| - sigma_Y_eff
    # with sigma_Y_eff = sigma_Y + H * |s_p| (linear isotropic hardening;
    # |s_p| serves as the accumulated plastic strain for monotonic loading).
    # Consistency gives plastic increment dp = (|sigma_tr| - sigma_Y_eff)/(k+H).
    if np.isfinite(p.sigma_Y):
        s_rest_post = p.s0 + state.s_p
        sigma = k_arr * (state.s - s_rest_post)
        sigma_Y_eff = p.sigma_Y + p.H_hard * np.abs(state.s_p)
        f = np.abs(sigma) - sigma_Y_eff
        over = f > 0.0
        if np.any(over):
            dp = f[over] / (k_arr[over] + p.H_hard)
            state.s_p[over] += np.sign(sigma[over]) * dp

    return residuals_dlam, residuals_gap, sweep_count, max_newton


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

@dataclass
class RunLog:
    times: np.ndarray
    KE: np.ndarray
    PE_g: np.ndarray
    PE_e: np.ndarray
    total: np.ndarray
    sweeps: np.ndarray
    xs: np.ndarray   # (steps+1, N)
    ss: np.ndarray   # (steps+1, N)
    vxs: np.ndarray  # (steps+1, N)
    sps: np.ndarray  # (steps+1, N) plastic strain offset
    ceilings: np.ndarray  # (steps+1,) ceiling_x at each recorded frame (nan if no ceiling)
    final_residuals: list  # residuals of the last step (for a representative convergence plot)
    stable: bool = True
    max_newton: int = 0    # worst-case Newton iters seen during free-flight update


def run_sim(p: Params, mode: str, record_every: int = 1, init: State = None,
            ceiling_fn: Optional[Callable[[float], float]] = None) -> RunLog:
    state = init.copy() if init is not None else make_initial_state(p)
    if ceiling_fn is not None:
        state.ceiling_x = float(ceiling_fn(0.0))
    T = p.steps
    nrec = T // record_every + 1
    times = np.zeros(nrec)
    KE = np.zeros(nrec)
    PE_g = np.zeros(nrec)
    PE_e = np.zeros(nrec)
    sweeps = np.zeros(nrec, dtype=int)
    xs = np.zeros((nrec, p.N))
    ss = np.zeros((nrec, p.N))
    vxs = np.zeros((nrec, p.N))
    sps = np.zeros((nrec, p.N))
    ceilings = np.full(nrec, np.nan)

    ke0, peg0, pee0 = energies(state, p)
    KE[0] = ke0; PE_g[0] = peg0; PE_e[0] = pee0
    xs[0] = state.x; ss[0] = state.s; vxs[0] = state.vx; sps[0] = state.s_p
    if state.ceiling_x is not None:
        ceilings[0] = state.ceiling_x

    last_residuals = []
    stable = True
    max_newton_run = 0
    ri = 1
    for step in range(T):
        if ceiling_fn is not None:
            state.ceiling_x = float(ceiling_fn((step + 1) * p.dt))
        res_dl, res_g, n, nit = si_step(state, p, mode=mode)
        last_residuals = res_dl
        max_newton_run = max(max_newton_run, nit)
        bad = not (np.all(np.isfinite(state.x)) and np.all(np.isfinite(state.s)))
        if not bad:
            # Center-ordering invariant: adjacent pairs must not swap, and
            # floor body must stay above the floor centre. Once this fails
            # the contact normal is inverted and the solve is meaningless —
            # catch it here so phantom "energy preservation" from pass-through
            # can't masquerade as a valid result.
            if p.floor and state.x[0] <= 0.0:
                print(f"  [run_sim] floor pass-through at step {step+1}: "
                      f"x[0]={state.x[0]:+.4f}")
                bad = True
            if not bad and p.N > 1:
                dx = np.diff(state.x)
                if np.any(dx <= 0.0):
                    c = int(np.argmin(dx)) + 1
                    print(f"  [run_sim] center pass-through at step {step+1}: "
                          f"x[{c-1}]={state.x[c-1]:+.4f} x[{c}]={state.x[c]:+.4f}")
                    bad = True
        if bad:
            stable = False
            for k in range(ri, nrec):
                times[k] = (step + 1) * p.dt
                KE[k] = np.nan; PE_g[k] = np.nan; PE_e[k] = np.nan
                xs[k] = np.nan; ss[k] = np.nan; vxs[k] = np.nan
                sps[k] = np.nan; ceilings[k] = np.nan
                sweeps[k] = 0
            break
        if (step + 1) % record_every == 0:
            times[ri] = (step + 1) * p.dt
            ke, peg, pee = energies(state, p)
            KE[ri] = ke; PE_g[ri] = peg; PE_e[ri] = pee
            xs[ri] = state.x; ss[ri] = state.s; vxs[ri] = state.vx
            sps[ri] = state.s_p
            if state.ceiling_x is not None:
                ceilings[ri] = state.ceiling_x
            sweeps[ri] = n
            ri += 1

    total = KE + PE_g + PE_e
    return RunLog(times=times, KE=KE, PE_g=PE_g, PE_e=PE_e, total=total,
                  sweeps=sweeps, xs=xs, ss=ss, vxs=vxs, sps=sps, ceilings=ceilings,
                  final_residuals=last_residuals, stable=stable,
                  max_newton=max_newton_run)
