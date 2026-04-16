"""2D affine body: uniform-density disk that deforms into an ellipse.

DoFs per body: 6 — (cx, cy, F11, F12, F21, F22).
A material point at reference position X maps to world position x(X) = c + F * X.

Mass matrix is block diagonal: diag(M, M, mu, mu, mu, mu)
where mu = M * r0^2 / 4 (second moment of a uniform disk).
"""
import numpy as np
from dataclasses import dataclass, field
from . import energy


@dataclass
class Body2D:
    mass: float
    r0: float
    k: float = 1000.0
    nu: float = 0.3
    energy_model: str = "snh"  # "snh" or "bower"

    static: bool = False

    c: np.ndarray = field(default_factory=lambda: np.zeros(2))
    F: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 1.0]))
    vc: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vF: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def __post_init__(self):
        self.c = np.array(self.c, dtype=float)
        self.F = np.array(self.F, dtype=float)
        self.vc = np.array(self.vc, dtype=float)
        self.vF = np.array(self.vF, dtype=float)

    @property
    def mu_inertia(self):
        return self.mass * self.r0 ** 2 / 4.0

    @property
    def inv_mass(self):
        return 1.0 / self.mass

    @property
    def inv_mu(self):
        return 1.0 / self.mu_inertia

    @property
    def mass_vec(self):
        mu = self.mu_inertia
        return np.array([self.mass, self.mass, mu, mu, mu, mu])

    @property
    def inv_mass_vec(self):
        if self.static:
            return np.zeros(6)
        return 1.0 / self.mass_vec

    @property
    def lame(self):
        return energy.lame_from_k(self.k, self.nu)

    @property
    def q(self):
        return np.concatenate([self.c, self.F])

    @q.setter
    def q(self, val):
        self.c[:] = val[:2]
        self.F[:] = val[2:]

    @property
    def v(self):
        return np.concatenate([self.vc, self.vF])

    @v.setter
    def v(self, val):
        self.vc[:] = val[:2]
        self.vF[:] = val[2:]

    def kinetic_energy(self):
        return 0.5 * (self.mass * np.dot(self.vc, self.vc) +
                      self.mu_inertia * np.dot(self.vF, self.vF))

    def potential_energy(self, gravity=np.array([0.0, -10.0])):
        return -self.mass * np.dot(gravity, self.c)

    def _energy_funcs(self):
        """Return (psi, pk1, hessian_spd) for the selected energy model."""
        if self.energy_model == "bower":
            return (energy.psi_bower, energy.pk1_bower,
                    energy.hessian_bower, energy.hessian_spd_bower)
        return (energy.psi, energy.pk1, energy.hessian, energy.hessian_spd)

    def elastic_energy(self):
        mu_l, lam_l = self.lame
        psi_fn = self._energy_funcs()[0]
        return psi_fn(self.F, mu_l, lam_l) * self._energy_scale

    @property
    def _energy_scale(self):
        """Scale factor: elastic energy density * reference area."""
        return np.pi * self.r0 ** 2

    def elastic_force_F(self):
        mu_l, lam_l = self.lame
        pk1_fn = self._energy_funcs()[1]
        return -pk1_fn(self.F, mu_l, lam_l) * self._energy_scale

    def elastic_hessian_F(self):
        mu_l, lam_l = self.lame
        hess_fn = self._energy_funcs()[2]
        return hess_fn(self.F, mu_l, lam_l) * self._energy_scale

    def elastic_hessian_spd_F(self):
        mu_l, lam_l = self.lame
        hess_spd_fn = self._energy_funcs()[3]
        return hess_spd_fn(self.F, mu_l, lam_l) * self._energy_scale

    def total_energy(self, gravity=np.array([0.0, -10.0])):
        return self.kinetic_energy() + self.potential_energy(gravity) + self.elastic_energy()


def integrate_backward_euler(body, dt, gravity=np.array([0.0, -10.0]),
                              max_newton=10, ls_max=20):
    """One step of backward Euler for internal (elastic) + external (gravity) forces.

    Minimises the incremental potential (IP):
        E(vF) = (mu_i/2)||vF - vF_old||^2  +  Psi(F_old + dt*vF) * scale

    using Newton iterations with backtracking line search.  The line search
    guarantees the IP decreases (Armijo condition) and — for barrier energies
    like Bower — that det(F) stays above a floor.

    One Newton step with alpha=1 is equivalent to the old single-step BE
    (exact recovery when the energy is smooth enough).
    """
    if body.static:
        return
    # --- Center of mass: gravity is constant, so BE is exact ---
    body.vc = body.vc + dt * gravity
    body.c = body.c + dt * body.vc

    # --- Deformation gradient: Newton + line search on IP ---
    mu_i = body.mu_inertia
    F_old = body.F.copy()
    vF_old = body.vF.copy()

    mu_l, lam_l = body.lame
    scale = body._energy_scale
    psi_fn, pk1_fn, _, hess_spd_fn = body._energy_funcs()
    need_det_guard = (body.energy_model == "bower")

    def ip_energy(vF):
        F_trial = F_old + dt * vF
        return (0.5 * mu_i * float(np.dot(vF - vF_old, vF - vF_old))
                + psi_fn(F_trial, mu_l, lam_l) * scale)

    vF = vF_old.copy()

    for _ in range(max_newton):
        F_cur = F_old + dt * vF
        f_el = -pk1_fn(F_cur, mu_l, lam_l) * scale
        H_el = hess_spd_fn(F_cur, mu_l, lam_l) * scale

        # Residual of the IP stationarity condition
        residual = mu_i * (vF - vF_old) - dt * f_el
        A = mu_i * np.eye(4) + dt ** 2 * H_el

        try:
            dvF = np.linalg.solve(A, -residual)
        except np.linalg.LinAlgError:
            break

        # Backtracking line search (Armijo)
        E_cur = ip_energy(vF)
        directional = float(np.dot(residual, dvF))  # should be negative
        alpha = 1.0
        for _ in range(ls_max):
            vF_trial = vF + alpha * dvF
            F_trial = F_old + dt * vF_trial
            if need_det_guard and energy._det2(F_trial) < energy._BOWER_J_FLOOR:
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
    body.F = F_old + dt * body.vF
