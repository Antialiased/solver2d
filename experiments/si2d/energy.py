"""Hyperelastic energies for 2D affine bodies.

Two models:

1. **SNH** (Stable Neo-Hookean) — Smith, de Goes, Kim (SIGGRAPH 2018)
   Psi(F) = (mu/2)(Ic - 2) + (lam/2)(J - alpha)^2
   where alpha = 1 + mu/lam ensures rest stability (PK1 = 0 at F = I).
   Handles inversion (J < 0) smoothly but has a FINITE energy barrier at J = 0.
   The Dirichlet term mu*Ic/2 actively drives toward collapse; under large
   deformation (e.g. cantilever bending) the balance tips and bodies can
   collapse to zero volume.

2. **Bower** (Isochoric Neo-Hookean) — Bower (2009), Bonet & Wood
   Psi(F) = (mu/2)(Ic/J - 2) + (lam/2)(J - 1)^2
   Replaces the raw Dirichlet term with its isochoric (volume-neutral) form
   Ic/J, which goes to infinity as J -> 0+, providing a natural barrier
   against collapse. No alpha offset needed (stress-free at F = I by
   construction). Trade-off: undefined for J <= 0 (no inversion support).

Both models share the same linearisation at F = I, so lame_from_k applies
to either.

F is stored as a flat 4-vector: [F11, F12, F21, F22] (row-major).
"""
import numpy as np


def _det2(F):
    """det of 2x2 matrix stored as flat [F11, F12, F21, F22]."""
    return F[0] * F[3] - F[1] * F[2]


def _cofactor2(F):
    """Cofactor matrix of 2x2, returned as flat 4-vector (same layout as F)."""
    return np.array([F[3], -F[2], -F[1], F[0]], dtype=float)


def psi(F, mu, lam):
    """SNH energy. F is flat (4,)."""
    Ic = float(np.dot(F, F))
    J = _det2(F)
    alpha = 1.0 + mu / lam
    return 0.5 * (mu * (Ic - 2.0) + lam * (J - alpha) ** 2)


def pk1(F, mu, lam):
    """First Piola-Kirchhoff stress dPsi/dF, returned as flat (4,)."""
    J = _det2(F)
    alpha = 1.0 + mu / lam
    cof = _cofactor2(F)
    return mu * F + lam * (J - alpha) * cof


def hessian(F, mu, lam):
    """Analytic Hessian d^2 Psi / dF^2, returned as (4,4).

    H = mu * I_4  +  lam * cof(F) cof(F)^T  +  lam*(J - alpha) * d^2J/dF^2

    d^2J/dFij dFkl for a 2x2 matrix has exactly 4 nonzero entries:
        (0,3) = +1,  (3,0) = +1,  (1,2) = -1,  (2,1) = -1
    corresponding to d^2(F11*F22 - F12*F21).
    """
    J = _det2(F)
    alpha = 1.0 + mu / lam
    cof = _cofactor2(F)
    s = lam * (J - alpha)

    H = mu * np.eye(4)
    H += lam * np.outer(cof, cof)

    # d^2 det / dF^2  (only 4 nonzero entries)
    H[0, 3] += s
    H[3, 0] += s
    H[1, 2] -= s
    H[2, 1] -= s

    return H


def hessian_spd(F, mu, lam):
    """SPD-projected Hessian (clamp negative eigenvalues to zero)."""
    H = hessian(F, mu, lam)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ── Bower (isochoric Neo-Hookean) ─────────────────────────────────────

_BOWER_J_FLOOR = 1e-6


def _bower_safe_J(F):
    """Return (J, J_safe) where J_safe >= _BOWER_J_FLOOR.

    Joint impulses bypass the energy and can push det(F) negative.
    The floor keeps the 1/J terms finite; the huge resulting force
    pulls det(F) back toward 1 at the next backward Euler step.
    """
    J = _det2(F)
    return J, max(J, _BOWER_J_FLOOR)


def psi_bower(F, mu, lam):
    """Bower isochoric energy. F is flat (4,).

    Psi = (mu/2)(Ic/J - 2) + (lam/2)(J - 1)^2
    """
    Ic = float(np.dot(F, F))
    J, Js = _bower_safe_J(F)
    return 0.5 * (mu * (Ic / Js - 2.0) + lam * (J - 1.0) ** 2)


def pk1_bower(F, mu, lam):
    """PK1 stress for Bower energy, returned as flat (4,).

    P = mu*F/J + (lam*(J-1) - mu*Ic/(2*J^2)) * cof(F)
    """
    Ic = float(np.dot(F, F))
    J, Js = _bower_safe_J(F)
    cof = _cofactor2(F)
    return mu * F / Js + (lam * (J - 1.0) - mu * Ic / (2.0 * Js ** 2)) * cof


def hessian_bower(F, mu, lam):
    """Analytic Hessian for Bower energy, returned as (4,4).

    H = (mu/J)*I_4
      - (mu/J^2)*(F x cof + cof x F)
      + (mu*Ic/J^3 + lam) * cof x cof
      + (lam*(J-1) - mu*Ic/(2*J^2)) * d^2J/dF^2
    """
    Ic = float(np.dot(F, F))
    J, Js = _bower_safe_J(F)
    cof = _cofactor2(F)

    H = (mu / Js) * np.eye(4)
    H -= (mu / Js ** 2) * (np.outer(F, cof) + np.outer(cof, F))
    H += (mu * Ic / Js ** 3 + lam) * np.outer(cof, cof)

    s = lam * (J - 1.0) - mu * Ic / (2.0 * Js ** 2)
    H[0, 3] += s
    H[3, 0] += s
    H[1, 2] -= s
    H[2, 1] -= s

    return H


def hessian_spd_bower(F, mu, lam):
    """SPD-projected Hessian for Bower energy."""
    H = hessian_bower(F, mu, lam)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def lame_from_k(k, nu=0.3):
    """Convert a 1D-style stiffness k and Poisson ratio nu to 2D Lame parameters.

    We define k so that uniaxial stretch F=diag(1,s) gives d^2 Psi/ds^2 = k
    at s=1, matching si1d's convention.  For SNH with F=diag(1,s):
        Ic = 1 + s^2,  J = s
        Psi = 0.5*[mu*(s^2 - 1) + lam*(s - alpha)^2]
        d^2 Psi/ds^2 = mu + lam

    So k = mu + lam.  With lam = 2*mu*nu / (1 - 2*nu):
        k = mu * (1 + 2*nu/(1-2*nu)) = mu / (1-2*nu)
        => mu = k * (1 - 2*nu)
        => lam = 2 * k * nu

    For nu=0.3: mu = 0.4*k, lam = 0.6*k.

    Note: nu must be > 0 because lam=0 makes alpha = 1 + mu/lam singular.
    For nearly compressible materials, use a small nu (e.g. 0.01).
    """
    if nu <= 0.0 or nu >= 0.5:
        raise ValueError(f"Poisson ratio must be in (0, 0.5), got {nu}")
    mu_l = k * (1.0 - 2.0 * nu)
    lam_l = 2.0 * k * nu
    return mu_l, lam_l
