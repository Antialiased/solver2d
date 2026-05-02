"""ARAP + volume hyperelastic energy for 2D affine bodies.

Following the Dynamic Deformables / HOBAK approach (Kim & Ebers, 2020):
analytic Hessian eigendecomposition via twist/flip modes avoids
differentiating through the SVD.

Energy:
    Psi(F) = mu * sum_i (sigma_i - 1)^2  +  (lam/2) * (J - 1)^2

where sigma_i are the (signed) singular values and J = det(F) = sigma_1 * sigma_2.

The ARAP term penalises deviation from a rotation (||F - R||^2_F).
The volume term penalises area change ((det(F) - 1)^2).

F is stored as a flat 4-vector: [F11, F12, F21, F22] (row-major).
"""
import numpy as np


def _det2(F):
    """det of 2x2 matrix stored as flat [F11, F12, F21, F22]."""
    return F[0] * F[3] - F[1] * F[2]


def _cofactor2(F):
    """Cofactor matrix of 2x2, returned as flat 4-vector (same layout as F)."""
    return np.array([F[3], -F[2], -F[1], F[0]], dtype=float)


def _svd2(F):
    """Signed SVD of 2x2 matrix from flat F.

    Returns (U, sigma, V) such that F_mat = U @ diag(sigma) @ V^T,
    where det(U) = det(V) = 1 (proper rotations).
    sigma[1] may be negative when det(F) < 0 (inverted element).
    """
    F_mat = F.reshape(2, 2)
    U, s, Vt = np.linalg.svd(F_mat)
    V = Vt.T

    # Ensure both U and V are proper rotations (det = +1).
    # numpy SVD returns s >= 0 always; to get signed singular values
    # when det(F) < 0, we flip the last column of U (or V) and negate s[1].
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1
        s[1] *= -1
    if np.linalg.det(V) < 0:
        V[:, 1] *= -1
        s[1] *= -1

    return U, s, V


# ── Energy, PK1, Hessian ────────────────────────────────────────────────

def psi(F, mu, lam):
    """ARAP + volume energy.

    Psi = mu * [(s1 - 1)^2 + (s2 - 1)^2]  +  (lam/2) * (J - 1)^2
    """
    _, s, _ = _svd2(F)
    J = s[0] * s[1]
    return mu * ((s[0] - 1.0) ** 2 + (s[1] - 1.0) ** 2) + 0.5 * lam * (J - 1.0) ** 2


def pk1(F, mu, lam):
    """First Piola-Kirchhoff stress dPsi/dF, returned as flat (4,).

    P = 2*mu*(F - R) + lam*(J - 1)*cof(F)

    where R = U @ V^T is the closest rotation to F.
    """
    U, _, V = _svd2(F)
    R = (U @ V.T).flatten()
    J = _det2(F)
    cof = _cofactor2(F)
    return 2.0 * mu * (F - R) + lam * (J - 1.0) * cof


def hessian(F, mu, lam):
    """Analytic Hessian d^2 Psi / dF^2, returned as (4,4).

    Uses the Dynamic Deformables twist/flip eigendecomposition.
    The 4x4 Hessian decomposes into:
      - 2x2 scaling block (stretches along principal directions)
      - twist mode (relative rotation of U vs V)
      - flip mode (reflection)

    Each has an analytic eigenvalue; eigenvectors are built from U, V.
    """
    U, s, V = _svd2(F)
    J = s[0] * s[1]

    u1, u2 = U[:, 0], U[:, 1]
    v1, v2 = V[:, 0], V[:, 1]

    # Eigenvector matrices (flattened to 4-vectors, row-major)
    q0 = np.outer(u1, v1).flatten()   # scaling mode 0
    q1 = np.outer(u2, v2).flatten()   # scaling mode 1
    # Off-diagonal modes: antisymmetric gets the "flip" eigenvalue,
    # symmetric gets the "twist" eigenvalue.
    q_flip = (np.outer(u1, v2) - np.outer(u2, v1)).flatten() / np.sqrt(2)
    q_twist = (np.outer(u1, v2) + np.outer(u2, v1)).flatten() / np.sqrt(2)

    # --- Scaling block eigenvalues (2x2 sub-problem) ---
    # ARAP contribution: diag(2mu, 2mu)
    # Volume contribution: [[lam*s2^2, lam*(2*s1*s2 - 1)],
    #                        [lam*(2*s1*s2 - 1), lam*s1^2]]
    A00 = 2.0 * mu + lam * s[1] ** 2
    A11 = 2.0 * mu + lam * s[0] ** 2
    A01 = lam * (2.0 * s[0] * s[1] - 1.0)

    tr = A00 + A11
    det_A = A00 * A11 - A01 ** 2
    disc = max(0.0, tr ** 2 - 4.0 * det_A)
    sqrt_disc = np.sqrt(disc)
    lam_s0 = (tr + sqrt_disc) / 2.0
    lam_s1 = (tr - sqrt_disc) / 2.0

    # Eigenvectors of scaling sub-block
    if abs(A01) > 1e-12:
        e0 = np.array([A01, lam_s0 - A00])
        e0 /= np.linalg.norm(e0)
        e1 = np.array([A01, lam_s1 - A00])
        e1 /= np.linalg.norm(e1)
    else:
        e0 = np.array([1.0, 0.0])
        e1 = np.array([0.0, 1.0])
        # Correct ordering when diagonal
        if A00 < A11:
            lam_s0, lam_s1 = A11, A00
            e0, e1 = e1, e0

    qs0 = e0[0] * q0 + e0[1] * q1
    qs1 = e1[0] * q0 + e1[1] * q1

    # --- Twist eigenvalue (symmetric eigenvector) ---
    # ARAP: 2mu,  Volume: -lam*(J - 1)
    lam_twist = 2.0 * mu - lam * (J - 1.0)

    # --- Flip eigenvalue (antisymmetric eigenvector) ---
    # ARAP: 2mu*(s1 + s2 - 2)/(s1 + s2),  Volume: lam*(J - 1)
    s_sum = s[0] + s[1]
    if abs(s_sum) > 1e-10:
        lam_flip = 2.0 * mu * (s_sum - 2.0) / s_sum + lam * (J - 1.0)
    else:
        lam_flip = -2.0 * mu + lam * (J - 1.0)

    # Reconstruct full Hessian
    H = (lam_s0 * np.outer(qs0, qs0) + lam_s1 * np.outer(qs1, qs1)
         + lam_twist * np.outer(q_twist, q_twist)
         + lam_flip * np.outer(q_flip, q_flip))

    return H


def hessian_spd(F, mu, lam):
    """SPD-projected Hessian (clamp negative eigenvalues to zero).

    Uses the analytic eigendecomposition — no numerical eigensolver needed.
    """
    U, s, V = _svd2(F)
    J = s[0] * s[1]

    u1, u2 = U[:, 0], U[:, 1]
    v1, v2 = V[:, 0], V[:, 1]

    q0 = np.outer(u1, v1).flatten()
    q1 = np.outer(u2, v2).flatten()
    q_flip = (np.outer(u1, v2) - np.outer(u2, v1)).flatten() / np.sqrt(2)
    q_twist = (np.outer(u1, v2) + np.outer(u2, v1)).flatten() / np.sqrt(2)

    # Scaling block
    A00 = 2.0 * mu + lam * s[1] ** 2
    A11 = 2.0 * mu + lam * s[0] ** 2
    A01 = lam * (2.0 * s[0] * s[1] - 1.0)

    tr = A00 + A11
    det_A = A00 * A11 - A01 ** 2
    disc = max(0.0, tr ** 2 - 4.0 * det_A)
    sqrt_disc = np.sqrt(disc)
    lam_s0 = (tr + sqrt_disc) / 2.0
    lam_s1 = (tr - sqrt_disc) / 2.0

    if abs(A01) > 1e-12:
        e0 = np.array([A01, lam_s0 - A00])
        e0 /= np.linalg.norm(e0)
        e1 = np.array([A01, lam_s1 - A00])
        e1 /= np.linalg.norm(e1)
    else:
        e0 = np.array([1.0, 0.0])
        e1 = np.array([0.0, 1.0])
        if A00 < A11:
            lam_s0, lam_s1 = A11, A00
            e0, e1 = e1, e0

    qs0 = e0[0] * q0 + e0[1] * q1
    qs1 = e1[0] * q0 + e1[1] * q1

    lam_twist = 2.0 * mu - lam * (J - 1.0)

    s_sum = s[0] + s[1]
    if abs(s_sum) > 1e-10:
        lam_flip = 2.0 * mu * (s_sum - 2.0) / s_sum + lam * (J - 1.0)
    else:
        lam_flip = -2.0 * mu + lam * (J - 1.0)

    # Reconstruct with clamped eigenvalues
    H = np.zeros((4, 4))
    for lam_i, qi in [(lam_s0, qs0), (lam_s1, qs1),
                      (lam_twist, q_twist), (lam_flip, q_flip)]:
        if lam_i > 0.0:
            H += lam_i * np.outer(qi, qi)

    return H


def lame_from_k(k, nu=0.3):
    """Convert stiffness k and Poisson ratio nu to Lame parameters for ARAP + volume.

    For ARAP + volume with F = diag(1, s):
        Psi = mu*(s - 1)^2 + (lam/2)*(s - 1)^2
        d^2 Psi / ds^2 = 2*mu + lam = k

    With lam = 2*mu*nu / (1 - 2*nu):
        k = 2*mu + 2*mu*nu/(1-2*nu) = 2*mu*(1-nu)/(1-2*nu)
        => mu = k*(1-2*nu) / (2*(1-nu))
        => lam = k*nu / (1-nu)

    nu = 0: no volume penalty (lam = 0), pure ARAP with mu = k/2.
    nu -> 0.5: incompressible limit (lam -> inf).

    For nu = 0.3: mu ≈ 0.286*k, lam ≈ 0.429*k.
    """
    if nu < 0.0 or nu >= 0.5:
        raise ValueError(f"Poisson ratio must be in [0, 0.5), got {nu}")
    if nu == 0.0:
        return k / 2.0, 0.0
    mu_l = k * (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
    lam_l = k * nu / (1.0 - nu)
    return mu_l, lam_l
