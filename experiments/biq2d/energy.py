"""ARAP + volume hyperelastic energy for 2D bilinear-quadrilateral bodies.

Body parameterisation
---------------------
Reference shape is the dimensionless square ξ ∈ [-1, 1]², physical shape is
[-h, h]² (h = half_extent).  A material point at ξ maps to world position

    x(ξ) = c + h · F · ξ + h · G · (ξ₁ ξ₂)

where F is the (constant) deformation gradient and G ∈ ℝ² is the bilinear
cross-coefficient.  This is the FEM-Q4 image, but with the four corner
positions reparameterised as (c, F, G).

The deformation gradient w.r.t. physical reference X = h·ξ is

    J(ξ) = ∂x/∂X = F + (linear contribution from G)

Specifically (with J flattened row-major as [J11, J12, J21, J22]):
    J11(ξ) = F11 + Gx · ξ₂      J12(ξ) = F12 + Gx · ξ₁
    J21(ξ) = F21 + Gy · ξ₂      J22(ξ) = F22 + Gy · ξ₁

J is independent of h: the half-extent only appears in the energy *integral*
multiplier (the body area is 4h²) and in the world-space corner positions.

Energy
------
Same per-point density as si2d:  Ψ(J) = μ Σ(σ_i − 1)² + (λ/2)(det J − 1)².
Integrated over the body via 2×2 Gauss quadrature in ξ — exact for the
polynomial-degree-≤3 part of the integrand.  ARAP itself isn't polynomial in
J because of the SVD, but 2×2 Gauss is the standard FEM-Q4 choice and matches
common practice.

Functions take F as a flat (4,), G as (2,), return arrays in matching layouts.
"""
import numpy as np


# ── ARAP + volume primitives (same as si2d/energy.py) ─────────────────

def _det2(F):
    """det of 2x2 stored as flat [F11, F12, F21, F22]."""
    return F[0] * F[3] - F[1] * F[2]


def _cofactor2(F):
    """Cofactor matrix of 2x2, returned as flat 4-vector."""
    return np.array([F[3], -F[2], -F[1], F[0]], dtype=float)


def _svd2(F):
    """Signed SVD of flat 2x2: F_mat = U @ diag(s) @ V^T, det(U)=det(V)=+1."""
    F_mat = F.reshape(2, 2)
    U, s, Vt = np.linalg.svd(F_mat)
    V = Vt.T
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1
        s[1] *= -1
    if np.linalg.det(V) < 0:
        V[:, 1] *= -1
        s[1] *= -1
    return U, s, V


def psi(F, mu, lam):
    """ARAP + volume energy density at deformation gradient F (flat 4-vector)."""
    _, s, _ = _svd2(F)
    J = s[0] * s[1]
    return mu * ((s[0] - 1.0) ** 2 + (s[1] - 1.0) ** 2) + 0.5 * lam * (J - 1.0) ** 2


def pk1(F, mu, lam):
    """First Piola–Kirchhoff stress dΨ/dF (flat 4-vector).

    P = 2μ(F - R) + λ(J - 1) cof(F)
    """
    U, _, V = _svd2(F)
    R = (U @ V.T).flatten()
    J = _det2(F)
    cof = _cofactor2(F)
    return 2.0 * mu * (F - R) + lam * (J - 1.0) * cof


def hessian_spd(F, mu, lam):
    """SPD-projected Hessian d²Ψ/dF² via analytic eigendecomposition (4×4)."""
    U, s, V = _svd2(F)
    J = s[0] * s[1]

    u1, u2 = U[:, 0], U[:, 1]
    v1, v2 = V[:, 0], V[:, 1]

    q0 = np.outer(u1, v1).flatten()
    q1 = np.outer(u2, v2).flatten()
    q_flip = (np.outer(u1, v2) - np.outer(u2, v1)).flatten() / np.sqrt(2)
    q_twist = (np.outer(u1, v2) + np.outer(u2, v1)).flatten() / np.sqrt(2)

    # Scaling block (2x2 sub-problem in the qs0/qs1 plane)
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

    H = np.zeros((4, 4))
    for lam_i, qi in [(lam_s0, qs0), (lam_s1, qs1),
                      (lam_twist, q_twist), (lam_flip, q_flip)]:
        if lam_i > 0.0:
            H += lam_i * np.outer(qi, qi)
    return H


def lame_from_k(k, nu=0.3):
    """Convert (k, nu) to Lame parameters for ARAP + volume.  See si2d/energy.py."""
    if nu < 0.0 or nu >= 0.5:
        raise ValueError(f"Poisson ratio must be in [0, 0.5), got {nu}")
    if nu == 0.0:
        return k / 2.0, 0.0
    mu_l = k * (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
    lam_l = k * nu / (1.0 - nu)
    return mu_l, lam_l


# ── Bilinear basis: J(ξ), chain rule, quadrature ─────────────────────

# 2×2 Gauss-Legendre nodes on [-1, 1]², all weights = 1 (sum = area = 4).
_g = 1.0 / np.sqrt(3.0)
GAUSS_PTS = np.array([
    (-_g, -_g),
    ( _g, -_g),
    (-_g,  _g),
    ( _g,  _g),
])

# Reference-square corners (used for det-J convexity check and barrier).
CORNERS = np.array([
    (-1.0, -1.0),
    ( 1.0, -1.0),
    (-1.0,  1.0),
    ( 1.0,  1.0),
])


def J_at(F, G, xi):
    """Deformation gradient at reference point ξ ∈ [-1, 1]², returned flat (4,).

    J11 = F11 + Gx ξ₂      J12 = F12 + Gx ξ₁
    J21 = F21 + Gy ξ₂      J22 = F22 + Gy ξ₁
    """
    x1, x2 = xi
    return np.array([
        F[0] + G[0] * x2,
        F[1] + G[0] * x1,
        F[2] + G[1] * x2,
        F[3] + G[1] * x1,
    ])


def chain_C(xi):
    """Chain-rule matrix dJ/dq at ξ, where q = (F11, F12, F21, F22, Gx, Gy).

    Shape (4, 6).  Used to lift gradients/Hessians from J-space to q-space.
    """
    x1, x2 = xi
    C = np.zeros((4, 6))
    C[0, 0] = 1.0;  C[0, 4] = x2          # J11
    C[1, 1] = 1.0;  C[1, 4] = x1          # J12
    C[2, 2] = 1.0;  C[2, 5] = x2          # J21
    C[3, 3] = 1.0;  C[3, 5] = x1          # J22
    return C


def integrated_energy(F, G, mu, lam, h):
    """∫_{[-h,h]²} Ψ(J(X)) dX via 2×2 Gauss in ξ.

    Equals h² Σ_n Ψ(J(ξ_n)) since the four Gauss weights are each 1.
    """
    total = 0.0
    for xi in GAUSS_PTS:
        total += psi(J_at(F, G, xi), mu, lam)
    return h * h * total


def integrated_grad(F, G, mu, lam, h):
    """Gradient of integrated_energy w.r.t. q = (F, G).

    Returns (6,):  h² Σ_n Cᵀ pk1(J(ξ_n)).
    """
    g = np.zeros(6)
    for xi in GAUSS_PTS:
        J = J_at(F, G, xi)
        p = pk1(J, mu, lam)
        C = chain_C(xi)
        g += C.T @ p
    return h * h * g


def integrated_hessian_spd(F, G, mu, lam, h):
    """SPD-projected Hessian of integrated_energy w.r.t. q = (F, G).

    Returns (6, 6):  h² Σ_n Cᵀ HSPD(J(ξ_n)) C.  Sum of SPD matrices is SPD.
    """
    H = np.zeros((6, 6))
    for xi in GAUSS_PTS:
        J = J_at(F, G, xi)
        H_J = hessian_spd(J, mu, lam)
        C = chain_C(xi)
        H += C.T @ H_J @ C
    return h * h * H


# ── Convexity / barrier on det J at corners ──────────────────────────

def det_J_at(F, G, xi):
    """det J(ξ) — scalar.  Linear in ξ given (F, G)."""
    return _det2(J_at(F, G, xi))


def det_J_at_corners(F, G):
    """(4,) det J values at the 4 reference-square corners."""
    return np.array([det_J_at(F, G, c) for c in CORNERS])


def barrier_energy_grad_hess(F, G, kappa, eps):
    """Soft barrier on min det J at corners.  Returns (E, grad_q, H_q_spd).

    Uses b(d) = κ · ((eps - d)/eps)² for d < eps, else 0.  Smooth at d = eps,
    polynomial growth as d → 0 (does NOT diverge — a conservative line-search
    will still occasionally cross if pushed hard, but for M0 this is fine).

    Shape: scalar, (6,), (6, 6).  H is SPD-clamped (rank-1 outer-product term
    only — d²d/dq² contribution dropped because it is not SPD in general).
    """
    E = 0.0
    grad = np.zeros(6)
    H = np.zeros((6, 6))
    inv_eps2 = 1.0 / (eps * eps)

    for xi in CORNERS:
        d = det_J_at(F, G, xi)
        if d >= eps:
            continue
        slack = eps - d
        # b(d) = κ · slack² / eps² ;  b'(d) = -2κ · slack / eps²
        # b''(d) = 2κ / eps² (constant, > 0 → SPD outer product)
        E += kappa * slack * slack * inv_eps2

        # dd/dq:  d = det F + ξ₁(F11 G2 - G1 F21) + ξ₂(G1 F22 - F12 G2)
        x1, x2 = xi
        F11, F12, F21, F22 = F
        Gx, Gy = G
        dd_dq = np.array([
            F22 + x1 * Gy,                   # d/dF11
            -F21 - x2 * Gy,                  # d/dF12
            -F12 - x1 * Gx,                  # d/dF21
            F11 + x2 * Gx,                   # d/dF22
            -x1 * F21 + x2 * F22,            # d/dGx
            x1 * F11 - x2 * F12,             # d/dGy
        ])
        bp = -2.0 * kappa * slack * inv_eps2
        bpp = 2.0 * kappa * inv_eps2
        grad += bp * dd_dq
        H += bpp * np.outer(dd_dq, dd_dq)

    return E, grad, H


# ── Full-quadratic basis: adds banana modes Hx, Hy ───────────────────
#
# Map uses mean-zero banana basis so c remains the centre of mass:
#     x(ξ) = c + h·( F·ξ + G·ξ₁ξ₂ + Hx·(ξ₁² − 1/3) + Hy·(ξ₂² − 1/3) )
#
# The constant shift does not affect ∂/∂ξ, so J(ξ) only sees the 2ξ terms:
#     J11 = F11 + Gx ξ₂ + 2 Hx0 ξ₁
#     J12 = F12 + Gx ξ₁ +             2 Hy0 ξ₂
#     J21 = F21 + Gy ξ₂ + 2 Hx1 ξ₁
#     J22 = F22 + Gy ξ₁ +             2 Hy1 ξ₂

# 3x3 grid for sampled min-det-J diagnostics (corners + edge mids + centre).
SAMPLE_3X3 = np.array([
    (sx, sy) for sy in (-1.0, 0.0, 1.0) for sx in (-1.0, 0.0, 1.0)
])


def J_at_full(F, G, Hx, Hy, xi):
    """Deformation gradient at ξ for the full-quadratic basis (flat (4,))."""
    x1, x2 = xi
    return np.array([
        F[0] + G[0] * x2 + 2.0 * Hx[0] * x1,
        F[1] + G[0] * x1 + 2.0 * Hy[0] * x2,
        F[2] + G[1] * x2 + 2.0 * Hx[1] * x1,
        F[3] + G[1] * x1 + 2.0 * Hy[1] * x2,
    ])


def chain_C_full(xi):
    """Chain-rule matrix dJ/dq at ξ, q = (F11,F12,F21,F22,Gx,Gy,Hx0,Hx1,Hy0,Hy1).

    Shape (4, 10).
    """
    x1, x2 = xi
    C = np.zeros((4, 10))
    # J11 = F11 + Gx·ξ₂ + 2·Hx0·ξ₁
    C[0, 0] = 1.0
    C[0, 4] = x2
    C[0, 6] = 2.0 * x1
    # J12 = F12 + Gx·ξ₁ + 2·Hy0·ξ₂
    C[1, 1] = 1.0
    C[1, 4] = x1
    C[1, 8] = 2.0 * x2
    # J21 = F21 + Gy·ξ₂ + 2·Hx1·ξ₁
    C[2, 2] = 1.0
    C[2, 5] = x2
    C[2, 7] = 2.0 * x1
    # J22 = F22 + Gy·ξ₁ + 2·Hy1·ξ₂
    C[3, 3] = 1.0
    C[3, 5] = x1
    C[3, 9] = 2.0 * x2
    return C


def integrated_energy_full(F, G, Hx, Hy, mu, lam, h):
    """∫_{[-h,h]²} Ψ(J(X)) dX via 2×2 Gauss for the full-quadratic basis."""
    total = 0.0
    for xi in GAUSS_PTS:
        total += psi(J_at_full(F, G, Hx, Hy, xi), mu, lam)
    return h * h * total


def integrated_grad_full(F, G, Hx, Hy, mu, lam, h):
    """Gradient of integrated_energy_full w.r.t. q = (F, G, Hx, Hy).  Returns (10,)."""
    g = np.zeros(10)
    for xi in GAUSS_PTS:
        J = J_at_full(F, G, Hx, Hy, xi)
        p = pk1(J, mu, lam)
        C = chain_C_full(xi)
        g += C.T @ p
    return h * h * g


def integrated_hessian_spd_full(F, G, Hx, Hy, mu, lam, h):
    """SPD-projected Hessian of integrated_energy_full.  Returns (10, 10)."""
    H = np.zeros((10, 10))
    for xi in GAUSS_PTS:
        J = J_at_full(F, G, Hx, Hy, xi)
        H_J = hessian_spd(J, mu, lam)
        C = chain_C_full(xi)
        H += C.T @ H_J @ C
    return h * h * H


def det_J_at_full(F, G, Hx, Hy, xi):
    """det J(ξ) for the full-quadratic basis — quartic in ξ."""
    return _det2(J_at_full(F, G, Hx, Hy, xi))


def min_det_J_sampled_full(F, G, Hx, Hy, grid_n=9):
    """Cheap min-det-J diagnostic: sample an n×n ξ-grid and take the min."""
    xs = np.linspace(-1.0, 1.0, grid_n)
    m = np.inf
    for x2 in xs:
        for x1 in xs:
            d = det_J_at_full(F, G, Hx, Hy, (x1, x2))
            if d < m:
                m = d
    return float(m)
