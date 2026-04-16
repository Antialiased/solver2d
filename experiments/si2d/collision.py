"""Collision detection and contact Jacobians for 2D affine ellipse bodies.

Floor contact (half-plane at y=0): closed-form.
Ellipse-ellipse: non-iterative analytic via frame transform (Mueller's
oriented particles approach — transform one ellipse to a circle, then
support-function evaluation gives the contact in closed form).
"""
import numpy as np


def _support_point(body, d):
    """Support point of body's ellipse in direction d (unit vector).

    The body occupies {c + F*n : |n| <= r0}. The support point in direction d is:
        c + r0 * F * (F^T d) / |F^T d|
    """
    FTd = np.array([body.F[0] * d[0] + body.F[2] * d[1],
                     body.F[1] * d[0] + body.F[3] * d[1]])
    norm = np.linalg.norm(FTd)
    if norm < 1e-15:
        return body.c.copy()
    Fn = np.array([body.F[0] * FTd[0] + body.F[1] * FTd[1],
                    body.F[2] * FTd[0] + body.F[3] * FTd[1]])
    return body.c + body.r0 * Fn / norm


def _inv2(F):
    """Inverse of a 2x2 matrix stored as flat (4,)."""
    det = F[0] * F[3] - F[1] * F[2]
    if abs(det) < 1e-30:
        return np.array([1.0, 0.0, 0.0, 1.0])
    inv_det = 1.0 / det
    return np.array([F[3] * inv_det, -F[1] * inv_det,
                     -F[2] * inv_det, F[0] * inv_det])


def _mat_mul2(A, B):
    """Multiply two 2x2 matrices stored as flat (4,)."""
    return np.array([A[0] * B[0] + A[1] * B[2],
                     A[0] * B[1] + A[1] * B[3],
                     A[2] * B[0] + A[3] * B[2],
                     A[2] * B[1] + A[3] * B[3]])


def _mat_vec2(F, v):
    """Multiply 2x2 matrix (flat 4,) by 2-vector."""
    return np.array([F[0] * v[0] + F[1] * v[1],
                     F[2] * v[0] + F[3] * v[1]])


# ── Floor contact (half-plane y = 0) ─────────────────────────────────────

def floor_gap(body):
    """Gap between body's ellipse and the floor at y=0.

    gap = cy - r0 * |F^T * ey|  where ey = (0,1), F^T*ey = (F21, F22).
    """
    L = np.sqrt(body.F[2] ** 2 + body.F[3] ** 2)
    return body.c[1] - body.r0 * L


def floor_jacobian(body):
    """Jacobian d(gap)/d(DoFs) for floor contact, returned as (6,).

    J = [0, 1, 0, 0, -r0*F21/L, -r0*F22/L]
    """
    L = np.sqrt(body.F[2] ** 2 + body.F[3] ** 2)
    J = np.zeros(6)
    J[1] = 1.0
    if L > 1e-15:
        J[4] = -body.r0 * body.F[2] / L
        J[5] = -body.r0 * body.F[3] / L
    return J


def floor_contact_point(body):
    """World-space contact point on the body for floor contact.

    p = c - r0 * F * (F^T * ey) / |F^T * ey|
    """
    FTey = np.array([body.F[2], body.F[3]])
    L = np.linalg.norm(FTey)
    if L < 1e-15:
        return body.c.copy()
    n_local = FTey / L
    Fn = np.array([body.F[0] * n_local[0] + body.F[1] * n_local[1],
                    body.F[2] * n_local[0] + body.F[3] * n_local[1]])
    return body.c - body.r0 * Fn


# ── Ellipse-ellipse contact ──────────────────────────────────────────────

def ellipse_ellipse(bodyA, bodyB):
    """Non-iterative closest points between two ellipses.

    Transform to B's frame (B becomes a circle), evaluate A's support
    function toward B's center, project onto B's circle, transform back.

    Returns (gap, normal, pA, pB) where:
        gap: signed distance (positive = separated)
        normal: unit vector from A toward B
        pA: closest point on A's boundary
        pB: closest point on B's boundary

    Returns None if bodies are concentric (degenerate).
    """
    d_AB = bodyB.c - bodyA.c
    dist = np.linalg.norm(d_AB)
    if dist < 1e-15:
        return None

    # Transform to B's frame: B becomes circle of radius rB at origin
    FB_inv = _inv2(bodyB.F)
    c_prime = _mat_vec2(FB_inv, bodyA.c - bodyB.c)
    F_prime = _mat_mul2(FB_inv, bodyA.F)

    c_prime_norm = np.linalg.norm(c_prime)
    if c_prime_norm < 1e-15:
        return None
    d_prime = c_prime / c_prime_norm

    # Support point of A' in direction -d' (toward B's center)
    neg_d = -d_prime
    FTd = np.array([F_prime[0] * neg_d[0] + F_prime[2] * neg_d[1],
                     F_prime[1] * neg_d[0] + F_prime[3] * neg_d[1]])
    FTd_norm = np.linalg.norm(FTd)
    if FTd_norm < 1e-15:
        pA_prime = c_prime.copy()
    else:
        Fn = np.array([F_prime[0] * FTd[0] + F_prime[1] * FTd[1],
                        F_prime[2] * FTd[0] + F_prime[3] * FTd[1]])
        pA_prime = c_prime + bodyA.r0 * Fn / FTd_norm

    # Closest point on B's circle (radius rB) toward pA'
    pA_prime_norm = np.linalg.norm(pA_prime)
    if pA_prime_norm < 1e-15:
        pB_prime = bodyB.r0 * d_prime
    else:
        pB_prime = bodyB.r0 * pA_prime / pA_prime_norm

    inside = pA_prime_norm < bodyB.r0

    # Transform contact points back to world frame
    pA = _mat_vec2(bodyB.F, pA_prime) + bodyB.c
    pB = _mat_vec2(bodyB.F, pB_prime) + bodyB.c

    # Normal and gap in world frame
    normal_raw = pB - pA
    normal_len = np.linalg.norm(normal_raw)
    if normal_len < 1e-15:
        normal = d_AB / dist
    else:
        normal = normal_raw / normal_len
        # Ensure normal is consistent with center-to-center direction
        if np.dot(normal, d_AB) < 0:
            normal = -normal

    gap = normal_len if not inside else -normal_len

    return gap, normal, pA, pB


def ellipse_ellipse_jacobian(bodyA, bodyB, normal, pA, pB):
    """Contact Jacobians for ellipse-ellipse contact.

    For body i with contact point pi = ci + Fi * ni_local:
        J_i = d(gap)/d(DoFs_i)

    gap = (pB - pA) . n, so:
        J_A = -n^T * [I_2 | nA_local^T kron I_2]   (1x6)
        J_B = +n^T * [I_2 | nB_local^T kron I_2]   (1x6)

    Returns (J_A, J_B), each shape (6,).
    """
    n = normal

    def _body_jacobian(body, sign):
        # Reference-frame contact point: the point on the body's boundary
        # at the contact. For A (sign=-1), contact is toward B: direction +n.
        # For B (sign=+1), contact is toward A: direction -n.
        contact_dir = -sign * n
        FTd = np.array([body.F[0] * contact_dir[0] + body.F[2] * contact_dir[1],
                         body.F[1] * contact_dir[0] + body.F[3] * contact_dir[1]])
        norm = np.linalg.norm(FTd)
        if norm < 1e-15:
            return np.zeros(6)
        u_local = FTd / norm * body.r0

        J = np.zeros(6)
        J[0] = sign * n[0]
        J[1] = sign * n[1]
        # d(gap)/d(F_kl) = sign * n_k * u_local_l
        # u_local points TOWARD the other body, so sign*n_k*u_local_l
        # gives compression (negative F change) for positive impulse.
        J[2] = sign * n[0] * u_local[0]
        J[3] = sign * n[0] * u_local[1]
        J[4] = sign * n[1] * u_local[0]
        J[5] = sign * n[1] * u_local[1]
        return J

    J_A = _body_jacobian(bodyA, -1.0)
    J_B = _body_jacobian(bodyB, +1.0)
    return J_A, J_B


# ── Tangent Jacobians (for friction) ────────────────────────────────────

def _contact_point_local(body, contact_dir):
    """Reference-frame contact point on body's boundary toward contact_dir."""
    FTd = np.array([body.F[0] * contact_dir[0] + body.F[2] * contact_dir[1],
                     body.F[1] * contact_dir[0] + body.F[3] * contact_dir[1]])
    norm = np.linalg.norm(FTd)
    if norm < 1e-15:
        return np.zeros(2)
    return body.r0 * FTd / norm


def tangent_jacobian(body, normal, sign):
    """Tangential Jacobian for a body at a contact.

    Same structure as the normal Jacobian but with tangent t = [-ny, nx].
    sign: +1 for floor body, -1 for pair body A, +1 for pair body B.
    Returns Jt (6,).
    """
    t = np.array([-normal[1], normal[0]])
    contact_dir = -sign * normal
    u_local = _contact_point_local(body, contact_dir)

    Jt = np.zeros(6)
    Jt[0] = sign * t[0]
    Jt[1] = sign * t[1]
    Jt[2] = sign * t[0] * u_local[0]
    Jt[3] = sign * t[0] * u_local[1]
    Jt[4] = sign * t[1] * u_local[0]
    Jt[5] = sign * t[1] * u_local[1]
    return Jt
