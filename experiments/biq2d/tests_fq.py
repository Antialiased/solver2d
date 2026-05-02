"""Free-flight tests for the full-quadratic body (BodyFQ2D, 12 DoFs).

Mirrors the M0 pattern in tests.py but exercises the new banana DoFs
(Hx, Hy) in addition to the existing (c, F, G) modes.  No joints, no
contacts — just per-body BE integration.
"""
import numpy as np
from .body import BodyFQ2D
from .solver import State, Params, step


def _check_state(body, label):
    assert np.all(np.isfinite(body.c)),   f"{label}: c not finite"
    assert np.all(np.isfinite(body.F)),   f"{label}: F not finite"
    assert np.all(np.isfinite(body.G)),   f"{label}: G not finite"
    assert np.all(np.isfinite(body.Hx)),  f"{label}: Hx not finite"
    assert np.all(np.isfinite(body.Hy)),  f"{label}: Hy not finite"
    assert np.all(np.isfinite(body.vc)),  f"{label}: vc not finite"
    assert np.all(np.isfinite(body.vF)),  f"{label}: vF not finite"
    assert np.all(np.isfinite(body.vG)),  f"{label}: vG not finite"
    assert np.all(np.isfinite(body.vHx)), f"{label}: vHx not finite"
    assert np.all(np.isfinite(body.vHy)), f"{label}: vHy not finite"
    md = body.min_det_J_sampled(grid_n=9)
    assert md > 0.0, f"{label}: sampled min det J = {md} (≤ 0)"


# ──────────────────────────────────────────────────────────────────────
# M0a-full: pure translation
# ──────────────────────────────────────────────────────────────────────

def test_fq_pure_translation():
    print("=== FQ-a: pure rigid translation ===")
    b = BodyFQ2D(mass=1.0, half_extent=1.0, k=1000.0, nu=0.3)
    b.vc = np.array([1.0, 0.0])

    state = State(bodies=[b])
    params = Params(gravity=np.array([0.0, 0.0]), dt=1.0 / 240.0)

    F0, G0, Hx0, Hy0 = b.F.copy(), b.G.copy(), b.Hx.copy(), b.Hy.copy()

    n_steps = 480
    for i in range(n_steps):
        step(state, params)
        _check_state(b, f"FQ-a step {i}")

    expected_x = n_steps * params.dt
    print(f"  c         = {b.c}")
    print(f"  F drift   = {np.max(np.abs(b.F - F0)):.3e}")
    print(f"  G drift   = {np.max(np.abs(b.G - G0)):.3e}")
    print(f"  Hx drift  = {np.max(np.abs(b.Hx - Hx0)):.3e}")
    print(f"  Hy drift  = {np.max(np.abs(b.Hy - Hy0)):.3e}")

    assert abs(b.c[0] - expected_x) < 1e-9
    assert abs(b.c[1]) < 1e-9
    assert np.max(np.abs(b.F  - F0))  < 1e-10
    assert np.max(np.abs(b.G  - G0))  < 1e-10
    assert np.max(np.abs(b.Hx - Hx0)) < 1e-10
    assert np.max(np.abs(b.Hy - Hy0)) < 1e-10
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────
# FQ-b: Hx banana oscillation (damped)
# ──────────────────────────────────────────────────────────────────────

def test_fq_Hx_oscillation():
    print("\n=== FQ-b: Hx banana oscillation ===")
    b = BodyFQ2D(mass=1.0, half_extent=1.0, k=1000.0, nu=0.3)
    b.Hx = np.array([0.0, 0.05])          # vertical banana along ξ₁

    state = State(bodies=[b])
    params = Params(gravity=np.array([0.0, 0.0]), dt=1.0 / 240.0)

    E0 = b.total_energy(params.gravity)
    Hx0 = b.Hx.copy()
    print(f"  E0 = {E0:.6f},  |Hx0| = {np.linalg.norm(Hx0):.4f}")

    n_steps = 1200
    for i in range(n_steps):
        step(state, params)
        _check_state(b, f"FQ-b step {i}")

    E_final = b.total_energy(params.gravity)
    print(f"  c (≈ 0)       = {b.c}")
    print(f"  F (≈ I)       = {b.F}")
    print(f"  G (≈ 0)       = {b.G}")
    print(f"  Hx_final      = {b.Hx}   (|Hx| = {np.linalg.norm(b.Hx):.4e})")
    print(f"  Hy (≈ 0)      = {b.Hy}")
    print(f"  E_final / E0  = {E_final / E0:.4f}")

    assert np.linalg.norm(b.c) < 1e-7
    assert np.max(np.abs(b.F - np.array([1., 0., 0., 1.]))) < 1e-7
    assert np.linalg.norm(b.G)  < 1e-6
    assert np.linalg.norm(b.Hy) < 1e-6
    assert np.linalg.norm(b.Hx) < np.linalg.norm(Hx0), "|Hx| should decrease under BE damping"
    assert E_final <= E0 * 1.001, f"energy grew: ratio {E_final/E0:.4f}"
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────
# FQ-c: Hy banana oscillation (damped, mirror of FQ-b)
# ──────────────────────────────────────────────────────────────────────

def test_fq_Hy_oscillation():
    print("\n=== FQ-c: Hy banana oscillation ===")
    b = BodyFQ2D(mass=1.0, half_extent=1.0, k=1000.0, nu=0.3)
    b.Hy = np.array([0.05, 0.0])          # horizontal banana along ξ₂

    state = State(bodies=[b])
    params = Params(gravity=np.array([0.0, 0.0]), dt=1.0 / 240.0)

    E0 = b.total_energy(params.gravity)
    Hy0 = b.Hy.copy()

    n_steps = 1200
    for i in range(n_steps):
        step(state, params)
        _check_state(b, f"FQ-c step {i}")

    print(f"  Hy_final = {b.Hy}   (|Hy| = {np.linalg.norm(b.Hy):.4e})")
    print(f"  Hx (≈ 0) = {b.Hx}")

    assert np.linalg.norm(b.c) < 1e-7
    assert np.max(np.abs(b.F - np.array([1., 0., 0., 1.]))) < 1e-7
    assert np.linalg.norm(b.G)  < 1e-6
    assert np.linalg.norm(b.Hx) < 1e-6
    assert np.linalg.norm(b.Hy) < np.linalg.norm(Hy0)
    assert b.total_energy(params.gravity) <= E0 * 1.001
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────
# FQ-d: rigid rotation — no phantom force on any elastic DoF
# ──────────────────────────────────────────────────────────────────────

def test_fq_rotation_no_phantom():
    """Body initialised with F = I and angular velocity ω·skew should spin as
    a rigid body.  ARAP is rotation-invariant, so Hx, Hy, G must stay ≈ 0."""
    print("\n=== FQ-d: pure rotation — no phantom banana forces ===")
    b = BodyFQ2D(mass=1.0, half_extent=1.0, k=1000.0, nu=0.3)
    omega = 2.0 * np.pi / 3.0     # one rotation in 3 s
    # vF so that Ḟ = ω · [[0, -1], [1, 0]] · F₀ = ω · skew (with F = I).
    b.vF = omega * np.array([0.0, -1.0, 1.0, 0.0])

    state = State(bodies=[b])
    params = Params(gravity=np.array([0.0, 0.0]), dt=1.0 / 240.0)

    n_steps = 720    # 3 seconds
    max_G  = 0.0
    max_Hx = 0.0
    max_Hy = 0.0

    for i in range(n_steps):
        step(state, params)
        _check_state(b, f"FQ-d step {i}")
        max_G  = max(max_G,  np.max(np.abs(b.G)))
        max_Hx = max(max_Hx, np.max(np.abs(b.Hx)))
        max_Hy = max(max_Hy, np.max(np.abs(b.Hy)))

    print(f"  max |G|  = {max_G:.3e}")
    print(f"  max |Hx| = {max_Hx:.3e}")
    print(f"  max |Hy| = {max_Hy:.3e}")

    # ARAP doesn't penalise rotation, but BE damping of radial modes will
    # slightly shrink F over time — that's fine; just ensure no spurious
    # excitation of G or H.  Tolerance 1e-5 is loose enough to absorb
    # floating-point noise accumulated over 720 steps.
    assert max_G  < 1e-5, f"G excited by pure rotation: {max_G}"
    assert max_Hx < 1e-5, f"Hx excited by pure rotation: {max_Hx}"
    assert max_Hy < 1e-5, f"Hy excited by pure rotation: {max_Hy}"
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────
# FQ-e: combined under gravity
# ──────────────────────────────────────────────────────────────────────

def test_fq_combined_under_gravity():
    print("\n=== FQ-e: combined modes under gravity ===")
    b = BodyFQ2D(mass=1.0, half_extent=1.0, k=1000.0, nu=0.3)
    b.c = np.array([0.0, 5.0])
    b.F = np.array([1.0, 0.1, 0.0, 1.0])   # mild shear

    state = State(bodies=[b])
    params = Params(gravity=np.array([0.0, -10.0]), dt=1.0 / 240.0)

    n_steps = 1200
    min_det = np.inf
    max_F_dev = 0.0
    max_G = 0.0
    max_Hx = 0.0
    max_Hy = 0.0

    for i in range(n_steps):
        step(state, params)
        _check_state(b, f"FQ-e step {i}")
        min_det = min(min_det, b.min_det_J_sampled(grid_n=9))
        max_F_dev = max(max_F_dev, np.max(np.abs(b.F - np.array([1.,0.,0.,1.]))))
        max_G  = max(max_G,  np.max(np.abs(b.G)))
        max_Hx = max(max_Hx, np.max(np.abs(b.Hx)))
        max_Hy = max(max_Hy, np.max(np.abs(b.Hy)))

    g_y = params.gravity[1]
    expected_y = 5.0 + g_y * params.dt ** 2 * n_steps * (n_steps + 1) / 2.0
    print(f"  c_final       = {b.c}  (BE expected y = {expected_y:.4f})")
    print(f"  min det J     = {min_det:.4f}")
    print(f"  max |F - I|   = {max_F_dev:.3e}")
    print(f"  max |G|       = {max_G:.3e}")
    print(f"  max |Hx|      = {max_Hx:.3e}")
    print(f"  max |Hy|      = {max_Hy:.3e}")

    assert abs(b.c[1] - expected_y) < 1e-6
    assert min_det > 0.0
    # Pure-F perturbation should not excite G, Hx, Hy (coupling-free at rest).
    assert max_G  < 1e-5
    assert max_Hx < 1e-5
    assert max_Hy < 1e-5
    print("  PASS")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    test_fq_pure_translation()
    test_fq_Hx_oscillation()
    test_fq_Hy_oscillation()
    test_fq_rotation_no_phantom()
    test_fq_combined_under_gravity()
    print("\nAll FQ free-flight tests passed.")
