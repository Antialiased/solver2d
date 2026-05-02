"""M0 milestone tests for biq2d prototype.

M0a: pure rigid translation — verify the c DoF integrates correctly while
     F and G stay quiescent.
M0b: pure G oscillation — verify bilinear DoFs participate in the elastic
     energy and BE damps them toward G = 0.
M0c: combined modes under gravity — verify nothing diverges, det J stays
     positive, c falls under gravity.
"""
import numpy as np
from .body import BodyBQ2D
from .solver import State, Params, step


def _check_state(body, label):
    """Assert no NaN/Inf and det J > 0 at all corners."""
    assert np.all(np.isfinite(body.c)),  f"{label}: c not finite: {body.c}"
    assert np.all(np.isfinite(body.F)),  f"{label}: F not finite: {body.F}"
    assert np.all(np.isfinite(body.G)),  f"{label}: G not finite: {body.G}"
    assert np.all(np.isfinite(body.vc)), f"{label}: vc not finite"
    assert np.all(np.isfinite(body.vF)), f"{label}: vF not finite"
    assert np.all(np.isfinite(body.vG)), f"{label}: vG not finite"
    md = body.min_det_J()
    assert md > 0.0, f"{label}: min det J at corners = {md} (≤ 0)"


def test_m0a_pure_translation():
    """M0a: no gravity, initial vc only.  c integrates linearly; F, G unchanged."""
    print("=== M0a: pure rigid translation ===")
    b = BodyBQ2D(mass=1.0, half_extent=1.0, k=1000.0, nu=0.3)
    b.vc = np.array([1.0, 0.0])

    state = State(bodies=[b])
    params = Params(gravity=np.array([0.0, 0.0]), dt=1.0 / 240.0)

    F0 = b.F.copy()
    G0 = b.G.copy()

    n_steps = 480     # 2 seconds
    for i in range(n_steps):
        step(state, params)
        _check_state(b, f"M0a step {i}")

    expected_x = 1.0 * (n_steps * params.dt)   # 2.0
    print(f"  c       = {b.c}    (expected ≈ ({expected_x:.3f}, 0))")
    print(f"  F drift = {np.max(np.abs(b.F - F0)):.3e}")
    print(f"  G drift = {np.max(np.abs(b.G - G0)):.3e}")

    assert abs(b.c[0] - expected_x) < 1e-9, f"x position drift: {b.c[0]:.6f} vs {expected_x}"
    assert abs(b.c[1]) < 1e-9, f"y should stay 0, got {b.c[1]}"
    assert np.max(np.abs(b.F - F0)) < 1e-10, "F should be unchanged"
    assert np.max(np.abs(b.G - G0)) < 1e-10, "G should be unchanged"
    print("  PASS")


def test_m0b_g_oscillation():
    """M0b: initial G ≠ 0, no gravity.  BE should damp G toward 0."""
    print("\n=== M0b: pure G oscillation (small amplitude) ===")
    b = BodyBQ2D(mass=1.0, half_extent=1.0, k=1000.0, nu=0.3)
    b.G = np.array([0.05, 0.0])    # small bilinear deformation

    state = State(bodies=[b])
    params = Params(gravity=np.array([0.0, 0.0]), dt=1.0 / 240.0)

    E0 = b.total_energy(params.gravity)
    G0 = b.G.copy()
    print(f"  E0 = {E0:.6f},  |G0| = {np.linalg.norm(G0):.4f}")

    n_steps = 1200    # 5 seconds
    energies = [E0]
    G_log = [G0.copy()]

    for i in range(n_steps):
        step(state, params)
        _check_state(b, f"M0b step {i}")
        if i % 100 == 99:
            energies.append(b.total_energy(params.gravity))
            G_log.append(b.G.copy())

    E_final = energies[-1]
    G_final = b.G

    print(f"  c (should ≈ 0)  = {b.c}")
    print(f"  F (should ≈ I)  = {b.F}")
    print(f"  G_final         = {G_final}    (|G| = {np.linalg.norm(G_final):.4e})")
    print(f"  E_final / E0    = {E_final / E0:.4f}    (BE damps → < 1)")

    # c and F should remain unchanged (no force on them from a G-only mode).
    assert np.linalg.norm(b.c)  < 1e-7,  f"c drifted: {b.c}"
    assert np.max(np.abs(b.F - np.array([1.,0.,0.,1.]))) < 1e-7, f"F drifted: {b.F}"
    # Energy must not grow.
    assert E_final <= E0 * 1.001, f"Energy grew under BE: ratio {E_final/E0:.4f}"
    # G should have decayed somewhat (BE damping is significant at this stiffness).
    assert np.linalg.norm(G_final) < np.linalg.norm(G0), \
        f"|G| not decreasing under BE damping"
    print("  PASS")


def test_m0c_combined_under_gravity():
    """M0c: c falls under gravity, F oscillates and damps, det J stays positive."""
    print("\n=== M0c: combined modes under gravity ===")
    b = BodyBQ2D(mass=1.0, half_extent=1.0, k=1000.0, nu=0.3)
    b.c = np.array([0.0, 5.0])
    # Mild shear in F: F = I + small off-diagonal
    b.F = np.array([1.0, 0.1, 0.0, 1.0])
    # G stays at zero initially.

    state = State(bodies=[b])
    params = Params(gravity=np.array([0.0, -10.0]), dt=1.0 / 240.0)

    n_steps = 1200    # 5 seconds (will fall ~125 m without floor — just checking BE is stable)

    min_det_seen = np.inf
    max_F_dev = 0.0
    max_G_seen = 0.0

    for i in range(n_steps):
        step(state, params)
        _check_state(b, f"M0c step {i}")
        min_det_seen = min(min_det_seen, b.min_det_J())
        max_F_dev = max(max_F_dev, np.max(np.abs(b.F - np.array([1.,0.,0.,1.]))))
        max_G_seen = max(max_G_seen, np.max(np.abs(b.G)))

    # BE on COM:  vc_n = vc_0 + n·dt·g,  c_n = c_0 + dt²·g·n(n+1)/2  (vc_0 = 0)
    g_y = params.gravity[1]
    expected_y = 5.0 + g_y * params.dt ** 2 * n_steps * (n_steps + 1) / 2.0
    print(f"  c_final          = {b.c}      (BE expected y = {expected_y:.4f})")
    print(f"  min det J seen   = {min_det_seen:.4f}")
    print(f"  max |F - I| seen = {max_F_dev:.4e}")
    print(f"  max |G| seen     = {max_G_seen:.4e}")

    # Free-fall on c (gravity-only on COM, BE-exact since g is constant).
    assert abs(b.c[1] - expected_y) < 1e-6, f"y position not BE free-fall: {b.c[1]} vs {expected_y}"
    # No floor → no contact, but min det J must stay > 0 throughout.
    assert min_det_seen > 0.0
    # Pure-F initial perturbation should not excite G (no coupling at G = 0, F near I).
    assert max_G_seen < 1e-6, f"G grew unexpectedly: {max_G_seen}"
    print("  PASS")


def test_m0d_g_near_convexity():
    """M0d: large initial G near convexity boundary.  Verify barrier keeps det J > 0.

    For G = (0.4, 0), F = I:  det J(ξ) = 1 + 0.4·ξ₂.  At ξ₂ = -1, det J = 0.6.
    For G = (0.9, 0):                     det J(ξ) = 1 + 0.9·ξ₂.  At ξ₂ = -1, det J = 0.1.
    Push G further to test the barrier.
    """
    print("\n=== M0d: G near convexity boundary (with barrier) ===")
    # Set kappa proportional to elastic stiffness so it's commensurate.
    b = BodyBQ2D(mass=1.0, half_extent=1.0, k=1000.0, nu=0.3,
                 barrier_kappa=10.0, barrier_eps=0.15)
    b.G = np.array([0.7, 0.0])

    state = State(bodies=[b])
    params = Params(gravity=np.array([0.0, 0.0]), dt=1.0 / 240.0)

    G0_norm = np.linalg.norm(b.G)
    md0 = b.min_det_J()
    print(f"  G0 = {b.G}    (|G0| = {G0_norm:.3f},  min det J = {md0:.3f})")

    n_steps = 600
    min_det_seen = md0
    for i in range(n_steps):
        step(state, params)
        _check_state(b, f"M0d step {i}")
        min_det_seen = min(min_det_seen, b.min_det_J())

    print(f"  G_final          = {b.G}    (|G| = {np.linalg.norm(b.G):.4f})")
    print(f"  min det J seen   = {min_det_seen:.4f}")
    assert min_det_seen > 0.0
    print("  PASS")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    test_m0a_pure_translation()
    test_m0b_g_oscillation()
    test_m0c_combined_under_gravity()
    test_m0d_g_near_convexity()
    print("\nAll M0 tests passed.")
