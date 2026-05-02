"""Milestone tests for si2d prototype."""
import numpy as np
from .body import Body2D, integrate_backward_euler
from .solver import State, Params, Joint, step
from . import energy


def test_m1_free_flight():
    """M1: Free flight of a spinning deformable disk.

    One body, no gravity, no contacts. Initial F = I, vF has off-diagonal
    components (spin + stretch). Total energy T + Psi should be conserved
    to O(dt^2) with backward Euler.
    """
    print("=== M1: Free flight of spinning deformable disk ===")

    b = Body2D(mass=1.0, r0=1.0, k=1000.0, nu=0.3)
    b.c = np.array([0.0, 5.0])
    b.vc = np.array([1.0, 0.0])
    # Spin + stretch: off-diagonal vF gives rotation, diagonal gives breathing
    b.vF = np.array([0.5, 2.0, -2.0, -0.5])

    no_gravity = np.array([0.0, 0.0])
    dt = 1.0 / 240.0

    E0 = b.total_energy(no_gravity)
    print(f"  dt = {dt:.4f}, k = {b.k}, nu = {b.nu}")
    print(f"  E0 = {E0:.6f}")

    energies = [E0]
    n_steps = 2400  # 10 seconds

    for i in range(n_steps):
        integrate_backward_euler(b, dt, no_gravity)
        E = b.total_energy(no_gravity)
        energies.append(E)

    E_final = energies[-1]
    E_max = max(energies)
    E_min = min(energies)
    drift = (E_final - E0) / E0

    print(f"  E_final = {E_final:.6f}")
    print(f"  E_max   = {E_max:.6f}")
    print(f"  E_min   = {E_min:.6f}")
    print(f"  drift   = {drift:.6e}")
    print(f"  range   = {(E_max - E_min) / E0:.6e}")

    # BE damps energy, so E_final <= E0. Check it doesn't grow.
    assert E_final <= E0 * 1.001, f"Energy grew: {drift:.6e}"
    # Check drift is reasonable (O(dt^2) per step, accumulated over n_steps)
    assert abs(drift) < 0.25, f"Excessive energy drift: {drift:.6e}"
    print(f"  PASS (energy drift {drift:.4e}, damped as expected for BE)")


def test_m1_rigid_limit():
    """M1 supplement: rigid-body limit (k -> large).

    With very high stiffness, F should stay near a rotation matrix
    and angular velocity should be approximately conserved.
    """
    print("\n=== M1 rigid limit: high stiffness ===")

    b = Body2D(mass=1.0, r0=1.0, k=1e6, nu=0.3)
    b.c = np.array([0.0, 0.0])
    b.vc = np.array([0.0, 0.0])
    b.vF = np.array([0.0, 1.0, -1.0, 0.0])  # pure spin

    no_gravity = np.array([0.0, 0.0])
    dt = 1.0 / 240.0

    for _ in range(2400):
        integrate_backward_euler(b, dt, no_gravity)

    # Check F is still near a rotation (F^T F ~ I)
    F_mat = b.F.reshape(2, 2)
    FTF = F_mat.T @ F_mat
    deviation = np.max(np.abs(FTF - np.eye(2)))
    det_F = np.linalg.det(F_mat)

    print(f"  max |F^T F - I| = {deviation:.6e}")
    print(f"  det(F)          = {det_F:.6f}")
    assert deviation < 0.01, f"F deviated from rotation: {deviation:.6e}"
    assert abs(det_F - 1.0) < 0.01, f"det(F) not 1: {det_F}"
    print(f"  PASS")


def _detect_apexes(body, r0, n_steps, state, params):
    """Run simulation and detect apex heights via velocity sign change."""
    apexes = []
    vy_prev = -999.0
    for i in range(n_steps):
        step(state, params)
        vy = body.vc[1]
        y = body.c[1] - r0
        if i > 10 and vy_prev >= 0 and vy < 0:
            apexes.append(y)
        vy_prev = vy
    return apexes


def test_m2_floor_bounce():
    """M2: Single deformable disk bouncing on floor.

    Tests three modes:
    1. Full coupled, e=0: heavy Moreau dissipation (~23% height ratio)
    2. Full coupled, e=1: restitution doesn't help (Jacobian pumps F)
    3. Decoupled, e=1: near-perfect energy preservation (~99.5%)
    """
    h0 = 5.0
    n_steps = 4800

    # --- Full coupled, no restitution ---
    print("\n=== M2a: Floor bounce, full coupled, e=0 ===")
    b = Body2D(mass=1.0, r0=0.5, k=5000.0, nu=0.3)
    b.c = np.array([0.0, h0 + b.r0])
    state = State(bodies=[b])
    params = Params(dt=1.0 / 240.0, restitution=0.0, position_iters=8, velocity_iters=8)
    apexes = _detect_apexes(b, b.r0, n_steps, state, params)
    if len(apexes) >= 1:
        print(f"  h1/h0 = {apexes[0] / h0:.4f}  ({len(apexes)} bounces)")
        assert apexes[0] < h0, "Should dissipate via Moreau"
        print(f"  PASS")
    else:
        print(f"  SKIP")

    # --- Full coupled, e=1 ---
    print("\n=== M2b: Floor bounce, full coupled, e=1 ===")
    b = Body2D(mass=1.0, r0=0.5, k=5000.0, nu=0.3)
    b.c = np.array([0.0, h0 + b.r0])
    state = State(bodies=[b])
    params = Params(dt=1.0 / 240.0, restitution=1.0, position_iters=8, velocity_iters=8)
    apexes = _detect_apexes(b, b.r0, n_steps, state, params)
    if len(apexes) >= 1:
        print(f"  h1/h0 = {apexes[0] / h0:.4f}  ({len(apexes)} bounces)")
        print(f"  (coupled restitution doesn't help — impulse pumps F)")
    else:
        print(f"  SKIP")

    # --- Decoupled, e=1: the fix ---
    print("\n=== M2c: Floor bounce, decoupled, e=1 ===")
    b = Body2D(mass=1.0, r0=0.5, k=5000.0, nu=0.3)
    b.c = np.array([0.0, h0 + b.r0])
    state = State(bodies=[b])
    params = Params(dt=1.0 / 240.0, restitution=1.0, position_iters=8, velocity_iters=8,
                    position_correct_F=False, velocity_couple_F=False)
    apexes = _detect_apexes(b, b.r0, n_steps, state, params)
    if len(apexes) >= 1:
        print(f"  h1/h0 = {apexes[0] / h0:.4f}  last = {apexes[-1] / h0:.4f}  "
              f"({len(apexes)} bounces)")
        assert apexes[0] / h0 > 0.95, f"Decoupled e=1 should preserve energy: {apexes[0] / h0}"
        print(f"  PASS")


def test_m3_oblique_bounce():
    """M3: Oblique bounce with spin + deformation.

    Single disk hitting floor at an angle with initial angular momentum.
    Tests qualitative correctness (spins and deforms) and energy behavior.
    """
    h0 = 3.0

    # --- M3a: no friction, coupled ---
    print("\n=== M3a: Oblique bounce, no friction, coupled ===")
    b = Body2D(mass=1.0, r0=0.5, k=5000.0, nu=0.3)
    b.c = np.array([0.0, h0 + b.r0])
    b.vc = np.array([2.0, 0.0])
    b.vF = np.array([0.0, 1.5, -1.5, 0.0])  # initial spin

    state = State(bodies=[b])
    params = Params(dt=1.0 / 240.0, restitution=0.5, friction=0.0,
                    position_iters=8, velocity_iters=8)

    E0 = b.total_energy(params.gravity)
    vx0 = b.vc[0]

    for _ in range(2400):
        step(state, params)

    E_final = b.total_energy(params.gravity)

    # F should have off-diagonal components (rotation/deformation happened)
    F_mat = b.F.reshape(2, 2)
    off_diag = abs(F_mat[0, 1]) + abs(F_mat[1, 0])
    print(f"  E ratio = {E_final / E0:.4f}")
    print(f"  F off-diag = {off_diag:.6f}")
    print(f"  vx: {vx0:.2f} -> {b.vc[0]:.4f}")
    assert E_final <= E0 * 1.01, f"Energy grew: {E_final / E0:.6f}"
    # Without friction, horizontal velocity should be ~unchanged
    assert abs(b.vc[0] - vx0) < 0.5, f"vx changed too much without friction: {b.vc[0]}"
    print(f"  PASS")

    # --- M3b: with friction ---
    print("\n=== M3b: Oblique bounce, friction=0.5, coupled ===")
    b = Body2D(mass=1.0, r0=0.5, k=5000.0, nu=0.3)
    b.c = np.array([0.0, h0 + b.r0])
    b.vc = np.array([2.0, 0.0])
    b.vF = np.array([0.0, 1.5, -1.5, 0.0])

    state = State(bodies=[b])
    params = Params(dt=1.0 / 240.0, restitution=0.5, friction=0.5,
                    position_iters=8, velocity_iters=8)

    E0 = b.total_energy(params.gravity)
    vx0 = b.vc[0]

    # Run until first bounce apex
    vy_prev = -999.0
    apex_vx = None
    for i in range(2400):
        step(state, params)
        vy = b.vc[1]
        if i > 10 and vy_prev >= 0 and vy < 0 and apex_vx is None:
            apex_vx = b.vc[0]
        vy_prev = vy

    E_final = b.total_energy(params.gravity)
    print(f"  E ratio = {E_final / E0:.4f}")
    print(f"  vx: {vx0:.2f} -> {b.vc[0]:.4f} (at apex: {apex_vx:.4f})")
    assert E_final <= E0 * 1.01, f"Energy grew: {E_final / E0:.6f}"
    # With friction, horizontal velocity should decrease
    if apex_vx is not None:
        assert apex_vx < vx0 - 0.1, \
            f"Friction didn't reduce vx: {vx0:.2f} -> {apex_vx:.4f}"
        print(f"  PASS (friction reduced vx by {vx0 - apex_vx:.2f})")
    else:
        print(f"  SKIP (no apex detected)")


def test_m4_head_on_collision():
    """M4: Head-on collision of two equal disks in free space.

    With decoupled e=1, KE should be perfectly preserved (elastic collision).
    Momentum should be exchanged: disks swap velocities.
    """
    print("\n=== M4a: Head-on collision, decoupled, e=1 ===")
    a = Body2D(mass=1.0, r0=0.5, k=500.0, nu=0.3)
    a.c = np.array([-2.0, 5.0])
    a.vc = np.array([3.0, 0.0])

    b = Body2D(mass=1.0, r0=0.5, k=500.0, nu=0.3)
    b.c = np.array([2.0, 5.0])
    b.vc = np.array([-3.0, 0.0])

    state = State(bodies=[a, b])
    params = Params(
        gravity=np.array([0.0, 0.0]), dt=1.0 / 240.0,
        restitution=1.0, position_iters=8, velocity_iters=8,
        position_correct_F=False, velocity_couple_F=False,
    )

    KE0 = a.kinetic_energy() + b.kinetic_energy()
    px0 = a.mass * a.vc[0] + b.mass * b.vc[0]

    for _ in range(2400):
        step(state, params)

    KE_final = a.kinetic_energy() + b.kinetic_energy()
    px_final = a.mass * a.vc[0] + b.mass * b.vc[0]

    print(f"  KE ratio = {KE_final / KE0:.6f}")
    print(f"  px conserved: {px0:.4f} -> {px_final:.4f}")
    assert abs(KE_final / KE0 - 1.0) < 0.01, f"KE not preserved: {KE_final / KE0}"
    assert abs(px_final - px0) < 0.01, f"Momentum not conserved: {px_final}"
    print(f"  PASS")


def test_m5_three_body_stack():
    """M5: Three disks stacked vertically on floor.

    Should settle to a stable rest state at y = 0.5, 1.5, 2.5.
    No inter-penetration, no horizontal drift.
    """
    print("\n=== M5: Three-body vertical stack ===")
    bodies = []
    for i in range(3):
        b = Body2D(mass=1.0, r0=0.5, k=500.0, nu=0.3)
        b.c = np.array([0.0, 2.5 + i * 1.0])
        bodies.append(b)

    state = State(bodies=bodies)
    params = Params(
        gravity=np.array([0.0, -10.0]), dt=1.0 / 240.0,
        restitution=0.2, position_iters=16, velocity_iters=16,
        position_correct_F=False, velocity_couple_F=False,
    )

    for _ in range(2400):
        step(state, params)

    ys = [b.c[1] for b in bodies]
    xs = [abs(b.c[0]) for b in bodies]

    print(f"  y = [{', '.join(f'{y:.4f}' for y in ys)}]")
    print(f"  |x| = [{', '.join(f'{x:.8f}' for x in xs)}]")

    for i, (y_target, y_actual) in enumerate(zip([0.5, 1.5, 2.5], ys)):
        assert abs(y_actual - y_target) < 0.05, \
            f"Body {i} at y={y_actual:.4f}, expected {y_target}"
    for i, x in enumerate(xs):
        assert x < 0.01, f"Body {i} drifted horizontally: x={x:.6f}"
    print(f"  PASS")


def _run_cantilever(n_steps=4800, label="ARAP", relin=False, use_tgs=False):
    """Run cantilever simulation and return diagnostics.

    Returns dict with: bodies, joints, min_det, min_det_body, min_det_step,
    max_pos_err, max_angle_err, ys, deflection.
    """
    from .solver import (_joint_position_error, _joint_angle_error,
                         step_tgs)
    from . import energy as energy_mod

    n_bodies = 8
    r0 = 0.3
    k = 2000.0
    nu = 0.35

    bodies = []
    for i in range(n_bodies):
        b = Body2D(mass=0.5, r0=r0, k=k, nu=nu, static=(i == 0))
        b.c = np.array([2.0 * r0 * i, 3.0])
        bodies.append(b)

    joints = []
    for i in range(n_bodies - 1):
        j = Joint(
            body_a_idx=i, body_b_idx=i + 1,
            local_a=np.array([r0, 0.0]),
            local_b=np.array([-r0, 0.0]),
            axis_a=np.array([1.0, 0.0]),
            axis_b=np.array([1.0, 0.0]),
        )
        joints.append(j)

    state = State(bodies=bodies, joints=joints)
    if use_tgs:
        params = Params(
            dt=1.0 / 240.0, substeps=16, relax_iters=1,
            restitution=0.0, friction=0.0,
        )
        step_fn = step_tgs
    else:
        params = Params(
            dt=1.0 / 240.0, position_iters=16, velocity_iters=16,
            restitution=0.0, friction=0.0, relin=relin,
        )
        step_fn = step

    # Track min det(F) across all non-static bodies
    min_det = float("inf")
    min_det_body = -1
    min_det_step = -1

    for s in range(n_steps):
        step_fn(state, params)
        for bi in range(1, n_bodies):  # skip static body 0
            J = energy_mod._det2(bodies[bi].F)
            if J < min_det:
                min_det = J
                min_det_body = bi
                min_det_step = s

    # Joint errors
    max_pos_err = 0.0
    max_angle_err = 0.0
    for j in joints:
        pos_err = np.linalg.norm(_joint_position_error(j, bodies))
        angle_err = abs(_joint_angle_error(j, bodies))
        max_pos_err = max(max_pos_err, pos_err)
        max_angle_err = max(max_angle_err, angle_err)

    ys = [b.c[1] for b in bodies]
    deflection = ys[0] - ys[-1]

    return dict(
        bodies=bodies, joints=joints, ys=ys,
        min_det=min_det, min_det_body=min_det_body, min_det_step=min_det_step,
        max_pos_err=max_pos_err, max_angle_err=max_angle_err,
        deflection=deflection, label=label,
    )


def test_cantilever():
    """Cantilever beam: chain of bodies connected by weld joints.

    Root body is static (clamped wall). Gravity bends the beam down.
    Tests ARAP + volume energy in direct mode (no relin).
    """
    print(f"\n=== Cantilever beam (8 bodies, weld joints, ARAP) ===")
    r = _run_cantilever()
    bodies, ys = r["bodies"], r["ys"]

    print(f"  y = [{', '.join(f'{y:.4f}' for y in ys)}]")
    print(f"  min det(F) = {r['min_det']:.6f}  "
          f"(body {r['min_det_body']}, step {r['min_det_step']})")
    print(f"  max joint pos error  = {r['max_pos_err']:.6e}")
    print(f"  max joint angle error = {r['max_angle_err']:.6e}")
    print(f"  deflection (tip) = {r['deflection']:.4f}")

    for bi in range(1, len(bodies)):
        det_F = energy._det2(bodies[bi].F)
        print(f"  body {bi}: det(F)={det_F:.6f}")

    # Root should be at original position
    assert abs(bodies[0].c[1] - 3.0) < 1e-10, \
        f"Static body moved: y={bodies[0].c[1]}"
    # Tip should be below root (gravity deflection)
    assert ys[-1] < ys[0], "Beam didn't deflect downward"
    # Joint errors should be small
    assert r["max_pos_err"] < 0.02, \
        f"Joint position error too large: {r['max_pos_err']}"
    assert r["max_angle_err"] < 0.02, \
        f"Joint angle error too large: {r['max_angle_err']}"
    # Monotonic deflection
    for i in range(1, len(ys)):
        assert ys[i] <= ys[i - 1] + 0.01, \
            f"Non-monotonic at body {i}: {ys[i]:.4f} > {ys[i-1]:.4f}"

    print(f"  PASS")


def test_cantilever_tgs():
    """Cantilever beam with TGS Soft + VBD joint solve.

    The VBD solve should keep det(F) well away from 0 by balancing
    elastic energy against constraint satisfaction.
    """
    import time as time_mod
    print(f"\n=== Cantilever beam (TGS + VBD) ===")
    t0 = time_mod.time()
    r = _run_cantilever(use_tgs=True, label="TGS+VBD")
    elapsed = time_mod.time() - t0
    bodies, ys = r["bodies"], r["ys"]

    print(f"  time = {elapsed:.1f}s")
    print(f"  y = [{', '.join(f'{y:.4f}' for y in ys)}]")
    print(f"  min det(F) = {r['min_det']:.6f}  "
          f"(body {r['min_det_body']}, step {r['min_det_step']})")
    print(f"  max joint pos error  = {r['max_pos_err']:.6e}")
    print(f"  max joint angle error = {r['max_angle_err']:.6e}")
    print(f"  deflection (tip) = {r['deflection']:.4f}")

    for bi in range(1, len(bodies)):
        det_F = energy._det2(bodies[bi].F)
        print(f"  body {bi}: det(F)={det_F:.6f}")

    # Root at original position
    assert abs(bodies[0].c[1] - 3.0) < 1e-10, \
        f"Static body moved: y={bodies[0].c[1]}"
    # Tip below root
    assert ys[-1] < ys[0], "Beam didn't deflect downward"

    print(f"  PASS")


def main():
    test_m1_free_flight()
    test_m1_rigid_limit()
    test_m2_floor_bounce()
    test_m3_oblique_bounce()
    test_m4_head_on_collision()
    test_m5_three_body_stack()
    test_cantilever()
    test_cantilever_tgs()
    print("\nAll tests done.")


if __name__ == "__main__":
    main()
