"""Diagnostic: energy accounting for a single floor bounce.

Compares coupled vs decoupled contact modes:
- Full: velocity+position pass couples F deformation DoFs (Moreau dissipation)
- Decoupled: position/velocity pass only acts on center-of-mass (energy preserving)
"""
import numpy as np
from .body import Body2D
from .solver import State, Params, step


def theoretical_moreau_tax(r0, mass):
    """Compute expected height ratio from a single Moreau projection at F=I.

    J = [0, 1, 0, 0, 0, -r0], M = diag(M, M, mu, mu, mu, mu), mu = M*r0^2/4
    K = 1/M + r0^2/mu = 5/M
    vy_new = vy * (1 - 1/(M*K)) = vy * 4/5
    """
    mu_inertia = mass * r0 ** 2 / 4.0
    K = 1.0 / mass + r0 ** 2 / mu_inertia
    vy_frac = 1.0 - (1.0 / mass) / K
    return vy_frac ** 2


def run_diagnostic():
    h0 = 5.0
    r0 = 0.5
    mass = 1.0

    h_theory = theoretical_moreau_tax(r0, mass)
    print(f"=== Theoretical single Moreau projection: h_ratio = {h_theory:.4f} ===\n")

    configs = [
        ("e=0, full",       0.0, True,  True),
        ("e=1, full",       1.0, True,  True),
        ("e=0, decoupled",  0.0, False, False),
        ("e=1, decoupled",  1.0, False, False),
    ]

    for label, e, pos_F, vel_F in configs:
        b = Body2D(mass=mass, r0=r0, k=5000.0, nu=0.3)
        b.c = np.array([0.0, h0 + r0])

        state = State(bodies=[b])
        params = Params(dt=1.0 / 240.0, restitution=e,
                        position_iters=8, velocity_iters=8,
                        position_correct_F=pos_F, velocity_couple_F=vel_F)

        apexes = []
        vy_prev = -999.0

        for i in range(4800):
            step(state, params)
            vy = b.vc[1]
            y = b.c[1] - r0

            if i > 10 and vy_prev >= 0 and vy < 0:
                apexes.append(y)
            vy_prev = vy

        if len(apexes) >= 1:
            last = apexes[-1] / h0
            print(f"  {label:20s}: h1/h0={apexes[0] / h0:.4f}  "
                  f"last={last:.4f}  ({len(apexes)} bounces)")
        else:
            print(f"  {label:20s}: no apex detected")

    print()
    print("Key finding: decoupled e=1 gives ~0.995 first-bounce ratio.")
    print("Full mode gives ~0.23 due to Jacobian coupling F into the impulse.")


if __name__ == "__main__":
    run_diagnostic()
