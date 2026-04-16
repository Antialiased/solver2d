"""Grid-search soft-contact params for two-ball-attract, with pass-through guard."""
import numpy as np
from .solver import Params, make_initial_state, run_sim


def run_one(k, dt, t_end, wn, zeta, mu_frac=1.0/3.0):
    steps = int(round(t_end / dt))
    g = 9.81
    g_vec = np.array([-g, +g], dtype=float)
    p = Params(N=2, k=k, dt=dt, steps=steps, g=g, damping=0.0,
               mu_frac=mu_frac, warm_start=False, floor=False,
               restitution=0.0, g_vec=g_vec,
               contact_wn=wn, contact_zeta=zeta)
    init = make_initial_state(p)
    init.x[0] = -1.0; init.s[0] = p.s0
    init.x[1] = +1.0; init.s[1] = p.s0
    log = run_sim(p, "exponential", init=init)
    if not log.stable:
        return None
    E = log.KE + log.PE_g + log.PE_e
    return float(E[-1] / E[0])


def main():
    dt = 5.0e-4
    t_end = 6.0
    ks = [1e3, 1e4, 1e5, 1e6]
    # ratios wn/omega_body
    ratios = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    zetas = [0.0, 0.1, 0.3, 1.0, 3.0]
    mu_frac = 1.0/3.0

    # hard baselines
    print("hard baselines:")
    for k in ks:
        r = run_one(k, dt, t_end, None, 0.0)
        print(f"  k={k:.0e}  E_end/E0 = {r}")
    print()

    for k in ks:
        ob = float(np.sqrt(k/mu_frac))
        print(f"=== k={k:.0e}  ob*dt={ob*dt:.3f} ===")
        hdr = "  wn/ob   " + "  ".join(f"z={z:<4.2f}" for z in zetas)
        print(hdr)
        for rr in ratios:
            wn = rr * ob
            row = [f"  {rr:<6.2f}"]
            for z in zetas:
                r = run_one(k, dt, t_end, wn, z)
                row.append("  UNSTB " if r is None else f"  {r:6.3f}")
            print(" ".join(row))
        print()


if __name__ == "__main__":
    main()
