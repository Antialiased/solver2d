"""Two deformable bodies attracting, restitution=1, energy preservation check."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

from .solver import Params, run_sim, make_initial_state


def run_two_attract(k, dt, t_end, restitution=1.0, mode="two_pass_exp",
                    mu_frac=1.0/3.0, record_every=1):
    steps = int(round(t_end / dt))
    g = 9.81
    g_vec = np.array([-g, +g], dtype=float)
    p = Params(N=2, k=k, dt=dt, steps=steps, g=g, damping=0.0,
               mu_frac=mu_frac, warm_start=False, floor=False,
               restitution=restitution, g_vec=g_vec)
    init = make_initial_state(p)
    init.x[0] = -1.0; init.s[0] = p.s0
    init.x[1] = +1.0; init.s[1] = p.s0
    log = run_sim(p, mode, init=init, record_every=record_every)
    return p, log


def main_report():
    dt = 5e-4
    t_end = 6.0
    print("two-ball attract, restitution=1, two_pass_exp:")
    print(f"  {'k':>9s} {'stable':>8s} {'E_end/E0':>10s} {'vx0_f':>9s} {'vx1_f':>9s}")
    for k in [1e3, 1e4, 1e5, 1e6]:
        p, log = run_two_attract(k, dt, t_end)
        if not log.stable:
            print(f"  {k:9.0e} UNSTABLE")
            continue
        E = log.KE + log.PE_g + log.PE_e
        ratio = float(E[-1] / E[0])
        print(f"  {k:9.0e} {str(log.stable):>8s} {ratio:10.4f} "
              f"{log.vxs[-1,0]:+9.4f} {log.vxs[-1,1]:+9.4f}")


def animate_two_body_sweep(outdir: str):
    print("\n=== ANIM: two-body attract sweep, restitution=1 ===")
    g = 9.81
    dt = 5e-4
    t_end = 6.0
    ks = [1e3, 1e4, 1e5, 1e6]
    steps = int(round(t_end / dt))
    record_every = max(1, steps // 180)

    logs = []
    for k in ks:
        p, log = run_two_attract(k, dt, t_end, record_every=record_every)
        logs.append(log)
        E = log.KE + log.PE_g + log.PE_e
        print(f"  k={k:.0e}  stable={log.stable}  E_end/E0={E[-1]/E[0]:.4f}")

    nrec = len(logs[0].times)
    fig, axes = plt.subplots(len(ks), 1, figsize=(7.5, 1.8 * len(ks)),
                             sharex=True)
    s0_const = Params().s0
    arts = []
    for ax, log, k in zip(axes, logs, ks):
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-0.9, 0.9)
        ax.set_aspect("equal")
        ax.set_title(f"k = {k:.0e}", fontsize=10, loc="left")
        ax.set_yticks([])
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        x0_0 = float(log.xs[0, 0]); x1_0 = float(log.xs[0, 1])
        s_0 = float(log.ss[0, 0]); s_1 = float(log.ss[0, 1])
        e0 = Ellipse((x0_0, 0.0), width=2 * s_0,
                     height=2 * (s0_const ** 2) / s_0,
                     facecolor="#3877e0", edgecolor="#1b3a70",
                     lw=1.2, alpha=0.9)
        e1 = Ellipse((x1_0, 0.0), width=2 * s_1,
                     height=2 * (s0_const ** 2) / s_1,
                     facecolor="#e04a3b", edgecolor="#6a1a14",
                     lw=1.2, alpha=0.9)
        ax.add_patch(e0); ax.add_patch(e1)
        tt = ax.text(0.98, 0.88, "", transform=ax.transAxes,
                     ha="right", va="top", fontsize=8, family="monospace")
        Et = log.KE + log.PE_g + log.PE_e
        arts.append((e0, e1, tt, log, float(Et[0])))
    axes[-1].set_xlabel("x")
    fig.suptitle("Two deformable bodies attracting, restitution = 1 "
                 "(body-body velocity reflection)", fontsize=11)

    def init_anim():
        out = []
        for e0, e1, tt, _, _ in arts:
            out.extend([e0, e1, tt])
        return out

    def update(fi):
        out = []
        for e0, e1, tt, log, E0 in arts:
            x0 = float(log.xs[fi, 0]); x1 = float(log.xs[fi, 1])
            s_0 = float(log.ss[fi, 0]); s_1 = float(log.ss[fi, 1])
            e0.set_center((x0, 0.0))
            e0.width = 2 * s_0
            e0.height = 2 * (s0_const ** 2) / max(s_0, 1e-6)
            e1.set_center((x1, 0.0))
            e1.width = 2 * s_1
            e1.height = 2 * (s0_const ** 2) / max(s_1, 1e-6)
            Et = float(log.KE[fi] + log.PE_g[fi] + log.PE_e[fi])
            tt.set_text(f"t={log.times[fi]:.2f}s E/E0={Et/E0:.3f}")
            out.extend([e0, e1, tt])
        return out

    anim = FuncAnimation(fig, update, frames=nrec,
                         init_func=init_anim, blit=True, interval=40)
    fig.tight_layout()
    out_path = os.path.join(outdir, "two_body_attract_restitution.gif")
    anim.save(out_path, writer="pillow", fps=25, dpi=90)
    plt.close(fig)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main_report()
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    os.makedirs(outdir, exist_ok=True)
    animate_two_body_sweep(outdir)
