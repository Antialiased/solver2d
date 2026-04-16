"""Bouncy ball and two-ball tests: hard-floor bounces, two-ball contact
bouncing, attractive two-ball free-space collision, plus the animations
for the attract scenario (BE vs Exp).
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

from .solver import Params, make_initial_state, run_sim


def test_bouncy_balls(outdir: str):
    print("\n=== TEST: bouncy balls - BE vs Exp vs rigid-contact ===")
    h0 = 1.0
    g = 9.81
    t_end = 4.0
    dt = 5.0e-4
    steps = int(round(t_end / dt))
    ks = [1.0e3, 1.0e4, 1.0e5, 1.0e6]
    mu_frac = 1.0 / 3.0

    modes = [
        ("BE + coupled",  "local_implicit"),
        ("Exp + coupled", "exponential"),
        ("BE + rigid",    "naive"),
    ]

    for run_label, e_rest, out_name in [
        ("explicit restitution e=1",   1.0, "bouncy_balls.png"),
        ("no restitution (plastic)",   0.0, "bouncy_balls_no_restitution.png"),
    ]:
        print(f"\n  --- {run_label} ---")
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(1, len(ks) - 1)) for i in range(len(ks))]
        print(f"  {'mode':<15s} {'k':>9s} {'omega*dt':>9s} "
              f"{'E_end/E0':>10s} {'h_peak_last/h0':>15s} {'bounces':>8s}")
        for col, (label, mode) in enumerate(modes):
            for k, color in zip(ks, colors):
                omega_int = np.sqrt(k / mu_frac)
                p = Params(N=1, k=k, dt=dt, steps=steps, g=g, damping=0.0,
                           mu_frac=mu_frac, warm_start=False, floor=True,
                           restitution=e_rest)
                init = make_initial_state(p)
                init.x[0] = p.s0 + h0
                init.vx[0] = 0.0
                init.s[0] = p.s0
                init.vs[0] = 0.0
                log = run_sim(p, mode, init=init)
                if not log.stable:
                    print(f"  {label:<15s} {k:9.0e} {omega_int*dt:9.3f} UNSTABLE")
                    continue
                h_t = log.xs[:, 0] - log.ss[:, 0]
                E_t = log.KE + log.PE_g + log.PE_e
                E0 = float(E_t[0])
                ratio = float(E_t[-1] / E0) if E0 > 0 else float("nan")
                touches = int(np.sum((h_t[1:] < 1.0e-4) & (h_t[:-1] >= 1.0e-4)))
                tail = h_t[int(0.75 * len(h_t)):]
                h_last_peak = float(np.nanmax(tail)) if len(tail) else float("nan")
                print(f"  {label:<15s} {k:9.0e} {omega_int*dt:9.3f} "
                      f"{ratio:10.3f} {h_last_peak/h0:15.3f} {touches:8d}")

                axes[0, col].plot(log.times, h_t, color=color, lw=1.0,
                                  label=f"k={k:.0e}  wdt={omega_int*dt:.2f}")
                axes[1, col].plot(log.times, E_t / max(E0, 1e-30),
                                  color=color, lw=1.0)
            axes[0, col].set_title(label)
            axes[0, col].set_ylabel("bottom height" if col == 0 else "")
            axes[0, col].grid(True, alpha=0.3)
            axes[0, col].legend(fontsize=7, loc="upper right")
            axes[0, col].set_ylim(bottom=-0.05)
            axes[1, col].set_ylabel("E / E0" if col == 0 else "")
            axes[1, col].set_xlabel("t (s)")
            axes[1, col].grid(True, alpha=0.3)
            axes[1, col].axhline(1.0, color="k", lw=0.5, ls=":")

        fig.suptitle(f"Bouncy balls (hard floor, no damping) - {run_label}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, out_name), dpi=120)
        plt.close(fig)
    return True


def test_two_ball_bouncy(outdir: str):
    print("\n=== TEST: two-ball bouncy (internal elasticity only) ===")
    h0 = 1.0
    g = 9.81
    t_end = 4.0
    dt = 5.0e-4
    steps = int(round(t_end / dt))
    ks = [1.0e3, 1.0e4, 1.0e5, 1.0e6]
    mu_frac = 1.0 / 3.0

    modes = [
        ("BE + coupled",  "local_implicit"),
        ("Exp + coupled", "exponential"),
        ("BE + rigid",    "naive"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(ks) - 1)) for i in range(len(ks))]

    print(f"  {'mode':<15s} {'k':>9s} {'omega*dt':>9s} "
          f"{'E_end/E0':>10s} {'gap_peak_last/h0':>17s} {'bounces':>8s}")
    for col, (label, mode) in enumerate(modes):
        for k, color in zip(ks, colors):
            omega_int = float(np.sqrt(k / mu_frac))
            p = Params(N=2, k=k, dt=dt, steps=steps, g=g, damping=0.0,
                       mu_frac=mu_frac, warm_start=True, floor=True,
                       restitution=0.0)
            init = make_initial_state(p)
            init.x[0] = p.s0
            init.s[0] = p.s0
            init.vx[0] = 0.0
            init.vs[0] = 0.0
            init.x[1] = p.s0 + 2.0 * p.s0 + h0
            init.s[1] = p.s0
            init.vx[1] = 0.0
            init.vs[1] = 0.0
            log = run_sim(p, mode, init=init)
            if not log.stable:
                print(f"  {label:<15s} {k:9.0e} {omega_int*dt:9.3f} UNSTABLE")
                continue
            x1_c = log.xs[:, 1]
            E_t = log.KE + log.PE_g + log.PE_e
            E0 = float(E_t[0])
            ratio = float(E_t[-1] / E0) if E0 > 0 else float("nan")
            vx1 = log.vxs[:, 1]
            sign_flip = (vx1[1:] > 0.0) & (vx1[:-1] <= 0.0)
            gap_bb_full = log.xs[:, 1] - log.ss[:, 1] - log.xs[:, 0] - log.ss[:, 0]
            touch_idx = np.argmax(gap_bb_full < 5.0e-4)
            touches = int(np.sum(sign_flip[touch_idx:])) if touch_idx < len(sign_flip) else 0
            tail = x1_c[int(0.75 * len(x1_c)):]
            x1_last_peak = float(np.nanmax(tail) - 3.0 * p.s0) if len(tail) else float("nan")
            print(f"  {label:<15s} {k:9.0e} {omega_int*dt:9.3f} "
                  f"{ratio:10.3f} {x1_last_peak/h0:17.4f} {touches:8d}")

            axes[0, col].plot(log.times, x1_c, color=color, lw=1.0,
                              label=f"k={k:.0e}  wdt={omega_int*dt:.2f}")
            axes[1, col].plot(log.times, E_t / max(E0, 1e-30),
                              color=color, lw=1.0)
        axes[0, col].set_title(label)
        axes[0, col].set_ylabel("body 1 center height" if col == 0 else "")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend(fontsize=7, loc="upper right")
        axes[0, col].axhline(3.0 * 0.5, color="k", lw=0.5, ls=":",
                             label="rest-on-top height")
        axes[1, col].set_ylabel("E / E0" if col == 0 else "")
        axes[1, col].set_xlabel("t (s)")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].axhline(1.0, color="k", lw=0.5, ls=":")

    fig.suptitle("Two-ball bouncy: body 1 dropped onto body 0 (on floor), "
                 "restitution purely from internal elasticity")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "two_ball_bouncy.png"), dpi=120)
    plt.close(fig)
    return True


def test_two_ball_attract(outdir: str):
    print("\n=== TEST: two balls attracting under gravity ===")
    g = 9.81
    t_end = 6.0
    dt = 5.0e-4
    steps = int(round(t_end / dt))
    ks = [1.0e3, 1.0e4, 1.0e5, 1.0e6]
    mu_frac = 1.0 / 3.0

    modes = [
        ("BE + coupled",  "local_implicit"),
        ("Exp + coupled", "exponential"),
        ("BE + rigid",    "naive"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(ks) - 1)) for i in range(len(ks))]

    print(f"  {'mode':<15s} {'k':>9s} {'omega*dt':>9s} "
          f"{'E_end/E0':>10s} {'|x0|_last':>11s} {'separations':>12s}")
    for col, (label, mode) in enumerate(modes):
        for k, color in zip(ks, colors):
            omega_int = float(np.sqrt(k / mu_frac))
            g_vec = np.array([-g, +g], dtype=float)
            p = Params(N=2, k=k, dt=dt, steps=steps, g=g, damping=0.0,
                       mu_frac=mu_frac, warm_start=False, floor=False,
                       restitution=0.0, g_vec=g_vec)
            init = make_initial_state(p)
            init.x[0] = -1.0
            init.s[0] = p.s0
            init.vx[0] = 0.0
            init.vs[0] = 0.0
            init.x[1] = +1.0
            init.s[1] = p.s0
            init.vx[1] = 0.0
            init.vs[1] = 0.0
            log = run_sim(p, mode, init=init)
            if not log.stable:
                print(f"  {label:<15s} {k:9.0e} {omega_int*dt:9.3f} UNSTABLE")
                continue
            x0 = log.xs[:, 0]
            x1 = log.xs[:, 1]
            gap_bb = x1 - log.ss[:, 1] - x0 - log.ss[:, 0]
            E_t = log.KE + log.PE_g + log.PE_e
            E0 = float(E_t[0])
            ratio = float(E_t[-1] / E0) if E0 > 0 else float("nan")
            separated = gap_bb > 0.01
            flips = int(np.sum((separated[1:]) & (~separated[:-1])))
            tail = np.abs(x0[int(0.8 * len(x0)):])
            x0_last = float(np.nanmax(tail)) if len(tail) else float("nan")
            print(f"  {label:<15s} {k:9.0e} {omega_int*dt:9.3f} "
                  f"{ratio:10.3f} {x0_last:11.4f} {flips:12d}")

            axes[0, col].plot(log.times, x0, color=color, lw=1.0,
                              label=f"k={k:.0e}")
            axes[0, col].plot(log.times, x1, color=color, lw=1.0, ls="--")
            axes[1, col].plot(log.times, np.maximum(gap_bb, -0.05),
                              color=color, lw=1.0)
            axes[2, col].plot(log.times, E_t / max(E0, 1e-30),
                              color=color, lw=1.0)
        axes[0, col].set_title(label)
        axes[0, col].set_ylabel("body 0 (solid) and body 1 (dashed) x" if col == 0 else "")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend(fontsize=7, loc="upper right")
        axes[0, col].axhline(0.0, color="k", lw=0.5, ls=":")
        axes[1, col].set_ylabel("inter-body gap" if col == 0 else "")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].axhline(0.0, color="k", lw=0.5, ls=":")
        axes[2, col].set_ylabel("E / E0" if col == 0 else "")
        axes[2, col].set_xlabel("t (s)")
        axes[2, col].grid(True, alpha=0.3)
        axes[2, col].axhline(1.0, color="k", lw=0.5, ls=":")

    fig.suptitle("Two balls attracting under gravity from x=+/-1 (no floor) - "
                 "restitution purely from internal elasticity")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "two_ball_attract.png"), dpi=120)
    plt.close(fig)
    return True


def animate_two_ball_attract(outdir: str):
    print("\n=== ANIM: two balls attracting, stiffness sweep ===")
    g = 9.81
    t_end = 4.0
    dt = 5.0e-4
    steps = int(round(t_end / dt))
    ks = [1.0e3, 1.0e4, 1.0e5, 1.0e6]
    mu_frac = 1.0 / 3.0

    record_every = max(1, steps // 120)

    logs = []
    for k in ks:
        p = Params(N=2, k=k, dt=dt, steps=steps, g=g, damping=0.0,
                   mu_frac=mu_frac, warm_start=False, floor=False,
                   restitution=0.0, g_vec=np.array([-g, g]))
        init = make_initial_state(p)
        init.x[0] = -1.0
        init.x[1] = +1.0
        init.s[0] = init.s[1] = p.s0
        log = run_sim(p, "exponential", init=init, record_every=record_every)
        logs.append(log)
        print(f"  k={k:.0e}  frames={len(log.times)}")

    nrec = len(logs[0].times)
    fig, axes = plt.subplots(len(ks), 1, figsize=(7.5, 1.8 * len(ks)),
                             sharex=True)
    pair_artists = []
    s0_const = Params().s0
    for ax, log, k in zip(axes, logs, ks):
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-0.9, 0.9)
        ax.set_aspect("equal")
        ax.set_title(f"k = {k:.0e}", fontsize=10, loc="left")
        ax.set_yticks([])
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.axhline(0, color="gray", lw=0.3, alpha=0.5)
        x0_0 = float(log.xs[0, 0])
        x1_0 = float(log.xs[0, 1])
        s0_0 = float(log.ss[0, 0])
        s1_0 = float(log.ss[0, 1])
        e0 = Ellipse((x0_0, 0.0), width=2 * s0_0,
                     height=2 * (s0_const ** 2) / s0_0,
                     facecolor="#3877e0", edgecolor="#1b3a70",
                     lw=1.2, alpha=0.85)
        e1 = Ellipse((x1_0, 0.0), width=2 * s1_0,
                     height=2 * (s0_const ** 2) / s1_0,
                     facecolor="#e04a3b", edgecolor="#6a1a14",
                     lw=1.2, alpha=0.85)
        ax.add_patch(e0)
        ax.add_patch(e1)
        time_txt = ax.text(0.98, 0.88, "", transform=ax.transAxes,
                           ha="right", va="top", fontsize=8,
                           family="monospace")
        pair_artists.append((e0, e1, time_txt))
    axes[-1].set_xlabel("x")
    fig.suptitle("Two balls attracting under gravity - "
                 "internal elasticity rebound, warm_start=False",
                 fontsize=11)

    def init_anim():
        arts = []
        for e0, e1, tt in pair_artists:
            arts.extend([e0, e1, tt])
        return arts

    def update(frame_i):
        arts = []
        for (e0, e1, tt), log in zip(pair_artists, logs):
            x0 = float(log.xs[frame_i, 0])
            x1 = float(log.xs[frame_i, 1])
            s0v = float(log.ss[frame_i, 0])
            s1v = float(log.ss[frame_i, 1])
            e0.set_center((x0, 0.0))
            e0.width = 2.0 * s0v
            e0.height = 2.0 * (s0_const ** 2) / max(s0v, 1e-6)
            e1.set_center((x1, 0.0))
            e1.width = 2.0 * s1v
            e1.height = 2.0 * (s0_const ** 2) / max(s1v, 1e-6)
            tt.set_text(f"t={log.times[frame_i]:.2f}s")
            arts.extend([e0, e1, tt])
        return arts

    anim = FuncAnimation(fig, update, frames=nrec,
                         init_func=init_anim, blit=True, interval=40)
    fig.tight_layout()
    out_path = os.path.join(outdir, "two_ball_attract.gif")
    anim.save(out_path, writer="pillow", fps=25, dpi=90)
    plt.close(fig)
    print(f"  saved {out_path}")
    return True


def animate_two_ball_attract_be_vs_exp(outdir: str):
    print("\n=== ANIM: two balls attracting, BE vs Exp side-by-side ===")
    g = 9.81
    t_end = 8.0
    dt = 5.0e-4
    steps = int(round(t_end / dt))
    ks = [1.0e3, 1.0e5, 1.0e6]
    mu_frac = 1.0 / 3.0
    modes = [("BE", "local_implicit"), ("Exp", "exponential")]

    record_every = max(1, steps // 200)

    logs_grid = {}
    for k in ks:
        for tag, mode in modes:
            p = Params(N=2, k=k, dt=dt, steps=steps, g=g, damping=0.0,
                       mu_frac=mu_frac, warm_start=False, floor=False,
                       restitution=0.0, g_vec=np.array([-g, g]))
            init = make_initial_state(p)
            init.x[0] = -1.0
            init.x[1] = +1.0
            init.s[0] = init.s[1] = p.s0
            logs_grid[(k, tag)] = run_sim(p, mode, init=init,
                                          record_every=record_every)
        print(f"  k={k:.0e}  wdt={np.sqrt(k/mu_frac)*dt:.3f}")

    nrec = len(logs_grid[(ks[0], "BE")].times)
    fig, axes = plt.subplots(len(ks), 2, figsize=(10, 1.9 * len(ks)),
                             sharex=True)
    pair_artists = []
    s0_const = Params().s0
    for row, k in enumerate(ks):
        for col, (tag, _mode) in enumerate(modes):
            ax = axes[row, col]
            log = logs_grid[(k, tag)]
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-0.9, 0.9)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"{tag}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"k={k:.0e}", fontsize=9)
            ax.set_yticks([])
            ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
            ax.axhline(0, color="gray", lw=0.3, alpha=0.5)
            x0_0 = float(log.xs[0, 0]); x1_0 = float(log.xs[0, 1])
            s0_0 = float(log.ss[0, 0]); s1_0 = float(log.ss[0, 1])
            e0 = Ellipse((x0_0, 0.0), width=2 * s0_0,
                         height=2 * (s0_const ** 2) / s0_0,
                         facecolor="#3877e0", edgecolor="#1b3a70",
                         lw=1.2, alpha=0.85)
            e1 = Ellipse((x1_0, 0.0), width=2 * s1_0,
                         height=2 * (s0_const ** 2) / s1_0,
                         facecolor="#e04a3b", edgecolor="#6a1a14",
                         lw=1.2, alpha=0.85)
            ax.add_patch(e0)
            ax.add_patch(e1)
            time_txt = ax.text(0.98, 0.88, "", transform=ax.transAxes,
                               ha="right", va="top", fontsize=8,
                               family="monospace")
            pair_artists.append((e0, e1, time_txt, log))
    for ax in axes[-1]:
        ax.set_xlabel("x")
    fig.suptitle("Two balls attracting - BE vs Exp internal integrator "
                 "(warm_start=False)", fontsize=11)

    def init_anim():
        arts = []
        for e0, e1, tt, _ in pair_artists:
            arts.extend([e0, e1, tt])
        return arts

    def update(frame_i):
        arts = []
        for e0, e1, tt, log in pair_artists:
            x0 = float(log.xs[frame_i, 0])
            x1 = float(log.xs[frame_i, 1])
            s0v = float(log.ss[frame_i, 0])
            s1v = float(log.ss[frame_i, 1])
            e0.set_center((x0, 0.0))
            e0.width = 2.0 * s0v
            e0.height = 2.0 * (s0_const ** 2) / max(s0v, 1e-6)
            e1.set_center((x1, 0.0))
            e1.width = 2.0 * s1v
            e1.height = 2.0 * (s0_const ** 2) / max(s1v, 1e-6)
            tt.set_text(f"t={log.times[frame_i]:.2f}s")
            arts.extend([e0, e1, tt])
        return arts

    anim = FuncAnimation(fig, update, frames=nrec,
                         init_func=init_anim, blit=True, interval=40)
    fig.tight_layout()
    out_path = os.path.join(outdir, "two_ball_attract_be_vs_exp.gif")
    anim.save(out_path, writer="pillow", fps=25, dpi=90)
    plt.close(fig)
    print(f"  saved {out_path}")
    return True
