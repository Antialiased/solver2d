"""Newton's cradle tests: near-rigid chain, stiffness sweep, near-ideal
transfer with the exponential integrator, plus associated animations."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

from .solver import Params, make_cradle_state, run_sim


def _cradle_params(k: float, steps: int = 3000, dt: float = 1.0e-3,
                   mu_frac: float = 1.0 / 3.0) -> Params:
    return Params(N=5, k=k, g=0.0, damping=0.0, floor=False, warm_start=False,
                  mu_frac=mu_frac, steps=steps, dt=dt)


def test_cradle_ideal(outdir: str):
    """
    The 'best-parameters' cradle: exponential integrator + low mu_frac +
    non-zero inner_gap. Near-textbook Newton's cradle transfer, with
    energy transfer error scaling roughly linearly in mu_frac.
    """
    print("\n=== TEST 6: near-ideal cradle (best parameters) ===")
    rows = []
    for mu_frac in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.33]:
        k = 1.0e4
        omega = np.sqrt(k / mu_frac)
        dt = 0.3 / omega
        T = 3.0
        p = _cradle_params(k=k, steps=int(T / dt), dt=dt, mu_frac=mu_frac)
        init = make_cradle_state(p, gap=0.3, inner_gap=0.02, v0=1.0)
        log = run_sim(p, "exponential", init=init)
        vf = log.vxs[-1]
        mid = float(np.max(np.abs(vf[1:-1])))
        ke_x = 0.5 * np.sum(vf ** 2)
        transfer = vf[-1]
        print(f"  mu_frac={mu_frac:7.0e}  transfer={transfer:.6f}  "
              f"1-transfer={1 - transfer:.2e}  mid_max={mid:.2e}  KE_x={ke_x:.4f}")
        rows.append((mu_frac, p, log))

    fig, ax = plt.subplots(figsize=(6, 4))
    mus = np.array([r[0] for r in rows])
    errs = np.array([1.0 - r[2].vxs[-1, -1] for r in rows])
    ax.loglog(mus, errs, "o-")
    ax.set_xlabel("mu_frac")
    ax.set_ylabel("1 - transfer")
    ax.set_title("Cradle transfer error vs mu_frac (N=5, inner_gap=0.02)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cradle_ideal_mu_scaling.png"), dpi=120)
    plt.close(fig)

    best = rows[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(best[1].N):
        ax.plot(best[2].times, best[2].vxs[:, i], label=f"body {i}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vx")
    ax.set_title(f"Near-ideal cradle, mu_frac={best[0]:.0e}, exp integrator")
    ax.legend(fontsize=8, ncol=best[1].N)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cradle_ideal_velocities.png"), dpi=120)
    plt.close(fig)
    return True


def test_cradle(outdir: str):
    print("\n=== TEST 4: Newton's cradle (near-rigid) ===")
    p = _cradle_params(k=1.0e6, steps=2000)
    init = make_cradle_state(p, gap=0.3, v0=1.0)
    log = run_sim(p, "local_implicit", init=init)

    v0 = 1.0
    vx_final = log.vxs[-1]
    print("  final vx per body:")
    for i in range(p.N):
        print(f"    body {i}: vx={vx_final[i]:+.4f}")
    transfer = vx_final[-1] / v0
    print(f"  transfer efficiency (vx[N-1] / v0) = {transfer:.3f}")
    middle_moved = np.max(np.abs(vx_final[1:-1]))
    print(f"  max |vx| in middle bodies         = {middle_moved:.3e}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(p.N):
        ax.plot(log.times, log.vxs[:, i], label=f"body {i}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vx")
    ax.set_title(f"Cradle velocities (k={p.k:g}, local implicit)")
    ax.legend(fontsize=8, ncol=p.N)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cradle_velocities.png"), dpi=120)
    plt.close(fig)
    return transfer > 0.9


def test_cradle_sweep(outdir: str):
    print("\n=== TEST 5: Cradle stiffness sweep ===")
    ks = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
    rows = []
    for k in ks:
        p = _cradle_params(k=k, steps=3000)
        init = make_cradle_state(p, gap=0.3, v0=1.0)
        log = run_sim(p, "local_implicit", init=init)
        transfer = log.vxs[-1, -1]
        incoming = log.vxs[-1, 0]
        print(f"  k={k:8.0f}  vx[0]_final={incoming:+.3f}  vx[N-1]_final={transfer:+.3f}")
        rows.append((k, p, log))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for idx, (k, p, log) in enumerate(rows):
        c = cmap(idx / max(1, len(rows) - 1))
        ax.plot(log.times, log.vxs[:, -1], color=c, label=f"k={k:g}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vx of last body")
    ax.set_title("Cradle transfer vs stiffness")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cradle_sweep_transfer.png"), dpi=120)
    plt.close(fig)
    return rows


def animate_cradle_ideal(outdir: str):
    """Animate a near-ideal cradle transfer using the best-parameters setup."""
    print("\n=== Ideal cradle animation ===")
    mu_frac = 1.0e-3
    k = 1.0e4
    omega = np.sqrt(k / mu_frac)
    dt = 0.3 / omega
    T = 2.0
    steps = int(T / dt)
    p = _cradle_params(k=k, steps=steps, dt=dt, mu_frac=mu_frac)
    init = make_cradle_state(p, gap=0.3, inner_gap=0.02, v0=1.0)
    record_every = max(1, steps // 240)
    log = run_sim(p, "exponential", record_every=record_every, init=init)

    vf = log.vxs[-1]
    print(f"  mu_frac={mu_frac:.0e}  transfer={vf[-1]:.5f}  mid_max={float(np.max(np.abs(vf[1:-1]))):.2e}")

    N = p.N
    s0 = p.s0
    x_min = 0.0
    x_max = float(np.nanmax(log.xs)) + 2.0 * s0 + 0.5
    row_height = 2.0 * s0 + 0.3

    fig, ax = plt.subplots(figsize=(max(9, 0.8 * (x_max - x_min)), 2.4))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.2, row_height + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_yticks([])
    ax.set_title(f"Near-ideal Newton's cradle - exponential integrator, "
                 f"mu_frac={mu_frac:.0e}, transfer={vf[-1]:.4f}")

    cy = s0
    ellipses = []
    for i in range(N):
        e = Ellipse((log.xs[0, i], cy),
                    width=2.0 * log.ss[0, i],
                    height=2.0 * p.s0 ** 2 / log.ss[0, i],
                    facecolor=f"C{i}", alpha=0.85, edgecolor="k")
        ax.add_patch(e)
        ellipses.append(e)

    vx_text = ax.text(0.01, 0.98, "", transform=ax.transAxes,
                      va="top", fontsize=9, family="monospace")

    def update(frame):
        for i, e in enumerate(ellipses):
            s = log.ss[frame, i]
            if not np.isfinite(s) or s <= 0:
                continue
            e.set_center((log.xs[frame, i], cy))
            e.width = 2.0 * s
            e.height = 2.0 * p.s0 ** 2 / s
        vxs = log.vxs[frame]
        vx_text.set_text(
            f"t={log.times[frame]:.3f}s\n"
            f"vx=[{vxs[0]:+.3f} {vxs[1]:+.3f} {vxs[2]:+.3f} {vxs[3]:+.3f} {vxs[4]:+.3f}]"
        )
        return ellipses + [vx_text]

    nframes = log.xs.shape[0]
    anim = FuncAnimation(fig, update, frames=nframes, interval=30, blit=False)
    try:
        anim.save(os.path.join(outdir, "cradle_ideal.gif"), writer="pillow", fps=30)
        print("  wrote cradle_ideal.gif")
    except Exception as e:
        print(f"  gif write failed ({e}); saving final frame instead")
        update(nframes - 1)
        fig.savefig(os.path.join(outdir, "cradle_ideal_final.png"), dpi=120)
    plt.close(fig)


def animate_cradle_sweep(outdir: str):
    print("\n=== Cradle sweep animation ===")
    ks = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
    logs = []
    params = []
    for k in ks:
        p = _cradle_params(k=k, steps=3000)
        init = make_cradle_state(p, gap=0.3, v0=1.0)
        log = run_sim(p, "local_implicit", record_every=10, init=init)
        logs.append(log); params.append(p)

    N = params[0].N
    s0 = params[0].s0
    x_min = 0.0
    x_max = max(np.nanmax(log.xs) for log in logs) + 2.0 * s0 + 0.5
    row_height = 2.0 * s0 + 0.3
    nrows = len(ks)
    fig_h = row_height * nrows + 0.6

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * (x_max - x_min)), fig_h))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.2, row_height * nrows + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_yticks([row_height * r + s0 for r in range(nrows)])
    ax.set_yticklabels([f"k={k:g}" for k in ks], fontsize=9)
    ax.set_title("Newton's cradle - stiffness sweep (local implicit)")

    all_ellipses = []
    for r, (p, log) in enumerate(zip(params, logs)):
        cy = row_height * r + s0
        row_ells = []
        for i in range(N):
            e = Ellipse((log.xs[0, i], cy),
                        width=2.0 * log.ss[0, i],
                        height=2.0 * p.s0 ** 2 / log.ss[0, i],
                        facecolor=f"C{i}", alpha=0.75, edgecolor="k")
            ax.add_patch(e)
            row_ells.append(e)
        all_ellipses.append(row_ells)

    nframes = min(log.xs.shape[0] for log in logs)
    time_text = ax.text(0.01, 0.98, "", transform=ax.transAxes,
                        va="top", fontsize=10, family="monospace")

    def update(frame):
        for r, (p, log, row_ells) in enumerate(zip(params, logs, all_ellipses)):
            cy = row_height * r + s0
            for i, e in enumerate(row_ells):
                s = log.ss[frame, i]
                if not np.isfinite(s) or s <= 0:
                    continue
                e.set_center((log.xs[frame, i], cy))
                e.width = 2.0 * s
                e.height = 2.0 * p.s0 ** 2 / s
        time_text.set_text(f"t = {logs[0].times[frame]:.2f} s")
        return [e for row in all_ellipses for e in row] + [time_text]

    anim = FuncAnimation(fig, update, frames=nframes, interval=30, blit=False)
    try:
        anim.save(os.path.join(outdir, "cradle_sweep.gif"), writer="pillow", fps=30)
        print("  wrote cradle_sweep.gif")
    except Exception as e:
        print(f"  gif write failed ({e}); saving final frame instead")
        update(nframes - 1)
        fig.savefig(os.path.join(outdir, "cradle_sweep_final.png"), dpi=120)
    plt.close(fig)
