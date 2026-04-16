"""Plasticity tests: single-body crush, stack crush, 2D sweep of (k, sigma_Y),
and the crushed-stack animation."""

import os
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

from .solver import Params, make_initial_state, run_sim


def _crush_ceiling_fn(t_press_start: float, t_press_end: float, t_lift_end: float,
                      x_top: float, x_crush: float) -> Callable[[float], float]:
    """Piecewise-linear ceiling schedule: hold top, press to x_crush, hold, lift."""
    def fn(t: float) -> float:
        if t < t_press_start:
            return x_top
        if t < t_press_end:
            u = (t - t_press_start) / (t_press_end - t_press_start)
            return x_top + (x_crush - x_top) * u
        t_lift_start = t_press_end + 0.2
        if t < t_lift_start:
            return x_crush
        if t < t_lift_end:
            u = (t - t_lift_start) / (t_lift_end - t_lift_start)
            return x_crush + (x_top - x_crush) * u
        return x_top
    return fn


def test_plasticity_single(outdir: str):
    """
    Single body compressed by a descending ceiling with perfect plasticity.
    """
    print("\n=== TEST 7: plasticity, single body crushed by ceiling ===")
    k = 1.0e3
    sigma_Y = 5.0
    p = Params(N=1, k=k, sigma_Y=sigma_Y, g=0.0, damping=2.0, dt=1.0e-3,
               steps=4000, mu_frac=0.1, warm_start=False)
    ceiling_fn = _crush_ceiling_fn(
        t_press_start=0.2, t_press_end=1.5, t_lift_end=3.5,
        x_top=1.2, x_crush=0.70)
    init = make_initial_state(p)
    log = run_sim(p, "local_implicit", record_every=10, init=init, ceiling_fn=ceiling_fn)

    s_final = log.ss[-1, 0]
    sp_final = log.sps[-1, 0]
    s_rest_final = p.s0 + sp_final
    print(f"  final s = {s_final:.5f}   (rest s_rest = {s_rest_final:.5f})")
    print(f"  plastic offset s_p = {sp_final:+.5f}")
    print(f"  after ceiling lift: |s - s_rest| = {abs(s_final - s_rest_final):.2e}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log.times, log.ss[:, 0], label="s (half-extent)")
    ax.plot(log.times, p.s0 + log.sps[:, 0], "--", label="s_rest = s0+s_p")
    ax.plot(log.times, (log.ceilings - 0.0) / 2.0, ":", label="ceiling/2")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("half-extent")
    ax.set_title(f"Plastic crush - k={k:g}, sigmaY={sigma_Y:g}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "plasticity_single.png"), dpi=120)
    plt.close(fig)

    ok = sp_final < -0.05 and abs(s_final - s_rest_final) < 1e-2
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_plasticity_stack(outdir: str):
    """5-body stack crushed by ceiling; bodies should retain shape after lift."""
    print("\n=== TEST 8: plasticity, 5-body stack crushed by ceiling ===")
    k = 1.0e5
    sigma_Y = 30.0
    H_hard = 5.0e3
    p = Params(N=5, k=k, sigma_Y=sigma_Y, H_hard=H_hard, g=0.0, damping=5.0, dt=5.0e-4,
               steps=12000, mu_frac=0.1, warm_start=False, max_sweeps=100)
    init = make_initial_state(p)
    x_top0 = 2.0 * p.s0 * p.N + 0.2
    x_crush = 2.0 * p.s0 * p.N * 0.55
    ceiling_fn = _crush_ceiling_fn(
        t_press_start=0.3, t_press_end=2.5, t_lift_end=5.0,
        x_top=x_top0, x_crush=x_crush)
    log = run_sim(p, "local_implicit", record_every=10, init=init, ceiling_fn=ceiling_fn)

    s_final = log.ss[-1]
    sp_final = log.sps[-1]
    print("  final (s, s_p, s_rest):")
    for i in range(p.N):
        print(f"    body {i}: s={s_final[i]:.4f}  s_p={sp_final[i]:+.4f}  "
              f"s_rest={p.s0+sp_final[i]:.4f}")
    print(f"  total column height final = {np.sum(2*s_final):.4f}  "
          f"(original = {2*p.s0*p.N:.4f}, crushed target ~ {x_crush:.4f})")

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for i in range(p.N):
        axes[0].plot(log.times, log.ss[:, i], color=f"C{i}", label=f"body {i}")
        axes[1].plot(log.times, log.sps[:, i], color=f"C{i}")
    axes[0].plot(log.times, log.ceilings / (2.0 * p.N), "k:", label="ceiling/(2N)")
    axes[0].set_ylabel("half-extent s")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set_ylabel("plastic offset s_p")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title(f"Stack crush - k={k:g}, sigmaY={sigma_Y:g}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "plasticity_stack.png"), dpi=120)
    plt.close(fig)

    ok = np.all(sp_final < 0.0)
    print(f"  {'PASS' if ok else 'FAIL'} (all bodies plastically shortened)")
    return ok


def test_plasticity_sweep(outdir: str):
    """2D sweep over (k, sigma_Y): single-body crush, snapshot at three phases."""
    print("\n=== TEST 9: plasticity 2D sweep (k x sigma_Y) ===")
    ks = [3.0e2, 1.0e3, 3.0e3, 1.0e4]
    sigmas = [2.0, 8.0, 32.0, 128.0]
    fig, axes = plt.subplots(len(sigmas), len(ks),
                             figsize=(2.6 * len(ks), 2.2 * len(sigmas)),
                             sharex=True, sharey=True)
    for si, sY in enumerate(sigmas):
        for ki, k in enumerate(ks):
            p = Params(N=1, k=k, sigma_Y=sY, g=0.0, damping=2.0, dt=1.0e-3,
                       steps=4000, mu_frac=0.1, warm_start=False)
            ceiling_fn = _crush_ceiling_fn(
                t_press_start=0.2, t_press_end=1.5, t_lift_end=3.5,
                x_top=1.2, x_crush=0.70)
            log = run_sim(p, "local_implicit", record_every=20, init=make_initial_state(p),
                          ceiling_fn=ceiling_fn)
            ax = axes[si, ki]
            ax.plot(log.times, log.ss[:, 0], "C0-", lw=1.2, label="s")
            ax.plot(log.times, p.s0 + log.sps[:, 0], "C3--", lw=1.0, label="s_rest")
            ax.plot(log.times, log.ceilings / 2.0, "k:", lw=0.8, label="ceil/2")
            ax.set_ylim(0.0, 0.62)
            ax.set_title(f"k={k:g}, sigmaY={sY:g}", fontsize=8)
            if si == len(sigmas) - 1:
                ax.set_xlabel("t (s)")
            if ki == 0:
                ax.set_ylabel("half-extent")
    axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Plasticity sweep: elastic <-> rigid-plastic spectrum")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "plasticity_sweep.png"), dpi=120)
    plt.close(fig)
    print(f"  wrote plasticity_sweep.png")
    return True


def animate_plasticity_stack(outdir: str):
    print("\n=== Plasticity stack animation ===")
    k = 1.0e5
    sigma_Y = 30.0
    H_hard = 5.0e3
    p = Params(N=5, k=k, sigma_Y=sigma_Y, H_hard=H_hard, g=0.0, damping=5.0, dt=5.0e-4,
               steps=12000, mu_frac=0.1, warm_start=False, max_sweeps=100)
    init = make_initial_state(p)
    x_top0 = 2.0 * p.s0 * p.N + 0.2
    x_crush = 2.0 * p.s0 * p.N * 0.55
    ceiling_fn = _crush_ceiling_fn(
        t_press_start=0.3, t_press_end=2.5, t_lift_end=5.0,
        x_top=x_top0, x_crush=x_crush)
    log = run_sim(p, "local_implicit", record_every=20, init=init, ceiling_fn=ceiling_fn)

    N = p.N
    y_top_plot = x_top0 + 0.4

    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, y_top_plot)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=2)
    ax.set_xticks([])
    ax.set_title(f"Rigid-plastic stack crush (k={k:g}, sigmaY={sigma_Y:g})")

    ellipses = []
    for i in range(N):
        e = Ellipse((0.0, log.xs[0, i]),
                    width=2.0 * p.s0 ** 2 / log.ss[0, i],
                    height=2.0 * log.ss[0, i],
                    facecolor=f"C{i}", alpha=0.8, edgecolor="k")
        ax.add_patch(e)
        ellipses.append(e)

    ceiling_line = ax.axhline(log.ceilings[0], color="0.2", lw=3)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                        fontsize=9, family="monospace")

    def update(frame):
        for i, e in enumerate(ellipses):
            s = log.ss[frame, i]
            if not np.isfinite(s) or s <= 0:
                continue
            e.set_center((0.0, log.xs[frame, i]))
            e.width = 2.0 * p.s0 ** 2 / s
            e.height = 2.0 * s
        ceiling_line.set_ydata([log.ceilings[frame], log.ceilings[frame]])
        sp = log.sps[frame]
        time_text.set_text(
            f"t={log.times[frame]:.2f}s\n"
            f"s_p=[{sp[0]:+.3f} {sp[1]:+.3f} {sp[2]:+.3f} {sp[3]:+.3f} {sp[4]:+.3f}]"
        )
        return ellipses + [ceiling_line, time_text]

    nframes = log.xs.shape[0]
    anim = FuncAnimation(fig, update, frames=nframes, interval=30, blit=False)
    try:
        anim.save(os.path.join(outdir, "plasticity_stack.gif"), writer="pillow", fps=30)
        print("  wrote plasticity_stack.gif")
    except Exception as e:
        print(f"  gif write failed ({e}); saving final frame instead")
        update(nframes - 1)
        fig.savefig(os.path.join(outdir, "plasticity_stack_final.png"), dpi=120)
    plt.close(fig)
