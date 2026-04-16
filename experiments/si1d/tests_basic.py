"""Basic sanity tests: single body, 5-body stack, stiffness sweep, and the
stack/sweep animations. These are the original smoke tests from the 1D SI
experiment's first iteration."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

from .solver import Params, analytic_single, analytic_equilibrium, run_sim


def test_single_body(outdir: str):
    print("\n=== TEST 1: single body on floor ===")
    p = Params(N=1, k=1.0e3, steps=8000, dt=1.0e-3)
    log = run_sim(p, "local_implicit")
    s_final = log.ss[-1, 0]
    s_ana = analytic_single(p)
    rel = abs(s_final - s_ana) / s_ana
    print(f"  k={p.k:g}  s_final={s_final:.6f}  analytic={s_ana:.6f}  rel_err={rel:.3e}")
    ok = rel < 1.0e-3
    print(f"  {'PASS' if ok else 'FAIL'} (< 0.1%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(log.times, log.ss[:, 0], label="s(t)")
    ax.axhline(s_ana, color="k", ls="--", label="analytic")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("half-extent s")
    ax.set_title("Single body on floor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "single_body.png"), dpi=120)
    plt.close(fig)
    return ok


def test_stack(outdir: str):
    print("\n=== TEST 2: 5-body stack ===")
    p = Params(N=5, k=1.0e3, steps=8000, dt=1.0e-3)

    log_li = run_sim(p, "local_implicit")
    log_nv = run_sim(p, "naive")

    x_eq, s_eq = analytic_equilibrium(p)
    print("  local_implicit final s vs analytic:")
    for i in range(p.N):
        print(f"    body {i}: s={log_li.ss[-1, i]:.6f}  analytic={s_eq[i]:.6f}  "
              f"rel={abs(log_li.ss[-1, i]-s_eq[i])/s_eq[i]:.3e}")
    max_rel = np.max(np.abs(log_li.ss[-1] - s_eq) / s_eq)
    print(f"  max rel err (local_implicit): {max_rel:.3e}")
    print(f"  naive stable: {log_nv.stable}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, log, title in zip(axes, [log_li, log_nv], ["local implicit", "naive"]):
        ax.plot(log.times, log.KE, label="KE")
        ax.plot(log.times, log.PE_g - log.PE_g[0], label="dPE_grav")
        ax.plot(log.times, log.PE_e, label="PE_elastic")
        ax.plot(log.times, log.total - log.total[0], label="dTotal", color="k")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("energy (J)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "stack_energy.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(p.N)
    ax.plot(idx, s_eq, "k--o", label="analytic")
    ax.plot(idx, log_li.ss[-1], "C0s", label="local implicit")
    if log_nv.stable:
        ax.plot(idx, log_nv.ss[-1], "C3x", label="naive")
    ax.set_xlabel("body index (bottom->top)")
    ax.set_ylabel("final s")
    ax.set_title(f"Final compression (k={p.k:g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "stack_compression.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if log_li.final_residuals:
        ax.semilogy(log_li.final_residuals, "o-")
    ax.set_xlabel("SI sweep")
    ax.set_ylabel("max |dlam|")
    ax.set_title("Convergence - final step (local implicit)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "stack_convergence.png"), dpi=120)
    plt.close(fig)

    return max_rel < 5.0e-2


def test_stiffness_sweep(outdir: str):
    print("\n=== TEST 3: stiffness sweep ===")
    ks = [3.0e2, 1.0e3, 3.0e3, 1.0e4, 1.0e5, 1.0e6]
    results = []
    for k in ks:
        p = Params(N=5, k=k, steps=6000, dt=1.0e-3)
        log_li = run_sim(p, "local_implicit")
        log_nv = run_sim(p, "naive")
        x_eq, s_eq = analytic_equilibrium(p)
        max_rel_li = np.max(np.abs(log_li.ss[-1] - s_eq) / s_eq)
        if log_nv.stable:
            max_rel_nv = np.max(np.abs(log_nv.ss[-1] - s_eq) / s_eq)
            nv_str = f"{max_rel_nv:.2e}"
        else:
            nv_str = "BLOWUP"
        mean_sweeps = np.mean(log_li.sweeps[log_li.sweeps > 0])
        print(f"  k={k:8.0f}  LI_err={max_rel_li:.2e}  naive_err={nv_str}  "
              f"mean_sweeps={mean_sweeps:5.2f}")
        results.append((k, p, log_li, log_nv, s_eq))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for idx, (k, p, log_li, _, _) in enumerate(results):
        color = cmap(idx / max(1, len(results) - 1))
        drift = log_li.total - log_li.total[0]
        ax.plot(log_li.times, drift, color=color, label=f"k={k:g}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("total energy drift (J)")
    ax.set_title("Energy drift across stiffness sweep (local implicit)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sweep_energy_drift.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(2.4 * len(results), 3), sharey=True)
    for ax, (k, p, log_li, log_nv, s_eq) in zip(axes, results):
        idx = np.arange(p.N)
        ax.plot(idx, s_eq, "k--")
        ax.plot(idx, log_li.ss[-1], "C0o-", label="local-impl")
        if log_nv.stable:
            ax.plot(idx, log_nv.ss[-1], "C3x-", label="naive")
        ax.set_title(f"k={k:g}")
        ax.set_xlabel("body")
    axes[0].set_ylabel("final s")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sweep_compression.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ks_arr = np.array([r[0] for r in results])
    mean_sweeps = np.array([np.mean(r[2].sweeps[r[2].sweeps > 0]) for r in results])
    ax.semilogx(ks_arr, mean_sweeps, "o-")
    ax.set_xlabel("stiffness k")
    ax.set_ylabel("mean SI sweeps / step")
    ax.set_title("SI cost vs stiffness")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sweep_cost.png"), dpi=120)
    plt.close(fig)

    return True


def animate_sweep(outdir: str):
    print("\n=== Sweep animation ===")
    ks = [3.0e2, 1.0e3, 3.0e3, 1.0e4, 1.0e5, 1.0e6]
    logs = []
    params = []
    for k in ks:
        p = Params(N=5, k=k, steps=3000, dt=1.0e-3)
        log = run_sim(p, "local_implicit", record_every=10)
        logs.append(log)
        params.append(p)

    N = params[0].N
    s0 = params[0].s0
    col_width = 4.0 * s0
    col_centers = np.arange(len(ks)) * col_width
    y_top = 2 * N * s0 + 0.5

    fig, ax = plt.subplots(figsize=(1.6 * len(ks), 8))
    ax.set_xlim(-col_width / 2, col_centers[-1] + col_width / 2)
    ax.set_ylim(-0.5, y_top)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=2)
    ax.set_xticks(col_centers)
    ax.set_xticklabels([f"k={k:g}" for k in ks], fontsize=9)
    ax.set_yticks([])
    ax.set_title("5-body stack settling - stiffness sweep (local implicit)")

    all_ellipses = []
    for col, (p, log) in enumerate(zip(params, logs)):
        cx = col_centers[col]
        col_ells = []
        for i in range(N):
            e = Ellipse((cx, log.xs[0, i]),
                        width=2.0 * p.s0 ** 2 / log.ss[0, i],
                        height=2.0 * log.ss[0, i],
                        facecolor=f"C{i}", alpha=0.75, edgecolor="k")
            ax.add_patch(e)
            col_ells.append(e)
        all_ellipses.append(col_ells)

    nframes = min(log.xs.shape[0] for log in logs)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        va="top", fontsize=10, family="monospace")

    def update(frame):
        for col, (p, log, col_ells) in enumerate(zip(params, logs, all_ellipses)):
            cx = col_centers[col]
            for i, e in enumerate(col_ells):
                s = log.ss[frame, i]
                if not np.isfinite(s) or s <= 0:
                    continue
                e.set_center((cx, log.xs[frame, i]))
                e.width = 2.0 * p.s0 ** 2 / s
                e.height = 2.0 * s
        time_text.set_text(f"t = {logs[0].times[frame]:.2f} s")
        return [e for col_ells in all_ellipses for e in col_ells] + [time_text]

    anim = FuncAnimation(fig, update, frames=nframes, interval=30, blit=False)
    try:
        anim.save(os.path.join(outdir, "sweep.gif"), writer="pillow", fps=30)
        print("  wrote sweep.gif")
    except Exception as e:
        print(f"  gif write failed ({e}); saving final frame instead")
        update(nframes - 1)
        fig.savefig(os.path.join(outdir, "sweep_final_frame.png"), dpi=120)
    plt.close(fig)


def animate_stack(outdir: str):
    print("\n=== Animation ===")
    p = Params(N=5, k=5.0e2, steps=3000, dt=1.0e-3)
    log = run_sim(p, "local_implicit", record_every=10)

    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.5, 2 * p.N * p.s0 + 0.5)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=2)

    ellipses = []
    for i in range(p.N):
        e = Ellipse((0.0, log.xs[0, i]), width=2.0 * p.s0 ** 2 / log.ss[0, i],
                    height=2.0 * log.ss[0, i], facecolor=f"C{i}", alpha=0.7,
                    edgecolor="k")
        ax.add_patch(e)
        ellipses.append(e)

    def update(frame):
        for i, e in enumerate(ellipses):
            s = log.ss[frame, i]
            e.set_center((0.0, log.xs[frame, i]))
            e.width = 2.0 * p.s0 ** 2 / s
            e.height = 2.0 * s
        return ellipses

    nframes = log.xs.shape[0]
    anim = FuncAnimation(fig, update, frames=nframes, interval=30, blit=False)
    try:
        anim.save(os.path.join(outdir, "stack.gif"), writer="pillow", fps=30)
        print("  wrote stack.gif")
    except Exception as e:
        print(f"  gif write failed ({e}); saving final frame instead")
        update(nframes - 1)
        fig.savefig(os.path.join(outdir, "stack_final_frame.png"), dpi=120)
    plt.close(fig)
