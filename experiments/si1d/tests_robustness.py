"""Robustness probes: fast impact, mixed stiffness/mass ratios, PGS
convergence scaling vs k, mass-ratio crush, cubic-force inversion stability.

These stress edge cases of the hard-contact PGS path — very high impact
speeds, extreme parameter ratios, and the compressive inversion regime
where the linear elastic well might be lost."""

import os

import numpy as np
import matplotlib.pyplot as plt

from .solver import Params, make_initial_state, run_sim, si_step
from .test_utils import descend_then_hold, sweep_mean


def test_fast_impact(outdir: str):
    print("\n=== TEST: fast-impact stability sweep ===")
    N = 5
    k = 1.0e5
    dt = 5.0e-4
    steps = 4000
    vs = [1.0, 10.0, 1.0e2, 1.0e3, 1.0e4]

    stack_top = N * 1.0
    squash = 0.5
    x_final = stack_top - squash

    rows = []
    for v in vs:
        p = Params(N=N, k=k, dt=dt, steps=steps, g=0.0, damping=2.0,
                   mu_frac=0.1, warm_start=False, max_sweeps=200)
        fn = descend_then_hold(v, stack_top, x_final)
        log = run_sim(p, "local_implicit", ceiling_fn=fn)
        stable = log.stable
        if stable:
            top_gap = log.ceilings - (log.xs[:, -1] + log.ss[:, -1])
            max_pen = float(max(0.0, -np.nanmin(top_gap)))
            s_min = float(np.nanmin(log.ss))
            vx_max = float(np.nanmax(np.abs(log.vxs)))
        else:
            max_pen = s_min = vx_max = float("nan")
        sw = sweep_mean(log) if stable else float("nan")
        rows.append((v, stable, max_pen, s_min, vx_max, sw))
        print(f"  v={v:8.1e}  stable={int(stable)}  max_pen={max_pen:.2e}  "
              f"s_min={s_min:+.4f}  |vx|max={vx_max:.2e}  mean_sweeps={sw:.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vs_arr = np.array([r[0] for r in rows])
    pen_arr = np.array([r[2] for r in rows])
    vx_arr = np.array([r[4] for r in rows])
    sw_arr = np.array([r[5] for r in rows])
    axes[0].loglog(vs_arr, np.maximum(pen_arr, 1e-18), "o-")
    axes[0].set_xlabel("impact velocity"); axes[0].set_ylabel("max residual penetration")
    axes[0].set_title("Gap violation vs v"); axes[0].grid(True, alpha=0.3)
    axes[1].loglog(vs_arr, vx_arr, "o-")
    axes[1].set_xlabel("impact velocity"); axes[1].set_ylabel("peak |vx|")
    axes[1].set_title("Rebound velocity"); axes[1].grid(True, alpha=0.3)
    axes[2].semilogx(vs_arr, sw_arr, "o-")
    axes[2].set_xlabel("impact velocity"); axes[2].set_ylabel("mean sweeps/step")
    axes[2].set_title("PGS work"); axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"Fast-impact sweep (N={N}, k={k:g}, dt={dt})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fast_impact.png"), dpi=120)
    plt.close(fig)
    return all(r[1] for r in rows)


def test_mixed_ratios(outdir: str):
    print("\n=== TEST: mixed stiffness+mass crush ===")
    N = 5
    dt = 5.0e-4
    steps = 4000
    configs = [
        ("uniform",     np.full(N, 1.0e5),                      np.full(N, 1.0)),
        ("k-alt 1e3/1e8", np.array([1e3, 1e8, 1e3, 1e8, 1e3]), np.full(N, 1.0)),
        ("m-alt 1/1e-3",  np.full(N, 1.0e5),                    np.array([1.0, 1e-3, 1.0, 1e-3, 1.0])),
        ("k+m alt",      np.array([1e3, 1e8, 1e3, 1e8, 1e3]),  np.array([1e-3, 1.0, 1e-3, 1.0, 1e-3])),
    ]
    rows = []
    for name, kv, mv in configs:
        p = Params(N=N, k=1e5, m=1.0, k_vec=kv, m_vec=mv, dt=dt, steps=steps,
                   g=0.0, damping=2.0, mu_frac=0.1, warm_start=False, max_sweeps=300)
        stack_top = N * 1.0
        fn = descend_then_hold(5.0, stack_top, stack_top - 0.5)
        log = run_sim(p, "local_implicit", ceiling_fn=fn)
        stable = log.stable
        s_end = log.ss[-1].copy() if stable else np.full(N, np.nan)
        sw = sweep_mean(log) if stable else float("nan")
        rows.append((name, stable, s_end, sw))
        print(f"  {name:<16s}  stable={int(stable)}  "
              f"s_end={np.array2string(s_end, precision=3, suppress_small=True)}  "
              f"mean_sweeps={sw:.1f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, stable, s_end, _ in rows:
        if stable:
            ax.plot(range(N), s_end, "o-", label=name)
    ax.set_xlabel("body index (bottom->top)")
    ax.set_ylabel("final s")
    ax.set_title("Mixed stiffness+mass: final compression")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mixed_ratios.png"), dpi=120)
    plt.close(fig)
    return all(r[1] for r in rows)


def test_convergence_scaling(outdir: str):
    print("\n=== TEST: PGS convergence scaling ===")
    N = 5
    dt = 5.0e-4

    ks = [1e3, 1e6, 1e9, 1e12]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in ks:
        p = Params(N=N, k=k, dt=dt, steps=1, g=0, damping=0, mu_frac=0.1,
                   warm_start=False, max_sweeps=500, tol=1e-14)
        state = make_initial_state(p)
        state.ceiling_x = float(state.x[-1] + state.s[-1] - 0.3)
        res_dl, _, _, _ = si_step(state, p, mode="local_implicit")
        axes[0].semilogy(np.arange(1, len(res_dl) + 1),
                         np.maximum(np.array(res_dl), 1e-18),
                         label=f"k={k:g}")
    axes[0].set_xlabel("PGS sweep")
    axes[0].set_ylabel("max |dlam|")
    axes[0].set_title("Single-step residual decay (stressed stack)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    ks2 = np.logspace(2, 12, 11)
    sweeps_li = []
    stable_li = []
    for k in ks2:
        p = Params(N=N, k=k, dt=dt, steps=2000, g=0, damping=2.0, mu_frac=0.1,
                   warm_start=False, max_sweeps=300, tol=1e-10)
        stack_top = N * 1.0
        fn = descend_then_hold(5.0, stack_top, stack_top - 0.5)
        log = run_sim(p, "local_implicit", ceiling_fn=fn)
        stable_li.append(log.stable)
        sweeps_li.append(sweep_mean(log) if log.stable else float("nan"))
        print(f"  k={k:8.1e}  stable={int(log.stable)}  mean_sweeps={sweeps_li[-1]:.1f}")
    axes[1].semilogx(ks2, sweeps_li, "o-")
    axes[1].set_xlabel("stiffness k")
    axes[1].set_ylabel("mean sweeps/step over crush")
    axes[1].set_title("Solver work vs stiffness")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "convergence_scaling.png"), dpi=120)
    plt.close(fig)
    return all(stable_li)


def test_mass_ratio(outdir: str):
    print("\n=== TEST: two-body mass ratio crush ===")
    N = 2
    dt = 5.0e-4
    steps = 4000
    ratios = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    rows = []
    for r in ratios:
        mv = np.array([r, 1.0])
        p = Params(N=N, k=1e5, dt=dt, steps=steps, g=0, damping=2.0,
                   mu_frac=0.1, warm_start=False, max_sweeps=200, m_vec=mv)
        stack_top = N * 1.0
        fn = descend_then_hold(5.0, stack_top, stack_top - 0.4)
        log = run_sim(p, "local_implicit", ceiling_fn=fn)
        stable = log.stable
        s_end = log.ss[-1].copy() if stable else np.full(N, np.nan)
        vx_max = float(np.nanmax(np.abs(log.vxs))) if stable else float("nan")
        sw = sweep_mean(log) if stable else float("nan")
        rows.append((r, stable, s_end, vx_max, sw))
        print(f"  m0/m1={r:.0e}  stable={int(stable)}  "
              f"s_end={np.array2string(s_end, precision=3, suppress_small=True)}  "
              f"|vx|max={vx_max:.2e}  mean_sweeps={sw:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rs = np.array([r[0] for r in rows])
    s0_arr = np.array([r[2][0] for r in rows])
    s1_arr = np.array([r[2][1] for r in rows])
    sw_arr = np.array([r[4] for r in rows])
    axes[0].semilogx(rs, s0_arr, "o-", label="bottom (m=r)")
    axes[0].semilogx(rs, s1_arr, "s-", label="top (m=1)")
    axes[0].set_xlabel("mass ratio m[0]/m[1]")
    axes[0].set_ylabel("final s")
    axes[0].set_title("Final compression")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].semilogx(rs, sw_arr, "o-")
    axes[1].set_xlabel("mass ratio m[0]/m[1]")
    axes[1].set_ylabel("mean sweeps/step")
    axes[1].set_title("PGS work")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mass_ratio.png"), dpi=120)
    plt.close(fig)
    return all(r[1] for r in rows)


def test_nonlinear_inversion(outdir: str):
    print("\n=== TEST: cubic-force inversion stability ===")
    N = 1
    k = 1.0e3
    dt = 5.0e-4
    steps = 4000

    t_start, t_end = 0.25, 1.25
    c0, c1 = 1.0, -0.2

    def ceiling_fn(t):
        if t < t_start:
            return c0
        if t < t_end:
            a = (t - t_start) / (t_end - t_start)
            return c0 + (c1 - c0) * a
        return c1

    betas = [0.0, 1.0e3, 1.0e4, 1.0e5]
    logs = []
    rows = []
    for beta in betas:
        p = Params(N=N, k=k, dt=dt, steps=steps, g=0.0, damping=2.0,
                   mu_frac=0.1, warm_start=False, max_sweeps=200,
                   beta3=beta, floor=True)
        log = run_sim(p, "local_implicit", ceiling_fn=ceiling_fn)
        logs.append(log)
        stable = log.stable
        if stable:
            s_min = float(np.nanmin(log.ss))
            s_end = float(log.ss[-1, 0])
            sw = sweep_mean(log)
        else:
            s_min = s_end = sw = float("nan")
        rows.append((beta, stable, s_min, s_end, sw, log.max_newton))
        print(f"  beta={beta:8.1e}  stable={int(stable)}  s_min={s_min:+.4f}  "
              f"s_end={s_end:+.4f}  mean_sweeps={sw:.1f}  max_newton={log.max_newton}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for (beta, *_), log in zip(rows, logs):
        axes[0].plot(log.times, log.ss[:, 0], label=f"b={beta:.0e}")
    axes[0].plot(logs[0].times, logs[0].ceilings / 2.0, "k--", lw=1.0, label="c(t)/2")
    axes[0].axhline(0.0, color="r", lw=0.5, ls=":")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("body half-extent s")
    axes[0].set_title("s(t) through inversion")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    u_range = np.linspace(-0.8, 0.4, 400)
    for beta, *_ in rows:
        Fe = -(k * u_range + beta * u_range ** 3)
        axes[1].plot(u_range, Fe, label=f"b={beta:.0e}")
    axes[1].axvline(-0.5, color="r", lw=0.5, ls=":", label="u=-s0 (s=0)")
    axes[1].set_xlabel("u = s - s0")
    axes[1].set_ylabel("elastic force F(u)")
    axes[1].set_title("Force law")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    for (beta, *_), log in zip(rows, logs):
        u = log.ss[:, 0] - (p.s0 + log.sps[:, 0])
        Fe_mag = np.abs(k * u + beta * u ** 3)
        axes[2].semilogy(log.times, np.maximum(Fe_mag, 1e-6), label=f"b={beta:.0e}")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("|F_elastic|  (log scale)")
    axes[2].set_title("Internal restoring force")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "nonlinear_inversion.png"), dpi=120)
    plt.close(fig)
    return all(r[1] for r in rows)
