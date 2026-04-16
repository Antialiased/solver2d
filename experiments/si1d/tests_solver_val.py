"""Solver validation probes from docs/plans/1d_solver_validation.md:
warm-start benefit, long-chain PGS scaling, dt refinement (linear and
nonlinear), mu_frac sensitivity, and substep vs inner-iteration
trade-off."""

import os
import time as _time

import numpy as np
import matplotlib.pyplot as plt

from .solver import (Params, make_cradle_state, make_initial_state, run_sim,
                     si_step)
from .test_utils import descend_then_hold, sweep_mean


def test_warm_start_benefit(outdir: str):
    print("\n=== TEST: warm-start benefit on PGS convergence ===")
    N = 5
    dt = 5.0e-4

    ks = [1e3, 1e6, 1e9]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in ks:
        for warm, ls in [(False, "--"), (True, "-")]:
            p = Params(N=N, k=k, dt=dt, steps=1, g=0, damping=0, mu_frac=0.1,
                       warm_start=warm, max_sweeps=500, tol=1e-14)
            state = make_initial_state(p)
            state.ceiling_x = float(state.x[-1] + state.s[-1] - 0.3)
            if warm:
                p_seed = Params(N=N, k=k, dt=dt, steps=1, g=0, damping=0,
                                mu_frac=0.1, warm_start=True, max_sweeps=500,
                                tol=1e-14)
                si_step(state, p_seed, mode="local_implicit")
            res_dl, _, _, _ = si_step(state, p, mode="local_implicit")
            axes[0].semilogy(np.arange(1, len(res_dl) + 1),
                             np.maximum(np.array(res_dl), 1e-18),
                             ls, label=f"k={k:g} {'warm' if warm else 'cold'}")
    axes[0].set_xlabel("PGS sweep")
    axes[0].set_ylabel("max |dlam|")
    axes[0].set_title("Single-step residual decay (warm vs cold)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    ks2 = np.logspace(2, 12, 11)
    rows = {"cold": [], "warm": []}
    stable = {"cold": [], "warm": []}
    for warm in (False, True):
        tag = "warm" if warm else "cold"
        for k in ks2:
            p = Params(N=N, k=k, dt=dt, steps=2000, g=0, damping=2.0,
                       mu_frac=0.1, warm_start=warm, max_sweeps=300, tol=1e-10)
            stack_top = N * 1.0
            fn = descend_then_hold(5.0, stack_top, stack_top - 0.5)
            log = run_sim(p, "local_implicit", ceiling_fn=fn)
            stable[tag].append(log.stable)
            rows[tag].append(sweep_mean(log) if log.stable else float("nan"))
            print(f"  {tag:<4s}  k={k:8.1e}  stable={int(log.stable)}  "
                  f"mean_sweeps={rows[tag][-1]:.1f}")
    axes[1].semilogx(ks2, rows["cold"], "o--", label="cold")
    axes[1].semilogx(ks2, rows["warm"], "s-",  label="warm")
    axes[1].axhline(50, color="r", lw=0.5, ls=":", label="plan target (<=50)")
    axes[1].set_xlabel("stiffness k")
    axes[1].set_ylabel("mean sweeps/step over crush")
    axes[1].set_title("Warm-start rescue? (solver work vs k)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "warm_start_benefit.png"), dpi=120)
    plt.close(fig)
    return all(stable["warm"])


def test_long_chain_scaling(outdir: str):
    print("\n=== TEST: long-chain PGS scaling ===")
    Ns = [5, 10, 20, 40, 80]
    k = 1.0e5
    dt = 5.0e-4
    rows = []
    for N in Ns:
        steps = 3000
        p = Params(N=N, k=k, dt=dt, steps=steps, g=9.81, damping=2.0,
                   mu_frac=0.1, warm_start=True, max_sweeps=200, tol=1e-8)
        stack_top_init = 2.0 * p.s0 * N
        c0 = stack_top_init + 0.05
        c1 = stack_top_init - 0.5
        t_start, t_end = 0.5, 1.0
        def fn(t, c0=c0, c1=c1, t_start=t_start, t_end=t_end):
            if t < t_start: return c0
            if t < t_end:
                a = (t - t_start) / (t_end - t_start)
                return c0 + (c1 - c0) * a
            return c1
        log = run_sim(p, "local_implicit", ceiling_fn=fn)
        if not log.stable:
            print(f"  N={N:4d}  UNSTABLE")
            rows.append((N, False, float("nan"), float("nan"), float("nan")))
            continue
        cut = int(0.7 * len(log.sweeps))
        sw = float(np.mean(log.sweeps[cut:][log.sweeps[cut:] > 0]))
        xs, ss = log.xs, log.ss
        floor_gap = xs[:, 0] - ss[:, 0]
        adj = xs[:, 1:] - ss[:, 1:] - xs[:, :-1] - ss[:, :-1]
        max_pen = float(max(0.0, -np.nanmin(np.concatenate([floor_gap.ravel(), adj.ravel()]))))
        ke = log.KE
        mask = log.times >= t_end
        if mask.any():
            ke_peak = float(np.nanmax(ke[mask]))
            below = np.where(mask & (ke < 0.01 * max(ke_peak, 1e-18)))[0]
            t_eq = float(log.times[below[0]] - t_end) if len(below) else float("nan")
        else:
            t_eq = float("nan")
        rows.append((N, True, sw, max_pen, t_eq))
        print(f"  N={N:4d}  mean_sweeps(tail)={sw:7.1f}  max_pen={max_pen:.2e}  "
              f"t_eq={t_eq:.3f}s")

    Ns_ok = np.array([r[0] for r in rows if r[1]], dtype=float)
    sws   = np.array([r[2] for r in rows if r[1]], dtype=float)
    pens  = np.array([r[3] for r in rows if r[1]], dtype=float)
    teqs  = np.array([r[4] for r in rows if r[1]], dtype=float)

    if len(Ns_ok) >= 2:
        a, b = np.polyfit(np.log(Ns_ok), np.log(sws), 1)
        fit_label = f"fit: sweeps ~ N^{a:.2f}"
    else:
        a = float("nan"); fit_label = "no fit"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].loglog(Ns_ok, sws, "o-", label="measured")
    if len(Ns_ok) >= 2:
        Nf = np.linspace(Ns_ok.min(), Ns_ok.max(), 50)
        axes[0].loglog(Nf, np.exp(b) * Nf ** a, "k--", lw=1, label=fit_label)
        axes[0].loglog(Nf, sws[0] * (Nf / Ns_ok[0]) ** 2, "r:", lw=1, label="N^2 guide")
    axes[0].set_xlabel("chain length N")
    axes[0].set_ylabel("mean sweeps/step (tail)")
    axes[0].set_title("PGS work vs N")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].loglog(Ns_ok, np.maximum(pens, 1e-18), "s-")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("max gap violation")
    axes[1].set_title("Penetration vs N")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].semilogx(Ns_ok, teqs, "d-")
    axes[2].set_xlabel("N")
    axes[2].set_ylabel("time to 1% KE (s)")
    axes[2].set_title("Settling time vs N")
    axes[2].grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Long-chain PGS scaling (k={k:g}, dt={dt}, warm on)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "long_chain_scaling.png"), dpi=120)
    plt.close(fig)
    return all(r[1] for r in rows)


def test_dt_refinement(outdir: str):
    print("\n=== TEST: dt refinement / temporal order ===")
    N = 2
    t_end = 0.1
    dts = [2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4]
    dt_ref = 1.0e-5

    def run_to(dt: float):
        steps = int(round(t_end / dt))
        p = Params(N=N, k=1.0e4, dt=dt, steps=steps, g=0.0, damping=0.0,
                   mu_frac=1.0/3.0, warm_start=True, max_sweeps=200, tol=1e-12,
                   floor=False)
        init = make_cradle_state(p, gap=0.05, v0=2.0)
        log = run_sim(p, "local_implicit", init=init)
        return float(log.xs[-1, 0]), float(log.xs[-1, -1]), log.stable

    print(f"  reference dt={dt_ref:g} ...")
    x0_ref, xN_ref, ok = run_to(dt_ref)
    print(f"  reference: x0={x0_ref:.8f}  xN={xN_ref:.8f}  stable={ok}")

    errs0 = []
    errsN = []
    for dt in dts:
        x0, xN, ok = run_to(dt)
        e0 = abs(x0 - x0_ref)
        eN = abs(xN - xN_ref)
        errs0.append(e0); errsN.append(eN)
        print(f"  dt={dt:8.2e}  x0={x0:.6f}  err0={e0:.3e}  errN={eN:.3e}  stable={ok}")

    dts_arr = np.array(dts)
    errs0 = np.array(errs0)
    errsN = np.array(errsN)

    def slope(y):
        m = y > 0
        if m.sum() < 2:
            return float("nan")
        return float(np.polyfit(np.log(dts_arr[m]), np.log(y[m]), 1)[0])

    s0 = slope(errs0)
    sN = slope(errsN)
    print(f"  slope err(x_0)  = {s0:.3f}")
    print(f"  slope err(x_-1) = {sN:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(dts_arr, np.maximum(errs0, 1e-18), "o-", label=f"|dx0| (slope {s0:.2f})")
    ax.loglog(dts_arr, np.maximum(errsN, 1e-18), "s-", label=f"|dxN| (slope {sN:.2f})")
    dtf = np.array([dts_arr.min(), dts_arr.max()])
    ax.loglog(dtf, errs0[0] * (dtf / dts_arr[0]), "k--", lw=0.8, label="slope 1 guide")
    ax.loglog(dtf, errs0[0] * (dtf / dts_arr[0]) ** 2, "r:", lw=0.8, label="slope 2 guide")
    ax.set_xlabel("dt")
    ax.set_ylabel(f"error vs dt={dt_ref:g} reference at t={t_end}s")
    ax.set_title("dt refinement - 2-body cradle")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dt_refinement.png"), dpi=120)
    plt.close(fig)
    return np.all(np.isfinite(errs0))


def test_dt_refinement_nonlinear(outdir: str):
    print("\n=== TEST: dt refinement, nonlinear elasticity ===")
    N = 2
    t_end = 0.1
    dts = [2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4, 6.25e-5]
    dt_ref = 2.0e-5
    betas = [0.0, 1.0e6, 1.0e7, 1.0e8]

    def run_to(dt, beta, mode="local_implicit", v0=3.0):
        steps = int(round(t_end / dt))
        p = Params(N=N, k=1.0e4, dt=dt, steps=steps, g=0.0, damping=0.0,
                   mu_frac=1.0/3.0, warm_start=True, max_sweeps=300, tol=1e-12,
                   floor=False, beta3=beta)
        init = make_cradle_state(p, gap=0.05, v0=v0)
        log = run_sim(p, mode, init=init)
        return log

    modes = [("local_implicit", "frozen"), ("local_implicit_relin", "relin")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    markers = ["o", "s", "^", "d"]
    print(f"  reference dt={dt_ref:g}, observable x_0(t={t_end})")
    for ax, (mode, tag) in zip(axes, modes):
        print(f"  -- mode: {tag} --")
        for beta, marker in zip(betas, markers):
            log_ref = run_to(dt_ref, beta, mode=mode)
            x0_ref = float(log_ref.xs[-1, 0])
            u = log_ref.ss - Params().s0
            k_val = 1.0e4
            ratio = (beta * float(np.max(np.abs(u))) ** 2) / k_val if beta > 0 else 0.0

            errs = []
            for dt in dts:
                log = run_to(dt, beta, mode=mode)
                x0 = float(log.xs[-1, 0])
                errs.append(abs(x0 - x0_ref))
            errs = np.array(errs)
            dts_arr = np.array(dts)
            mvalid = errs > 0
            if mvalid.sum() >= 2:
                slope_all = float(np.polyfit(np.log(dts_arr[mvalid]), np.log(errs[mvalid]), 1)[0])
                if mvalid.sum() >= 4:
                    slope_coarse = float(np.polyfit(np.log(dts_arr[:3]), np.log(np.maximum(errs[:3], 1e-18)), 1)[0])
                    slope_fine   = float(np.polyfit(np.log(dts_arr[-3:]), np.log(np.maximum(errs[-3:], 1e-18)), 1)[0])
                else:
                    slope_coarse = slope_fine = slope_all
            else:
                slope_all = slope_coarse = slope_fine = float("nan")
            print(f"  [{tag}] beta3={beta:8.1e}  nonlin_ratio~{ratio:5.3f}  "
                  f"slope_all={slope_all:+.2f}  slope_coarse={slope_coarse:+.2f}  "
                  f"slope_fine={slope_fine:+.2f}")
            for dt, e in zip(dts, errs):
                print(f"    dt={dt:8.2e}  dx0={e:.3e}")
            ax.loglog(dts_arr, np.maximum(errs, 1e-18), marker + "-",
                      label=f"b3={beta:.0e} ({slope_coarse:+.2f}->{slope_fine:+.2f})")

        dtf = np.array([min(dts), max(dts)])
        e_anchor = 1.0e-4
        ax.loglog(dtf, e_anchor * (dtf / dtf[-1]), "k--", lw=0.7, label="slope 1")
        ax.loglog(dtf, e_anchor * (dtf / dtf[-1]) ** 2, "r:", lw=0.7, label="slope 2")
        ax.set_xlabel("dt")
        if ax is axes[0]:
            ax.set_ylabel(f"|dx0| vs dt={dt_ref:g} reference")
        ax.set_title(f"dt refinement - {tag} tangent")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Nonlinear dt refinement: frozen vs relinearized tangent")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dt_refinement_nonlinear.png"), dpi=120)
    plt.close(fig)
    return True


def test_mu_frac_sensitivity(outdir: str):
    print("\n=== TEST: mu_frac sensitivity ===")
    N = 5
    k = 1.0e5
    dt = 5.0e-4
    steps = 3000
    mfs = [0.01, 0.1, 0.33, 1.0, 3.0]
    stack_top = N * 1.0
    rows = []
    for mf in mfs:
        for beta in (0.0, 1.0e5):
            p = Params(N=N, k=k, dt=dt, steps=steps, g=0.0, damping=2.0,
                       mu_frac=mf, warm_start=True, max_sweeps=300, tol=1e-10,
                       beta3=beta)
            fn = descend_then_hold(5.0, stack_top, stack_top - 0.5)
            log = run_sim(p, "local_implicit", ceiling_fn=fn)
            stable = log.stable
            sw = sweep_mean(log) if stable else float("nan")
            if stable:
                tail = log.ss[int(0.9 * len(log.ss)):]
                s_mean = np.nanmean(tail, axis=0)
                std_end = float(np.nanstd(tail[-1]))
            else:
                s_mean = np.full(N, np.nan)
                std_end = float("nan")
            rows.append((mf, beta, stable, sw, std_end, log.max_newton))
            print(f"  mu_frac={mf:5.2f}  beta3={beta:6.0e}  stable={int(stable)}  "
                  f"mean_sweeps={sw:7.2f}  std_end={std_end:.3e}  "
                  f"max_newton={log.max_newton}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    mfs_arr = np.array(mfs)
    for beta_tag, beta_val, marker in [("b=0", 0.0, "o"), ("b=1e5", 1.0e5, "s")]:
        sub = [r for r in rows if r[1] == beta_val]
        sw_v = np.array([r[3] for r in sub])
        std_v = np.array([r[4] for r in sub])
        nit_v = np.array([r[5] for r in sub])
        axes[0].semilogx(mfs_arr, sw_v, marker + "-", label=beta_tag)
        axes[1].loglog(mfs_arr, np.maximum(std_v, 1e-18), marker + "-", label=beta_tag)
        axes[2].semilogx(mfs_arr, nit_v, marker + "-", label=beta_tag)
    axes[0].set_xlabel("mu_frac"); axes[0].set_ylabel("mean sweeps/step")
    axes[0].set_title("Solver work"); axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("mu_frac"); axes[1].set_ylabel("std of final s")
    axes[1].set_title("Steady-state variance"); axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[2].set_xlabel("mu_frac"); axes[2].set_ylabel("max Newton iters (free flight)")
    axes[2].set_title("Nonlinear inner cost"); axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"mu_frac sensitivity (N={N}, k={k:g})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mu_frac_sensitivity.png"), dpi=120)
    plt.close(fig)
    return all(r[2] for r in rows)


def test_substep_tradeoff(outdir: str):
    print("\n=== TEST: substep vs inner-iteration trade-off ===")
    budget = 200
    Ks = [1, 2, 4, 8, 16]
    configs = [(K, budget // K) for K in Ks]

    print("  -- cradle impact --")
    N_c = 2
    t_end = 0.1
    dt_macro = 1.0e-3
    rows_c = []

    def cradle_run(dt, steps, ms):
        p = Params(N=N_c, k=1.0e4, dt=dt, steps=steps, g=0.0, damping=0.0,
                   mu_frac=1.0/3.0, warm_start=True, max_sweeps=ms, tol=1e-12,
                   floor=False)
        init = make_cradle_state(p, gap=0.05, v0=2.0)
        t0 = _time.perf_counter()
        log = run_sim(p, "local_implicit", init=init)
        dt_wall = _time.perf_counter() - t0
        return log, dt_wall

    log_ref_c, _ = cradle_run(2.0e-5, int(round(t_end / 2.0e-5)), 400)
    x0_ref_c = float(log_ref_c.xs[-1, 0])

    for K, ms in configs:
        dt = dt_macro / K
        steps = int(round(t_end / dt))
        log, wt = cradle_run(dt, steps, ms)
        x0_err = abs(float(log.xs[-1, 0]) - x0_ref_c)
        drift = float(log.total[-1] - log.total[0])
        rows_c.append((K, ms, x0_err, drift, wt, log.stable))
        print(f"    K={K:3d}  ms={ms:4d}  dx0={x0_err:.2e}  drift={drift:+.2e}  "
              f"wall={wt*1000:6.1f}ms  stable={int(log.stable)}")

    print("  -- ceiling crush --")
    N_s = 5
    t_end_s = 1.5
    dt_macro_s = 5.0e-4
    stack_top = N_s * 1.0
    crush_fn = descend_then_hold(5.0, stack_top, stack_top - 0.5)
    rows_s = []

    def crush_run(dt, steps, ms):
        p = Params(N=N_s, k=1.0e5, dt=dt, steps=steps, g=0.0, damping=2.0,
                   mu_frac=0.1, warm_start=True, max_sweeps=ms, tol=1e-10)
        t0 = _time.perf_counter()
        log = run_sim(p, "local_implicit", ceiling_fn=crush_fn)
        dt_wall = _time.perf_counter() - t0
        return log, dt_wall

    log_ref_s, _ = crush_run(5.0e-5, int(round(t_end_s / 5.0e-5)), 300)
    s_ref = log_ref_s.ss[-1].copy()

    for K, ms in configs:
        dt = dt_macro_s / K
        steps = int(round(t_end_s / dt))
        log, wt = crush_run(dt, steps, ms)
        if not log.stable:
            rows_s.append((K, ms, float("nan"), float("nan"), wt, False))
            print(f"    K={K:3d}  ms={ms:4d}  UNSTABLE  wall={wt:.2f}s")
            continue
        s_err = float(np.max(np.abs(log.ss[-1] - s_ref)))
        xs, ss = log.xs, log.ss
        floor_gap = xs[:, 0] - ss[:, 0]
        adj = xs[:, 1:] - ss[:, 1:] - xs[:, :-1] - ss[:, :-1]
        ceil_gap = log.ceilings - (xs[:, -1] + ss[:, -1])
        max_pen = float(max(0.0, -np.nanmin(np.concatenate(
            [floor_gap.ravel(), adj.ravel(), ceil_gap.ravel()]))))
        rows_s.append((K, ms, s_err, max_pen, wt, True))
        print(f"    K={K:3d}  ms={ms:4d}  s_err={s_err:.2e}  max_pen={max_pen:.2e}  "
              f"wall={wt:.2f}s  stable=1")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    Ks_arr = np.array([r[0] for r in rows_c], dtype=float)

    err_c  = np.array([r[2] for r in rows_c])
    drift_c = np.abs(np.array([r[3] for r in rows_c]))
    wall_c = np.array([r[4] for r in rows_c])

    axes[0, 0].loglog(Ks_arr, np.maximum(err_c, 1e-18), "o-")
    axes[0, 0].set_xlabel("K (substeps)"); axes[0, 0].set_ylabel("|dx0| vs reference")
    axes[0, 0].set_title("Cradle: accuracy")
    axes[0, 0].grid(True, which="both", alpha=0.3)

    axes[0, 1].loglog(Ks_arr, np.maximum(drift_c, 1e-18), "o-")
    axes[0, 1].set_xlabel("K"); axes[0, 1].set_ylabel("|total energy drift|")
    axes[0, 1].set_title("Cradle: energy drift")
    axes[0, 1].grid(True, which="both", alpha=0.3)

    axes[0, 2].loglog(wall_c, np.maximum(err_c, 1e-18), "o-")
    for (K, ms, e, *_), wt in zip(rows_c, wall_c):
        axes[0, 2].annotate(f"K={K}", (wt, max(e, 1e-18)), fontsize=7)
    axes[0, 2].set_xlabel("wall time (s)"); axes[0, 2].set_ylabel("|dx0|")
    axes[0, 2].set_title("Cradle: Pareto (error vs cost)")
    axes[0, 2].grid(True, which="both", alpha=0.3)

    err_s  = np.array([r[2] for r in rows_s])
    pen_s  = np.array([r[3] for r in rows_s])
    wall_s = np.array([r[4] for r in rows_s])

    axes[1, 0].loglog(Ks_arr, np.maximum(err_s, 1e-18), "o-")
    axes[1, 0].set_xlabel("K"); axes[1, 0].set_ylabel("max |ds| vs reference")
    axes[1, 0].set_title("Crush: accuracy")
    axes[1, 0].grid(True, which="both", alpha=0.3)

    axes[1, 1].loglog(Ks_arr, np.maximum(pen_s, 1e-18), "o-")
    axes[1, 1].set_xlabel("K"); axes[1, 1].set_ylabel("max gap violation")
    axes[1, 1].set_title("Crush: penetration")
    axes[1, 1].grid(True, which="both", alpha=0.3)

    axes[1, 2].loglog(wall_s, np.maximum(err_s, 1e-18), "o-")
    for (K, ms, e, *_), wt in zip(rows_s, wall_s):
        axes[1, 2].annotate(f"K={K}", (wt, max(e, 1e-18)), fontsize=7)
    axes[1, 2].set_xlabel("wall time (s)"); axes[1, 2].set_ylabel("max |ds|")
    axes[1, 2].set_title("Crush: Pareto (error vs cost)")
    axes[1, 2].grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Substep vs sweep trade-off  (budget K*ms = {budget})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "substep_tradeoff.png"), dpi=120)
    plt.close(fig)
    return all(r[5] for r in rows_c) and all(r[5] for r in rows_s)
