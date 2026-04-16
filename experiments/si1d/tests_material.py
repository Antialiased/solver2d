"""Material-model probes from docs/plans/1d_material_richness.md:
soft-contact COR, asymmetric piecewise-cubic law, and free-oscillation
BE numerical damping."""

import os

import numpy as np
import matplotlib.pyplot as plt

from .solver import Params, make_initial_state, run_sim


def test_cor_soft_contact(outdir: str):
    print("\n=== TEST 6: COR via compliant contact ===")
    dt = 5.0e-4
    mu_frac = 1.0
    k_body = 1.0e10

    wns = [200.0, 1000.0, 5000.0]
    zetas = [0.0, 0.1, 0.3, 0.5, 0.7]
    v_ins = [1.0, 5.0, 20.0]

    def theory(zeta):
        return float(np.exp(-np.pi * zeta / np.sqrt(max(1.0 - zeta * zeta, 1e-12)))) if zeta < 1 else 0.0

    rows = []
    print(f"  dt={dt}, k_body={k_body:g}, mu_frac={mu_frac:g}")
    print(f"  {'wn':>7s} {'wn*dt':>7s} {'zeta':>5s} {'v_in':>6s} "
          f"{'COR_m':>7s} {'COR_t':>7s} {'rel':>8s}")
    for wn in wns:
        for zeta in zetas:
            for v_in in v_ins:
                p = Params(N=1, k=k_body, dt=dt, steps=3000, g=0.0, damping=0.0,
                           mu_frac=mu_frac, warm_start=False, floor=True,
                           contact_wn=wn, contact_zeta=zeta)
                init = make_initial_state(p)
                init.x[0] = p.s0 + 0.5
                init.vx[0] = -v_in
                init.s[0] = p.s0
                init.vs[0] = 0.0
                log = run_sim(p, "local_implicit", init=init)
                gap = log.xs[:, 0] - log.ss[:, 0]
                vx = log.vxs[:, 0]
                touch = gap < 1.0e-5
                if not touch.any():
                    cor = float("nan")
                else:
                    start = int(np.argmax(touch))
                    released = (gap[start + 1:] > 1.0e-4) & (vx[start + 1:] > 0.0)
                    if released.any():
                        rel_idx = int(np.argmax(released)) + start + 1
                        cor = float(abs(vx[rel_idx]) / v_in)
                    else:
                        cor = float("nan")
                cor_t = theory(zeta)
                rel = abs(cor - cor_t) / max(cor_t, 1e-6) if not np.isnan(cor) else float("nan")
                rows.append((wn, zeta, v_in, cor, cor_t, rel))
                print(f"  {wn:7.0f} {wn*dt:7.3f} {zeta:5.2f} {v_in:6.1f} "
                      f"{cor:7.4f} {cor_t:7.4f} {rel:7.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    zetas_arr = np.array(zetas)
    for wn in wns:
        sub = [r for r in rows if r[0] == wn and r[2] == 5.0]
        cor_arr = np.array([r[3] for r in sub])
        axes[0].plot(zetas_arr, cor_arr, "o-", label=f"wn={wn:.0f} (wn*dt={wn*dt:.2f})")
    theory_arr = np.array([theory(z) for z in zetas_arr])
    axes[0].plot(zetas_arr, theory_arr, "k--", lw=1.2, label="exp(-pi*zeta/sqrt(1-zeta^2))")
    axes[0].set_xlabel("damping ratio zeta")
    axes[0].set_ylabel("COR")
    axes[0].set_title("COR vs zeta (v_in=5)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    mid_wn = 1000.0
    for zeta in [0.0, 0.1, 0.3, 0.5]:
        sub = [r for r in rows if r[0] == mid_wn and r[1] == zeta]
        if not sub:
            continue
        v_arr = np.array([r[2] for r in sub])
        cor_arr = np.array([r[3] for r in sub])
        line, = axes[1].plot(v_arr, cor_arr, "o-", label=f"zeta={zeta}")
        axes[1].axhline(theory(zeta), ls=":", color=line.get_color(), alpha=0.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("v_in")
    axes[1].set_ylabel("COR")
    axes[1].set_title(f"COR vs v_in (wn={mid_wn:.0f}) - flat if linear")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cor_soft_contact.png"), dpi=120)
    plt.close(fig)

    print("  -- dt convergence at fixed wn=500, zeta=0.3 --")
    dts_probe = [2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4]
    wn_fixed = 500.0
    zeta_fixed = 0.3
    v0 = 5.0
    cor_theory_fixed = theory(zeta_fixed)
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    errs = []
    for dtp in dts_probe:
        p = Params(N=1, k=k_body, dt=dtp, steps=int(round(1.0 / dtp)),
                   g=0.0, damping=0.0, mu_frac=mu_frac, warm_start=False,
                   floor=True, contact_wn=wn_fixed, contact_zeta=zeta_fixed)
        init = make_initial_state(p)
        init.x[0] = p.s0 + 0.5
        init.vx[0] = -v0
        init.s[0] = p.s0
        init.vs[0] = 0.0
        log = run_sim(p, "local_implicit", init=init)
        gap = log.xs[:, 0] - log.ss[:, 0]
        vx = log.vxs[:, 0]
        touch = gap < 1.0e-5
        if touch.any():
            start = int(np.argmax(touch))
            released = (gap[start + 1:] > 1.0e-4) & (vx[start + 1:] > 0.0)
            if released.any():
                rel_idx = int(np.argmax(released)) + start + 1
                cor = float(abs(vx[rel_idx]) / v0)
            else:
                cor = float("nan")
        else:
            cor = float("nan")
        err = abs(cor - cor_theory_fixed)
        errs.append(err)
        print(f"    dt={dtp:8.2e}  wn*dt={wn_fixed*dtp:6.3f}  "
              f"COR_m={cor:.4f}  err={err:.4f}")
    ax2.loglog(dts_probe, np.maximum(errs, 1e-6), "o-")
    ax2.set_xlabel("dt")
    ax2.set_ylabel(f"|COR - theory| (theory={cor_theory_fixed:.3f})")
    ax2.set_title(f"dt convergence (wn={wn_fixed}, zeta={zeta_fixed})")
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "cor_soft_dt_convergence.png"), dpi=120)
    plt.close(fig2)
    return True


def test_asymmetric_cubic(outdir: str):
    print("\n=== TEST: asymmetric cubic (tension-soft / compression-hard) ===")
    u_range = np.linspace(-0.3, 0.3, 401)
    k_val = 1.0e4
    cases = [
        ("symmetric 1e6",      dict(beta3=1.0e6)),
        ("comp-hard  1e7/0",   dict(beta_comp=1.0e7, beta_tens=0.0)),
        ("tens-hard  0/1e7",   dict(beta_comp=0.0, beta_tens=1.0e7)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_law = axes[0, 0]
    for name, kw in cases:
        p_plot = Params(N=1, k=k_val, **kw)
        beta_arr = p_plot.beta_of(u_range) if p_plot.has_piecewise_beta() else \
                   np.full_like(u_range, p_plot.beta3)
        F = -(k_val * u_range + beta_arr * u_range ** 3)
        ax_law.plot(u_range, F, label=name)
    ax_law.axvline(0, color="k", lw=0.5, ls=":")
    ax_law.axhline(0, color="k", lw=0.5, ls=":")
    ax_law.set_xlabel("u = s - s0")
    ax_law.set_ylabel("F(u)")
    ax_law.set_title("Force law (C1 at u=0)")
    ax_law.legend(fontsize=8)
    ax_law.grid(True, alpha=0.3)

    print("  -- scenario A: free oscillation through u=0 --")
    ax_os = axes[0, 1]
    dt = 1.0e-3
    steps = 6000
    mu_frac = 1.0 / 3.0
    u0 = 0.15
    for name, kw in cases:
        p_osc = Params(N=1, k=k_val, dt=dt, steps=steps, g=0.0, damping=0.0,
                       mu_frac=mu_frac, warm_start=False, floor=False, **kw)
        init = make_initial_state(p_osc)
        init.s[0] = p_osc.s0 + u0
        init.vs[0] = 0.0
        log = run_sim(p_osc, "local_implicit", init=init)
        u_t = log.ss[:, 0] - p_osc.s0
        ax_os.plot(log.times, u_t, label=f"{name}  Newton<={log.max_newton}")
        u_min = float(np.nanmin(u_t))
        u_max = float(np.nanmax(u_t))
        print(f"    {name:<22s}  u_max={u_max:+.4f}  u_min={u_min:+.4f}  "
              f"max_newton={log.max_newton}")
    ax_os.axhline(0, color="k", lw=0.5, ls=":")
    ax_os.set_xlabel("t (s)")
    ax_os.set_ylabel("u = s - s0")
    ax_os.set_title("Free oscillation through u=0")
    ax_os.legend(fontsize=7)
    ax_os.grid(True, alpha=0.3)

    print("  -- scenario B: drop onto floor --")
    ax_s = axes[1, 0]
    ax_x = axes[1, 1]
    k_drop = 1.0e4
    drop_cases = [
        ("linear",           dict(beta3=0.0)),
        ("sym b3=1e7",       dict(beta3=1.0e7)),
        ("comp-hard 1e7/0",  dict(beta_comp=1.0e7, beta_tens=0.0)),
    ]
    rows_drop = []
    for name, kw in drop_cases:
        p_drop = Params(N=1, k=k_drop, dt=5.0e-4, steps=6000, g=9.81,
                        damping=0.5, mu_frac=mu_frac, warm_start=True,
                        floor=True, **kw)
        init = make_initial_state(p_drop)
        init.x[0] = p_drop.s0 + 0.5
        init.vx[0] = 0.0
        init.s[0] = p_drop.s0
        init.vs[0] = 0.0
        log = run_sim(p_drop, "local_implicit", init=init)
        s_min = float(np.nanmin(log.ss[:, 0]))
        u_min = s_min - p_drop.s0
        x_peak = float(np.nanmax(log.xs[:, 0]))
        xs_arr = log.xs[:, 0]
        if len(xs_arr) > 10:
            bot = int(np.argmin(xs_arr[:len(xs_arr)//2]))
            if bot + 10 < len(xs_arr):
                reb = float(np.nanmax(xs_arr[bot:]))
            else:
                reb = x_peak
        else:
            reb = x_peak
        rows_drop.append((name, u_min, reb, log.max_newton))
        print(f"    {name:<18s}  u_min={u_min:+.4f}  rebound_x_peak={reb:.4f}  "
              f"max_newton={log.max_newton}")
        ax_s.plot(log.times, log.ss[:, 0] - p_drop.s0, label=name)
        ax_x.plot(log.times, log.xs[:, 0], label=name)
    ax_s.axhline(0, color="k", lw=0.5, ls=":")
    ax_s.set_xlabel("t (s)")
    ax_s.set_ylabel("u = s - s0")
    ax_s.set_title("Internal strain during drop")
    ax_s.legend(fontsize=7)
    ax_s.grid(True, alpha=0.3)
    ax_x.set_xlabel("t (s)")
    ax_x.set_ylabel("x center height")
    ax_x.set_title("Drop + rebound trajectory")
    ax_x.legend(fontsize=7)
    ax_x.grid(True, alpha=0.3)

    fig.suptitle("Asymmetric tension/compression cubic law")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "asymmetric_cubic.png"), dpi=120)
    plt.close(fig)
    return all(r[3] >= 0 for r in rows_drop)


def test_free_oscillation_drift(outdir: str):
    print("\n=== TEST: free-oscillation energy drift (BE L-stability cost) ===")
    dt = 1.0e-3
    mu_frac = 1.0 / 3.0
    mu_val = 1.0 * mu_frac
    omega_dts = [0.1, 0.3, 1.0, 3.0, 10.0]
    n_steps_min = 4000

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(omega_dts) - 1)) for i in range(len(omega_dts))]

    for w_dt, color in zip(omega_dts, colors):
        omega = w_dt / dt
        k_val = mu_val * omega * omega
        T = 2.0 * np.pi / omega
        steps = max(n_steps_min, int(round(30.0 * T / dt)))
        for mode, ls, tag in (("local_implicit", "-", "BE"),
                              ("exponential",     "--", "Exp")):
            p = Params(N=1, k=k_val, dt=dt, steps=steps, g=0.0, damping=0.0,
                       mu_frac=mu_frac, warm_start=False, floor=False)
            init = make_initial_state(p)
            init.s[0] = p.s0 + 0.1
            init.vs[0] = 0.0
            log = run_sim(p, mode, init=init)
            E = log.KE + log.PE_e
            valid = (E > 1.0e-14) & np.isfinite(E)
            if valid.sum() >= 5:
                slope_E = float(np.polyfit(log.times[valid], np.log(E[valid]), 1)[0])
                lam_meas = -slope_E / 2.0
            else:
                if E[0] > 0 and E[1] > 0:
                    lam_meas = -0.5 * float(np.log(E[1] / E[0])) / dt
                else:
                    lam_meas = float("nan")
            lam_theory_be = float(np.log(1.0 + omega * omega * dt * dt) / (2.0 * dt))
            lam_theory = lam_theory_be if mode == "local_implicit" else 0.0
            rows.append((w_dt, mode, lam_meas, lam_theory, omega))
            print(f"  omega*dt={w_dt:5.2f}  mode={tag:<3s}  "
                  f"lam_meas={lam_meas:+.3e}  lam_theory={lam_theory:.3e}  "
                  f"(lam/omega)_meas={lam_meas/omega:+.3e}")
            label = f"omega*dt={w_dt} {tag}"
            t_plot = log.times
            E_plot = np.maximum(E / max(E[0], 1e-30), 1e-18)
            axes[0].semilogy(t_plot, E_plot, ls, color=color, alpha=0.9, lw=1.2,
                             label=label if tag == "BE" else None)

    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("E(t) / E(0)")
    axes[0].set_title("Energy decay (solid=BE, dashed=Exp)")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[0].grid(True, which="both", alpha=0.3)

    be_rows = [r for r in rows if r[1] == "local_implicit"]
    w_arr = np.array([r[0] for r in be_rows])
    lam_meas = np.array([r[2] for r in be_rows])
    omega_arr = np.array([r[4] for r in be_rows])
    ratio_meas = lam_meas / omega_arr
    ratio_theo = np.log(1.0 + w_arr * w_arr) / (2.0 * w_arr)

    axes[1].loglog(w_arr, np.maximum(ratio_meas, 1e-18), "o-", label="measured (BE)")
    axes[1].loglog(w_arr, ratio_theo, "k--", lw=1, label="log(1+(wdt)^2)/(2 wdt)")
    axes[1].loglog(w_arr, w_arr / 2.0, "r:", lw=0.8, label="wdt/2 asymptote")
    axes[1].set_xlabel("omega * dt")
    axes[1].set_ylabel("lambda / omega")
    axes[1].set_title("BE numerical damping vs theory")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    print("  -- BE measured vs theory --")
    ok = True
    for (w, _mode, lm, lt, _w), rt in zip(be_rows, ratio_theo):
        if lt > 0:
            rel = abs(lm - lt) / lt
            flag = "" if rel < 0.2 else "  *"
            print(f"    omega*dt={w:5.2f}  lam_meas={lm:.3e}  lam_theory={lt:.3e}  "
                  f"rel_err={rel:.2%}{flag}")
            if rel >= 0.2:
                ok = False

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "free_oscillation_drift.png"), dpi=120)
    plt.close(fig)
    return ok
