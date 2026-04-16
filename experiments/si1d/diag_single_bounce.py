"""Narrow single-bounce diagnostic.

One deformable body, no gravity, dropped onto the floor with initial downward
velocity. We step manually and decompose the energy at every step into KE_x,
KE_s, PE_e. Contact activity is tracked by the per-step change in state.lam[0]
(floor impulse). The goal is to localise where the ~0.5 energy loss happens:
entry impulse, spring phase, or release impulse.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.animation import FuncAnimation

from .solver import Params, State, si_step, energies, run_sim, make_initial_state


def build_state(p: Params, V: float, gap0: float) -> State:
    # one body, at rest shape, starts `gap0` above floor, moving downward
    x = np.array([p.s0 + gap0], dtype=float)
    s = np.array([p.s0], dtype=float)
    vx = np.array([-V], dtype=float)
    vs = np.array([0.0], dtype=float)
    lam = np.array([0.0], dtype=float)
    return State(x=x, s=s, vx=vx, vs=vs, lam=lam)


def run_diag(k: float, dt: float, V: float = 2.0, mu_frac: float = 1.0/3.0,
             n_steps: int = 4000, gap0: float = 0.05, mode: str = "exponential",
             restitution: float = 0.0):
    p = Params(N=1, k=k, dt=dt, steps=n_steps, g=0.0, damping=0.0,
               mu_frac=mu_frac, warm_start=False, floor=True,
               restitution=restitution, g_vec=np.array([0.0]))
    st = build_state(p, V, gap0)

    T = n_steps
    rec = {
        "t": np.zeros(T + 1), "x": np.zeros(T + 1), "s": np.zeros(T + 1),
        "vx": np.zeros(T + 1), "vs": np.zeros(T + 1),
        "KE_x": np.zeros(T + 1), "KE_s": np.zeros(T + 1),
        "PE_e": np.zeros(T + 1), "E": np.zeros(T + 1),
        "gap": np.zeros(T + 1), "lam_cum": np.zeros(T + 1),
        "dlam": np.zeros(T + 1),
    }

    def snap(i):
        ke, peg, pee = energies(st, p)
        rec["t"][i] = i * dt
        rec["x"][i] = st.x[0]; rec["s"][i] = st.s[0]
        rec["vx"][i] = st.vx[0]; rec["vs"][i] = st.vs[0]
        rec["KE_x"][i] = 0.5 * p.m * st.vx[0]**2
        rec["KE_s"][i] = 0.5 * p.mu * st.vs[0]**2
        rec["PE_e"][i] = pee
        rec["E"][i] = ke + pee  # g=0 so no PE_g
        rec["gap"][i] = st.x[0] - st.s[0]
        rec["lam_cum"][i] = st.lam[0]

    snap(0)
    last_lam = 0.0
    for step in range(T):
        si_step(st, p, mode=mode)
        snap(step + 1)
        rec["dlam"][step + 1] = st.lam[0] - last_lam
        last_lam = st.lam[0]

    return p, rec


def find_contact_window(rec):
    """Return (entry_idx, release_idx) — gap<=0 window (fallback: dlam>0)."""
    touched = rec["gap"] <= 1e-9
    if np.any(touched):
        idxs = np.where(touched)[0]
        return int(idxs[0]), int(idxs[-1])
    active = rec["dlam"] > 1e-12
    if not np.any(active):
        return None, None
    idxs = np.where(active)[0]
    return int(idxs[0]), int(idxs[-1])


def count_impacts(rec):
    gap = rec["gap"]
    below = gap < 0.0
    # rising edges: step where below goes False->True
    starts = np.where((~below[:-1]) & (below[1:]))[0] + 1
    return starts


def report(k: float, V: float, mu_frac: float, rec: dict, ent: int, rel: int):
    mu_x = 1.0 - mu_frac
    mu_s = mu_frac
    # predicted entry-step inelastic drop:
    # constraint x - s = 0, velocity jump: vx_new = vs_new
    # pre: vx=-V, vs=0 -> post: v' = (mu_x*(-V) + mu_s*0)/(mu_x+mu_s) = -V*mu_x
    v_after = -V * mu_x
    KE_pre = 0.5 * 1.0 * V**2
    KE_after_entry_pred = 0.5 * mu_x * v_after**2 + 0.5 * mu_s * v_after**2
    # = 0.5 * (mu_x+mu_s) * v_after^2 = 0.5 * V^2 * mu_x^2
    ratio_entry_pred = KE_after_entry_pred / KE_pre  # = mu_x^2

    print(f"\n--- k={k:.0e} V={V} mu_frac={mu_frac} ---")
    E0 = rec["E"][0]
    print(f"  E before first contact (step 0)       : {E0:.6f}")
    print(f"  entry step = {ent}   release step = {rel}   dur = {rel-ent+1} steps")
    print(f"  analytic entry ratio (mu_x^2)          : {ratio_entry_pred:.4f}")
    print(f"  E at entry-1     : {rec['E'][ent-1]:.6f}")
    print(f"  E at entry       : {rec['E'][ent]:.6f}   ({rec['E'][ent]/E0:.4f}·E0)")
    print(f"  E at release-1   : {rec['E'][rel-1]:.6f} ({rec['E'][rel-1]/E0:.4f}·E0)")
    print(f"  E at release     : {rec['E'][rel]:.6f}   ({rec['E'][rel]/E0:.4f}·E0)")
    print(f"  E at release+1   : {rec['E'][rel+1]:.6f} ({rec['E'][rel+1]/E0:.4f}·E0)")
    # Split losses:
    dE_entry = rec['E'][ent] - rec['E'][ent-1]
    dE_spring = rec['E'][rel-1] - rec['E'][ent]
    dE_release = rec['E'][rel+1] - rec['E'][rel-1]
    print(f"  dE entry step    : {dE_entry:+.6f}  ({dE_entry/E0:+.4f}·E0)")
    print(f"  dE spring phase  : {dE_spring:+.6f}  ({dE_spring/E0:+.4f}·E0)")
    print(f"  dE release step  : {dE_release:+.6f}  ({dE_release/E0:+.4f}·E0)")
    # Final far after bounce
    print(f"  E final (last)   : {rec['E'][-1]:.6f}  ({rec['E'][-1]/E0:.4f}·E0)")
    print(f"  vx final         : {rec['vx'][-1]:+.4f}    vs final : {rec['vs'][-1]:+.4f}")


def plot_bounce(p, rec, ent, rel, outpath):
    t = rec["t"]; E0 = rec["E"][0]
    lo = max(ent - 10, 0); hi = min(rel + 30, len(t) - 1)
    sl = slice(lo, hi + 1)
    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(t[sl], rec["gap"][sl], label="gap")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_ylabel("gap"); ax[0].legend()
    ax[1].plot(t[sl], rec["vx"][sl], label="vx")
    ax[1].plot(t[sl], rec["vs"][sl], label="vs")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("velocity"); ax[1].legend()
    ax[2].plot(t[sl], rec["KE_x"][sl], label="KE_x")
    ax[2].plot(t[sl], rec["KE_s"][sl], label="KE_s")
    ax[2].plot(t[sl], rec["PE_e"][sl], label="PE_e")
    ax[2].plot(t[sl], rec["E"][sl], "k--", label="E tot")
    ax[2].set_ylabel("energy"); ax[2].legend()
    ax[3].plot(t[sl], rec["E"][sl] / E0, "k-")
    ax[3].axvline(t[ent], color="g", lw=0.5, label="entry")
    ax[3].axvline(t[rel], color="r", lw=0.5, label="release")
    ax[3].set_ylabel("E/E0"); ax[3].set_xlabel("t"); ax[3].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def animate_bounce_sweep(outdir: str):
    """Single deformable body bouncing on a floor under gravity, with
    restitution=1. Sweeps stiffness; demonstrates energy-preserving bounce."""
    print("\n=== ANIM: bounce sweep with restitution=1 ===")
    g_ = 9.81
    t_end = 4.0
    dt = 5.0e-4
    steps = int(round(t_end / dt))
    ks = [1.0e3, 1.0e4, 1.0e5, 1.0e6]
    mu_frac = 1.0 / 3.0
    record_every = max(1, steps // 150)

    logs = []
    for k in ks:
        p = Params(N=1, k=k, dt=dt, steps=steps, g=g_, damping=0.0,
                   mu_frac=mu_frac, warm_start=False, floor=True,
                   restitution=1.0)
        init = make_initial_state(p)
        # drop from height 1.0 above the floor centre
        init.x[0] = 1.0 + p.s0
        init.s[0] = p.s0
        log = run_sim(p, "two_pass_exp", init=init, record_every=record_every)
        logs.append(log)
        E0 = float(log.KE[0] + log.PE_g[0] + log.PE_e[0])
        Ef = float(log.KE[-1] + log.PE_g[-1] + log.PE_e[-1])
        print(f"  k={k:.0e}  stable={log.stable}  E_end/E0={Ef/E0:.4f}")

    nrec = len(logs[0].times)
    fig, axes = plt.subplots(len(ks), 1, figsize=(6.5, 1.7 * len(ks)),
                             sharex=True)
    s0_const = Params().s0
    artists = []
    for ax, log, k in zip(axes, logs, ks):
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.1, 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"k = {k:.0e}", fontsize=10, loc="left")
        ax.set_yticks([])
        ax.axhline(0.0, color="k", lw=1.0)
        ax.add_patch(Rectangle((-1.3, -0.2), 2.6, 0.2,
                               facecolor="#dddddd", edgecolor="none"))
        y0 = float(log.xs[0, 0])
        s_ = float(log.ss[0, 0])
        e = Ellipse((0.0, y0), width=2 * s_,
                    height=2 * (s0_const ** 2) / s_,
                    facecolor="#3877e0", edgecolor="#1b3a70",
                    lw=1.2, alpha=0.9)
        ax.add_patch(e)
        tt = ax.text(0.98, 0.88, "", transform=ax.transAxes,
                     ha="right", va="top", fontsize=8, family="monospace")
        Et = log.KE + log.PE_g + log.PE_e
        E0 = float(Et[0])
        artists.append((e, tt, log, E0))
    axes[-1].set_xlabel("y (floor at 0)")
    fig.suptitle("Deformable bounce under gravity, restitution = 1 "
                 "(two_pass_exp)", fontsize=11)

    def init_anim():
        out = []
        for e, tt, _, _ in artists:
            out.extend([e, tt])
        return out

    def update(fi):
        out = []
        for e, tt, log, E0 in artists:
            y = float(log.xs[fi, 0])
            s_ = float(log.ss[fi, 0])
            e.set_center((0.0, y))
            e.width = 2.0 * (s0_const ** 2) / max(s_, 1e-6)  # horizontal
            e.height = 2.0 * s_                               # vertical = s
            Et = float(log.KE[fi] + log.PE_g[fi] + log.PE_e[fi])
            tt.set_text(f"t={log.times[fi]:.2f}s  E/E0={Et/E0:.3f}")
            out.extend([e, tt])
        return out

    anim = FuncAnimation(fig, update, frames=nrec,
                         init_func=init_anim, blit=True, interval=40)
    fig.tight_layout()
    out_path = os.path.join(outdir, "bounce_sweep_restitution.gif")
    anim.save(out_path, writer="pillow", fps=25, dpi=90)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    os.makedirs(outdir, exist_ok=True)

    V = 2.0
    restitution = 1.0
    mode = "two_pass_exp"
    print(f"restitution = {restitution}  mode = {mode}")
    for k, dt in [(1e3, 1e-4), (1e4, 1e-4), (1e5, 1e-4), (1e6, 2e-5)]:
        n_steps = int(1.5 / dt)  # 1.5 s worth
        p, rec = run_diag(k=k, dt=dt, V=V, n_steps=n_steps,
                          restitution=restitution, mode=mode)
        ent, rel = find_contact_window(rec)
        if ent is None:
            print(f"k={k:.0e}: no contact detected")
            continue
        report(k, V, p.mu_frac, rec, ent, rel)
        impacts = count_impacts(rec)
        print(f"  impacts (gap crossings to negative): {len(impacts)}")
        if len(impacts) > 0:
            print(f"    first 5 at steps: {list(impacts[:5])}")
        # energy at each impact
        E0 = rec["E"][0]
        for m, s in enumerate(impacts[:6]):
            print(f"    impact {m} step={s} E/E0={rec['E'][s]/E0:.4f} gap_min_window={rec['gap'][s]:.2e}")
        plot_bounce(p, rec, ent, rel,
                    os.path.join(outdir, f"diag_bounce_k{int(np.log10(k))}.png"))
    animate_bounce_sweep(outdir)
    print(f"\nplots -> {outdir}")


if __name__ == "__main__":
    main()
