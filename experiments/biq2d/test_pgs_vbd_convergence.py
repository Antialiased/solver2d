"""Convergence probe for the PGS-style (Gauss-Seidel) VBD solver in `solver.py`.

The 'PGS-VBD' name refers to `step_vbd_fq`'s outer iteration: a forward+backward
Gauss-Seidel sweep over bodies, with each visit running a small per-body Newton.
That's structurally a block-PGS over the per-body 12-DoF blocks.

This file does NOT modify the solver. It instruments it:
  - run a single deterministic cantilever step,
  - run K outer sweeps (each = one fwd + one bwd pass),
  - between sweeps, snapshot (q_all, v_all),
  - log per-sweep ||Δv||, ||Δq||, total IP energy and global gradient norm.

If the iterates settle, the solver converges; if they keep moving with no
asymptote, it does not.
"""
import numpy as np
from .body import BodyFQ2D
from .solver import (
    State, Params, JointFQ, point_position_fq,
    vbd_body_step_fq, _collect_incident_constraints_fq,
)
from . import energy
from .test_cantilever_fq import make_cantilever_fq


def snapshot_qv(state):
    """Concatenate all non-static bodies' (q, v) into flat arrays."""
    qs, vs = [], []
    for b in state.bodies:
        if b.static:
            continue
        qs.append(np.concatenate([b.c, b.F, b.G, b.Hx, b.Hy]))
        vs.append(np.concatenate([b.vc, b.vF, b.vG, b.vHx, b.vHy]))
    return np.concatenate(qs), np.concatenate(vs)


def total_ip_energy(state, dt, gravity, alpha):
    """Sum of every non-static body's IP energy at the current (q, v).

    Mirrors `vbd_body_step_fq.ip_energy` but evaluated globally so each joint
    contributes once (we sum half-joints from both endpoints).
    """
    E = 0.0
    for bi, b in enumerate(state.bodies):
        if b.static:
            continue
        M = b.mass_vec
        M_c = b.mass
        mu_l, lam_l = b.lame
        h = b.h
        v_pre = np.concatenate([b._vc_pre, b._vF_pre, b._vG_pre, b._vHx_pre, b._vHy_pre])
        v = np.concatenate([b.vc, b.vF, b.vG, b.vHx, b.vHy])
        q_pre = np.concatenate([b._c_pre, b._F_pre, b._G_pre, b._Hx_pre, b._Hy_pre])
        q = q_pre + dt * v
        dv = v - v_pre
        E += 0.5 * float(np.dot(dv * M, dv))
        E -= M_c * float(np.dot(gravity, q[:2]))
        E += energy.integrated_energy_full(q[2:6], q[6:8], q[8:10], q[10:12], mu_l, lam_l, h)
        # Per-body half-joint penalty mirrors what its local Newton sees.
        for J_q, target in _collect_incident_constraints_fq(bi, state):
            r = J_q @ q - target
            E += 0.5 * alpha * float(np.dot(r, r))
    return E


def max_joint_gap(state):
    g = 0.0
    for j in state.joints:
        Pa = point_position_fq(state.bodies[j.body_a_idx], j.xi_a)
        Pb = point_position_fq(state.bodies[j.body_b_idx], j.xi_b)
        g = max(g, np.linalg.norm(Pa - Pb))
    return g


def stash_pre(state):
    """Store the (q_pre, v_pre) snapshot the body solver expects."""
    for b in state.bodies:
        if b.static:
            continue
        b._c_pre   = b.c.copy()
        b._F_pre   = b.F.copy()
        b._G_pre   = b.G.copy()
        b._Hx_pre  = b.Hx.copy()
        b._Hy_pre  = b.Hy.copy()
        b._vc_pre  = b.vc.copy()
        b._vF_pre  = b.vF.copy()
        b._vG_pre  = b.vG.copy()
        b._vHx_pre = b.vHx.copy()
        b._vHy_pre = b.vHy.copy()


def run_sweeps_with_log(state, params, alpha, n_sweeps, newton_iters):
    """Run n_sweeps GS sweeps from current state, logging per-sweep diagnostics.

    Returns list of dicts: {sweep, dv_norm, dq_norm, ip_energy, joint_gap}.
    """
    stash_pre(state)
    n = len(state.bodies)
    log = []
    q_prev, v_prev = snapshot_qv(state)
    E0 = total_ip_energy(state, params.dt, params.gravity, alpha)
    g0 = max_joint_gap(state)
    log.append(dict(sweep=0, dv_norm=0.0, dq_norm=0.0, ip_energy=E0, joint_gap=g0))
    for s in range(1, n_sweeps + 1):
        for i in range(n):
            vbd_body_step_fq(state.bodies[i], i, state, params.dt, params.gravity,
                             alpha, max_newton=newton_iters)
        for i in range(n - 1, -1, -1):
            vbd_body_step_fq(state.bodies[i], i, state, params.dt, params.gravity,
                             alpha, max_newton=newton_iters)
        q_now, v_now = snapshot_qv(state)
        dv = np.linalg.norm(v_now - v_prev)
        dq = np.linalg.norm(q_now - q_prev)
        E = total_ip_energy(state, params.dt, params.gravity, alpha)
        g = max_joint_gap(state)
        log.append(dict(sweep=s, dv_norm=float(dv), dq_norm=float(dq),
                        ip_energy=float(E), joint_gap=float(g)))
        q_prev, v_prev = q_now, v_now
    return log


def warmup(state, params, alpha, n_warmup_steps, sweeps_per_step, newton_iters):
    """Advance the cantilever for n_warmup_steps using nominal sweeps, so the
    convergence probe runs from a non-trivial mid-fall state."""
    from .solver import step_vbd_fq
    p = Params(dt=params.dt, vbd_sweeps=sweeps_per_step, vbd_newton=newton_iters,
               gravity=params.gravity)
    for _ in range(n_warmup_steps):
        step_vbd_fq(state, p, alpha=alpha)


def integrate_freebody_predict(state, params):
    """Apply pre-iteration prediction so v_pre/c_pre carry gravity acceleration —
    matches what step_vbd_fq does at the top of a real step before sweeps run."""
    # The actual solver doesn't pre-predict; the per-body Newton starts from
    # v = v_pre (which already includes the previous step's velocity) and
    # gravity enters only via the IP energy gradient. We mirror that exactly.
    pass


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=== PGS-VBD cantilever convergence probe ===\n")
    for alpha_mult in (1.0, 10.0, 100.0):
        # Build the standard 4-body cantilever, warm up 60 steps so we are
        # somewhere in the swing (not a trivial linearised t=0 step).
        state = make_cantilever_fq(n_bodies=4, k=2000.0)
        params = Params(dt=1.0/240.0, vbd_sweeps=5, vbd_newton=5)
        b_ref = next(b for b in state.bodies if not b.static)
        alpha = alpha_mult * b_ref.k * (b_ref.h ** 2)

        warmup(state, params, alpha, n_warmup_steps=60,
               sweeps_per_step=5, newton_iters=5)

        # Now: take a single test step. Disable any post-step bookkeeping so
        # we can re-run sweeps from the same v_pre snapshot for diagnostics.
        log = run_sweeps_with_log(state, params, alpha,
                                  n_sweeps=80, newton_iters=10)

        print(f"--- alpha_mult={alpha_mult}, alpha={alpha:.1f} ---")
        print("  sweep   ||Δv||        ||Δq||        IP energy        joint_gap")
        # Print a subset of sweeps + the final tail for readability.
        idxs = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80]
        for k in idxs:
            if k >= len(log): continue
            r = log[k]
            print(f"  {r['sweep']:5d}   {r['dv_norm']:.4e}   {r['dq_norm']:.4e}   "
                  f"{r['ip_energy']:+.6e}   {r['joint_gap']:.4e}")
        # Convergence rate estimate from the last decade of sweeps.
        dvs = [r['dv_norm'] for r in log[-20:] if r['dv_norm'] > 0]
        if len(dvs) >= 2:
            ratios = [dvs[i+1] / max(dvs[i], 1e-30) for i in range(len(dvs) - 1)]
            geom = float(np.exp(np.mean(np.log(np.clip(ratios, 1e-30, 1e30)))))
            print(f"  geometric ||Δv|| contraction over last 20 sweeps: {geom:.4f}")
        print()


if __name__ == "__main__":
    main()
