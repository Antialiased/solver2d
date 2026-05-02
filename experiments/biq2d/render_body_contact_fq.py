"""Render GIFs of FQ2D body-body contact (frozen-normal IPC barrier).

Two scenes:
  - body_body_drop_fq.gif       — Body A static on floor, B drops onto it.
  - body_body_collision_fq.gif  — Two bodies colliding side-on, no floor.

Reuses the floor-renderer machinery for body drawing, sub-cell rendering,
and chord-polygon overlays.  Adds a per-frame highlight of *currently
penetrating* body-body contact points (red dots at any A-vertex inside
any B-sub-cell — under the barrier these should essentially never appear).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io

from .body import BodyFQ2D
from .solver import (Params, State, step_vbd_fq, _OUTER_XI,
                     _outer_world_pts_from_q, _subcell_polygons_world,
                     _vertex_in_subcell)
from .render_cantilever_fq import _sub_cell_colors, _chain_colors, NUM_EDGE_SAMPLES
from .render_floor_fq import _draw_body, _draw_chord_polygon


def _active_body_body_pts(state):
    """List of world points where some non-static body's outer ξ vertex is
    currently penetrating some other body's sub-cell.  Used as red-dot
    overlay in the renderer."""
    polys_by = {i: _subcell_polygons_world(b) for i, b in enumerate(state.bodies)}
    out = []
    for i, ba in enumerate(state.bodies):
        if ba.static:
            continue
        pts = _outer_world_pts_from_q(
            np.concatenate([ba.c, ba.F, ba.G, ba.Hx, ba.Hy]), ba.h)
        for j, _bb in enumerate(state.bodies):
            if j == i:
                continue
            for sc in polys_by[j]:
                for P in pts:
                    active, _depth, _ = _vertex_in_subcell(P, sc)
                    if active:
                        out.append(P.copy())
    return out


def render_scene(output_path, build_state, n_steps, dt, alpha_mult,
                 barrier_kappa_floor, barrier_kappa_body, barrier_dhat_frac,
                 k, y_floor, view, title,
                 vbd_sweeps=5, vbd_newton=5, render_stride=4,
                 show_chord_polygon=True, gravity=(0.0, -9.81)):
    state = build_state()
    params = Params(gravity=np.array(gravity, dtype=float),
                    dt=dt, vbd_sweeps=vbd_sweeps, vbd_newton=vbd_newton)
    b_ref = next(b for b in state.bodies if not b.static)
    h = b_ref.h
    alpha = alpha_mult * b_ref.k * (h ** 2)
    kappa_floor = barrier_kappa_floor if y_floor is not None else 0.0
    kappa_bb    = barrier_kappa_body
    barrier_dhat = barrier_dhat_frac * h
    n_bodies = len(state.bodies)

    print(f"{output_path}: n={n_bodies}, alpha={alpha:.2f}, "
          f"barrier_kappa_floor={kappa_floor:.3f}, "
          f"barrier_kappa_body={kappa_bb:.3f}, dhat={barrier_dhat:.4f}, "
          f"k={k}, y_floor={y_floor}")

    def snapshot():
        return [(b.c.copy(), b.F.copy(), b.G.copy(),
                 b.Hx.copy(), b.Hy.copy()) for b in state.bodies]

    snapshots = [snapshot()]
    contact_pts = [_active_body_body_pts(state)]
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha,
                    barrier_kappa_floor=kappa_floor,
                    y_floor=y_floor,
                    barrier_kappa_body=kappa_bb,
                    barrier_dhat=barrier_dhat)
        snapshots.append(snapshot())
        contact_pts.append(_active_body_body_pts(state))
        max_F = max(np.max(np.abs(b.F)) for b in state.bodies if not b.static)
        if not np.isfinite(max_F) or max_F > 100:
            print(f"  DIVERGED at step {i}, truncating")
            break
        if i % 120 == 0:
            min_det = min(b.min_det_J_sampled(grid_n=9)
                          for b in state.bodies if not b.static)
            print(f"  step {i:4d}: min_det={min_det:.4f}  "
                  f"n_contacts={len(contact_pts[-1])}")

    print(f"Simulated {len(snapshots)} frames.")

    render_indices = list(range(0, len(snapshots), render_stride))
    n_frames = len(render_indices)
    print(f"Rendering {n_frames} frames at {render_stride}× stride...")

    colors = _chain_colors(n_bodies)
    images = []
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    x_min, x_max, y_min, y_max = view

    for frame_idx in render_indices:
        ax.clear()

        if y_floor is not None:
            ax.fill_between([x_min - 1.0, x_max + 1.0], y_min - 1.0, y_floor,
                            color="#888888", alpha=0.3)
            ax.plot([x_min - 1.0, x_max + 1.0], [y_floor, y_floor],
                    color="#555555", linewidth=1.5)

        snap = snapshots[frame_idx]
        for bi in range(n_bodies):
            face, edge = colors[bi]
            _draw_body(ax, snap[bi], face, edge,
                       h=state.bodies[bi].h)
            if show_chord_polygon and not state.bodies[bi].static:
                _draw_chord_polygon(ax, snap[bi], state.bodies[bi].h)

        for P in contact_pts[frame_idx]:
            ax.plot(P[0], P[1], "o", color="#cc1133", markersize=8,
                    markeredgecolor="#660011", markeredgewidth=0.8, zorder=6)

        t = frame_idx * dt
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.tick_params(labelsize=7, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.text(0.95, 0.02, f"t = {t:.2f} s", fontsize=9, ha="right",
                 color="#666666")
        plt.subplots_adjust(left=0.06, right=0.96, bottom=0.05, top=0.91)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    duration = max(int(6000 / max(n_frames, 1)), 16)
    print(f"Saving GIF to {output_path} ({n_frames} frames, {duration} ms/frame)...")
    images[0].save(
        output_path, save_all=True, append_images=images[1:],
        duration=duration, loop=0,
    )
    print("Done.")


def _build_drop_onto_static(h=0.5, k=2000.0):
    def builder():
        A = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35, static=True)
        A.c = np.array([0.0, h])
        B = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
        B.c = np.array([0.0, 2.5])
        return State(bodies=[A, B], joints=[])
    return builder


def _build_side_collision(h=0.5, k=2000.0, v0=2.0):
    def builder():
        A = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
        A.c = np.array([-1.5, 1.0]); A.vc = np.array([+v0, 0.0])
        B = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
        B.c = np.array([+1.5, 1.0]); B.vc = np.array([-v0, 0.0])
        return State(bodies=[A, B], joints=[])
    return builder


if __name__ == "__main__":
    render_scene(
        output_path="experiments/biq2d/body_body_drop_fq.gif",
        build_state=_build_drop_onto_static(h=0.5, k=2000.0),
        n_steps=960, dt=1.0 / 240.0,
        alpha_mult=10.0,
        barrier_kappa_floor=100.0, barrier_kappa_body=1000.0, barrier_dhat_frac=0.05,
        k=2000.0, y_floor=0.0,
        view=(-1.5, 1.5, -0.4, 3.0),
        title="FQ2D body-body drop — A static, B onto A; frozen-normal IPC barrier",
    )
    render_scene(
        output_path="experiments/biq2d/body_body_collision_fq.gif",
        build_state=_build_side_collision(h=0.5, k=2000.0, v0=2.0),
        n_steps=720, dt=1.0 / 240.0,
        alpha_mult=10.0,
        barrier_kappa_floor=0.0, barrier_kappa_body=1000.0, barrier_dhat_frac=0.05,
        k=2000.0, y_floor=None,
        view=(-3.0, 3.0, -0.5, 2.5),
        title="FQ2D body-body collision — symmetric ±2 m/s, barrier contact, no friction",
        gravity=(0.0, 0.0),
    )
    # Soft variants (k=200, 10x softer) — bananas activate visibly during
    # contact; quadratic-vs-chord contrast becomes obvious mid-impact.
    # The barrier defaults are unchanged from the stiff variants:
    # penetration is feasibility-bounded by the barrier itself, not by κ_b.
    render_scene(
        output_path="experiments/biq2d/body_body_drop_fq_soft.gif",
        build_state=_build_drop_onto_static(h=0.5, k=200.0),
        n_steps=1200, dt=1.0 / 240.0,
        alpha_mult=10.0,
        barrier_kappa_floor=100.0, barrier_kappa_body=1000.0, barrier_dhat_frac=0.05,
        k=200.0, y_floor=0.0,
        view=(-1.5, 1.5, -0.4, 3.0),
        title="FQ2D body-body drop (soft, k=200) — barrier contact, defaults",
    )
    render_scene(
        output_path="experiments/biq2d/body_body_collision_fq_soft.gif",
        build_state=_build_side_collision(h=0.5, k=200.0, v0=2.0),
        n_steps=900, dt=1.0 / 240.0,
        alpha_mult=10.0,
        barrier_kappa_floor=0.0, barrier_kappa_body=1000.0, barrier_dhat_frac=0.05,
        k=200.0, y_floor=None,
        view=(-3.0, 3.0, -0.5, 2.5),
        title="FQ2D body-body collision (soft, k=200) — barrier contact, defaults",
        gravity=(0.0, 0.0),
    )
