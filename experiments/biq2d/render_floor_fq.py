"""Render GIFs of FQ2D bodies dropping onto a floor (frozen-normal IPC barrier).

Mirrors `render_cantilever_fq.py`: bodies drawn as 4 sub-cells with split
lines.  Floor visualised as a filled grey band below `y_floor`.  Currently
penetrating outer-ξ contact points are highlighted with red dots on top of
the existing white joint markers (under the barrier these should be
essentially absent — the barrier guarantees feasibility).

Two GIFs:
  experiments/biq2d/floor_drop_fq.gif            — single body drops
  experiments/biq2d/cantilever_drop_floor_fq.gif — chain swings onto floor
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io

from .body import BodyFQ2D
from .solver import (Params, State, step_vbd_fq, point_position_fq,
                     _OUTER_XI, _outer_py_from_q)
from .test_cantilever_fq import make_cantilever_fq
from .render_cantilever_fq import _sub_cell_colors, _chain_colors, NUM_EDGE_SAMPLES


def _draw_body(ax, snap_entry, face, edge, h):
    """Render one body at (c, F, G, Hx, Hy) into ax."""
    c, F, G, Hx, Hy = snap_entry
    tmp = BodyFQ2D(mass=1.0, half_extent=h)
    tmp.c, tmp.F, tmp.G, tmp.Hx, tmp.Hy = (
        c.copy(), F.copy(), G.copy(), Hx.copy(), Hy.copy())
    sub_cells = tmp.sub_cell_corners(num_samples=NUM_EDGE_SAMPLES)
    sc_colors = _sub_cell_colors(face)
    for poly_pts, sc_face in zip(sub_cells, sc_colors):
        ax.add_patch(Polygon(poly_pts, closed=True, facecolor=sc_face,
                             edgecolor=edge, linewidth=1.2, alpha=0.9))
    vert, horz = tmp.split_line_xi()
    ax.plot(vert[:, 0], vert[:, 1], "-", color=edge, linewidth=1.0, alpha=0.7)
    ax.plot(horz[:, 0], horz[:, 1], "-", color=edge, linewidth=1.0, alpha=0.7)
    return tmp


def _outer_world_pts(snap_entry, h):
    """(8, 2) world positions of the outer ξ points for this snapshot."""
    c, F, G, Hx, Hy = snap_entry
    tmp = BodyFQ2D(mass=1.0, half_extent=h)
    tmp.c, tmp.F, tmp.G, tmp.Hx, tmp.Hy = (
        c.copy(), F.copy(), G.copy(), Hx.copy(), Hy.copy())
    return tmp.map_points(_OUTER_XI)


def _draw_chord_polygon(ax, snap_entry, h, color="#222222", lw=1.4):
    """Overlay the 8-vertex outer chord polygon (straight edges between
    outer ξ points) — the geometry the contact code actually uses for
    detection.  Drawn as a dashed loop on top of the smooth body."""
    pts = _outer_world_pts(snap_entry, h)            # (8, 2) in cyclic order
    loop = np.concatenate([pts, pts[:1]], axis=0)
    ax.plot(loop[:, 0], loop[:, 1], "--", color=color, linewidth=lw,
            alpha=0.85, zorder=4)
    ax.plot(pts[:, 0], pts[:, 1], "o", color=color, markersize=3.5,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4)


def render_drop(output_path, build_state, n_steps, dt, alpha_mult,
                barrier_kappa, barrier_dhat_frac, k, y_floor, view, title,
                vbd_sweeps=5, vbd_newton=5, render_stride=4,
                show_chord_polygon=False):
    state = build_state()
    params = Params(gravity=np.array([0.0, -9.81]),
                    dt=dt, vbd_sweeps=vbd_sweeps, vbd_newton=vbd_newton)
    b_ref = next(b for b in state.bodies if not b.static)
    h = b_ref.h
    alpha = alpha_mult * b_ref.k * (h ** 2)
    barrier_dhat = barrier_dhat_frac * h
    n_bodies = len(state.bodies)

    print(f"{output_path}: n={n_bodies}, alpha={alpha:.2f}, "
          f"barrier_kappa={barrier_kappa:.3f}, barrier_dhat={barrier_dhat:.4f}, "
          f"k={k}, dt={dt}, sweeps={vbd_sweeps}x newton={vbd_newton}")

    def snapshot():
        return [(b.c.copy(), b.F.copy(), b.G.copy(),
                 b.Hx.copy(), b.Hy.copy()) for b in state.bodies]

    snapshots = [snapshot()]
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha,
                    barrier_kappa_floor=barrier_kappa, y_floor=y_floor,
                    barrier_dhat=barrier_dhat)
        snapshots.append(snapshot())
        max_F = max(np.max(np.abs(b.F)) for b in state.bodies if not b.static)
        if not np.isfinite(max_F) or max_F > 100:
            print(f"  DIVERGED at step {i}, truncating")
            break
        if i % 120 == 0:
            min_y = min(float(np.min(_outer_py_from_q(
                np.concatenate([b.c, b.F, b.G, b.Hx, b.Hy]), b.h)))
                        for b in state.bodies if not b.static)
            min_det = min(b.min_det_J_sampled(grid_n=9)
                          for b in state.bodies if not b.static)
            print(f"  step {i:4d}: min_outer_y={min_y:.3f}  "
                  f"min_det={min_det:.4f}")

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

        # Floor: filled band + line
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

        # White joint markers (body_a side)
        tmp = BodyFQ2D(mass=1.0, half_extent=h)
        for j in state.joints:
            ca, Fa, Ga, Hxa, Hya = snap[j.body_a_idx]
            tmp.c, tmp.F, tmp.G, tmp.Hx, tmp.Hy = (
                ca.copy(), Fa.copy(), Ga.copy(), Hxa.copy(), Hya.copy())
            P = point_position_fq(tmp, j.xi_a)
            ax.plot(P[0], P[1], "o", color="white", markersize=5,
                    markeredgecolor="#222", markeredgewidth=1.0, zorder=5)

        # Red dots on currently penetrating outer ξ points
        for bi in range(n_bodies):
            if state.bodies[bi].static:
                continue
            pts = _outer_world_pts(snap[bi], state.bodies[bi].h)
            below = pts[:, 1] < y_floor
            if np.any(below):
                ax.plot(pts[below, 0], pts[below, 1], "o", color="#cc1133",
                        markersize=6, markeredgecolor="#660011",
                        markeredgewidth=0.8, zorder=6)

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


def _build_single_drop(h=0.5, k=2000.0):
    def builder():
        b = BodyFQ2D(mass=1.0, half_extent=h, k=k, nu=0.35)
        b.c = np.array([0.0, 2.0])
        return State(bodies=[b], joints=[])
    return builder


def _build_cantilever_drop(n_bodies=4, h=0.5, k=2000.0):
    def builder():
        return make_cantilever_fq(n_bodies=n_bodies, h=h, k=k, y0=0.6)
    return builder


if __name__ == "__main__":
    render_drop(
        output_path="experiments/biq2d/floor_drop_fq.gif",
        build_state=_build_single_drop(h=0.5, k=2000.0),
        n_steps=720, dt=1.0 / 240.0,
        alpha_mult=10.0, barrier_kappa=100.0, barrier_dhat_frac=0.05,
        k=2000.0, y_floor=0.0,
        view=(-2.0, 2.0, -0.6, 2.4),
        title="FQ2D drop — pure VBD, frozen-normal IPC barrier (κ_b=1, dhat=0.05·h)",
    )
    render_drop(
        output_path="experiments/biq2d/cantilever_drop_floor_fq.gif",
        build_state=_build_cantilever_drop(n_bodies=4, h=0.5, k=2000.0),
        n_steps=720, dt=1.0 / 240.0,
        alpha_mult=10.0, barrier_kappa=100.0, barrier_dhat_frac=0.05,
        k=2000.0, y_floor=0.0,
        view=(-1.0, 4.5, -2.0, 1.6),
        title="FQ2D cantilever onto floor — barrier contact (κ_b=1, dhat=0.05·h)",
    )
    # Soft single-body drop with chord-polygon overlay so the curved
    # full-quadratic outline (filled) and the 8-vertex contact polygon
    # (dashed) are visible side-by-side.  Same barrier defaults as the
    # stiff variant — barrier penetration doesn't depend on `k`.
    render_drop(
        output_path="experiments/biq2d/floor_drop_fq_soft.gif",
        build_state=_build_single_drop(h=0.5, k=200.0),
        n_steps=960, dt=1.0 / 240.0,
        alpha_mult=10.0, barrier_kappa=100.0, barrier_dhat_frac=0.05,
        k=200.0, y_floor=0.0,
        view=(-2.0, 2.0, -0.6, 2.4),
        title=("FQ2D drop (soft, k=200) — barrier contact, smooth quadratic body "
               "vs 8-vertex chord polygon (dashed)"),
        show_chord_polygon=True,
    )
