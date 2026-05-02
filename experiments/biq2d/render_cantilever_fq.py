"""Render a GIF of the FQ2D cantilever (pure VBD with smooth penalty).

Mirrors `render_cantilever.py` but uses full-quadratic bodies: each body is
drawn as 4 sub-cells (4-way split along ξ₁=0, ξ₂=0) so banana curvature is
visually obvious.  Joint markers at all 3 edge-match points per edge.

Saves two GIFs:
  experiments/biq2d/cantilever_fq.gif       — k=2000 (matches tests)
  experiments/biq2d/cantilever_fq_soft.gif  — k=200 (softer, larger strain)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io

from .body import BodyFQ2D
from .solver import Params, step_vbd_fq, point_position_fq
from .test_cantilever_fq import make_cantilever_fq


NUM_EDGE_SAMPLES = 16


def _sub_cell_colors(face):
    """Four slightly different shades per sub-cell so the split is visible."""
    rgb = np.array([int(face[1:3], 16), int(face[3:5], 16),
                    int(face[5:7], 16)]) / 255.0
    out = []
    for t in (0.00, 0.12, 0.24, 0.12):
        mix = rgb * (1.0 - t) + np.array([1.0, 1.0, 1.0]) * t
        out.append("#{:02x}{:02x}{:02x}".format(
            int(mix[0] * 255), int(mix[1] * 255), int(mix[2] * 255)))
    return out


def _chain_colors(n):
    """Gradient root → tip, (face, edge) pairs."""
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        r_l = int(207 + (1 - t) * (245 - 207))
        g_l = int(231 + (1 - t) * (224 - 231))
        b_l = int(255 + (1 - t) * (196 - 255))
        light = f"#{r_l:02x}{g_l:02x}{b_l:02x}"
        r_d = int(59 + (1 - t) * (217 - 59))
        g_d = int(130 + (1 - t) * (122 - 130))
        b_d = int(196 + (1 - t) * (58  - 196))
        edge = f"#{r_d:02x}{g_d:02x}{b_d:02x}"
        colors.append((light, edge))
    return colors


def render_gif(output_path="experiments/biq2d/cantilever_fq.gif",
               n_bodies=4, n_steps=720, dt=1.0 / 240.0,
               alpha_mult=10.0, k=2000.0,
               vbd_sweeps=5, vbd_newton=5,
               render_stride=4,
               title_suffix=""):
    state = make_cantilever_fq(n_bodies=n_bodies, k=k)
    params = Params(dt=dt, vbd_sweeps=vbd_sweeps, vbd_newton=vbd_newton)
    b_ref = next(b for b in state.bodies if not b.static)
    alpha = alpha_mult * b_ref.k * (b_ref.h ** 2)

    print(f"FQ2D cantilever: n={n_bodies}, alpha={alpha:.2f} (mult={alpha_mult}), "
          f"k={k}, sweeps={vbd_sweeps}x newton={vbd_newton}")

    def snapshot():
        return [(b.c.copy(), b.F.copy(), b.G.copy(),
                 b.Hx.copy(), b.Hy.copy()) for b in state.bodies]

    snapshots = [snapshot()]
    for i in range(n_steps):
        step_vbd_fq(state, params, alpha=alpha)
        snapshots.append(snapshot())

        max_F = max(np.max(np.abs(b.F)) for b in state.bodies if not b.static)
        if not np.isfinite(max_F) or max_F > 100:
            print(f"  DIVERGED at step {i}, truncating")
            break
        if i % 120 == 0:
            tip_y = state.bodies[-1].c[1]
            min_det = min(b.min_det_J_sampled(grid_n=9)
                          for b in state.bodies if not b.static)
            max_G  = max(np.linalg.norm(b.G)  for b in state.bodies if not b.static)
            max_Hx = max(np.linalg.norm(b.Hx) for b in state.bodies if not b.static)
            max_Hy = max(np.linalg.norm(b.Hy) for b in state.bodies if not b.static)
            print(f"  step {i:4d}: tip_y={tip_y:.3f}  |G|={max_G:.4f}  "
                  f"|Hx|={max_Hx:.4f}  |Hy|={max_Hy:.4f}  "
                  f"min_det={min_det:.4f}")

    print(f"Simulated {len(snapshots)} frames.")

    render_indices = list(range(0, len(snapshots), render_stride))
    n_frames = len(render_indices)
    print(f"Rendering {n_frames} frames at {render_stride}× stride...")

    colors = _chain_colors(n_bodies)

    images = []
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    n_h = state.bodies[0].h
    x_max = 2.0 * n_h * (n_bodies - 1) + 2.0
    x_min = -1.0
    y_min = 3.0 - x_max
    y_max = 4.5

    tmp = BodyFQ2D(mass=1.0, half_extent=n_h)

    for frame_idx in render_indices:
        ax.clear()

        # Wall at x = -h (left edge of body 0 = static anchor)
        wall_x = -n_h
        ax.fill_between([wall_x - 0.5, wall_x], y_min, y_max,
                        color="#888888", alpha=0.3)
        ax.plot([wall_x, wall_x], [y_min, y_max], color="#555555", linewidth=1.5)

        snap = snapshots[frame_idx]

        # Draw each body as 4 sub-cells + split lines
        for bi in range(n_bodies):
            c, F, G, Hx, Hy = snap[bi]
            face, edge = colors[bi]
            tmp.c, tmp.F, tmp.G, tmp.Hx, tmp.Hy = (
                c.copy(), F.copy(), G.copy(), Hx.copy(), Hy.copy())

            sub_cells = tmp.sub_cell_corners(num_samples=NUM_EDGE_SAMPLES)
            sc_colors = _sub_cell_colors(face)
            for poly_pts, sc_face in zip(sub_cells, sc_colors):
                ax.add_patch(Polygon(poly_pts, closed=True, facecolor=sc_face,
                                     edgecolor=edge, linewidth=1.2, alpha=0.9))

            vert, horz = tmp.split_line_xi()
            ax.plot(vert[:, 0], vert[:, 1], "-", color=edge,
                    linewidth=1.0, alpha=0.7)
            ax.plot(horz[:, 0], horz[:, 1], "-", color=edge,
                    linewidth=1.0, alpha=0.7)

        # Joint markers at the body_a point of each JointFQ
        for j in state.joints:
            ca, Fa, Ga, Hxa, Hya = snap[j.body_a_idx]
            tmp.c, tmp.F, tmp.G, tmp.Hx, tmp.Hy = (
                ca.copy(), Fa.copy(), Ga.copy(), Hxa.copy(), Hya.copy())
            P = point_position_fq(tmp, j.xi_a)
            ax.plot(P[0], P[1], "o", color="white", markersize=5,
                    markeredgecolor="#222", markeredgewidth=1.0, zorder=5)

        t = frame_idx * dt
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        title = (f"FQ2D cantilever — pure VBD, α = {alpha:.0f}, "
                 f"n_bodies = {n_bodies}{title_suffix}")
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

    duration = max(int(6000 / n_frames), 16)
    print(f"Saving GIF to {output_path} ({n_frames} frames, {duration} ms/frame)...")
    images[0].save(
        output_path, save_all=True, append_images=images[1:],
        duration=duration, loop=0,
    )
    print("Done.")


if __name__ == "__main__":
    render_gif(output_path="experiments/biq2d/cantilever_fq.gif",
               k=2000.0, alpha_mult=10.0)
    render_gif(output_path="experiments/biq2d/cantilever_fq_soft.gif",
               k=200.0, alpha_mult=10.0, title_suffix=" (soft k=200)")
