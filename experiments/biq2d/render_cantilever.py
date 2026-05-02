"""Render a GIF of the BQ2D cantilever (pure VBD with smooth penalty).

Mirrors `render_modes.py` style:  one matplotlib axis, polygon patches per body,
joint markers at corners.  Saves to `experiments/biq2d/cantilever_vbd.gif`.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io

from .body import BodyBQ2D
from .solver import State, Params, Joint, step_vbd, corner_position
from .test_cantilever import make_cantilever


def render_gif(output_path="experiments/biq2d/cantilever_vbd.gif",
               n_bodies=4, n_steps=720, dt=1.0 / 240.0,
               alpha_mult=10.0, k=2000.0,
               vbd_sweeps=5, vbd_newton=5,
               render_stride=4):
    state = make_cantilever(n_bodies=n_bodies, k=k)
    params = Params(dt=dt, vbd_sweeps=vbd_sweeps, vbd_newton=vbd_newton)
    b_ref = next(b for b in state.bodies if not b.static)
    alpha = alpha_mult * b_ref.k * (b_ref.h ** 2)

    print(f"Cantilever: n={n_bodies}, alpha={alpha:.2f} (mult={alpha_mult}), "
          f"k={k}, sweeps={vbd_sweeps}x newton={vbd_newton}")

    snapshots = [[(b.c.copy(), b.F.copy(), b.G.copy()) for b in state.bodies]]
    diverged = False
    for i in range(n_steps):
        step_vbd(state, params, alpha=alpha)
        snapshots.append([(b.c.copy(), b.F.copy(), b.G.copy()) for b in state.bodies])

        max_F = max(np.max(np.abs(b.F)) for b in state.bodies if not b.static)
        if not np.isfinite(max_F) or max_F > 100:
            print(f"  DIVERGED at step {i}, truncating")
            diverged = True
            break
        if i % 120 == 0:
            tip_y = state.bodies[-1].c[1]
            min_det = min(b.min_det_J() for b in state.bodies if not b.static)
            max_G = max(np.linalg.norm(b.G) for b in state.bodies if not b.static)
            print(f"  step {i:4d}: tip_y={tip_y:.3f}  |G|={max_G:.4f}  "
                  f"min_det={min_det:.4f}  max|F|={max_F:.4f}")

    print(f"Simulated {len(snapshots)} frames.")

    render_indices = list(range(0, len(snapshots), render_stride))
    n_frames = len(render_indices)
    print(f"Rendering {n_frames} frames at {render_stride}× stride...")

    # Color gradient root → tip
    n = n_bodies
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

    images = []
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Visual extents — leave room for the chain to droop
    n_h = state.bodies[0].h
    x_max = 2.0 * n_h * (n_bodies - 1) + 2.0
    x_min = -1.0
    y_min = 3.0 - x_max  # generous downward room
    y_max = 4.5

    for frame_idx in render_indices:
        ax.clear()

        # Wall at x = -h (left edge of body 0 = static anchor)
        wall_x = -n_h
        ax.fill_between([wall_x - 0.5, wall_x], y_min, y_max,
                        color="#888888", alpha=0.3)
        ax.plot([wall_x, wall_x], [y_min, y_max], color="#555555", linewidth=1.5)

        snap = snapshots[frame_idx]
        # Reusable temp body for quick corner computation
        tmp = BodyBQ2D(mass=1.0, half_extent=n_h)

        # Draw bodies
        for bi in range(n_bodies):
            c, F, G = snap[bi]
            face, edge = colors[bi]
            tmp.c, tmp.F, tmp.G = c.copy(), F.copy(), G.copy()
            corners = tmp.corners()        # order: (--), (+-), (-+), (++)
            poly_pts = corners[[0, 1, 3, 2]]   # CCW
            ax.add_patch(Polygon(poly_pts, closed=True, facecolor=face,
                                 edgecolor=edge, linewidth=1.4, alpha=0.9))

        # Draw joint markers (anchor corners)
        for j in state.joints:
            ca, Fa, Ga = snap[j.body_a_idx]
            tmp.c, tmp.F, tmp.G = ca.copy(), Fa.copy(), Ga.copy()
            P = corner_position(tmp, j.corner_a)
            ax.plot(P[0], P[1], "o", color="white", markersize=5,
                    markeredgecolor="#222", markeredgewidth=1.0, zorder=5)

        t = frame_idx * dt
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        title = (f"BQ2D cantilever — pure VBD, α = {alpha:.0f}, "
                 f"n_bodies = {n_bodies}")
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
    render_gif()
