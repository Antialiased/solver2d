"""Render a GIF of M3: oblique bounce with spin and friction."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io

from .body import Body2D
from .solver import State, Params, step
from .render_collision import ellipse_points, quadrant_patches


def render_gif(output_path="experiments/si2d/oblique.gif"):
    dt = 1.0 / 240.0
    n_steps = 1920  # 8 seconds
    r0 = 0.5
    k = 200.0
    nu = 0.35

    b = Body2D(mass=1.0, r0=r0, k=k, nu=nu)
    b.c = np.array([-1.0, 4.0 + r0])
    b.vc = np.array([1.5, 0.0])
    b.vF = np.array([0.0, 2.0, -2.0, 0.0])  # spin

    state = State(bodies=[b])
    params = Params(
        dt=dt, restitution=0.6, friction=0.4, position_iters=8, velocity_iters=8,
    )

    frames = []
    for i in range(n_steps):
        frames.append((b.c.copy(), b.F.copy()))
        step(state, params)
    frames.append((b.c.copy(), b.F.copy()))

    # Adaptive frame selection
    render_indices = []
    for i in range(n_steps):
        c = frames[i][0]
        near_floor = c[1] < 2.5
        if near_floor:
            if i % 2 == 0:
                render_indices.append(i)
        else:
            if i % 6 == 0:
                render_indices.append(i)

    n_frames = len(render_indices)
    print(f"Rendering {n_frames} frames...")

    fill_light, fill_dark, edge = "#4A90D9", "#2563a0", "#1a3d6e"
    images = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for frame_idx in render_indices:
        ax.clear()

        ax.fill_between([-5, 7], -0.5, 0, color="#D2C5A0", alpha=0.6)
        ax.plot([-5, 7], [0, 0], color="#8B7355", linewidth=2.5)

        c, F = frames[frame_idx]

        # Ghost trail
        for ti, offset in enumerate([20, 10, 5]):
            prev_idx = frame_idx - offset
            if prev_idx >= 0:
                pc, pF = frames[prev_idx]
                body_tmp = Body2D(mass=1.0, r0=r0)
                body_tmp.c = pc
                body_tmp.F = pF
                pts = ellipse_points(body_tmp, n=64)
                poly = Polygon(pts, closed=True, facecolor=fill_light,
                               edgecolor="none", alpha=0.05 + 0.04 * ti)
                ax.add_patch(poly)

        body_tmp = Body2D(mass=1.0, r0=r0)
        body_tmp.c = c
        body_tmp.F = F
        for patch in quadrant_patches(body_tmp, fill_light, fill_dark, edge):
            ax.add_patch(patch)

        pts = ellipse_points(body_tmp, n=64)
        poly = Polygon(pts, closed=True, facecolor="none",
                       edgecolor=edge, linewidth=1.5)
        ax.add_patch(poly)
        ax.plot(c[0], c[1], "o", color="white", markersize=3,
                markeredgecolor=edge, markeredgewidth=0.7)

        t = frame_idx * dt
        ax.set_xlim(-4, 6)
        ax.set_ylim(-0.5, 5.5)
        ax.set_aspect("equal")
        ax.set_title(f"M3: Oblique Bounce with Spin + Friction",
                     fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.tick_params(labelsize=7, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.text(0.95, 0.02, f"t = {t:.2f} s", fontsize=9, ha="right",
                 color="#666666")
        plt.subplots_adjust(left=0.06, right=0.95, bottom=0.05, top=0.91)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    duration = max(int(8000 / n_frames), 16)
    print(f"Saving GIF to {output_path} ({n_frames} frames, {duration}ms/frame)...")
    images[0].save(
        output_path, save_all=True, append_images=images[1:],
        duration=duration, loop=0,
    )
    print("Done.")


if __name__ == "__main__":
    render_gif()
