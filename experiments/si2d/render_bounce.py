"""Render a GIF of the M2 floor bounce: decoupled e=1 vs full coupled e=0."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io

from .body import Body2D
from .solver import State, Params, step


def ellipse_points(body, n=64):
    """Return (n,2) array of points on the ellipse boundary."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = body.r0 * np.column_stack([np.cos(theta), np.sin(theta)])
    F_mat = body.F.reshape(2, 2)
    pts = (F_mat @ circle.T).T + body.c
    return pts


def simulate(params_kwargs, h0=5.0, r0=0.5, k=50.0, nu=0.45, n_steps=1920):
    """Run simulation and record body state each frame."""
    b = Body2D(mass=1.0, r0=r0, k=k, nu=nu)
    b.c = np.array([0.0, h0 + r0])
    state = State(bodies=[b])
    params = Params(dt=1.0 / 240.0, position_iters=8, velocity_iters=8, **params_kwargs)

    frames = []
    for i in range(n_steps):
        frames.append((b.c.copy(), b.F.copy(), b.vc.copy()))
        step(state, params)
    frames.append((b.c.copy(), b.F.copy(), b.vc.copy()))
    return frames


def render_gif(output_path="experiments/si2d/bounce.gif"):
    h0 = 5.0
    r0 = 0.5
    n_steps = 1920  # 8 seconds at 240 Hz

    print("Simulating relin e=1...")
    frames_good = simulate(
        dict(restitution=1.0, relin=True),
        h0=h0, r0=r0, n_steps=n_steps)

    print("Simulating coupled e=1...")
    frames_bad = simulate(
        dict(restitution=1.0, position_correct_F=True, velocity_couple_F=True),
        h0=h0, r0=r0, n_steps=n_steps)

    # Adaptive frame selection: denser sampling near contacts
    render_indices = []
    for i in range(n_steps):
        c_good = frames_good[i][0]
        c_bad = frames_bad[i][0]
        near_floor = min(c_good[1], c_bad[1]) < 2.0
        if near_floor:
            if i % 4 == 0:
                render_indices.append(i)
        else:
            if i % 12 == 0:
                render_indices.append(i)

    n_frames = len(render_indices)
    print(f"Rendering {n_frames} frames...")
    images = []

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 5))

    for frame_idx in render_indices:
        for ax in (ax_left, ax_right):
            ax.clear()

        for ax, frames, title, fill_color, edge_color in [
            (ax_left, frames_good, "Relin  e = 1", "#4A90D9", "#1a3d6e"),
            (ax_right, frames_bad, "Coupled  e = 1", "#D94A4A", "#6e1a1a"),
        ]:
            c, F, vc = frames[frame_idx]

            # Floor
            ax.fill_between([-5, 5], -0.6, 0, color="#D2C5A0", alpha=0.6)
            ax.plot([-5, 5], [0, 0], color="#8B7355", linewidth=2.5)

            # Ghost trail
            trail_alpha = [0.06, 0.10, 0.16]
            for ti, offset in enumerate([24, 16, 8]):
                prev_idx = frame_idx - offset
                if prev_idx >= 0:
                    pc, pF, _ = frames[prev_idx]
                    body_tmp = Body2D(mass=1.0, r0=r0)
                    body_tmp.c = pc
                    body_tmp.F = pF
                    pts = ellipse_points(body_tmp, n=64)
                    poly = Polygon(pts, closed=True, facecolor=fill_color,
                                   edgecolor="none", alpha=trail_alpha[ti])
                    ax.add_patch(poly)

            # Current ellipse
            body_tmp = Body2D(mass=1.0, r0=r0)
            body_tmp.c = c
            body_tmp.F = F
            pts = ellipse_points(body_tmp, n=64)
            poly = Polygon(pts, closed=True, facecolor=fill_color,
                           edgecolor=edge_color, linewidth=2.0, alpha=0.9)
            ax.add_patch(poly)

            # Center dot
            ax.plot(c[0], c[1], "o", color="white", markersize=3,
                    markeredgecolor=edge_color, markeredgewidth=0.7)

            # Drop height reference
            ax.plot([-2.0, 2.0], [h0, h0], color="#BBBBBB", linewidth=0.8,
                    linestyle="--", zorder=0)
            ax.text(2.1, h0, "h₀", fontsize=8, color="#999999", va="center")

            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-0.6, 6.5)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
            ax.set_xticks([])
            ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
            ax.tick_params(labelsize=7, length=3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        t = frame_idx / 240.0
        fig.suptitle(f"2D Deformable Bounce  —  t = {t:.2f} s    (k = 50, ν = 0.45)",
                     fontsize=12, fontweight="bold", y=0.97)
        plt.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.89, wspace=0.18)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    # Target ~30 fps display. Adaptive sampling means variable dt per frame.
    # Use uniform duration for simplicity.
    duration = max(int(8000 / n_frames), 16)  # aim for 8s playback
    print(f"Saving GIF to {output_path} ({n_frames} frames, {duration}ms/frame)...")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    print("Done.")


if __name__ == "__main__":
    render_gif()
