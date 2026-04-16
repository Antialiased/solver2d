"""Render a GIF of body-body collision: oblique drop onto a resting ball."""
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
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = body.r0 * np.column_stack([np.cos(theta), np.sin(theta)])
    F_mat = body.F.reshape(2, 2)
    pts = (F_mat @ circle.T).T + body.c
    return pts


def quadrant_patches(body, fill_light, fill_dark, edge_color, n_arc=32, alpha=0.9):
    """Return 4 matplotlib Polygon patches for the body's quadrants."""
    F_mat = body.F.reshape(2, 2)
    patches = []
    for qi, (t0, t1) in enumerate([(0, np.pi/2), (np.pi/2, np.pi),
                                    (np.pi, 3*np.pi/2), (3*np.pi/2, 2*np.pi)]):
        theta = np.linspace(t0, t1, n_arc)
        arc = body.r0 * np.column_stack([np.cos(theta), np.sin(theta)])
        ref_pts = np.vstack([[0.0, 0.0], arc])
        world_pts = (F_mat @ ref_pts.T).T + body.c
        color = fill_light if qi % 2 == 0 else fill_dark
        poly = Polygon(world_pts, closed=True, facecolor=color,
                       edgecolor=edge_color, linewidth=0.5, alpha=alpha)
        patches.append(poly)
    return patches


def render_gif(output_path="experiments/si2d/collision.gif"):
    dt = 1.0 / 240.0
    n_steps = 1680  # 7 seconds
    r0 = 0.5
    k = 50.0
    nu = 0.45

    # Ball A: dropped from height with lateral offset
    a = Body2D(mass=1.0, r0=r0, k=k, nu=nu)
    a.c = np.array([-0.8, 5.0 + r0])
    a.vc = np.array([0.5, 0.0])

    # Ball B: resting near floor
    b = Body2D(mass=1.0, r0=r0, k=k, nu=nu)
    b.c = np.array([0.0, r0])

    state = State(bodies=[a, b])
    params = Params(
        dt=dt, restitution=0.5, friction=0.3, position_iters=8, velocity_iters=8,
    )

    frames = []
    for i in range(n_steps):
        frames.append([(body.c.copy(), body.F.copy()) for body in state.bodies])
        step(state, params)
    frames.append([(body.c.copy(), body.F.copy()) for body in state.bodies])

    # Adaptive frame selection
    render_indices = []
    for i in range(n_steps):
        ca, cb = frames[i][0][0], frames[i][1][0]
        near_action = min(ca[1], cb[1]) < 2.0 or np.linalg.norm(ca - cb) < 2.0
        if near_action:
            if i % 3 == 0:
                render_indices.append(i)
        else:
            if i % 8 == 0:
                render_indices.append(i)

    n_frames = len(render_indices)
    print(f"Rendering {n_frames} frames...")

    colors = [("#4A90D9", "#2563a0", "#1a3d6e"),
              ("#D94A4A", "#a03030", "#6e1a1a")]
    images = []
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    for frame_idx in render_indices:
        ax.clear()

        ax.fill_between([-4, 4], -0.5, 0, color="#D2C5A0", alpha=0.6)
        ax.plot([-4, 4], [0, 0], color="#8B7355", linewidth=2.5)

        for bi, (fill_light, fill_dark, edge) in enumerate(colors):
            c, F = frames[frame_idx][bi]

            # Ghost trail
            for ti, offset in enumerate([16, 8, 4]):
                prev_idx = frame_idx - offset
                if prev_idx >= 0:
                    pc, pF = frames[prev_idx][bi]
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

            # Outline
            pts = ellipse_points(body_tmp, n=64)
            poly = Polygon(pts, closed=True, facecolor="none",
                           edgecolor=edge, linewidth=1.5)
            ax.add_patch(poly)

            ax.plot(c[0], c[1], "o", color="white", markersize=3,
                    markeredgecolor=edge, markeredgewidth=0.7)

        t = frame_idx * dt
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-0.5, 6.5)
        ax.set_aspect("equal")
        ax.set_title(f"Ellipse-Ellipse Collision  (k={k}, nu={nu})",
                     fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax.tick_params(labelsize=7, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.text(0.95, 0.02, f"t = {t:.2f} s", fontsize=9, ha="right",
                 color="#666666")
        plt.subplots_adjust(left=0.08, right=0.95, bottom=0.05, top=0.91)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    duration = max(int(7000 / n_frames), 16)
    print(f"Saving GIF to {output_path} ({n_frames} frames, {duration}ms/frame)...")
    images[0].save(
        output_path, save_all=True, append_images=images[1:],
        duration=duration, loop=0,
    )
    print("Done.")


if __name__ == "__main__":
    render_gif()
