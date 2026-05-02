"""Render a GIF of a cantilever beam using the TGS + VBD solver."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io

from .body import Body2D
from .solver import State, Params, Joint, step_tgs
from .render_collision import ellipse_points, quadrant_patches
from . import energy


def render_gif(output_path="experiments/si2d/cantilever_tgs.gif"):
    dt = 1.0 / 240.0
    n_steps = 720  # 3 seconds (stays in stable region)
    n_bodies = 8
    r0 = 0.3
    k = 2000.0
    nu = 0.35

    bodies = []
    for i in range(n_bodies):
        b = Body2D(mass=0.5, r0=r0, k=k, nu=nu, static=(i == 0))
        b.c = np.array([2.0 * r0 * i, 3.0])
        bodies.append(b)

    joints = []
    for i in range(n_bodies - 1):
        joints.append(Joint(
            body_a_idx=i, body_b_idx=i + 1,
            local_a=np.array([r0, 0.0]),
            local_b=np.array([-r0, 0.0]),
        ))

    state = State(bodies=bodies, joints=joints)
    params = Params(dt=dt, substeps=4)

    frames = []
    import warnings
    warnings.filterwarnings("ignore")
    for i in range(n_steps):
        frames.append([(b.c.copy(), b.F.copy()) for b in bodies])
        step_tgs(state, params)

        # Diagnostics every 120 steps
        if i % 120 == 0:
            max_F = max(np.max(np.abs(b.F)) for b in bodies if not b.static)
            min_det = min(energy._det2(b.F) for b in bodies if not b.static)
            print(f"  step {i:4d}: tip_y={bodies[-1].c[1]:.3f}, "
                  f"max|F|={max_F:.4f}, min_det={min_det:.6f}")

        # Safety: bail if exploded
        max_F = max(np.max(np.abs(b.F)) for b in bodies if not b.static)
        if max_F > 50:
            print(f"  DIVERGED at step {i}, truncating")
            break

    frames.append([(b.c.copy(), b.F.copy()) for b in bodies])

    render_indices = list(range(0, len(frames) - 1, 3))
    n_frames = len(render_indices)
    print(f"Rendering {n_frames} frames...")

    # Color gradient from root (dark) to tip (light)
    colors = []
    for i in range(n_bodies):
        t = i / max(n_bodies - 1, 1)
        r_c = int(74 + t * 100)
        g_c = int(144 + t * 60)
        b_c = int(217 - t * 40)
        light = f"#{r_c:02x}{g_c:02x}{b_c:02x}"
        r_d = int(37 + t * 70)
        g_d = int(99 + t * 40)
        b_d = int(160 - t * 30)
        dark = f"#{r_d:02x}{g_d:02x}{b_d:02x}"
        edge_c = f"#{max(r_d-30,0):02x}{max(g_d-30,0):02x}{max(b_d-30,0):02x}"
        colors.append((light, dark, edge_c))

    images = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for frame_idx in render_indices:
        ax.clear()

        # Wall at x=0
        ax.fill_between([-0.6, -0.05], -0.5, 5, color="#888888", alpha=0.3)
        ax.plot([-0.05, -0.05], [-0.5, 5], color="#555555", linewidth=2)

        for bi in range(n_bodies):
            c, F = frames[frame_idx][bi]
            fl, fd, edge_c = colors[bi]

            body_tmp = Body2D(mass=0.5, r0=r0)
            body_tmp.c = c
            body_tmp.F = F
            for patch in quadrant_patches(body_tmp, fl, fd, edge_c):
                ax.add_patch(patch)

            pts = ellipse_points(body_tmp, n=64)
            poly = Polygon(pts, closed=True, facecolor="none",
                           edgecolor=edge_c, linewidth=1.0)
            ax.add_patch(poly)

        # Draw joint markers
        for j in joints:
            ca, Fa = frames[frame_idx][j.body_a_idx]
            la = j.local_a
            px = ca[0] + Fa[0]*la[0] + Fa[1]*la[1]
            py = ca[1] + Fa[2]*la[0] + Fa[3]*la[1]
            ax.plot(px, py, "o", color="white", markersize=3,
                    markeredgecolor="#333", markeredgewidth=0.7, zorder=5)

        t = frame_idx * dt
        ax.set_xlim(-0.8, 5.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_aspect("equal")
        ax.set_title(f"Cantilever (TGS + VBD, {n_bodies} bodies)",
                     fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_yticks([0, 1, 2, 3, 4])
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

    duration = max(int(6000 / n_frames), 16)
    print(f"Saving GIF to {output_path} ({n_frames} frames, {duration}ms/frame)...")
    images[0].save(
        output_path, save_all=True, append_images=images[1:],
        duration=duration, loop=0,
    )
    print("Done.")


if __name__ == "__main__":
    render_gif()
