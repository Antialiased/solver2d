"""Render a side-by-side GIF showing the full-quadratic (12 DoF) modes.

Five panels, each a single BodyFQ2D with one mode excited:
  1. Hx₀      (ξ₁² in x output — horizontal parabolic spread)
  2. Hx₁      (ξ₁² in y output — vertical banana / arch)
  3. Hy₀      (ξ₂² in x output — horizontal banana / sideways bend)
  4. Hy₁      (ξ₂² in y output — vertical parabolic spread)
  5. Pure spin (F = I, vF = ω·skew — sanity check that banana DoFs are not
                excited by rigid rotation)

Every body is rendered as **four sub-cells** split along the banana
symmetry axes ξ₁ = 0 and ξ₂ = 0.  For banana-active modes each sub-cell
has one curved (parabolic) edge; polyline sampling at num_samples = 16
makes the curvature visible.  On each sub-cell the parabolic edge is
monotonic in its arc parameter, so the sub-cell stays boundary-convex
even though the full body is not.

Saved to experiments/biq2d/modes_fq.gif.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io

from .body import BodyFQ2D
from .solver import State, Params, step
from . import energy as energy_mod


NUM_EDGE_SAMPLES = 16   # samples per sub-cell edge for polyline rendering


def make_panel_bodies():
    # Soft material so bananas decay slowly enough to inspect across the GIF.
    common = dict(mass=1.0, half_extent=1.0, k=40.0, nu=0.3)

    b1 = BodyFQ2D(**common)
    b1.Hx = np.array([0.4, 0.0])    # horizontal parabolic spread (ξ₁² → x)

    b2 = BodyFQ2D(**common)
    b2.Hx = np.array([0.0, 0.4])    # vertical arch (ξ₁² → y)

    b3 = BodyFQ2D(**common)
    b3.Hy = np.array([0.4, 0.0])    # sideways bend (ξ₂² → x)

    b4 = BodyFQ2D(**common)
    b4.Hy = np.array([0.0, 0.4])    # vertical spread (ξ₂² → y)

    # Pure spin: F = I, vF = ω · skew.  Hx, Hy, G must stay ≈ 0.
    b5 = BodyFQ2D(**common)
    omega = 2.0 * np.pi / 3.0
    b5.vF = omega * np.array([0.0, -1.0, 1.0, 0.0])

    titles = [
        "Hx[0]  (xi1^2 -> x)\nhorizontal spread",
        "Hx[1]  (xi1^2 -> y)\nvertical arch / banana",
        "Hy[0]  (xi2^2 -> x)\nsideways banana",
        "Hy[1]  (xi2^2 -> y)\nvertical spread",
        "Pure spin  omega = 2pi/3\n(no phantom excitation)",
    ]
    return [b1, b2, b3, b4, b5], titles


def _snapshot(b):
    """Freeze everything we need to re-render the body from a cached state."""
    return (b.c.copy(), b.F.copy(), b.G.copy(), b.Hx.copy(), b.Hy.copy())


def _body_from_snapshot(snap, h=1.0):
    c, F, G, Hx, Hy = snap
    tmp = BodyFQ2D(mass=1.0, half_extent=h)
    tmp.c, tmp.F, tmp.G, tmp.Hx, tmp.Hy = c.copy(), F.copy(), G.copy(), Hx.copy(), Hy.copy()
    return tmp


def render_gif(output_path="experiments/biq2d/modes_fq.gif"):
    dt = 1.0 / 240.0
    n_steps = 720
    render_stride = 4

    bodies, titles = make_panel_bodies()
    states = [State(bodies=[b]) for b in bodies]
    params = Params(gravity=np.array([0.0, 0.0]), dt=dt)

    snapshots = [[_snapshot(b) for b in bodies]]
    for i in range(n_steps):
        for s in states:
            step(s, params)
        snapshots.append([_snapshot(b) for b in bodies])

    print(f"Simulated {n_steps} steps.  Final diagnostics:")
    I_flat = np.array([1., 0., 0., 1.])
    for bi, b in enumerate(bodies):
        U, _, V = energy_mod._svd2(b.F.copy())
        R = (U @ V.T).flatten()
        print(f"  panel {bi+1}: |F-I|={np.linalg.norm(b.F - I_flat):.4e}  "
              f"|F-R|={np.linalg.norm(b.F - R):.4e}  "
              f"|G|={np.linalg.norm(b.G):.4e}  "
              f"|Hx|={np.linalg.norm(b.Hx):.4e}  "
              f"|Hy|={np.linalg.norm(b.Hy):.4e}  "
              f"min det J={b.min_det_J_sampled(9):.4f}")

    render_indices = list(range(0, len(snapshots), render_stride))
    n_frames = len(render_indices)
    print(f"Rendering {n_frames} frames at {render_stride} step stride...")

    # Per-panel palette: (sub-cell base fill, shared edge, label dark)
    palette = [
        ("#cfe7ff", "#3b82c4", "#1e3a5f"),
        ("#fde0c4", "#d97a3a", "#7a3a18"),
        ("#d4f0d0", "#3aa05a", "#1c5030"),
        ("#e8d5f5", "#8a4ec0", "#4a1f70"),
        ("#f5e0e0", "#c44a4a", "#702020"),
    ]

    # 4 slight hue offsets per sub-cell so the split is visually obvious
    # without fighting the panel colour.  Simple: lerp toward white at
    # different alphas per sub-cell.
    def sub_cell_colors(face):
        rgb = np.array([int(face[1:3], 16), int(face[3:5], 16), int(face[5:7], 16)]) / 255.0
        out = []
        for t in (0.00, 0.12, 0.24, 0.12):   # slightly different per cell
            mix = rgb * (1.0 - t) + np.array([1.0, 1.0, 1.0]) * t
            out.append("#{:02x}{:02x}{:02x}".format(
                int(mix[0] * 255), int(mix[1] * 255), int(mix[2] * 255)))
        return out

    images = []
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))

    for frame_idx in render_indices:
        for ax in axes:
            ax.clear()

        snap = snapshots[frame_idx]
        for ax, body_snap, (face, edge, dark), title in zip(axes, snap, palette, titles):
            # Reference square outline (faint)
            ref_pts = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)])
            ax.add_patch(Polygon(ref_pts, closed=True, facecolor="none",
                                 edgecolor="#cccccc", linestyle="--", linewidth=0.8))

            tmp = _body_from_snapshot(body_snap)
            sub_cells = tmp.sub_cell_corners(num_samples=NUM_EDGE_SAMPLES)
            sc_colors = sub_cell_colors(face)

            # Draw the 4 sub-cells with a shared edge color but slightly
            # varying fill so the decomposition is visible.
            for poly_pts, sc_face in zip(sub_cells, sc_colors):
                ax.add_patch(Polygon(poly_pts, closed=True, facecolor=sc_face,
                                     edgecolor=edge, linewidth=1.2, alpha=0.9))

            # Draw internal split lines (ξ₁=0 and ξ₂=0 under the map)
            vert, horz = tmp.split_line_xi()
            ax.plot(vert[:, 0], vert[:, 1], "-", color=edge, linewidth=1.3, alpha=0.9)
            ax.plot(horz[:, 0], horz[:, 1], "-", color=edge, linewidth=1.3, alpha=0.9)

            # Grid markers at the 9 sub-cell corner points
            grid_pts = tmp.sample_grid_3x3()
            ax.plot(grid_pts[:, 0], grid_pts[:, 1], "o", color="white",
                    markersize=4, markeredgecolor=dark, markeredgewidth=1.0, zorder=5)

            # Centre marker (c)
            ax.plot(tmp.c[0], tmp.c[1], "+", color=dark, markersize=8, mew=1.5, zorder=6)

            # Readout
            f_dev = np.linalg.norm(tmp.F - np.array([1., 0., 0., 1.]))
            U, _, V = energy_mod._svd2(tmp.F.copy())
            R = (U @ V.T).flatten()
            f_strain = np.linalg.norm(tmp.F - R)
            min_det = tmp.min_det_J_sampled(9)
            readout = (
                f"|F-I|={f_dev:.3f}\n"
                f"|F-R|={f_strain:.3f}\n"
                f"|G|  ={np.linalg.norm(tmp.G):.3f}\n"
                f"|Hx| ={np.linalg.norm(tmp.Hx):.3f}\n"
                f"|Hy| ={np.linalg.norm(tmp.Hy):.3f}\n"
                f"min|J|={min_det:.3f}"
            )
            ax.text(0.02, 0.98, readout,
                    transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                    family="monospace", color=dark,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor=dark, alpha=0.85))

            ax.set_xlim(-1.9, 1.9)
            ax.set_ylim(-1.9, 1.9)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=9, pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ("top", "right", "bottom", "left"):
                ax.spines[s].set_color("#cccccc")

        t = frame_idx * dt
        fig.suptitle(f"FQ2D modes — BE relaxation, 4-way sub-cell rendering   t = {t:.2f} s",
                     fontsize=12, fontweight="bold", y=0.98)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.04, wspace=0.08)

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
