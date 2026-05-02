"""Render a side-by-side GIF showing the BQ2D modes being excited and decaying.

Five panels, each a single body initialised with one mode excited:
  1. F shear         (F12 nonzero, parallelogram)
  2. F stretch       (F11 > 1, F22 < 1, anisotropic rectangle)
  3. Pure rotation   (F = R(45°), zero velocity — must stay still: rotations
                      are zero-energy in ARAP, so any motion would indicate
                      a phantom restoring force)
  4. G_x trapezoid   (Gx nonzero, horizontal taper)
  5. G_y trapezoid   (Gy nonzero, vertical taper)

Each body relaxes under backward Euler with no gravity.  The GIF illustrates
the deformation modes the bilinear basis can represent and their damping.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io

from .body import BodyBQ2D
from .solver import State, Params, step
from . import energy as energy_mod


def make_panel_bodies():
    """One body per panel with a single mode excited."""
    common = dict(mass=1.0, half_extent=1.0, k=400.0, nu=0.3)

    b1 = BodyBQ2D(**common)
    b1.F = np.array([1.0, 0.35, 0.0, 1.0])     # F12 shear

    b2 = BodyBQ2D(**common)
    b2.F = np.array([1.30, 0.0, 0.0, 0.77])    # anisotropic stretch (volume preserved-ish)

    b3 = BodyBQ2D(**common)
    # Pure spin: F starts at I, vF = ω · skew matrix (= dR/dθ at θ=0).
    # Rotation is a zero-energy mode of ARAP, so the SPD Hessian vanishes in
    # that direction → BE has no stiffness to damp against → angular velocity
    # should be preserved exactly across the whole simulation.
    omega = 2.0 * np.pi / 3.0    # one full rotation in 3 s
    b3.F = np.array([1.0, 0.0, 0.0, 1.0])
    b3.vF = omega * np.array([0.0, -1.0, 1.0, 0.0])

    b4 = BodyBQ2D(**common)
    b4.G = np.array([0.45, 0.0])               # Gx trapezoid

    b5 = BodyBQ2D(**common)
    b5.G = np.array([0.0, 0.45])               # Gy trapezoid

    titles = ["F shear  (F₁₂)",
              "F stretch  (F₁₁, F₂₂)",
              "Pure spin  ω = 2π/3 rad/s\n(must keep spinning, ω constant)",
              "G trapezoid  (Gₓ)",
              "G trapezoid  (Gᵧ)"]
    return [b1, b2, b3, b4, b5], titles


def quad_patch(corners, face, edge):
    """Polygon patch for a deformed quad given (4,2) corners in CORNERS order:
    (-1,-1), (1,-1), (-1,1), (1,1).  We need to close the loop in CCW winding."""
    # Reorder to CCW: bottom-left, bottom-right, top-right, top-left
    order = [0, 1, 3, 2]
    pts = corners[order]
    return Polygon(pts, closed=True, facecolor=face, edgecolor=edge,
                   linewidth=1.5, alpha=0.85)


def render_gif(output_path="experiments/biq2d/modes.gif"):
    dt = 1.0 / 240.0
    n_steps = 720      # 3 seconds
    render_stride = 4  # ~60 fps render

    bodies, titles = make_panel_bodies()
    states = [State(bodies=[b]) for b in bodies]
    params = Params(gravity=np.array([0.0, 0.0]), dt=dt)

    # Snapshot every step so we have full motion for the GIF.
    snapshots = [[(b.c.copy(), b.F.copy(), b.G.copy()) for b in bodies]]
    spin_log = []   # (t, |vF|, |F-R|) for the rotation panel (index 2)
    for i in range(n_steps):
        for s in states:
            step(s, params)
        snapshots.append([(b.c.copy(), b.F.copy(), b.G.copy()) for b in bodies])
        if i % 60 == 59:
            b3 = bodies[2]
            U, _, V = energy_mod._svd2(b3.F.copy())
            R = (U @ V.T).flatten()
            spin_log.append((((i + 1) * dt), float(np.linalg.norm(b3.vF)),
                             float(np.linalg.norm(b3.F - R))))

    print("\nRotation panel angular-velocity tracking (vF norm should stay ≈ ω·√2 = "
          f"{2.0 * np.pi / 3.0 * np.sqrt(2):.4f}):")
    for t, v, fr in spin_log:
        print(f"  t={t:.2f}s  |vF|={v:.4f}  |F-R|={fr:.4e}")

    print(f"Simulated {n_steps} steps.  Sample diagnostics:")
    I_flat = np.array([1., 0., 0., 1.])
    for bi, b in enumerate(bodies):
        F_dev = np.linalg.norm(b.F - I_flat)
        U, _, V = energy_mod._svd2(b.F.copy())
        R = (U @ V.T).flatten()
        F_strain = np.linalg.norm(b.F - R)
        v_norm = np.linalg.norm(b.v)
        print(f"  panel {bi+1}: |F-I|={F_dev:.4e}  |F-R|={F_strain:.4e}  "
              f"|G|={np.linalg.norm(b.G):.4e}  |v|={v_norm:.4e}  "
              f"min det J={b.min_det_J():.4f}")

    render_indices = list(range(0, len(snapshots), render_stride))
    n_frames = len(render_indices)
    print(f"Rendering {n_frames} frames at {render_stride} step stride...")

    # Distinct colors per panel
    palette = [
        ("#cfe7ff", "#3b82c4", "#1e3a5f"),   # blue-ish (F shear)
        ("#fde0c4", "#d97a3a", "#7a3a18"),   # orange (F stretch)
        ("#f5e0e0", "#c44a4a", "#702020"),   # red (rotation — should stay still)
        ("#d4f0d0", "#3aa05a", "#1c5030"),   # green (Gx trapezoid)
        ("#e8d5f5", "#8a4ec0", "#4a1f70"),   # purple (Gy trapezoid)
    ]

    images = []
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.0))

    for frame_idx in render_indices:
        for ax in axes:
            ax.clear()

        snap = snapshots[frame_idx]
        for ax, (c, F, G), (face, edge, dark), title, body in zip(
                axes, snap, palette, titles, bodies):

            # Reference square outline (faint)
            ref_pts = np.array([(-1,-1),(1,-1),(1,1),(-1,1)])
            ax.add_patch(Polygon(ref_pts, closed=True, facecolor="none",
                                 edgecolor="#cccccc", linestyle="--", linewidth=0.8))

            # Deformed corners using current (c, F, G)
            tmp = BodyBQ2D(mass=1.0, half_extent=1.0)
            tmp.c, tmp.F, tmp.G = c.copy(), F.copy(), G.copy()
            corners = tmp.corners()

            ax.add_patch(quad_patch(corners, face, edge))

            # Corner markers
            order = [0, 1, 3, 2]
            for p in corners[order]:
                ax.plot(p[0], p[1], "o", color="white", markersize=5,
                        markeredgecolor=dark, markeredgewidth=1.2, zorder=5)

            # Centre marker
            ax.plot(c[0], c[1], "+", color=dark, markersize=8, mew=1.5)

            # Mode amplitude readout
            # |F-I| includes rotations; |F-R| is the actual elastic strain
            # (distance from nearest rotation — this is what ARAP penalises).
            f_dev = np.linalg.norm(F - np.array([1., 0., 0., 1.]))
            U, _, V = energy_mod._svd2(F.copy())
            R = (U @ V.T).flatten()
            f_strain = np.linalg.norm(F - R)
            g_norm = np.linalg.norm(G)
            ax.text(0.02, 0.98,
                    f"|F-I|={f_dev:.3f}\n|F-R|={f_strain:.3f}\n|G|  ={g_norm:.3f}",
                    transform=ax.transAxes, fontsize=8, va="top", ha="left",
                    family="monospace", color=dark,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor=dark, alpha=0.85))

            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=10, pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ("top", "right", "bottom", "left"):
                ax.spines[s].set_color("#cccccc")

        t = frame_idx * dt
        fig.suptitle(f"BQ2D modes — backward-Euler relaxation   t = {t:.2f} s",
                     fontsize=12, fontweight="bold", y=0.98)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.04, wspace=0.08)

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
