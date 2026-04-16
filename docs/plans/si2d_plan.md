# si2d: 2D Deformable Body Prototype in Python

Companion to the C retrofit plan (`plan.md`). This doc covers the **Python-only 2D prototype** (`experiments/si2d/`) — the sandbox where we validate the 2D math (mass matrix, strain energy, contact Jacobians, integrator) before touching solver2d's C code. Everything here is ellipse-only: a deformable body is a circle at rest that deforms into an ellipse under affine motion.

See `../insights.md` for running experimental findings. The 1D prototype (`experiments/si1d/`) is the direct predecessor; design choices here should reduce to si1d's when you restrict to one axis.

## Body model

### Reference shape and DoFs

The reference body is a **uniform-density disk** of radius `r0`, centered at the origin.

A material point at reference position `X` (with `|X| <= r0`) maps to world position:

    x(X) = c + F * X

where `c = (cx, cy)` is the center and `F` is a 2x2 deformation gradient.

**DoFs per body: 6** — `(cx, cy, F11, F12, F21, F22)`.

No explicit rotation angle. Rigid rotation, stretch, and shear are all encoded in `F`. The polar decomposition `F = R * S` is computed when needed (e.g. for the corotated elastic energy) but is not part of the state.

**1D reduction check:** restrict to `c = (cx, 0)`, `F = diag(1, s/s0)`. Then the body is a vertical interval `[-s, s]` centered at `(cx, 0)`. The DoFs reduce to `(cx, s)`, matching si1d.

### Mass matrix

Kinetic energy:

    T = 1/2 * integral_body rho * |c_dot + F_dot * X|^2 dA

For a centred disk of radius `r0` with uniform density `rho`:
- `integral 1 dA = pi * r0^2 = M / rho` (total mass `M`)
- `integral X dA = 0` (centred)
- `integral Xi Xj dA = (pi * r0^4 / 4) * delta_ij` (second moment of disk)

So:

    T = 1/2 * M * |c_dot|^2  +  1/2 * mu * ||F_dot||_F^2

where `mu = M * r0^2 / 4` is the **affine inertia** (scalar, same for all four `F` components).

The generalized mass matrix is **block diagonal**: `diag(M, M, mu, mu, mu, mu)`.

**1D check:** for a 1D interval of length `2*r0`, the second moment gives `mu_1d = M * r0^2 / 3`, i.e. `mu_frac = 1/3`. The 2D disk gives `mu_frac_2d = r0^2 / 4`. These are just the normalised second moments of the respective reference shapes.

### Elastic energy

**Stable Neo-Hookean (SNH)** energy on the deformation gradient `F`, following Kim & Eberle's "Dynamic Deformables" SIGGRAPH course (2020/2022) and the HOBAK codebase (`github.com/theodorekim/HOBAKv1`).

2D specialisation:

    Ic = ||F||_F^2 = F11^2 + F12^2 + F21^2 + F22^2
    J  = det(F)   = F11*F22 - F12*F21
    alpha = 1 + mu / lam

    Psi(F) = 0.5 * [mu * (Ic - 2) + lam * (J - alpha)^2]

Gradient (PK1 stress):

    P(F) = dPsi/dF = mu * F + lam * (J - alpha) * cof(F)

where `cof(F) = [F22, -F21, -F12, F11]` (cofactor of 2x2).

Hessian (4x4):

    H = mu * I_4  +  lam * cof(F) cof(F)^T  +  lam*(J-alpha) * d^2J/dF^2

where `d^2J/dF^2` has exactly 4 nonzero entries: `(0,3)=+1, (3,0)=+1, (1,2)=-1, (2,1)=-1`. No polar decomposition derivatives needed — this is the key advantage of SNH over corotated linear.

SPD projection: eigendecompose the 4x4 Hessian, clamp negative eigenvalues to zero, reconstruct. Trivially cheap for a 4x4 matrix. Kim & Eberle provide an analytic eigensystem in 3D (twist/flip/scaling modes); the 2D version is simpler but the numeric eigendecomposition is already fast enough.

**Lame parameter mapping:**  user-facing stiffness `k` and Poisson ratio `nu` (must be > 0, since `lam = 0` makes `alpha` singular):

    mu  = k * (1 - 2*nu)
    lam = 2 * k * nu

This gives `d^2 Psi/ds^2 = mu + lam = k` for uniaxial stretch `F = diag(1, s)` at `s = 1`, matching si1d's convention. Uniform scaling stiffness is `4 * lam = 8*k*nu` (stiffer, because it fights area preservation).

**1D reduction check:** with `F = diag(1, s)`, `Ic = 1 + s^2`, `J = s`, we get `Psi = 0.5 * [mu*(s^2-1) + lam*(s-alpha)^2]`, and `d^2Psi/ds^2|_{s=1} = mu + lam = k`. Verified symbolically and via finite differences in `experiments/si2d/verify_energy.py`.

### Area preservation

Built in to SNH via the `lam * (J - alpha)^2` term. The Poisson ratio `nu` controls the strength: `nu = 0.3` gives moderate area preservation, `nu -> 0.49` approaches incompressibility. No separate penalty or constraint needed.

## Collision detection

### Ellipse-floor (half-plane at y = 0)

The deformed body is the set `{c + F * n : |n| <= r0}`. The lowest world-space point is in direction `-ey`:

    gap = cy - r0 * |F^T * ey|

where `ey = (0, 1)` and `F^T * ey = (F21, F22)`.

This is **closed-form**. The contact normal is `ey` (upward), and the contact point on the body is:

    p_contact = c - r0 * F * (F^T * ey) / |F^T * ey|

### Ellipse-ellipse (non-iterative, Mueller oriented particles)

Each body `i` is the set `{ci + Fi * n : |n| <= r0_i}`. The support point in direction `d` is:

    support_i(d) = ci + r0_i * Fi * (Fi^T * d) / |Fi^T * d|

**Non-iterative algorithm:** transform to B's frame so B becomes a circle, then a single support-function evaluation gives the contact:
1. Transform: `c' = FB^{-1} * (cA - cB)`, `F' = FB^{-1} * FA`. B is now a circle of radius `rB` at the origin.
2. Direction: `d' = c' / |c'|` (from B's center toward A's center in B's frame).
3. Support point on A': `pA' = c' + rA * F' * (F'^T * (-d')) / |F'^T * (-d')|`.
4. Project onto B's circle: `pB' = rB * pA' / |pA'|`.
5. Transform back: `pA = FB * pA' + cB`, `pB = FB * pB' + cB`.
6. Gap = world-space distance between pA and pB, signed by whether pA' is inside B's circle.

Zero iterations, no convergence issues. Exact for circles, good approximation for moderate deformations. No GJK/EPA or broadphase needed for this prototype.

## Contact Jacobians

### Floor contact

    gap = cy - r0 * sqrt(F21^2 + F22^2)

Jacobian `J = d(gap)/d(DoFs)` for body `i`:

    J = [0, 1, 0, 0, -r0*F21/L, -r0*F22/L]

where `L = sqrt(F21^2 + F22^2)`. Only `(cy, F21, F22)` participate.

**1D check:** with `F = diag(1, s/s0)`, `F21 = 0`, `F22 = s/s0`, `L = s/s0`. Then `J_s = -r0 * (s/s0) / (s/s0) = -r0/s0... ` Hmm — in 1D we had `J = [+1, -1]` acting on `(x, s)`. Here `gap = cy - r0 * |s/s0| = cy - s` (since `r0 = s0`). So `J = [d/dcy, d/dF22] = [1, -r0] = [1, -s0]`, and in the 1D model `d(gap)/ds = -1` because `s` is the half-extent directly. The factor of `s0` is a coordinate scaling — in si1d `s` is the half-extent (world units), here `F22 = s/s0` is dimensionless. Both are correct.

### Ellipse-ellipse contact

For two bodies `i, j` with GJK-computed closest points `pi` on body `i` and `pj` on body `j`, contact normal `n` (from `i` to `j`), and gap `g = (pj - pi) . n`:

The contact point `pi` on body `i` is `ci + Fi * ni_local` where `ni_local` is a point on the reference disk. The Jacobian row for body `i` is:

    J_i = d(g)/d(DoFs_i) = -n^T * [I_2 | ni_local^T kron I_2]  (block form)

This is 1x6: `-[nx, ny, nx*nix, nx*niy, ny*nix, ny*niy]` where `ni = ni_local`.

Similarly for body `j` (with `+n` sign).

The effective mass is `K = J_i * M_i^{-1} * J_i^T + J_j * M_j^{-1} * J_j^T`, a scalar.

## Integrator

### Free flight (between contacts)

The internal elastic force on `F` is `-dPsi/dF`. For the corotated model this is nonlinear in `F` (via the polar decomposition).

Options:
1. **Backward Euler** on the full `(c, F, vc, vF)` state. Unconditionally stable, damps stiff modes (the 1D finding). Requires solving a 6x6 nonlinear system per body per step — Newton with 1-2 iterations using the Hessian.
2. **Exponential integrator** on the linearised elastic oscillator. In 1D this was exact for the harmonic oscillator. In 2D, linearise `Psi` around the current `F` to get `d^2 Psi/dF^2` (a 4x4 SPD matrix), then the affine DoFs obey `mu * F_ddot = -H * (F - F_eq)`. Eigendecompose `H` (4x4, once per step) and apply the scalar exponential solution per eigenmode.
3. **Semi-implicit Euler** (symplectic). Cheapest, but dt-limited by the stiffest eigenmode of `H`.

**Plan: start with backward Euler for robustness.** Switch to exponential once the contact solver is working and energy preservation becomes the focus (exactly the 1D trajectory).

### Contact solver

Sequential impulses, matching si1d's structure:

1. **Velocity pass**: for each active contact, compute `Jv`, apply impulse `dlam = -Jv / K` with clamping `lam >= 0`, update velocities of both bodies.
2. **Position pass** (split from velocity pass, as in si1d's `two_pass` mode): correct penetration via `dlam = -bias * gap / K`, update positions only.

The two-pass split is essential for energy preservation (si1d finding: single-pass couples position correction back into velocity, pumping energy after a velocity reflection).

Warm starting: carry `lam` from previous step, scale by `dt_ratio` if dt changed.

## Code structure

```
experiments/si2d/
    __init__.py
    energy.py          # SNH energy, PK1 gradient, Hessian, SPD projection, Lame conversion
    verify_energy.py   # sympy + finite-diff + 1D-reduction + SPD verification (run as module)
    body.py            # Body2D dataclass, mass matrix, integration
    collision.py       # gap functions, contact Jacobians (floor + GJK)
    solver.py          # Params, State, step function, SI sweeps
    tests.py           # milestone scenarios (see below)
    diag_bounce.py     # energy-preservation diagnostic (port of si1d/diag_single_bounce.py)
```

## Milestones

Each milestone has a clear pass/fail criterion. Findings go into `docs/insights.md`.

### M1: Free flight of a spinning deformable disk
- One body, no gravity, no contacts. Initial: `F = I`, `vF` has off-diagonal components (spin + stretch).
- **Pass:** total energy `T + Psi` conserved to machine precision over 10k steps (exponential integrator) or to O(dt^2) (BE). Rigid-body limit (k -> inf): `F` stays near `R(theta)`, angular velocity constant.

### M2: Single deformable disk bouncing on floor
- Gravity, floor contact, `F` initially `I`, dropped from height `h0`.
- **Pass (no restitution):** ball settles. Measure `E_n / E_0` per bounce (apex-height ratio). Compare with the Moreau-projection prediction: energy tax per bounce = `1 - 1/(1 + mu/M)` per participating DoF. In 1D this was 25% (mu_frac=1/3); in 2D it should be `1 - 1/(1 + r0^2/4)` per contact, depending on how many F-components the Jacobian excites.
- **Pass (with velocity restitution e=1):** `E_n / E_0 ~ 1.000` across stiffness sweep, matching 1D result.

### M3: Oblique bounce (rotation + deformation)
- Single disk hitting floor at an angle with initial angular momentum.
- **Pass:** qualitative correctness (it spins and deforms), energy ratio consistent with M2 predictions.
- This is the test 1D literally could not do.

### M4: Two deformable disks attracting (free space)
- Port of si1d's `diag_two_body.py` to 2D. Two disks, opposing gravity, no floor.
- **Pass:** with restitution e=1, `E_end/E_0 > 0.95` at moderate stiffness (matching si1d's 0.985).

### M5: Three-disk vertical stack under gravity
- Three disks stacked on the floor. First multi-contact scenario.
- **Pass:** stable rest with visually correct deformation (bottom disk squished more). No inter-penetration, no energy growth.

## Open design questions

1. **Friction.** Omitted for now. Floor friction is needed for M3 (oblique bounce) to be physically meaningful, but the energy analysis is cleaner without it. Add after M3 if oblique bounces look wrong.

2. **Plastic deformation.** The si1d prototype has yield + hardening on the s-DoF. The 2D equivalent is a yield surface on the Biot strain `S - I` (von Mises / J2 in 2D). Deferred to post-M5.

3. **Exponential integrator in 2D.** The 4x4 eigendecomposition per body per step is cheap for small body counts but scales as O(N) with a non-trivial constant. For the prototype this is fine; flag if it becomes a bottleneck.

4. **SNH alpha singularity at nu=0.** The `alpha = 1 + mu/lam` term diverges when `lam -> 0` (Poisson ratio -> 0). HOBAK reparametrises with `lam' = lam + mu` to avoid this. For now we require `nu > 0`; if compressible materials are needed, either adopt HOBAK's reparametrisation or switch to a pure `Ic`-based energy for the `lam = 0` case.
