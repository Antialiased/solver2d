# Research: Deformable Body Dynamics

A working name for this project. "Affine Body Dynamics" would be the natural label, but Lan et al. (SIGGRAPH 2022) published under that name first; we use **Deformable Body Dynamics (DBD)** to distinguish our real-time, substepped-compliant-constraint, convex-2D-first line of work.

**Companions:** `code_map.md` (file-by-file index of the solver2d codebase), `insights.md` (running catalogue of experimental findings that validate or revise what's in this doc), `followups.md` (downstream applications of DBD material state — fracture, hallucinated modes, render/FX), `plans/` (active implementation and experiment plans, including `plans/plan.md` for the staged solver build-out).

## Question

**Can we efficiently include deformable-body behaviour (solid mechanics; elasto-plasto-rigid) in a real-time physics engine — without paying the cost of a mesh-based FEM/XPBD pipeline?**

## Motivation

Real-time deformable simulation today typically means XPBD cloth or tetrahedral/FEM soft bodies. Compared to rigid-body dynamics (RBD), these are roughly an order of magnitude more expensive, for two structural reasons:

1. **DoF explosion.** RBD stores one transform (position + rotation, 3 DoFs in 2D / 6 in 3D) per body. Mesh-based deformables concentrate DoFs at every vertex. A single soft body can have hundreds to thousands of DoFs, each participating in constraints.
2. **Loss of convexity.** RBD collision can exploit convex shapes — GJK, SAT, MPR — which are fast, robust, and closed-form for common primitives. Arbitrary deformable meshes are non-convex and often require CCD or IPC-style barrier methods to remain robust under adversarial input.

Robustness under arbitrary user input is the **first** requirement for a game physics engine — it must not blow up, tunnel, or jitter regardless of what the player does. Current deformable methods sacrifice this, or buy it back at great cost (IPC).

## Proposed approach: affine bodies

Replace the rigid transform of each body with a general **affine transform**. In 2D that is a 2×2 linear map `A` plus a translation `t`:

```
x_world = A * x_local + t
```

where `A ∈ R^{2×2}` is unconstrained (vs. rigid, which constrains `A = R` with `R ∈ SO(2)`). This adds only **one extra scalar DoF per body in 2D** (4 entries in `A` minus the 3 rigid DoFs of rotation+uniform-scale-removed… actually: rigid uses 1 angle, affine uses 4 matrix entries, so +3 DoFs in 2D / +6 in 3D). Still `O(1)` per body, not `O(#vertices)`.

Why this is the right amount of softening:

- **Squash, stretch, shear, and volume change** all fall out of `A` directly. Plasticity is a rest-state update on `A`. Elasticity is a restoring force pulling `A` toward `R` (or toward the plastic rest `A_p`).
- **Affine is the richest per-body deformation that preserves convexity.** Linear maps send convex sets to convex sets. Anything more expressive (bending, free-form deformation) breaks convexity and forces us back onto mesh collision.
- Because shapes stay convex, **GJK/SAT/MPR still apply** — with the local shape transformed by `A` each query, or equivalently with support points `s_world(d) = A * s_local(A^T d) + t`. Broadphase AABBs still work; CCD, if needed, works on the same support functions.
- Warm-started contact and constraint machinery from the existing solver carries over almost unchanged: contacts still live between two bodies, anchors are still local points, impulses still apply to a small fixed-size state vector per body.

## Relationship to prior work

### Affine Body Dynamics (Lan et al., SIGGRAPH 2022)
The closest named predecessor and the reason we have to call ours something else. ABD uses the same core idea — per-body affine state as a replacement for a rigid transform — but lives in an IPC / implicit-integration world aimed at offline-ish robustness. DBD takes the same state representation into the **substepped compliant-constraint** regime of solver2d's TGS Soft pipeline, 2D first, aimed at real-time game-grade robustness. The bet is that IPC-style guarantees aren't the only way to stay robust once you already have convex collision and aggressive substepping.

### Material Point Method (MPM)
The other useful comparison, and probably the richer source of mathematical scaffolding.

In MPM, each **particle** already carries exactly the state we are proposing per body: a position, an affine velocity (often called `C` in APIC/MLS-MPM), and — implicitly via the deformation gradient `F` — an affine "pose". A particle **is** a small elasto-plastic body that owns an affine transform and affine velocity. Each timestep:

1. Particle state is scattered to a background grid (P2G).
2. Forces and momentum are resolved on the grid.
3. Grid state is interpolated back to particles (G2P), updating `v`, `C`, and `F`.

MPM is, from this angle, essentially a **reconstruction filter** (the grid round trip) sitting on top of a **hypoelastic / finite-strain material model** carried on particles. The grid exists because MPM interprets particles as point samples of an underlying continuum field, and the filter is what keeps neighboring samples coupled.

**DBD treats its bodies as discrete objects, not point samples of a field.** Two DBD bodies are coupled by contact constraints, not by spatial kernel overlap on a shared grid. So we can drop the reconstruction filter entirely and keep the per-particle material model. That gives us, essentially, "MPM particles without the grid" — which is the same shape as a compliant-constraint rigid body solver where the body state has been widened from `SE(2)` to `GA(2)` (general affine).

A suggestive checkpoint: **for a single MPM particle in isolation, the P2G/G2P round trip is (to the degree interpolation permits) lossless** — the particle's affine state in and out matches. In that limit, MPM integration of one particle reduces to exactly the kind of discrete affine-body update we want. This is strong evidence that the DBD integrator should look like "an isolated MPM particle, stepped forward with a hypoelastic material, coupled to others through explicit contact/joint constraints instead of through a grid."

The paper **"A Position-Based Material Point Method"** (and the broader position-based / XPBD-flavored MPM line) points in the same direction from the opposite side: it shows MPM material models transpose cleanly into compliant (position-based / XPBD) constraint form. We want the same transposition but applied to a *single* particle blown up to body scale, with convex collision replacing the grid coupling.

### What we inherit from each

From ABD: the state representation `(A, t)`, the basic observation that affine preserves convexity, the polar-decomposition elastic energy `‖A − R(A)‖²` and its plastic extensions.

From MPM: the material model toolbox (hypoelastic updates, `F = F_e F_p` multiplicative plasticity, yield surfaces, J-based volumetric response), the APIC-style affine velocity `C` as the correct conjugate variable to `A`, and the intuition that the affine DoFs are "cheap" because nature already uses them at particle granularity in a working real-time method (MLS-MPM runs at interactive rates for tens of thousands of particles).

From solver2d / TGS Soft: substepping, compliant constraints parameterized by Hertz/damping, warm starting, and convex contact. This is the delivery vehicle.

### What's genuinely new (we think)

- Per-body affine state in a **real-time compliant-constraint** solver, not an IPC solver and not a grid-based continuum solver.
- Explicit bet that **convex collision + substepping + compliance** is sufficient for robustness at game scale, so we never need a grid or a barrier.
- Framing a DBD body as "one MPM particle, scaled up, with its own convex shape, coupled via contacts." This reframing — dropping the reconstruction filter and keeping the material model — is the specific theoretical move we want to validate.

## Hypothesis

A TGS Soft solver whose "body state" is an affine map `(A, t)` instead of a rigid transform `(R, t)` can:

- **H1 (performance):** run at close to rigid-body cost — within a small constant factor (target: <2×) of solver2d's existing TGS Soft, regardless of visual deformation amplitude.
- **H2 (robustness):** retain the robustness properties of convex RBD under adversarial stacking, high mass ratios, and fast input, because collision detection remains convex and substepping + compliance keep the constraint solve well-conditioned.
- **H3 (expressiveness):** capture the qualitatively interesting behaviours of elasto-plasto-rigid solids — squash on impact, plastic dents, jelly wobble, near-incompressibility — with material parameters that map cleanly onto frequency/damping/yield knobs.
- **H4 (authoring):** require no mesh, no tet generation, no per-vertex authoring. The artist supplies the same convex shape(s) they would for a rigid body plus a small material block.

## What counts as success

We will consider the technique proven out in 2D if we can demonstrate, inside solver2d:

1. A working `s2_solverAffine` (or extension of `s2_solverTGS_Soft`) where each dynamic body carries an affine state and an elastic + plastic material.
2. Stable stacks and piles of affine bodies under gravity, matching rigid-body stacking quality (no creeping, no explosion) at comparable iteration counts.
3. Visible squash/stretch/volume-preservation on impact, controllable via stiffness (Hz) and Poisson-like parameter.
4. Plasticity: permanent deformation after yield, rest shape update, no energy drift.
5. A "rigid limit": as stiffness → ∞ the solver reproduces rigid-body behaviour (sanity check against existing TGS Soft).
6. Wall-clock cost per step within a small constant factor of rigid TGS Soft on matched scenes.
7. Robustness under adversarial input (fast mouse drag, high mass ratio, thin walls) comparable to rigid baseline.

## Open questions / footnotes

These are the "subject to some footnotes" — things we expect to have to resolve during the work.

- **Inertia under deformation.** The inertia tensor is no longer constant in body frame once `A` is non-rigid. How do we integrate angular/affine momentum cheaply? Do we reuse a reference inertia and correct, or recompute each substep?
- **Affine momentum / conjugate variables.** What are the correct generalized velocities and masses for the 4 entries of `A`? (Lan et al. give one answer; we need the 2D TGS Soft form.)
- **Compliance formulation.** The elastic "restoring" energy keeping `A` near `R` (or near `A_p`) needs to plug into the soft-constraint (Hertz/damping) framework the rest of the solver already uses, so the substep loop stays unchanged.
- **Degenerate `A`.** What prevents `det(A) → 0` or flipping? Barrier? Projection? Large restoring force? This is where robustness could leak.
- **Plasticity model.** Simple yield surface on the deviatoric part of `A`? Rate-independent vs. rate-dependent? Hardening?
- **Multi-shape bodies.** A body with several attached convex shapes: do they all share one `A`, or can they deform independently (giving up global convexity but keeping per-shape convexity)?
- **Contact anchors.** Local anchor points `rA`, `rB` used by the contact solver become `A * rA_local`, which changes each substep — mostly already handled by TGS Soft's "updated anchors" path, but needs auditing.
- **Friction.** Does the existing Coulomb tangent-impulse model still behave well when the contact frame is being stretched by `A`?
- **Coupling to joints.** Revolute/mouse joints currently assume rigid anchors. How do they interact with affine bodies — do we project, or generalize the joint Jacobians?
- **The "+1 DoF" footnote.** In 2D, rigid has 3 DoFs, affine has 6. The claim of "essentially zero extra DoFs" is about *scaling* (still `O(1)` per body vs. `O(#vertices)`), not literal zero. Worth stating carefully in any write-up.

## Material range and the near-rigid limit

A core design requirement: **DBD must span the full material range, from soft jelly all the way down to almost-rigid**, with the same solver and no tuning cliffs. Rigidity should be a limit, not a special case.

### Why "almost-rigid" rather than "rigid"

Modern rigid-body solvers already do not aim for *true* rigidity. Strictly rigid contact is indeterminate under redundant constraints (stacks, boxed-in objects, closed kinematic loops) and produces solver stalls or chatter. The pragmatic answer, embodied in TGS Soft and similar compliant-constraint solvers, is to introduce a **small amount of compliance** — just enough to make the constraint matrix full-rank and let forces redistribute smoothly. Contact Hertz ≈ 30 in a 60 Hz sim is already "almost-rigid" in this sense; nobody runs games with literally infinite stiffness.

Read this way, a rigid TGS Soft solver already contains a trivial material model: a **diagonal compliance** tying each constraint to its own local impulse, with no cross-coupling. That diagonal is the entire "material" through which contact forces communicate and equilibrate inside a single rigid body. The reason it works is precisely that a small regularization breaks indeterminacy without changing the large-scale dynamics.

### What changes in DBD

In DBD, contacts and joints no longer talk to each other through a trivial diagonal. They talk through the **actual body material model** — a nonlinear hypoelastic (or similar) response on the affine state. Two contacts on opposite sides of a squishy block communicate via the block's elastic response, not via a per-constraint compliance term bolted on for regularization.

This is a feature, not a bug:

- It **subsumes** the diagonal compliance trick. The material model is itself a source of well-conditioning; we no longer need to sprinkle epsilon compliance on every constraint to avoid indeterminacy.
- It gives physically meaningful force flow: a stack of DBD blocks equilibrates through the same elastic channel that makes them squash, so the "regularization" and the "deformation" are the same mechanism.
- In the stiff limit, the block's material response approaches a diagonal (plus cross-terms vanishing like `1/stiffness`), so we should smoothly recover rigid TGS Soft behaviour.

But it also **breaks an assumption that goes all the way back to Baraff**: the derivation of a rigid-body time-stepper typically starts by linearizing constraint forces through time derivatives (velocity-level constraints, Jacobians, diagonal regularization). Pseudo-linearity through `d/dt` is the first move. With a nonlinear material model carrying the coupling, that first move is no longer free — the "constraint compliance" is now a nonlinear operator depending on the current affine state, not a fixed (or diagonal) matrix.

### The high-frequency problem

"Arbitrary material parameters" implies arbitrarily high natural frequencies in the simulation. A stiff DBD body under our substep `h` can have material modes with `ω h ≫ 1`. A naive explicit integrator of those modes blows up immediately; a naive implicit one damps them out entirely. Neither is acceptable:

- **Explicit / symplectic Euler / RK**: unstable beyond the CFL-like limit set by the stiffest mode. Since we want the solver to remain stable as stiffness → ∞ at fixed `h`, explicit integration of material forces is a non-starter.
- **Implicit / backward Euler / projected Newton with line search**: unconditionally stable, but aggressively dissipative. It will kill the very squash-and-stretch behaviour the system exists to produce. Crucially, it also breaks symplecticity: the rest of the rigid-body machinery (semi-implicit Euler + constraint projection) is approximately symplectic and conserves energy on long timescales, which is why stacks don't drift and orbits don't spiral. A dissipative material integrator bolted onto a symplectic constraint solver will leak energy out of deformation modes preferentially, producing a "soft bodies feel dead, rigid bodies feel alive" pathology.

We would like the material integrator to be **both symplectic (or close to it) and stable up to the stiff limit**. These two properties are famously in tension: A-stable linear multistep methods cannot be higher than second order (Dahlquist), and symplectic implicit methods (implicit midpoint, Gauss–Legendre) are only conditionally stable for nonlinear stiff problems. The candidates we have in mind to investigate:

- **Variational / discrete-Lagrangian integrators.** Symplectic by construction, and the implicit variants (e.g. variational backward Euler, SIPIC-style) can be made unconditionally stable for convex potentials. The compliant-constraint form of TGS Soft is already close to a variational integrator in disguise, which is encouraging.
- **Exponential integrators / IMEX splits.** Treat the stiff linear part of the material model (the Hessian at the current `A`) implicitly and exactly, and the nonlinear remainder explicitly. For a per-body `2×2` (2D) or `3×3` (3D) affine block, the implicit step is cheap — it's a tiny local solve, not a global one.
- **Position-based / XPBD-style projection on the material constraint.** This is how the rest of solver2d handles compliance, and it is known to be stable at arbitrary stiffness because it is a projection, not a force integration. The open question is whether a nonlinear material energy can be projected cleanly in the same per-substep inner loop that currently handles scalar-valued contact constraints, and whether doing so remains symplectic in the relevant sense.
- **Local implicit, global explicit.** Because affine state is *per body* (not per vertex), the stiff coupling is *intra-body*. A per-body implicit solve of size 4–6 (2D) or 9–12 (3D) is essentially free and can be done inside the substep loop without any global linear system. This is probably the single biggest structural advantage DBD has over FEM/XPBD for handling stiff materials.

The last point is the one we are most optimistic about: **DBD's stiffness is local**, so the stiff solve is local, so we can afford a proper implicit treatment per body per substep without the usual global-Newton cost. That makes the symplectic-*and*-stable goal more reachable than it would be in a mesh-based setting.

This is, however, still an open design question — arguably **the** central technical risk of the project. It does not block early stages of the plan (carrying affine state, free flight, basic contacts) but must be resolved before we can claim DBD meets H1–H4 across the full material range.

## Restitution as a material property

A third theoretical payoff — after "no DoF explosion" and "coupling through a real material model" — is a **principled model of restitution**.

### Why restitution is awkward in standard solvers

In a conventional rigid-body solver, restitution is not really part of the dynamics; it's a post-hoc adjustment. The velocity-level contact constraint gets a bounce term `-e * v_n^-` added to its target, where `e` is a per-material coefficient and `v_n^-` is the pre-impact normal velocity. This works for isolated collisions and breaks in a predictable set of cases:

- **Chained impacts in one step.** The pre-impact velocity is a snapshot of one body, but the bounce must propagate through a chain of bodies — by the time the Nth body in the chain is processed, `v_n^-` for it has already been contaminated by the impulses of earlier pairs.
- **Simultaneous resting and impacting contacts.** A bouncing ball sitting on a table: the resting contact wants `e = 0`, the impact wants `e > 0`, and the solver has to decide per-contact which it is, usually via a velocity threshold that is a tuning parameter, not a physical quantity.
- **Restitution + friction coupling.** Bounce and slip are solved in separate passes, so energy leaks in ways that don't correspond to any material.
- **Restitution + stacking.** Turning `e > 0` on in a stack of boxes produces jitter, because the solver re-injects energy into resting contacts that should be dead.

Standard engines paper over these with special-case logic: velocity thresholds, restitution "slop," separating rest from impact by heuristic, disabling bounce when penetration is large. The result is a parameter that is nominally physical (coefficient of restitution) but behaves like a knob the user tunes until it looks right. **Newton's Cradle is the showpiece failure case**: a row of touching balls where an impact on one end must travel through `N−2` resting contacts and eject the far ball with nearly all of the incoming energy, in a single step, without disturbing the middle balls. Many real-time solvers simply cannot reproduce this — the restitution gets dissipated or distributed incorrectly because the "chain" is not a single pairwise impact the restitution model can see.

### Why DBD should get restitution for free

In DBD, restitution is not a coefficient — it is an **emergent consequence of the material model on the affine state**. The spectrum of real materials from "perfectly elastic" to "perfectly plastic" is already parameterized by the ratio of elastic to plastic response on `A`:

- **Fully elastic (`e = 1`).** A stiff elastic material with no yield. The affine state compresses on impact, stores energy in the elastic potential, and releases it back into velocity as it relaxes. No bounce term is needed; the bounce is kinetic → elastic → kinetic energy flow through the body's own deformation.
- **Fully plastic (`e = 0`).** A rigid-plastic material with a low yield threshold. Impact deforms `A_p` permanently, dissipating the kinetic energy into plastic work. No "restitution = 0" special case; the plasticity simply eats the energy.
- **Partial restitution.** Any elasto-plastic mix in between, with the fraction of energy returned determined by how much of the impact's elastic strain lies below versus above the yield surface. This is how real materials actually behave — a steel ball on a steel plate bounces more than a lead ball on the same plate because steel's yield is higher, not because somebody typed `e = 0.8`.

Critically, this is **the same material model that handles static contact, stacking, and joint coupling.** There is no separate "restitution pass" and no threshold deciding whether a contact is resting or impacting. A resting contact and an impacting contact are both just the body's material responding to whatever compression the constraint solver imposes — one at low rate, one at high rate. The solver does not need to know the difference.

### Newton's Cradle as a benchmark

We should take the Cradle seriously as a benchmark. For DBD, it's not a special case — it's a clean stress test of whether the **intra-body force propagation** works correctly through the substep loop.

What has to happen, physically, in one cradle "click":

1. Ball 1 strikes ball 2. Ball 2's affine state compresses on the impact side.
2. Ball 2's material response pushes back — on the impact side *and* on the far side where it touches ball 3. The far-side push is the internal elastic wave arriving at the far contact.
3. Ball 3 receives that impulse through its contact with ball 2, its own affine state compresses, its material response pushes ball 4, and so on down the chain.
4. At the far end, ball N has nothing to push against except empty space, so its material response accelerates the whole ball outward instead — and that's the "eject" at the end of the cradle.

Every step of that is just DBD doing its normal thing: contact constraints coupling to the affine state, the material model redistributing the compression, the substep loop letting the response propagate. The cradle is "solved" if our substep count and our intra-body implicit solve (see the material-range section) are fast enough that the elastic wave can cross all `N` balls inside one frame. Given that propagation through `N` balls needs `O(N)` substeps of information flow (each substep moves one contact worth of coupling), we need enough substeps per frame to cover the longest expected chain — or, better, an intra-substep ordering / Gauss-Seidel sweep across contacts that lets information flow further per substep.

Importantly, **none of this is special-cased**. If DBD can do Newton's Cradle, it is because the same mechanism that makes a single ball bounce also makes a chain of balls transmit. And if we can handle the Cradle, we almost certainly handle easier restitution scenarios (bouncing ball, rolling ball, ball-on-stack) without any of the usual heuristics.

### Requirements this places on the rest of the design

- **Stiff materials must be stable at full stiffness.** A realistic elastic bounce needs a high natural frequency; the integrator-stability question from the previous section is directly blocking this use case.
- **Energy conservation at the substep level.** A dissipative material integrator will silently eat cradle energy. This is the other reason we care about symplectic-or-near-symplectic integration: not just aesthetics, but benchmark correctness.
- **Contact ordering / information flow across a chain.** We need the substep loop to propagate a contact impulse through multiple bodies fast enough. This may influence how we sweep contacts inside a substep (pure parallel Jacobi won't work for chains; Gauss-Seidel order, or multiple inner sweeps, will).
- **Gap between touching-but-not-colliding and resting contact must vanish.** Because there is no separate restitution threshold, a "resting" contact in a cradle must be indistinguishable from a "barely touching, about to transmit a wave" contact. The solver should already treat them uniformly; this is a check, not a new feature.

### Added to success criteria

Appending to the "What counts as success" list earlier in this doc:

8. **Newton's Cradle.** A row of `N ≥ 5` stiff-elastic DBD balls in contact, one swung in, must eject one ball from the far end with most of the incoming energy, leaving the middle balls approximately at rest, and must sustain this over many cycles without drift or collapse. This must work with the same material and solver settings used for generic bouncing and stacking tests — no cradle-specific tuning.

## Collapse and inversion

Allowing `A` to vary freely opens a failure mode that strict rigidity forbids: **degenerate or inverted affine state**. `det(A) → 0` is collapse (the body squashes to a line or a point); `det(A) < 0` is inversion (the body has been reflected through itself). Nonlinear FEM hits exactly this problem on tets and handles it at the cost of significant complexity (invertible FEM, rotation-aware constitutive models, barrier-based contact). We need a story for it in DBD too, and per our "robustness first" principle we must both **prevent it where we can** and **recover gracefully when we can't**.

### Why this is especially dangerous in DBD

In a mesh-based deformable solver, an inverted element is a local problem — it affects visual quality, stability, and element forces, but collision detection still runs on the mesh surface and remains well-defined (if slow). In DBD, the affine transform **is** the thing collision detection queries through. A support function `s_world(d) = A s_local(A^T d) + t` assumes `A` is a well-behaved linear map. With `det(A) = 0` the support function collapses to a lower-dimensional set; with `det(A) < 0` the orientation flips and GJK/SAT can silently start returning "collisions" on the wrong side of the shape. Either case is a loaded gun pointed at the broadphase and narrowphase. **Collapse/inversion is not just a quality issue; it is a correctness issue for the collision pipeline.**

This means the solver cannot be lazy about it. We need the integrator, the material model, and the collision pre-pass to collectively guarantee that `A` presented to collision detection is always non-degenerate and orientation-preserving.

### When we expect it to actually happen

Worth being honest about the threat model. In normal gameplay with high stiffness, the elastic restoring force pulling `A` toward `R(A)` is strong, the timestep is small, and the typical deformation is a few percent. Runaway compression is rare. The realistic failure modes are:

- **Sandwiching between static colliders.** A body wedged into a gap narrower than its rest shape, driven by gameplay logic (a door closing, a crusher, a moving platform). Static geometry cannot yield, so the body's only options are collapse, tunnel, or pop out.
- **Extreme mass-ratio stacks.** Heavy objects on top of a soft one, where the elastic restoring force is simply outmatched.
- **Adversarial user input.** Mouse-drag into a wall, teleporting bodies into overlap, scripted forces.
- **Numerical slop at very high stiffness.** A stiff body under a stiff constraint can, in principle, oscillate through collapse if the integrator mishandles a high-frequency mode (connecting directly to the stability question from the material-range section).

The first case is the dominant one in practice, and it's usually a **gameplay bug** expressed as a physics bug — but that doesn't let us off the hook, because "the engine exploded when the level designer made a mistake" is exactly the kind of non-robustness we're trying to avoid. The engine's job is to fail safely.

### Defense in depth

No single mechanism will cover all cases. We need layered safeguards.

**Layer 1 — Material model chooses the right energy.** The elastic potential on `A` should diverge as `det(A) → 0^+`. Classic choices: Neo-Hookean (`∝ (log J)^2` or `∝ (J − 1)^2 − log J`), or any energy with a `log det` term. These give an ever-growing restoring force as the body approaches collapse, so a well-conditioned solver simply cannot cross the degenerate surface in finite energy. This is the cheapest and most principled line of defense — it costs nothing at runtime beyond picking the right formula.

**Layer 2 — Singular-value clamp as a projection.** Before handing `A` to collision detection (and possibly after every substep), run an SVD `A = U Σ V^T`, clamp `Σ` entrywise to `[σ_min, σ_max]` with `σ_min > 0`, and reassemble. This is exactly the "invertible FEM" trick. It throws away whatever elastic energy was stored in the clamped modes — we accept that loss as the cost of robustness. In 2D the SVD of a `2×2` is closed-form and cheap, so this is basically free per body per substep.

**Layer 3 — Orientation guard.** If after a substep `det(A) < 0` (inversion), fix it explicitly: either reflect one singular value's sign (`V → V diag(1, −1)` and flip `σ_2`) to restore `det > 0`, or snap `A` back to its polar-decomposition rotation `R(A)` with current singular values' magnitudes. Either way, we guarantee the post-step `A` is orientation-preserving before anything downstream looks at it.

**Layer 4 — Collision pre-filter.** Broadphase AABBs and support queries should be computed on a **sanitized copy** of `A` regardless of what the solver state holds internally. Even if an internal substep briefly walks through a bad state, collision sees only the projected version. This decouples solver bugs from pipeline crashes.

**Layer 5 — Substep CFL on deformation rate.** As a last line of defense, cap the per-substep change in singular values. If a single substep tries to compress a singular value by more than (say) 50%, the substep is too coarse for the forces present and we should subdivide, damp, or clamp. This is analogous to a CFL condition and is cheap to check.

The order matters. Layers 1–2 are always on. Layer 3 is a cheap correctness check. Layer 4 is the firewall that lets the collision pipeline trust its input. Layer 5 is diagnostic — if it triggers often, something upstream is wrong.

### Graceful degradation, not error-free operation

It is worth being explicit: **we are not promising that DBD bodies are inviolable.** A body crushed between two static walls narrower than any `σ_min` we choose is not physically resolvable — no continuous material can do it. Our promise is that in this case the body **remains a valid simulation object** (non-degenerate `A`, well-defined support function, bounded forces, no NaNs, no tunneling) even though it may visibly clip into the geometry or pop out when the squeeze releases. The engine must never crash, explode, or poison neighboring bodies. Those are the contract; perfect resolution of unphysical scenarios is not.

### Open questions in this corner

- **Choice of `σ_min`.** Too large and bodies can't legitimately squash very far; too small and collision support functions get ill-conditioned. Probably some fraction of the rest-shape inradius, tuned once and fixed.
- **Energy accounting on clamp.** When layer 2 throws away energy, where does it go in the material bookkeeping? Silently discarded? Counted as plastic work? The answer may matter for long-term drift on borderline cases.
- **Volume-preserving materials in the clamp.** If the constitutive model wants near-incompressibility (`det(A) ≈ 1`), clamping singular values independently can push against the volumetric constraint. We may need a clamp that respects `det`.
- **Interaction with plasticity.** Does a clamp event update `A_p`? Otherwise a body that gets crushed and released will snap back to a rest shape it never experienced the energy to earn.
- **Interaction with the intra-body implicit solve.** The per-body implicit material step proposed in the material-range section should ideally already enforce non-degeneracy through the energy's `log det` term — layer 2 becomes a safety net for when the implicit solver bails out (line search failure, max iterations).


## Downstream applications

Several payoffs of carrying a real material model per body — fracture/destruction driven by stress state, hallucinated high-frequency vibration modes for audio and visual detail, and plastic state as a render/FX signal — have been split out into `docs/followups.md`. Those threads are out of scope for early project stages. The core-sim implication is that we must expose material state (`A_e`, `A_p`, `C`, accumulated plastic work, per-contact impact events) as a clean read-only API so the follow-on systems can be added later without re-plumbing.
## Running log

Theoretical notes, decisions, and surprises. Date each entry. Implementation progress lives in `docs/plans/plan.md`.

### 2026-04-14 — split out follow-ups
Moved fracture/destruction, hallucinated vibration modes, and plastic-state-as-FX into `docs/followups.md`. These are downstream applications of DBD's material state, not core sim research. Research.md now focuses on: material model, near-rigid limit, integrator stability, collision/restitution, and collapse/inversion. Left behind a short "Downstream applications" pointer section with the requirement that the core sim exposes the state the follow-ups will need (`A_e`, `A_p`, `C`, plastic work, contact impact events) as a clean read-only API so we don't re-plumb later.

### 2026-04-14 — research doc created
Framing written up. Hypotheses H1–H4 and open footnotes recorded. Implementation staging split out into `docs/plans/plan.md`.

### 2026-04-14 — affine state as a gameplay signal
Added the "Affine state as a gameplay signal" section. Two distinct opportunities: (1) **hallucinated vibration modes** from affine velocity — precompute a modal basis per body from a detailed tet mesh of the graphical shape, identify the affine subspace as the simulated modes, cascade energy into unsimulated high-frequency modes via a precomputed spectrum model. Direct analogy to wavelet turbulence (Kim et al. 2008): cheap coarse sim + procedural high-frequency detail driven by a physically-justified spectrum. Outputs are procedural impact audio (builds on James/van den Doel/O'Brien modal sound lineage) and procedural visual wobble detail, both `O(#modes)` per body at runtime with zero marginal simulation cost. (2) **Plastic state as render/FX signal** — accumulated plastic strain drives damage textures, work-hardening drives roughness/gloss, plastic dissipation drives heat emissive, distance-to-failure drives warning tells, durability bars become literal measurements of `1 − accumulated plastic work / fracture threshold`. Theme: DBD's material state is a shared source of truth for rendering/audio/UI/gameplay, replacing the proxy-fictions each layer invents today. Both threads queued as follow-on work; the physics side (carrying the right state) is already in scope from day one.

### 2026-04-14 — fracture from material state
Added the "Fracture and destruction from material state" section. Core points: (1) standard RBD fracture is blind because a rigid body has no internal stress representation, so engines use impulse-threshold heuristics; (2) DBD's affine state is a deformation gradient up to a constant, so real stress tensors are available and brittle/ductile/fatigue distinctions fall out of the material model, along with work hardening and the Bauschinger effect via standard tensor-valued backstress; (3) gameplay upside: the player can read physical history off visible body shape (dents, bends, fatigue). Flagged open problem: **cascading destruction inside a single timestep** — needs topology mutation during the substep loop, warm-start invalidation, re-fracture handling, deterministic ordering, joint remap, and a per-frame budget. Parked as out-of-scope for early stages but we commit to carrying the *information* (stress, `A_p`, damage) from day one so fracture can be added without re-plumbing. Literature review queued: O'Brien & Hodgins, Houdini FEM destruction, NVIDIA Blast/Apex, peridynamics.

### 2026-04-14 — collapse and inversion
Added the "Collapse and inversion" section. Core points: (1) unlike mesh FEM where inversion is a local quality problem, in DBD a bad `A` corrupts the collision pipeline itself (support functions, GJK orientation), so this is a correctness issue not a visual one; (2) realistic threat model is sandwiching by static geometry from gameplay bugs, not normal dynamics; (3) defense in depth — `log det`-containing energy (Neo-Hookean-ish), closed-form `2×2` SVD clamp of singular values, orientation guard on `det(A) < 0`, sanitized-copy firewall between solver state and collision queries, and a deformation-rate CFL as diagnostic; (4) explicit contract: we promise the body remains a valid simulation object under unresolvable squeezes (no crash, no NaN, no tunneling) but not perfect physical fidelity in those cases.

### 2026-04-14 — restitution as material property
Added the "Restitution as a material property" section. Core points: (1) standard solvers treat restitution as a post-hoc velocity adjustment `-e*v_n^-`, which fails on chained impacts, rest/impact ambiguity, and stacking; (2) in DBD, restitution is emergent from the elastic/plastic mix on the affine state — fully elastic = stiff + no yield, fully plastic = low yield, partial = yield surface placement; (3) Newton's Cradle is the showpiece benchmark we commit to — it must work with *no* cradle-specific tuning, which places concrete requirements on integrator stability at high stiffness, near-symplectic energy behaviour, and contact ordering / info flow across chains within the substep loop. Added as success criterion #8.

### 2026-04-14 — material range + integrator question
Added the "material range and the near-rigid limit" section. Core points: (1) rigid TGS Soft already uses a trivial diagonal compliance as a minimum regularizer, and DBD's nonlinear material model subsumes that role; (2) nonlinear material coupling breaks the Baraff-style "linearize through d/dt" starting move for time stepping; (3) arbitrary stiffness implies arbitrary frequencies, so the material integrator must be stable to the stiff limit *and* ideally symplectic to match the rest of the rigid-body machinery. Most optimistic route: DBD stiffness is local to a body (4–6 DoFs in 2D), so a proper per-body implicit solve inside the substep loop is cheap and may give us symplectic-and-stable without a global Newton. This is flagged as the central open technical risk.

### 2026-04-14 — naming + MPM framing
Settled on "Deformable Body Dynamics (DBD)" as the working name (ABD is taken). Added relationship-to-prior-work section positioning DBD against ABD and MPM. Key theoretical move: a DBD body is "one MPM particle scaled up, stripped of its grid reconstruction filter, coupled to others through explicit convex contact constraints instead of through spatial kernels." The APIC affine velocity `C`, the `F = F_e F_p` plasticity split, and hypoelastic updates are the concrete tools we expect to borrow. Position-based MPM literature already shows the material models transpose into compliant-constraint form.
