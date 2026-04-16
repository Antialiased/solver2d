# Follow-on Work: Applications of DBD State

Companion to `research.md`. That doc is the core simulation research (material model, integrator, collision, restitution, collapse handling). This doc collects **downstream applications** of DBD's per-body material state — things that depend on the core sim working but are not themselves part of the core sim.

**Other companions:** `code_map.md` (file-by-file index of the codebase), `insights.md` (running catalogue of experimental findings), `plans/` (active implementation and experiment plans).

Everything here is out of scope for early project stages. The purpose of this doc is to (a) keep the ideas from rotting, and (b) make sure the core sim carries the information these applications will eventually need, so we don't have to re-plumb later.

## Fracture and destruction from material state

A further payoff of carrying a real material model per body: **destruction decisions can be driven by the physical state of the body, not by heuristics on incident contacts.** This upgrades fracture from a scripted event into a physically-informed gameplay mechanic.

### The problem with fracture in standard rigid body solvers

In a conventional RBD pipeline, the body has no internal state to consult. Everything it "knows" about its stress is a list of incoming contact impulses this frame. Games build fracture on top of this with some variant of:

- **Impulse magnitude threshold.** "If any contact impulse exceeds `K`, shatter." Crude, ignores contact location, ignores geometry, ignores material direction.
- **Accumulated damage counter.** "Sum per-frame impulse into a damage scalar, shatter at `D_max`." Better, but still a 1D summary of a tensor-valued quantity and still blind to distribution.
- **Scripted / tag-based.** "This crate is destructible, this one isn't; breakpoints are authored." Physics has left the building entirely.

The fundamental limitation is that **a rigid body has no notion of internal stress**, because its DoFs cannot represent one. Any "stress" the engine computes is reconstructed from the outside, from contacts, as a best-guess lower bound on what the body is actually experiencing. Two identical contacts on opposite sides of a body — a clean compression — look to the engine the same as two contacts on the same side — a bending moment with very different failure implications.

### What DBD buys us

A DBD body's affine state `A` (and its elastic/plastic decomposition `A = A_e A_p`) is, up to a constant, a **deformation gradient**. From a deformation gradient and a material model we get a **stress tensor** directly — Cauchy, Piola–Kirchhoff, whatever the constitutive model produces. That stress is the same quantity a materials scientist would use to predict failure, and it responds correctly to:

- **Contact distribution.** Bending vs. compression vs. shear show up as different components of the stress tensor, not as the same scalar impulse total.
- **Material direction.** If we want anisotropic materials (grain, fiber), the stress picks it up naturally.
- **Rate.** Strain rate is `dA/dt`, available from the affine velocity. Rate-dependent failure (brittle at high rate, ductile at low rate) falls out.
- **History.** The plastic state `A_p` encodes accumulated permanent deformation, which is exactly the channel real materials use to remember their loading history.

This gives us, essentially for free, a physically-motivated basis for distinguishing failure modes:

- **Brittle fracture** = elastic stress exceeds a threshold (Rankine / max-principal-stress, or energy-release-rate criterion) while `A_e` is still close to identity. The body shatters without significant plastic deformation. Glass, ceramic, frozen materials.
- **Ductile fracture** = failure after significant accumulated plastic work. The body has been bent, yielded, bent further, yielded further, and eventually tears. Metal, clay, hot materials. The threshold is on accumulated plastic strain, not instantaneous stress.
- **Fatigue / repeated-loading failure** = failure driven by cyclic changes in `A_p`, tracked as a damage variable per body. "Break after being bent back and forth five times," without any per-frame scripting.
- **Work hardening.** As plasticity accumulates, yield stress rises. The body gets harder to deform further, until it reaches a brittle regime and shatters. This is a one-line modification to the yield surface on top of the plastic update.
- **Bauschinger effect.** Yield in one direction reduces yield in the opposite direction. Also a local modification to the yield surface, parameterized by a tensor-valued backstress stored alongside `A_p`.

None of these require new solver machinery — they are small additions to the material update. The reason they're hard in conventional RBD is not that the models are complicated, it's that **the solver has nowhere to put them**.

### Gameplay modalities this unlocks

Stepping back from the physics: because destruction is now driven by a continuous, physically-meaningful state, gameplay can read that state for purposes beyond "is it broken yet."

- A crowbar that deforms permanently after enough prying, degrading as a tool over its use.
- A metal gate you can pry open by repeated loading until it yields plastically and stays bent.
- A glass sheet that cracks where stress concentrates, not where an authored hit-volume says.
- A rope bridge whose beams sag permanently under load, warning the player before they fail.
- Enemies whose armor dents before it breaks, visibly tracking the damage they've taken.

The common thread: the player can **read the world's physical history** off the body's visible shape, because the visible shape is a faithful display of the material state. Classical RBD can't do this; everything is either whole or broken.

### Open problem: cascading destruction inside a single timestep

For destruction to feel good, a single initial failure often has to cause a chain of further failures **within the same frame**: a pillar cracks, drops its load onto a floor, the floor yields and collapses, the debris triggers more fractures as it falls. If destruction is resolved once per frame — "after the solver finishes, check each body's state and decide whether to break it" — this cascade plays out over many frames, which looks terrible (the pillar breaks, then a frame later the floor breaks, then a frame later...). Standard destruction systems paper over this with slow-motion effects, pre-baked animations, or scripted sequences.

In DBD, the solver itself already has the right structure: the substep loop is continuously updating material state, stresses are continuously re-evaluated, and cascades can in principle happen at substep granularity. But turning "break this body" into a solver operation that is safe to perform **inside the substep loop** is non-trivial:

- **Topology changes mid-solve.** Fracture typically means creating new bodies (or shapes) and removing old ones. The solver's contact graph, constraint arrays, and stack allocations are built per-step. Mutating them mid-step without invalidating iterators or warm-starts is a real engineering problem.
- **Warm-start validity after fracture.** An impulse warm-started from the previous step's contact is meaningless for a body that just split. We need to invalidate or remap warm-start state per affected body.
- **Re-fracture in the same substep.** A fragment created by a fracture may itself immediately exceed its own stress threshold (e.g., it's still being crushed by the thing that broke the parent). We need the solver to be able to re-fracture it later in the same substep without stack overflow or infinite loops.
- **Determinism and ordering.** Which fracture "goes first" matters when two can happen simultaneously. We need a principled ordering — probably by stress magnitude relative to threshold — and we need the ordering to be stable under trivial reordering of bodies.
- **Constraint / joint reassignment.** A joint attached to the parent body must be reattached to the correct fragment. For contacts this is implicit (rebuilt next substep); for persistent joints it needs explicit handling.
- **Budget.** A single frame's worth of cascading destruction cannot be allowed to eat unbounded CPU. We probably need a per-frame fracture budget, overflowing work into "queued" fractures that resolve next frame.

Relevant prior art to digest before designing this:

- FEM destruction literature (O'Brien & Hodgins 1999 onward) for the stress-based fracture criteria and crack propagation, though their topology-mutation machinery is heavier than we want.
- Destruction in offline film-grade simulators (Houdini RBD + FEM) for the authoring story — how they let artists tune material parameters to shape destruction behaviour.
- Real-time destruction systems like NVIDIA Blast / Apex Destruction for the engineering side of topology changes in a real-time loop, and the tricks they use for budget and warm-start handling.
- Peridynamics for a constitutive framework that treats fracture as native (bonds break, no mesh re-meshing) — may offer ideas applicable to our "body splits into bodies" level even though we're not running peridynamics itself.

### What the core sim must carry for this to be possible later

- Per-body elastic and plastic affine states (`A_e`, `A_p`).
- Per-body accumulated plastic work / damage scalar.
- A way to query current stress tensor from material state (a function, not a stored field).
- Hooks in the solver to notify an external system of large stress events per substep — we'll want these anyway for audio.

## Hallucinated vibration modes from affine velocity

The affine velocity tensor (the APIC-style `C` matrix conjugate to `A`) contains, in its spectral decomposition, **the body's current low-frequency vibrational state projected onto the affine basis**. A symmetric rate means uniform compression/expansion; an anti-symmetric rate means rotation rate; off-diagonals mean shear rates; signed singular values tell us how fast the body is stretching along each principal direction. This is 3 numbers in 2D (and 6 in 3D) per body — tiny, but carrying the low-wavenumber end of a much richer vibrational spectrum.

The useful observation: *we don't have to actually simulate the rest of that spectrum*. We can **hallucinate it** from the simulated modes plus a precomputed model of how the body *would* respond, exactly analogous to how **wavelet turbulence** (Kim et al., SIGGRAPH 2008) hallucinates sub-grid smoke detail from the resolved velocity field using a Kolmogorov-spectrum model. In their setting, the expensive resolved simulation gives the low-wavenumber motion and a cheap procedural process rides a turbulence spectrum into the high wavenumbers. In our setting:

1. **Precompute a modal basis per mesh.** For each physics body, build a modal basis *offline* from a detailed representation — probably a tetrahedral meshing of the graphical (not collision) shape — by solving the generalized eigenproblem of the elastic Hessian around rest pose. This gives us a spectrum of vibration modes `{φ_k, ω_k}` for the body, sorted by frequency. Standard linear modal analysis; well-established in the computer animation literature (James & Pai, O'Brien et al. on modal sound).
2. **Identify the "affine subspace" in that basis.** The affine DoFs the physics is simulating — translation, rotation, scale, shear — correspond to the lowest-frequency rigid and affine modes. That's 6 modes in 2D (2 trans + 1 rot + 3 affine) out of the full spectrum. The physics simulates these directly, with the real material response and the real contact forces.
3. **Cascade simulated mode energy into unsimulated modes.** The unresolved high-frequency modes are driven procedurally from the simulated affine velocity, using a **precomputed coupling / spectrum model**. This is the direct analogue of wavelet turbulence's Kolmogorov cascade: we have a physically-motivated spectrum (material-dependent, falling off with mode frequency), and we pump amplitude into high-frequency modes as a function of low-frequency energy and the body's impact/stress events. The coupling model is precomputed per body offline.
4. **Output the hallucinated modes for rendering and audio.** Summing high-frequency mode amplitudes `× φ_k` gives a hallucinated vibration displacement field across the body's surface, usable for visual wobble detail. Summing `φ_k`-evaluated-at-surface-points times `ω_k` gives a pressure signal, usable for impact/contact sound.

The efficiency win is the same as wavelet turbulence's: we pay nothing at runtime for the unsimulated modes beyond updating a handful of scalar amplitudes per mode per body, using a precomputed coupling. The *physics* stays cheap (6 DoFs, convex collision) while the *sensory experience* carries the detail of a high-resolution modal model.

Two distinct use cases fall out:

- **Procedural impact/contact audio.** Modal sound has a long history in computer animation — O'Brien & Hodgins 1998, van den Doel & Pai 2003, James et al. 2006 on precomputed acoustic transfer. Our input is cleaner than theirs because we already have a physically-correct low-frequency affine velocity coming out of the solver, not a contact-impulse proxy. Each contact event excites the modal cascade; the audio is the radiated pressure summed across surface mode amplitudes. Because the basis is precomputed and mode integration is linear, per-body audio cost is `O(#active modes)` per sample.
- **Procedural visual vibration detail.** The same hallucinated mode amplitudes, applied as vertex displacements on the graphical mesh, give visible high-frequency wobble after impacts, compression ringing, stress-driven shimmer. Free, because it rides the audio machinery.

The connection to wavelet turbulence is not just rhetorical — it's the right mental model. In both cases, a cheap coarse simulation produces the physically-meaningful energy input to a procedurally-synthesized high-frequency detail field, driven by a spectrum model the underlying physics justifies. We pay the expensive simulation cost only where the physics is actually needed for *interaction*, and we hallucinate detail everywhere the result is only observed.

### Open points

- Where exactly does the "simulated → hallucinated" cutoff sit? The affine subspace is only 6 modes. The next unsimulated modes (low-order bending, higher-order stretch) may still be physically important for some gameplay — so "just 6 modes" may be too few for, say, a long beam, and we may need a small per-body extended modal state beyond the affine DoFs. Possible middle ground: simulate the affine subspace with full nonlinear material, and layer a handful of linear modal extensions on top, integrated cheaply in generalized coordinates.
- Precomputation cost: modal analysis of a detailed tet mesh is offline but nontrivial. Needs a tool in the pipeline.
- Interaction with large deformation: linear modal bases are anchored at rest pose, so they degrade when the body is significantly deformed. Either warp the basis by `A` (cheap, first-order correct) or use rotation-invariant modal bases (rigid-mode factored out).
- Plastic state shifts the rest pose. When `A_p` changes, the basis technically should too. Probably acceptable to interpolate or re-center on plasticity update events.

### What the core sim must carry for this to be possible later

- Affine velocity `C` exposed as a read-only per-body value.
- Per-contact impact events exposed as a stream (magnitude, location, time within substep).
- Optional per-body material ID for offline basis lookup.

## Plastic state as a render/FX signal

The same deformation bookkeeping supports a different and simpler gameplay hook: **use the plastic state to drive graphical and UI effects that communicate material history to the player.** This is the visible analogue of the "player reads world history off body shape" idea from the fracture section, but it doesn't require fracture to be useful.

Concrete examples:

- **Damage textures fading in with accumulated plastic strain.** A metal plate that has been beaten on tracks a scalar damage variable (e.g. `tr(A_p − I)^2` integrated over time, or accumulated plastic work). The rendering system reads that scalar and blends a dents/scratches texture layer proportionally. No scripted break points; damage visibly accumulates in the exact regions the physics was stressed.
- **Surface roughness / glossiness driven by work hardening.** Real metals become less glossy and more matte as they work-harden, because the microscale deformation roughens the surface. We can mimic this directly: the work-hardening variable the material already tracks (backstress magnitude, or accumulated plastic strain) drives a `roughness` / `metallic` PBR parameter. A metal joint that has been bent many times looks visibly dulled before it fails.
- **Heat / glow from plastic dissipation.** Plastic work is dissipated energy, and in real materials it becomes heat. A debug-channel first, possibly a real gameplay signal later: emissive glow on a hot-worked body, cooling over time via a simple exponential decay. Gives the player a readable cue that *this* part of the object is the one about to fail.
- **"Failure imminent" visual tell.** As the damage scalar approaches the fracture threshold, amplify the effect — crack decals starting to show, color shift toward the failed-material look. Gives the player a warning that is still driven by the same physics state the fracture system will eventually read.
- **UI readouts on tools/weapons.** A weapon's durability bar is literally `1 − (accumulated plastic work / fracture threshold)`, not a scripted counter. "Durability" becomes a physical measurement of the object's real state, not an abstract gameplay resource.

These are essentially free from the physics side: the plastic state is already being computed and already being stored per body for the material model. Exposing it to the renderer and the gameplay layer is a matter of adding a read-only accessor. Per body, the quantities the render layer might want are: accumulated plastic strain (scalar), max principal plastic strain (scalar), a local "hot spot" direction if we can spatialize (stretch goal), and a normalized "distance to failure" if a threshold is defined.

### What the core sim must carry for this to be possible later

- `A_p` exposed as a read-only per-body value.
- Scalar accumulated-plastic-work / damage counter per body.
- Optional backstress tensor for Bauschinger-aware hardening.

## Shared theme

Across all three threads: **because the simulation is carrying a real material state, every layer above it (rendering, audio, UI, gameplay logic) can read that state as ground truth instead of inventing proxies.** Classical RBD forces every layer to invent its own fiction — audio guesses impacts from contact events, rendering uses scripted break animations, UI uses abstract HP — and those fictions drift apart and contradict each other under close inspection. DBD offers a single physical state as the shared source of truth.

The design implication for the core sim: **expose material state cleanly as a first-class read-only surface.** We should not bury `A_e`, `A_p`, `C`, plastic work, and contact impulse events inside the solver's internal structs. A thin API around them is cheap to add and saves us from a painful refactor when any of these follow-on threads comes online.
