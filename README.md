# Team Plasma GS: The Sun Also Places Macros

[![Submission](https://img.shields.io/badge/submission-submissions%2Fteam__plasma-blue)](#run-it)
[![Mode](https://img.shields.io/badge/mode-pure%20plasma-orange)](#pure-vs-augmented)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Benchmarks](https://img.shields.io/badge/valid-17%2F17-brightgreen)](#results)

> The Sun gave us light, weather, photosynthesis, seasons, suspiciously cinematic sunsets, life itself and the long-term dream of fusion energy. We asked for one more thing: macro placement.
>
> I'd put macro placement in the top 2 (with life) among the greatest gifts of our star, the Sun.

**Team Plasma GS** is a pure plasma / fusion-physics macro placement solver for the Partcl / HRT Macro Placement Challenge 2026. It treats chip placement as a Grad-Shafranov / MHD equilibrium problem: macros are current-bearing particles, nets induce current topology, congestion behaves like pressure, and placement is a relaxation toward a low-energy plasma state.

This repository is intentionally small. It contains the public README, Apache 2.0 license, and the single locked submission package. Research notes, sweeps, logs, and augmented configs are not included.

The active competition package is locked to:

```text
submissions/team_plasma/
```

**Team Plasma GS** is the algorithm/model name. **`submissions/team_plasma/`** is the folder that should be copied into, run from, and submitted for the challenge.

The augmented TILOS-assisted versions still exist for research, but they are not the main submission.

---

## Reviewer Quick Path

If you only have two minutes, read these sections:

1. [What This Submission Is](#what-this-submission-is)
2. [Why Plasma?](#why-plasma)
3. [Pure vs Augmented](#pure-vs-augmented)
4. [How Plasma Becomes Placement](#how-plasma-becomes-placement)
5. [Results](#results)
6. [Research Lineage](#research-lineage)
7. [Next Steps](#next-steps)

---

## What This Submission Is

| Question | Answer |
|---|---|
| Main submission | Pure plasma Team Plasma GS |
| Entry point | `submissions/team_plasma/placer.py` |
| Active config | `submissions/team_plasma/config.json` |
| License | Apache 2.0 |
| Benchmark hardcoding in active config | None |
| Active TILOS soft optimization | No |
| Active TILOS PLC warm start | No, disabled by purity mode |
| Internal time budget | `1300s` |
| Intended award story | Innovation / methodology |

The short version:

```text
We solve macro placement as plasma equilibrium.
The proxy score is worse than augmented TILOS-assisted placement.
The physics story is much stronger.
```

## Why Plasma?

I came into this challenge with no prior experience in macro placement, physical design, or hardware engineering. My background is in computational plasma physics, fusion, and scientific simulation, so I approached the problem from a completely different direction.

The idea that immediately caught my attention was that macro placement is not just a set of independent positioning decisions. Every component affects many others through connectivity, congestion, density, and geometry. That felt familiar. In plasma physics, every particle, current, pressure gradient, and boundary condition influences the global equilibrium. Local changes can reshape the entire field.

That connection made the challenge feel less like a hardware problem I had never seen before and more like an unfamiliar version of a problem I already cared about: how do complex interacting systems relax toward a constrained low-energy state?

Team Plasma GS came from that analogy. Instead of starting from a classical placement heuristic, I tried to translate plasma ideas directly into placement: Grad-Shafranov equilibrium, current topology, pressure, two-fluid transport, sheath effects, and instability-driven refinements. The result is not the strongest proxy-score placer, but it is a genuine attempt to solve macro placement through the lens of plasma physics.

---

## Run It

This repository is a clean public release of the submission package. To run it inside the official challenge repository, copy or keep `submissions/team_plasma` under that repository's `submissions/` directory and evaluate the placer:

```powershell
uv run evaluate submissions/team_plasma/placer.py -b ibm02
```

For the pure submission:

```text
Do not set TEAM_PLASMA_GS_CONFIG.
```

The default adjacent `config.json` is the clean pure-plasma config. Do not point `TEAM_PLASMA_GS_CONFIG` at an experimental config for the pure competition run.

---

## Submission Readiness

The active source package was audited before submission.

| Check | Status |
|---|---|
| Active pure config | Passed |
| Public release contains only required solver package | Passed |
| JSON config parses | Passed |
| Python modules compile | Passed |
| Apache 2.0 `LICENSE` present | Passed |
| No `ibmXX` hardcoding in active config | Passed |
| No nonempty `*_by_benchmark` / `*_auto_benchmarks` routing in active config | Passed |
| TILOS soft optimizer disabled | Passed |
| Purity assertion crash risk disabled | Passed |
| Internal budget reduced for runtime safety | Passed, `1300s` |

The active config intentionally chooses purity and compliance over the absolute best proxy score.

---

## One-Screen Architecture

```mermaid
flowchart LR
    A[Netlist + Canvas] --> B[R1 Plasma Startup]
    B --> C[Grad-Shafranov Field Solve]
    C --> D[R4 Two-Fluid Transport]
    D --> E[R2 Sheath Legalization]
    E --> F[Instability Refinement]
    F --> G[Exact Proxy Gate]
    G --> H[Legal Placement]

    B:::plasma
    C:::plasma
    D:::plasma
    E:::plasma
    F:::plasma
    G:::gate

    classDef plasma fill:#e9f6ff,stroke:#0b5d7a,color:#111,stroke-width:2px;
    classDef gate fill:#fff3cd,stroke:#9a6a00,color:#111,stroke-width:2px;
```

Plain-text fallback:

```text
Netlist + canvas
  -> R1 plasma startup
  -> Grad-Shafranov equilibrium solve
  -> R4 two-fluid soft-macro transport
  -> R2 Bohm-sheath / continuation legalization
  -> plasma instability refinements
  -> exact proxy gate
  -> legal placement
```

The important design choice is that the active placement-producing stages are plasma stages:

| Stage | Classical TILOS Role | Pure Plasma Replacement |
|---|---|---|
| Initial macro placement | PLC / placement initialization | R1 plasma startup from current topology |
| Global field | Analytical placement forces | Grad-Shafranov equilibrium field |
| Soft macro motion | `optimize_stdcells` force-directed relaxation | R4 two-fluid transport |
| Legalization | strict greedy legalization | R2 sheath / continuation legalization |
| Refinement | local move search | exact-gated plasma instability moves |

---

## Watch It Place (Pure Plasma, ibm03)

![Pure-plasma staged optimization on ibm03](media/staged_ibm03_pure.gif)

Every frame is a real, accepted placement captured live from the running
solver. The animation walks the actual pipeline, with a persistent progress
bar (top) showing which stage is active and the placement and descent graph
held on screen throughout:

- **R1 Plasma Startup** (purple) - graph current topology to flux bands to an initial macro plasma.
- **Grad-Shafranov Equilibrium** (blue) - macros flow under the self-consistent `psi` field (one frame per Picard iteration).
- **R4 Two-Fluid Transport** (teal) - soft macros relax as the fast electron species in the `psi` field.
- **Instability Refinement** (orange) - Mercier / tearing / flux-rope cluster moves, each accepted only through the exact proxy gate.

Every frame is also legalized through the **R2 Bohm-sheath gate**, so the
animation is simultaneous evidence that the placement stays overlap-free. The
banner (top-right) and the curve (bottom-right, dots colored by stage, black
line = best so far) show the placement relaxing from the startup plasma toward
a lower-energy equilibrium.

This GIF is illustrative: the live capture adds per-iteration scoring overhead
and uses a denser Picard schedule, so its exact final proxy can differ
slightly from the clean official-harness numbers in [Results](#results).

### Plasma View of the Same Run

![Plasma physics view: Grad-Shafranov relaxation on ibm03](media/plasma_physics_ibm03.gif)

The placer isn't placing macros with plasma metaphors. It's solving a real
Grad-Shafranov plasma equilibrium where **macros *are* current-carrying
coils**, **nets *are* the toroidal current topology**, and **congestion *is*
pressure**. The equilibrium coil positions the solver converges to *are*
the macro placement; the animation above just renders that same equilibrium
in its native plasma form instead of as a chip layout. Every value is read
live from the placer's tensors at each Picard iteration.

- **Main panel**: `psi(R,Z)` poloidal flux with nested **flux-surface
  contours**, the **magnetic axis** (yellow star), and the
  **Mercier-unstable region** (cyan).
- **Macros**: hard macros as current-bearing coils, sized by `|I_i|`;
  soft macros as a faint electron cloud.
- **Radial profiles**: `p'(psi)` and `F(psi)`, the two RHS terms of the
  Grad-Shafranov equation.
- **GS residual** (log scale): `||Delta* psi - rhs||` descending across
  all 30 Picard iterations.

The plasma view is GS-only; the macro-placement GIF above covers all four
pipeline stages.

---

## How Plasma Becomes Placement

The core analogy is variational: both placement and plasma equilibrium seek a low-energy state under constraints.

```text
Placement objective:

    W_place(x) = WL(x) + 0.5 * Density(x) + 0.5 * Congestion(x)

MHD equilibrium energy:

    W_MHD = integral( B^2 / 2mu0 + p/(gamma - 1) + 0.5 rho v^2 ) dV

Shared algorithmic idea:

    relax toward a low-energy state while preserving topology and legality
```

### Translation Table

| Fusion / Plasma Term | Placement Interpretation | Why It Matters |
|---|---|---|
| Poloidal flux `psi` | Placement potential over the chip | Gives a global field that macros respond to |
| Grad-Shafranov equation | Self-consistent equilibrium PDE | Couples pressure, current, and geometry |
| Plasma pressure `p(psi)` | Density / congestion demand | Hot dense regions push back |
| Toroidal current `F(psi)` | Netlist connectivity current | Connected macros shape the field |
| Internal coils | Hard macros | Current-bearing objects in the plasma |
| Electrons | Soft macros | Fast species; equilibrate quickly |
| Ions | Hard macros | Heavy species; move slowly |
| Debye shielding | Finite-radius repulsion | Prevents local collapse |
| Plasma sheath | Canvas boundary pressure | Keeps macros inside the chamber |
| Mercier stability | Congestion-tail stabilization | Suppresses unstable hot spots |
| Taylor relaxation | Global energy relaxation | Moves toward a lower-energy topology |
| Tearing modes | Coordinated cluster moves | Escapes local basins |
| PIC / P3M | Particle-grid-particle solver | Plasma-native proposal generator |
| Biot-Savart attraction | Pairwise current attraction | Plasma analog of connected-pair attraction |

---

## The Pipeline, Stage by Stage

### R1: Plasma Startup

R1 builds the initial plasma state. Instead of using a classical TILOS PLC warm start, it places macros using graph/current structure and flux-band ideas.

```text
netlist graph -> current topology -> flux bands -> initial macro plasma
```

What it does:

- Converts graph structure into current-bearing topology.
- Places hard macros on flux-like bands.
- Initializes soft macros with plasma density / phase-locking rules.
- Avoids TILOS PLC warm start in pure mode.

### Grad-Shafranov Field Solve

The solver then computes a self-consistent `psi` field. This is the central plasma equilibrium equation used in tokamak modeling.

```text
Delta-star(psi) = -mu0 * R^2 * p'(psi) - F(psi) * F'(psi) - current_sources
```

In placement language:

- `psi` is the global placement potential.
- `p'(psi)` represents density / congestion pressure.
- `F(psi)F'(psi)` represents net-induced current structure.
- hard macros act as internal current sources.

### R4: Two-Fluid Transport

R4 handles soft macros as the fast species in a two-fluid plasma.

```text
hard macros = ions      slow, massive
soft macros = electrons fast, responsive
```

The soft macros equilibrate against the current `psi` field before the next hard-macro update. This gives a plasma-native alternative to classical soft-cell optimization.

### R2: Bohm-Sheath Legalization

R2 removes overlaps using continuation and sheath-style forces.

```text
overlap pressure + wall sheath + continuation beta -> legal hard macros
```

This stage is why the pure submission can return zero-overlap placements without making strict TILOS legalization the load-bearing path.

### Instability Refinement

Real plasmas do not only relax smoothly. They also reorganize through instabilities. Team Plasma GS includes exact-gated versions of:

- Mercier interchange shaping.
- Tearing-mode cluster movement.
- NTM-suppression-inspired targeted refinement.
- Flux-rope cluster moves.
- Snowflake-divertor research moves.

The official proxy gate decides whether a proposed move is kept.

### R5: PIC / P3M Research Module

R5 exists in the codebase as a plasma Particle-In-Cell / P3M proposal module.

```mermaid
flowchart TD
    P[Macro particles] --> Q[Deposit charge/current to grid]
    Q --> R[Solve E/B fields]
    R --> S[Interpolate fields to particles]
    S --> T[Boris / plasma push]
    T --> U[Exact proxy gate]
```

R5 added real plasma-PIC ideas:

- charge deposition,
- field solves,
- Debye-shielded interactions,
- magnetic forces,
- pairwise Biot-Savart current attraction.

It remains disabled in the final active pure config because it was not robust enough across the suite before deadline.

---

## Pure vs Augmented

This distinction is important.

### Pure Plasma Submission

This is what the final active config runs.

```text
submissions/team_plasma/config.json
```

Characteristics:

- plasma startup instead of TILOS PLC,
- two-fluid transport instead of TILOS soft optimization,
- plasma legalization instead of strict legalization as the load-bearing path,
- no benchmark-name routing in active config,
- worse proxy score,
- stronger Innovation claim.

### Augmented / Research Versions

The broader internal research project also has augmented configs. They are intentionally not bundled in this clean public release, because this repository is meant to be unambiguous: it ships the pure plasma competition package.

Those augmented versions are useful for research and proxy hunting. They can be much stronger because TILOS remains load-bearing in certain areas, especially classical soft-cell optimization and refinement. Plasma physics then acts as an additional layer.

That is a practical strategy, but it is not the pure plasma submission.

### Comparison

| Mode | Proxy Strength | Physics Purity | Main Use |
|---|---|---|---|
| Pure plasma | Worse (3.37) | Highest | Innovation submission |
| Augmented TILOS + plasma | Better (1.41) | Medium | Research / leaderboard-style experiments |
| Classical TILOS-style placement | Best baseline behavior | Low for this project | Reference point |

---

## Results

### Full Pure-Plasma Suite Summary

The table below summarizes the best full official-harness pure-plasma suite run from the final internal audit. That run predates the final `time_budget_sec=1300` safety reduction, so very long benchmarks may return slightly earlier in the public submission configuration.

| Summary Metric | Value |
|---|---:|
| Valid benchmarks | `17 / 17` |
| Hard overlaps | `0` |
| Average proxy | `3.3653` |
| Total runtime | `25570.11s` |
| Average runtime | `1504.1s` |

### Per-Benchmark Results

| Benchmark | Proxy | WL | Density | Congestion | Runtime | Valid |
|---|---:|---:|---:|---:|---:|---|
| ibm01 | 2.2431 | 0.218 | 0.823 | 3.226 | 533.38s | yes |
| ibm02 | 2.5615 | 0.228 | 0.752 | 3.915 | 727.11s | yes |
| ibm03 | 2.7403 | 0.239 | 0.860 | 4.142 | 430.31s | yes |
| ibm04 | 2.6192 | 0.214 | 0.863 | 3.947 | 583.91s | yes |
| ibm06 | 3.3843 | 0.228 | 0.785 | 5.527 | 1517.98s | yes |
| ibm07 | 3.0691 | 0.205 | 0.892 | 4.836 | 767.90s | yes |
| ibm08 | 3.3196 | 0.221 | 0.859 | 5.337 | 1421.88s | yes |
| ibm09 | 2.6965 | 0.199 | 0.882 | 4.113 | 1378.98s | yes |
| ibm10 | 2.9904 | 0.201 | 0.763 | 4.815 | 1821.44s | yes |
| ibm11 | 2.9010 | 0.207 | 0.905 | 4.482 | 1383.37s | yes |
| ibm12 | 3.9146 | 0.194 | 0.859 | 6.583 | 2159.33s | yes |
| ibm13 | 3.0825 | 0.200 | 0.961 | 4.803 | 1601.90s | yes |
| ibm14 | 4.5315 | 0.185 | 0.939 | 7.753 | 2445.39s | yes |
| ibm15 | 4.1704 | 0.219 | 1.027 | 6.876 | 1552.03s | yes |
| ibm16 | 3.8474 | 0.190 | 0.879 | 6.435 | 1756.23s | yes |
| ibm17 | 5.4864 | 0.185 | 0.886 | 9.717 | 3407.46s | yes |
| ibm18 | 3.6531 | 0.156 | 0.908 | 6.085 | 2081.52s | yes |

### Final Clean Config Smoke Check

After removing benchmark-name routing and preparing the clean pure config, we ran a targeted smoke check on representative benchmarks:

| Benchmark | Proxy | WL | Density | Congestion | Runtime | Purity |
|---|---:|---:|---:|---:|---:|---|
| ibm02 | 2.5977 | 0.2282 | 0.7680 | 3.9710 | 766.31s | yes |
| ibm03 | 2.7371 | 0.2391 | 0.8618 | 4.1342 | 365.50s | yes |
| ibm06 | 3.3849 | 0.2285 | 0.7883 | 5.5245 | 769.59s | yes |

The final active config uses `time_budget_sec=1300`. The smoke benchmarks above all ran under that limit. The expected suite-average cost of the budget reduction is roughly `+1%` to `+2.5%`, but it reduces timeout risk on the longest cases.

---

## Why the Proxy Is Worse Than Augmented

The weakness is congestion.

The competition proxy strongly rewards classical placement behavior that TILOS is already very good at. Pure plasma removes those classical load-bearing pieces and replaces them with physical analogs: pressure, flux, current, transport, sheath forces, and instabilities.

That makes the method scientifically cleaner but less directly aligned with the proxy.

```text
Augmented version:
    TILOS does the heavy classical placement work.
    Plasma modules add refinements.
    Proxy score is better.

Pure version:
    Plasma modules carry the placement pipeline.
    Proxy score is worse.
    Innovation claim is stronger.
```

---

## Repository Layout

```text
Plasma-Fusion-Physics-Macro-Place-Algo/
  README.md
  LICENSE
  submissions/
    team_plasma/
      placer.py                     top-level TeamPlasmaPlacer
      config.json                   active pure submission config
      config.py                     config schema / loader
      plasma_init.py                R1 plasma startup
      gs_solver.py                  Grad-Shafranov solver
      two_fluid.py                  R4 two-fluid transport
      legalization.py               R2 legalization
      pic_placer.py                 optional R5 PIC/P3M research module
      stability.py                  Mercier / stability shaping
      smooth_proxy.py               smooth guidance fields
      LICENSE                       Apache 2.0
```

---

## File Guide

| File | Purpose |
|---|---|
| `placer.py` | Main orchestrator. Wires all physics stages together. |
| `config.json` | Active pure-plasma submission settings. |
| `config.py` | Config schema and JSON loader. |
| `plasma_init.py` | R1 startup, flux bands, soft initialization. |
| `gs_solver.py` | Grad-Shafranov finite-difference solver. |
| `profiles.py` | Density, RUDY, pressure and current profiles. |
| `two_fluid.py` | R4 soft-macro two-fluid transport. |
| `legalization.py` | R2 hard-macro overlap removal. |
| `sheath_legalize.py` | Bohm-sheath legalization tools. |
| `pic_placer.py` | Optional R5 PIC / P3M proposal module. |
| `stability.py` | Mercier and stability-inspired shaping. |
| `smooth_proxy.py` | Smooth proxy / hotspot guidance fields. |
| `coils.py` | Macro current and coil force helpers. |
| `geometry.py` | Cylindrical embedding and interpolation. |

---

## What Is Active in the Final Pure Config?

| Mechanism | Active? | Notes |
|---|---|---|
| R1 plasma startup | yes | replaces TILOS PLC warm start |
| Grad-Shafranov solve | yes | central equilibrium field |
| R4 two-fluid transport | yes | replaces TILOS soft optimization |
| R2 continuation legalization | yes | pure legality path |
| Mercier / NTM-style refinement | yes | exact-gated |
| Flux-rope cluster moves | yes | budget-capped |
| R5 PIC / P3M | no | available as research, disabled in final config |
| Soft TILOS optimization | no | disabled in pure submission |
| Benchmark-name routing | no | active config is clean |

---

## Known Limitations

- Congestion is still the dominant error term.
- Pure plasma is much worse than the augmented TILOS-assisted version on proxy score.
- R5 PIC/P3M works on some cases but is not robust enough for global activation.
- Long benchmarks require conservative budget caps.
- Future work should move dimensionless plasma regime selection earlier into R1/R4, not only late proposal stages.

---

## Research Lineage

The internal research docs for this project were intentionally left out of the public repo, but the ideas came from a real paper trail. This section gives the reviewer-friendly version: what each reference contributed to the implementation story.

### Plasma Simulation And Kinetic Theory

| Reference | What We Borrowed |
|---|---|
| [Birdsall and Langdon, *Plasma Physics via Computer Simulation*](https://www.osti.gov/biblio/6702524) | Particle-in-cell thinking: particle-to-grid deposition, field solves, and grid-to-particle interpolation. |
| [Qin et al., "Why is Boris algorithm so good?"](https://www.osti.gov/biblio/1090047) | The reason Boris-style charged-particle updates are stable enough to be useful as proposal dynamics. |
| [Boris pusher historical reference](https://www.sciencedirect.com/science/article/pii/S0010465522002788) | The Lorentz-force particle-push lineage behind the optional R5 PIC module. |
| [Bohm, *The Characteristics of Electrical Discharges in Magnetic Fields*](https://books.google.com/books/about/The_Characteristics_of_Electrical_Discha.html?id=NHV5AAAAIAAJ) | Bohm sheath criterion and boundary-wall physics used as the legalization analogy. |
| [Stangeby, *The Plasma Boundary of Magnetic Fusion Devices*](https://openlibrary.org/books/OL22380912M/The_plasma_boundary_of_magnetic_fusion_devices) | Plasma-wall / scrape-off-layer intuition for boundary pressure and sheath handling. |

### MHD, Equilibrium, And Instability

| Reference | What We Borrowed |
|---|---|
| [Grad-Shafranov equation overview](https://en.wikipedia.org/wiki/Grad%E2%80%93Shafranov_equation) | The central equilibrium template: pressure, current, and geometry coupled through a flux field. |
| [Boozer, "Plasma equilibrium with rational magnetic surfaces"](https://www.osti.gov/biblio/6063300) | Flux-coordinate and rational-surface thinking used in field-aligned transport and stability language. |
| [Furth, Killeen, and Rosenbluth, "Finite-Resistivity Instabilities of a Sheet Pinch"](https://cir.nii.ac.jp/crid/1363107370207531008) | Tearing-mode inspiration for topology-changing refinement moves. |
| [Ryutov, "Geometrical properties of a snowflake divertor"](https://doi.org/10.1063/1.2738399) | Snowflake-divertor intuition for splitting intense hot spots into multiple channels. |
| [Ware, "Pinch Effect for Trapped Particles in a Tokamak"](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.25.15) | Ware-pinch style congestion/current coupling experiments. |
| [Taylor, "Relaxation of Toroidal Plasma and Generation of Reverse Magnetic Fields"](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.33.1139) | Taylor relaxation as the conceptual basis for energy minimization under topology constraints. |

### Placement And Optimization Context

| Reference | What We Borrowed |
|---|---|
| [ePlace: Electrostatics Based Placement Using Nesterov's Method](https://cseweb.ucsd.edu/~jlu/papers/eplace-dac14.pdf) | The classical electrostatic placement baseline we deliberately did not use as the load-bearing mechanism in pure mode. |
| [RePlAce: Advancing Solution Quality and Routability Validation in Global Placement](https://vlsicad.ucsd.edu/Publications/Journals/j126.pdf) | Modern analytical placement context: density, step-size control, and routability pressure. |
| [DREAMPlace](https://research.nvidia.com/publication/2019-06_dreamplace-deep-learning-toolkit-enabled-gpu-acceleration-modern-vlsi-placement) | Differentiable / tensorized placement framing and the lesson that EDA objectives can be recast in other computational languages. |
| [SIMSOPT: A flexible framework for stellarator optimization](https://doi.org/10.21105/joss.03525) | The strongest "two-way street" inspiration: plasma equilibria are optimized with software patterns that look a lot like constrained placement search. |

These references are not a claim that the solver is a perfect physical simulator. They are the source map for the design vocabulary: fields, currents, sheaths, instabilities, exact gates, and variational relaxation.

---

## Next Steps

The internal research notes point to a clear next generation of this project. None of these are required for the submitted package, but they are the most promising directions.

### 1. Ab-Initio Plasma Parameters

The current config is clean, but it still has many engineering constants. A stronger scientific version would derive nearly everything from a small set of plasma quantities:

```text
macro density -> Debye length
Debye length  -> grid resolution
thermal speed -> stable timestep
Larmor radius -> background magnetic field
collision rate -> damping / relaxation schedule
```

That would move the solver from "plasma-inspired with tuned parameters" toward "ab-initio plasma placement."

### 2. Adjoint Proxy Gradients

Stellarator optimization tools such as STELLOPT and SIMSOPT use adjoint gradients to optimize plasma equilibria. The same idea can be applied here:

```text
forward placement state
  -> smooth proxy field
  -> adjoint gradient
  -> plasma-constrained update
```

This is the most direct path to improving congestion, because congestion dominates the current proxy gap.

### 3. Stronger PIC / P3M

The optional R5 module is a first pass at plasma Particle-In-Cell placement. The next version should make PIC more load-bearing:

- spatially varying background `B` field,
- magnetic mirror effects,
- grad-B drift,
- pairwise Biot-Savart attraction,
- stronger exact-gated proposal portfolios.

The useful research lesson was that one current loop per net loses too much pairwise information. P3M-style short-range pair forces plus long-range mesh fields are a better plasma analogue.

### 4. Dimensionless Regime Selection

Instead of benchmark-name routing, future configs should classify layouts with continuous plasma-like numbers:

| Number | Placement Meaning |
|---|---|
| packing fraction | total macro area / canvas area |
| net density | nets per unit canvas area |
| hard ratio | hard macros / all macros |
| connectivity | average weighted net degree |
| aspect ratio | canvas anisotropy |

That keeps the solver general while still letting it adapt to very different placement regimes.

### 5. Plasma Instability Escape Moves

Classical placers use annealing and local search to escape basins. Plasma has its own escape vocabulary:

- tearing reconnection,
- sawtooth crashes,
- edge-localized mode bursts,
- Taylor relaxation,
- resonant magnetic perturbations.

Those are not just metaphors. They suggest structured, topology-changing proposal moves that can be exact-gated by the placement proxy.

---

## A Two-Way Street

The fun part is not only that plasma physics can be used for macro placement. The reverse direction is interesting too.

Macro placement is a brutal optimization laboratory: rank-based congestion, hard legality constraints, topology preservation, limited runtime, and highly nonconvex objectives. If plasma-inspired algorithms can survive here, the tricks we learn may feed back into plasma simulation and fusion optimization:

```mermaid
flowchart LR
    A[Plasma Physics] -->|fields, transport, sheaths, instabilities| B[Macro Placement]
    B -->|fast proxy gates, discrete topology moves, layout stress tests| A

    A:::plasma
    B:::eda

    classDef plasma fill:#e9f6ff,stroke:#0b5d7a,color:#111,stroke-width:2px;
    classDef eda fill:#fff3cd,stroke:#9a6a00,color:#111,stroke-width:2px;
```

Possible feedback back into plasma work:

- exact-gated proposal portfolios for expensive simulation loops,
- discrete topology moves inspired by placement legalization,
- congestion-like hotspot metrics for transport barriers and divertor loads,
- fast surrogate fields that approximate expensive equilibrium solves,
- benchmark-style stress tests for optimization robustness.

That is the bigger bet behind Team Plasma GS: chip placement is not just a place to borrow plasma ideas. It can also become a playground for inventing optimization patterns that plasma simulation may borrow back.

---

## Final Note

This is not a claim that plasma physics beats mature classical placement today. It does not.

It is a claim that macro placement can be formulated, implemented, and submitted as a plasma equilibrium problem:

```text
R1 plasma startup
+ Grad-Shafranov equilibrium
+ R4 two-fluid transport
+ R2 sheath legalization
+ plasma instability refinements
= legal pure-plasma macro placement
```

The Sun has not dethroned classical placers. But it did produce 17 legal IBM placements, which is a wonderfully unreasonable thing for a star to help with.
