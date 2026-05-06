# Team Plasma Macro Placer

Team Plasma is a challenge-compatible macro placer for the Partcl/HRT Macro
Placement Challenge. The implementation lives under `submissions/team_plasma`
so it can be dropped directly into the official challenge repository.

This repository contains only the Team Plasma submission code, its current
mainline configuration, the candidate configs referenced by that mainline, and
focused tests. It intentionally does not vendor the official evaluator,
benchmarks, or competition README.

## Challenge Links

- Challenge repository: https://github.com/partcleda/partcl-macro-place-challenge
- Setup and evaluator usage: https://github.com/partcleda/partcl-macro-place-challenge/blob/main/SETUP.md
- Scoring details: https://github.com/partcleda/partcl-macro-place-challenge/blob/main/SCORING.md
- Submission form and rules: https://github.com/partcleda/partcl-macro-place-challenge/blob/main/README.md

## Current Result

Latest trusted IBM-suite checkpoint:

- Average proxy cost: `1.4204550552`
- Result artifact: `docs/official_fullsuite_v22_summary.json`
- Route proof: `docs/official_fullsuite_v22_route_proof.json`
- Zero overlaps on all evaluated public IBM benchmarks

The remaining high-cost cases are mostly congestion and density dominated:

| Benchmark | Proxy | Wirelength | Density | Congestion |
| --- | ---: | ---: | ---: | ---: |
| `ibm18` | `1.7085441` | `0.0563` | `0.7889` | `2.5157` |
| `ibm17` | `1.6791` | `0.0550` | `0.7360` | `2.5120` |
| `ibm06` | `1.6583` | `0.0640` | `0.7230` | `2.4670` |
| `ibm12` | `1.6298` | `0.0600` | `0.7700` | `2.3700` |
| `ibm02` | `1.5736223` | `0.0755` | `0.7215` | `2.2748` |

## Why Plasma?

Macro placement has a familiar tension: connected macros want to cluster, but
routing and density constraints need empty channels. Fusion plasma has a similar
mathematical flavor: pressure pushes outward, fields confine and shape the
plasma, boundaries matter, and useful equilibrium is not found by optimizing one
force in isolation.

Team Plasma borrows that equilibrium idea. It does not simulate real plasma. It
uses a Grad-Shafranov-inspired nonlinear field solve as a placement engine.

In axisymmetric magnetohydrodynamics, the Grad-Shafranov equation solves for a
magnetic flux function `psi` whose contours describe plasma equilibrium:

```text
Delta* psi = source terms from pressure and magnetic field profiles
```

This placer uses the same kind of idea:

```text
placement potential psi = equilibrium of net attraction, density pressure,
routing pressure, wall pressure, pin/port influence, and local source shaping
```

Macros then move along forces derived from that potential. The goal is not just
short wirelength; it is a balanced field where macros settle into legal,
routable, low-density-pressure positions.

Background on the physics inspiration:

- Grad-Shafranov equation: https://en.wikipedia.org/wiki/Grad%E2%80%93Shafranov_equation
- Magnetohydrodynamic equilibrium: https://en.wikipedia.org/wiki/Magnetohydrodynamics

## Physics-To-Placement Map

| Fusion / plasma idea | Macro placement analogue |
| --- | --- |
| Plasma pressure | Macro density and congestion pressure |
| Magnetic flux function `psi` | Scalar placement potential over the chip canvas |
| Current / source profile | Net, pin, port, and soft-cluster source fields |
| Conducting wall / vessel | Chip boundary, blockages, and wall repulsion |
| Equilibrium solve | Iterative global placement update |
| Field gradients | Macro movement forces |
| Relaxation to a stable state | Legalization plus local refinement |

The useful part of this analogy is that it turns placement into a global field
problem. A macro does not only react to nearby overlaps or direct net edges. It
reacts to a canvas-wide potential that combines connectivity, pressure, and
routability.

## Algorithm Overview

Team Plasma is a hybrid placer with five major stages:

1. Feature routing

The placer extracts benchmark-scale features and chooses a route family from
`config_local.json`. This keeps the mainline general: it routes by geometric and
netlist features, not by hardcoded benchmark names.

2. Plasma global solve

`plasma_core.py` builds the field terms and solves a nonlinear elliptic
potential problem with damped Picard iterations and red-black Gauss-Seidel
relaxation. The resulting potential produces vector forces for hard macros.

3. Source shaping

`placer.py` prepares route-specific source terms such as source quench,
pin-flux tubes, wall pressure, bundle terms, and optional cluster-aware hooks.
These terms alter the global field before legalization, which is where the
plasma analogy matters most.

4. Legalization

After the global field moves macros into a useful basin, deterministic
legalization removes hard-macro overlaps while trying to preserve the field's
structure.

5. Local refinement and soft-macro passes

Route-scoped exact refinement, soft-macro optimization, and optional portfolios
make small end-to-end improvements while preserving zero overlap.

## Code Tour

```text
submissions/team_plasma/
  placer.py
    TeamPlasmaPlacer entrypoint.
    Owns benchmark parsing, route selection, global solve orchestration,
    legalization, exact refinement, and soft-macro refinement.

  plasma_core.py
    Numerical plasma-field core.
    Builds density/routing/source fields, solves the nonlinear potential,
    samples vector fields, and converts potential gradients into macro forces.

  config.py
    Config loading and merge logic.
    Supports constructor-provided configs, TEAM_PLASMA_CONFIG, config_local.json,
    and defaults.

  config_local.json
    Current trusted mainline router/config.

  config_submission.json
    Longer-budget submission-oriented profile.

  config_iter_best.json
    Historical iterative profile.

  jax_accel.py
    Optional JAX helpers for acceleration experiments.

  isolation_worker.py
    Helper used by isolated portfolio/evaluation flows.

  candidates/
    Candidate JSONs referenced by the current configs and tests.

test/
  test_team_plasma.py
    Focused invariants for field construction, legalization behavior, config
    routing, and regression guards around experimental hooks.

docs/
  official_fullsuite_v22_summary.json
  official_fullsuite_v22_route_proof.json
```

## What The Current Mainline Is Good At

- Strong average public IBM proxy score versus the official RePlAce baseline.
- Zero-overlap public IBM placements in the trusted checkpoint.
- Good wirelength control on the hardest remaining cases.
- Route-scoped behavior that avoids benchmark-name hardcoding.
- Extensible plasma-native hooks for source shaping and local refinement.

## Current Technical Bottleneck

The remaining high-cost cases are mostly density and congestion dominated. The
mainline can find decent global basins, but the hardest cases still need a
better representation of hidden macro-to-standard-cell-cluster structure and a
stronger handoff from global field placement to soft/legalization stages.

## Running Inside The Official Challenge Repo

1. Clone and set up the official challenge repository:

```powershell
git clone https://github.com/partcleda/partcl-macro-place-challenge.git
cd partcl-macro-place-challenge
git submodule update --init external/MacroPlacement
uv sync
```

2. Copy this repository's `submissions/team_plasma` folder into the challenge
   repo at the same relative path:

```powershell
Copy-Item -Recurse -Force path\to\this\repo\submissions\team_plasma .\submissions\
```

3. Evaluate a single benchmark:

```powershell
uv run evaluate submissions/team_plasma/placer.py -b ibm18
```

4. Evaluate the public IBM suite:

```powershell
uv run evaluate submissions/team_plasma/placer.py --all
```

## Configuration

`TeamPlasmaPlacer` loads configuration in this priority order:

1. An explicit constructor `config_path`
2. `TEAM_PLASMA_CONFIG`
3. `submissions/team_plasma/config_local.json`
4. Defaults from `submissions/team_plasma/config.py`

For normal evaluation, use `config_local.json`. The included candidate JSONs are
the files referenced by the current mainline config and the focused tests.

## Tests

Run the focused tests from the official challenge repository after copying the
submission folder:

```powershell
uv run python -m pytest test/test_team_plasma.py
```

If `pytest` is not installed in the challenge environment, install the
challenge's development dependencies or use direct import/compile checks:

```powershell
uv run python -m compileall submissions/team_plasma test/test_team_plasma.py
```

## License

Apache-2.0. See `LICENSE`.
