"""Team Plasma GS — macro placement as a Grad-Shafranov equilibrium.

The public entry point is `TeamPlasmaPlacer` in `placer.py`.

Submodules:
    geometry      cylindrical-coordinate embedding of the canvas
    gs_solver     Delta-star FD operator + RB-GS + Picard
    profiles      online p(psi), F(psi) fitting
    coils         macros as internal current-bearing coils
    stability     Mercier interchange tail-aware shaping
    two_fluid     adiabatic-electron soft macros
    legalization  pairwise push-apart hard-macro legalization
    config        configuration loading
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from placer import TeamPlasmaPlacer  # noqa: E402

__all__ = ["TeamPlasmaPlacer"]
