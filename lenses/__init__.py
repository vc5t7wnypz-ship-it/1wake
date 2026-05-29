"""lenses — WAKE interpretive lens system.

Exports every public symbol from the lens subsystem so that callers can do::

    from lenses import Lens, LensConfig, ComposedLens, ALL_LENSES
    from lenses import ViconianLens, KabbalisticLens, FreudianLens
    from lenses import IrishMythologyLens, NorseLens, BrunianLens
    from lenses import get_lens, compose_lenses, list_lenses
"""

from .base import Lens, LensConfig
from .brunian import BrunianLens
from .freudian import FreudianLens
from .irish_mythology import IrishMythologyLens
from .kabbalistic import KabbalisticLens
from .norse import NorseLens
from .registry import ALL_LENSES, ComposedLens, compose_lenses, get_lens, list_lenses
from .viconian import ViconianLens

__all__ = [
    # Base abstractions
    "Lens",
    "LensConfig",
    # Concrete lens classes
    "ViconianLens",
    "KabbalisticLens",
    "FreudianLens",
    "IrishMythologyLens",
    "NorseLens",
    "BrunianLens",
    # Registry / ensemble
    "ALL_LENSES",
    "ComposedLens",
    "get_lens",
    "compose_lenses",
    "list_lenses",
]
