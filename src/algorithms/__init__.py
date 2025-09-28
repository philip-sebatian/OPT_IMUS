"""
Optimization algorithms for the Optimus routing system.

This module contains the main optimization algorithms including:
- Refill optimization
- Split delivery algorithms
- Route optimization engine
"""

from .cuopt_enhanced import CuOptEnhancedPlanner


def _unavailable(name: str, exc: Exception):
    class _Unavailable:
        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise RuntimeError(
                f"{name} requires a functional cuOpt GPU runtime."
                f" Original error: {exc}"
            )

    _Unavailable.__name__ = name
    return _Unavailable


try:  # pragma: no cover
    from .refill_optimizer import OptimizedRefillSystem
except Exception as exc:  # pragma: no cover
    OptimizedRefillSystem = _unavailable("OptimizedRefillSystem", exc)

try:  # pragma: no cover
    from .split_delivery import EnhancedRefillSystem
except Exception as exc:  # pragma: no cover
    EnhancedRefillSystem = _unavailable("EnhancedRefillSystem", exc)

try:  # pragma: no cover
    from .route_optimizer import RouteOptimizer
except Exception as exc:  # pragma: no cover
    RouteOptimizer = _unavailable("RouteOptimizer", exc)

__all__ = [
    "OptimizedRefillSystem",
    "EnhancedRefillSystem", 
    "RouteOptimizer",
    "CuOptEnhancedPlanner",
]
