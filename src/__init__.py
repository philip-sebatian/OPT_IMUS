"""
Optimus: Advanced Vehicle Routing with Refill Optimization

A comprehensive vehicle routing optimization system that extends the Pickup and Delivery Problem (PDP)
with advanced refill capabilities, split delivery optimization, and intelligent depot selection.
"""

from .core.vehicle import VehicleState
from .core.delivery_task import DeliveryTask
from .core.depot_manager import DepotManager
from .core.cost_calculator import CostCalculator
from .algorithms.cuopt_enhanced import CuOptEnhancedPlanner


def _unavailable(name: str, exc: Exception):
    class _Unavailable:
        def __init__(self, *args, **kwargs):  # pragma: no cover - runtime guard
            raise RuntimeError(
                f"{name} requires a functional cuOpt GPU runtime."
                f" Original error: {exc}"
            )

    _Unavailable.__name__ = name
    return _Unavailable


try:  # pragma: no cover - exercised only when GPU runtime is present
    from .algorithms.refill_optimizer import OptimizedRefillSystem
except Exception as exc:  # pragma: no cover - handled during CPU-only testing
    OptimizedRefillSystem = _unavailable("OptimizedRefillSystem", exc)

try:  # pragma: no cover
    from .algorithms.split_delivery import EnhancedRefillSystem
except Exception as exc:  # pragma: no cover
    EnhancedRefillSystem = _unavailable("EnhancedRefillSystem", exc)

try:  # pragma: no cover
    from .algorithms.route_optimizer import RouteOptimizer
except Exception as exc:  # pragma: no cover
    RouteOptimizer = _unavailable("RouteOptimizer", exc)

__version__ = "1.0.0"
__author__ = "workspace_optimus Team"
__email__ = "workspace_optimus@example.com"

__all__ = [
    "VehicleState",
    "DeliveryTask", 
    "DepotManager",
    "CostCalculator",
    "OptimizedRefillSystem",
    "EnhancedRefillSystem",
    "CuOptEnhancedPlanner",
    "RouteOptimizer",
]
