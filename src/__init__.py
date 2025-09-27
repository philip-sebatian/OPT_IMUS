"""
Optimus: Advanced Vehicle Routing with Refill Optimization

A comprehensive vehicle routing optimization system that extends the Pickup and Delivery Problem (PDP)
with advanced refill capabilities, split delivery optimization, and intelligent depot selection.
"""

from .core.vehicle import VehicleState
from .core.delivery_task import DeliveryTask
from .core.depot_manager import DepotManager
from .core.cost_calculator import CostCalculator
from .algorithms.refill_optimizer import OptimizedRefillSystem
from .algorithms.split_delivery import EnhancedRefillSystem

__version__ = "1.0.0"
__author__ = "Optimus Team"
__email__ = "optimus@example.com"

__all__ = [
    "VehicleState",
    "DeliveryTask", 
    "DepotManager",
    "CostCalculator",
    "OptimizedRefillSystem",
    "EnhancedRefillSystem"
]
