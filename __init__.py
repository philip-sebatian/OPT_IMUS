"""
Optimus: Advanced Vehicle Routing with Refill Optimization

A comprehensive vehicle routing optimization system that extends the Pickup and Delivery Problem (PDP)
with advanced refill capabilities, split delivery optimization, and intelligent depot selection.
"""

# Import the main classes from the src module
from .src import (
    CuOptEnhancedPlanner,
    EnhancedRefillSystem,
    OptimizedRefillSystem,
    RouteOptimizer,
)

__version__ = "1.0.0"
__author__ = "workspace_optimus Team"
__email__ = "workspace_optimus@example.com"

__all__ = [
    "OptimizedRefillSystem",
    "EnhancedRefillSystem",
    "RouteOptimizer",
    "CuOptEnhancedPlanner",
]
