# API Reference

This document provides detailed API reference for the Workspace Workspace Optimus routing system.

## Core Classes

### VehicleState

Represents the current state of a vehicle in the routing system.

```python
class VehicleState:
    def __init__(self, id: int, capacity: int, current_stock: int, position: int, 
                 total_cost: float = 0.0, route: List[int] = None)
```

**Attributes:**
- `id`: Unique identifier for the vehicle
- `capacity`: Maximum carrying capacity of the vehicle
- `current_stock`: Current amount of stock in the vehicle
- `position`: Current location of the vehicle
- `total_cost`: Total cost accumulated by this vehicle
- `route`: List of locations visited by this vehicle

**Methods:**
- `can_carry(amount: int) -> bool`: Check if the vehicle can carry the specified amount
- `has_stock(amount: int) -> bool`: Check if the vehicle has sufficient stock
- `can_deliver(amount: int) -> bool`: Check if the vehicle can deliver the specified amount
- `refill() -> None`: Refill the vehicle to maximum capacity
- `deliver(amount: int) -> bool`: Deliver the specified amount and update stock
- `move_to(location: int) -> None`: Move the vehicle to a new location
- `add_cost(cost: float) -> None`: Add cost to the vehicle's total cost

### DeliveryTask

Represents a delivery task that can be split across multiple vehicles.

```python
class DeliveryTask:
    def __init__(self, location: int, demand: int, assigned_vehicle: Optional[Union[int, List[int]]] = None,
                 completed: bool = False, split_deliveries: List[Tuple[int, int]] = None)
```

**Attributes:**
- `location`: Target location for delivery
- `demand`: Amount to be delivered
- `assigned_vehicle`: ID of the vehicle assigned to this task
- `completed`: Whether the task has been completed
- `split_deliveries`: List of (vehicle_id, amount) tuples for split deliveries

**Methods:**
- `is_split_delivery() -> bool`: Check if this task is a split delivery
- `get_total_assigned() -> int`: Get the total amount assigned across all vehicles
- `is_fully_assigned() -> bool`: Check if the full demand has been assigned
- `add_split_delivery(vehicle_id: int, amount: int) -> bool`: Add a split delivery assignment

### DepotManager

Manages depot operations including closest depot selection and distance calculations.

```python
class DepotManager:
    def __init__(self, depot_locations: List[int], cost_matrix: np.ndarray, 
                 location_to_index: dict)
```

**Methods:**
- `find_nearest_depot(current_location: int) -> Tuple[int, float]`: Find the nearest depot
- `get_distance(from_location: int, to_location: int) -> float`: Get distance between locations
- `calculate_refill_cost(from_location: int, to_delivery: int) -> float`: Calculate refill cost

### CostCalculator

Calculates costs for various delivery strategies and options.

```python
class CostCalculator:
    def __init__(self, depot_manager: DepotManager)
```

**Methods:**
- `calculate_direct_delivery_cost(vehicle: VehicleState, task: DeliveryTask) -> float`
- `calculate_refill_cost(vehicle: VehicleState, task: DeliveryTask) -> float`
- `calculate_new_vehicle_cost(vehicles: List[VehicleState], task: DeliveryTask) -> float`
- `calculate_split_delivery_cost(vehicles: List[VehicleState], task: DeliveryTask) -> List[DeliveryOption]`
- `find_optimal_delivery_strategy(vehicles: List[VehicleState], task: DeliveryTask) -> DeliveryOption`

## Main System Classes

### OptimizedRefillSystem

Basic refill optimization system without split delivery.

```python
class OptimizedRefillSystem:
    def __init__(self, offsets: np.ndarray, edges: np.ndarray, weights: np.ndarray,
                 time_to_travel: np.ndarray, target_locations: np.ndarray,
                 depot_locations: List[int], vehicle_capacities: List[int],
                 initial_stock: List[int], delivery_demands: List[int])
```

**Methods:**
- `solve_optimized() -> Dict`: Solve the routing problem with optimal vehicle assignment

### EnhancedRefillSystem

Enhanced refill system with split delivery optimization.

```python
class EnhancedRefillSystem:
    def __init__(self, offsets: np.ndarray, edges: np.ndarray, weights: np.ndarray,
                 time_to_travel: np.ndarray, target_locations: np.ndarray,
                 depot_locations: List[int], vehicle_capacities: List[int],
                 initial_stock: List[int], delivery_demands: List[int])
```

**Methods:**
- `solve_enhanced() -> Dict`: Solve with enhanced optimization including split delivery

### CuOptEnhancedPlanner

GPU-accelerated planner that builds a cuOpt data model, launches the solver, and returns a normalized result object.

```python
class CuOptEnhancedPlanner:
    def __init__(self, offsets, edges, weights, time_to_travel,
                 target_locations, depot_locations,
                 vehicle_capacities, initial_stock,
                 delivery_demands, solver_time_limit: float = 30.0,
                 vehicle_start_locations: Optional[List[int]] = None)

    def solve(self) -> PlannerResult
```

**Key behaviours:**
- Automatically splits oversized customer demand into multiple cuOpt orders.
- Configures vehicle capacity, time, and depot constraints before calling the GPU solver.
- Returns a `PlannerResult` object that can be converted to the legacy dictionary structure via `as_dict()`.
- Accepts per-vehicle starting locations (defaults to the first depot) and ensures every vehicle finishes at the nearest depot to its final task location.
- Falls back to a deterministic CPU heuristic with identical output fields when the cuOpt runtime is unavailable.

## Utility Classes

### GraphUtils

Utility class for graph manipulation and analysis.

```python
class GraphUtils:
    @staticmethod
    def create_graph_from_waypoint_matrix(offsets: np.ndarray, edges: np.ndarray, 
                                        weights: np.ndarray) -> nx.Graph
    
    @staticmethod
    def validate_graph_connectivity(offsets: np.ndarray, edges: np.ndarray, 
                                  weights: np.ndarray) -> bool
    
    @staticmethod
    def find_shortest_path(offsets: np.ndarray, edges: np.ndarray, weights: np.ndarray,
                          source: int, target: int) -> Tuple[List[int], float]
```

### DistanceCalculator

Utility class for distance calculations.

```python
class DistanceCalculator:
    @staticmethod
    def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float
    
    @staticmethod
    def manhattan_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float
    
    @staticmethod
    def create_distance_matrix(coordinates: List[Tuple[float, float]], 
                             distance_func: Callable = None) -> np.ndarray
```

### InputValidator

Utility class for input validation.

```python
class InputValidator:
    @staticmethod
    def validate_waypoint_matrix(offsets: np.ndarray, edges: np.ndarray, 
                               weights: np.ndarray) -> Tuple[bool, List[str]]
    
    @staticmethod
    def validate_locations(locations: np.ndarray, num_nodes: int) -> Tuple[bool, List[str]]
    
    @staticmethod
    def validate_vehicle_capacities(capacities: List[int]) -> Tuple[bool, List[str]]
```

## Data Structures

### DeliveryOption

Represents a delivery option with cost analysis.

```python
@dataclass
class DeliveryOption:
    strategy: str                    # Delivery strategy ('direct', 'refill', 'new_vehicle', 'split')
    vehicles: List[int]              # List of vehicle IDs involved
    cost: float                      # Total cost for this option
    details: Dict[str, Any]          # Additional details about the option
```

## Return Types

### Solution Dictionary

All solvers expose a dictionary with the following structure (the cuOpt planner adds the `vehicle_plans` and `meta` entries while preserving backwards compatibility):

```python
{
    'total_cost': float,                    # Total cost of the solution
    'assignments': List[Dict],              # List of delivery assignments
    'vehicles_used': int,                   # Number of vehicles used
    'strategy_counts': Dict[str, int],      # Count of each strategy used
    'vehicle_routes': Dict[int, List[int]], # Routes for each vehicle
    'vehicle_plans': List[Dict[str, Any]],  # Detailed per-vehicle actions (CuOpt planner)
    'meta': Dict[str, Any]                  # Solver metadata (cuOpt status, fallback reason, etc.)
}
```

### Assignment Dictionary

Each assignment in the solution contains:

```python
{
    'task_id': int,                         # Task identifier
    'strategy': str,                        # Strategy used ('direct', 'refill', 'new_vehicle', 'split')
    'vehicles': Union[int, List[int]],      # Vehicle ID(s) involved
    'cost': Optional[float],                # Cost of this assignment (legacy heuristics)
    'location': int,                        # Delivery location
    'demand': int,                          # Delivery demand
    'delivery_time': Optional[float],       # Delivery timestamp (CuOpt planner)
    'details': Dict[str, Any]               # Additional details
}
```

## Error Handling

The system raises the following exceptions:

- `ValueError`: Invalid input parameters or no feasible solution found
- `TypeError`: Incorrect parameter types
- `RuntimeError`: System runtime errors

## Configuration

The system uses configuration from `workspace_optimus.config.default_config`:

```python
from workspace_optimus.config.default_config import DEFAULT_CONFIG

# Access configuration
optimization_config = DEFAULT_CONFIG['optimization']
vehicle_config = DEFAULT_CONFIG['vehicle']
# ... etc
```

## Examples

See the `workspace_optimus/examples/` directory for comprehensive usage examples:

- `basic_usage.py`: Basic system usage
- `split_delivery_demo.py`: Split delivery demonstration
- `multi_vehicle_example.py`: Multi-vehicle scenarios
- `large_scale_example.py`: Large-scale problem examples
