"""
Route optimization engine for the Optimus routing system.

This module provides the RouteOptimizer class which handles the core
routing optimization using the cuOpt library.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import time

from cuopt.distance_engine import WaypointMatrix
from cuopt.routing import DataModel, SolverSettings, Solve

from ..core.vehicle import VehicleState
from ..core.delivery_task import DeliveryTask


class RouteOptimizer:
    """
    Core route optimization engine using cuOpt.
    
    This class handles the underlying routing optimization using NVIDIA's cuOpt
    library, providing a clean interface for the higher-level optimization systems.
    """
    
    def __init__(self, offsets: np.ndarray, edges: np.ndarray, weights: np.ndarray, 
                 time_to_travel: np.ndarray, target_locations: np.ndarray, 
                 depot_locations: List[int], vehicle_capacities: List[int]):
        """
        Initialize the route optimizer.
        
        Args:
            offsets: Graph offsets for WaypointMatrix
            edges: Graph edges for WaypointMatrix  
            weights: Edge weights for WaypointMatrix
            time_to_travel: Travel times for edges
            target_locations: Delivery locations
            depot_locations: List of depot locations
            vehicle_capacities: Maximum capacity for each vehicle
        """
        self.offsets = offsets
        self.edges = edges
        self.weights = weights
        self.time_to_travel = time_to_travel
        self.target_locations = target_locations
        self.depot_locations = depot_locations
        self.vehicle_capacities = vehicle_capacities
        
        # Create waypoint matrix
        self.w_matrix = WaypointMatrix(offsets, edges, weights)
        
        # All locations (targets + depots)
        self.all_locations = np.unique(np.concatenate([target_locations, depot_locations]))
        
        # Compute cost and time matrices for all locations
        self.cost_matrix = self.w_matrix.compute_cost_matrix(self.all_locations)
        self.time_matrix = self.w_matrix.compute_shortest_path_costs(self.all_locations, time_to_travel)
        
        # Create location mapping
        self.location_to_index = {loc: idx for idx, loc in enumerate(self.all_locations)}
        self.index_to_location = {idx: loc for idx, loc in enumerate(self.all_locations)}
    
    def solve_basic_routing(self, time_limit: float = 1.0) -> Optional[Dict]:
        """
        Solve basic routing problem without refill considerations.
        
        Args:
            time_limit: Maximum solve time in seconds
            
        Returns:
            Solution dictionary or None if no solution found
        """
        try:
            # Create data model
            data_model = DataModel(
                n_locations=len(self.all_locations), 
                n_fleet=len(self.vehicle_capacities)
            )
            data_model.add_cost_matrix(self.cost_matrix)
            data_model.add_transit_time_matrix(self.time_matrix)
            
            # Solver settings
            solver_settings = SolverSettings()
            solver_settings.set_time_limit(time_limit)
            solver_settings.set_verbose_mode(False)
            
            # Solve the problem
            solution = Solve(data_model, solver_settings)
            
            if solution is not None and solution.status == 0:
                return self._process_solution(solution)
            else:
                return None
                
        except Exception as e:
            print(f"Error in basic routing: {e}")
            return None
    
    def _process_solution(self, solution) -> Dict:
        """
        Process cuOpt solution into standardized format.
        
        Args:
            solution: cuOpt solution object
            
        Returns:
            Processed solution dictionary
        """
        routes_df = solution.get_route()
        if hasattr(routes_df, 'to_pandas'):
            routes_df = routes_df.to_pandas()
        
        vehicle_routes = {}
        for truck_id in routes_df['truck_id'].unique():
            vehicle_route = routes_df[routes_df['truck_id'] == truck_id]
            original_locations = vehicle_route['location'].tolist()
            
            # Convert indices back to actual locations
            route = [self.index_to_location[idx] for idx in original_locations]
            vehicle_routes[truck_id] = route
        
        return {
            'status': solution.status,
            'objective_value': solution.total_objective_value,
            'vehicle_count': solution.vehicle_count,
            'message': solution.message,
            'vehicle_routes': vehicle_routes,
            'routes_df': routes_df
        }
    
    def calculate_route_cost(self, route: List[int]) -> float:
        """
        Calculate the total cost of a route.
        
        Args:
            route: List of locations in the route
            
        Returns:
            Total cost of the route
        """
        if len(route) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(route) - 1):
            from_loc = route[i]
            to_loc = route[i + 1]
            
            if from_loc in self.location_to_index and to_loc in self.location_to_index:
                from_idx = self.location_to_index[from_loc]
                to_idx = self.location_to_index[to_loc]
                total_cost += self.cost_matrix[from_idx][to_idx]
        
        return total_cost
    
    def get_distance(self, from_location: int, to_location: int) -> float:
        """
        Get distance between two locations.
        
        Args:
            from_location: Source location ID
            to_location: Destination location ID
            
        Returns:
            Distance between locations
        """
        if (from_location not in self.location_to_index or 
            to_location not in self.location_to_index):
            return float('inf')
        
        from_idx = self.location_to_index[from_location]
        to_idx = self.location_to_index[to_location]
        return self.cost_matrix[from_idx][to_idx]
    
    def get_time(self, from_location: int, to_location: int) -> float:
        """
        Get travel time between two locations.
        
        Args:
            from_location: Source location ID
            to_location: Destination location ID
            
        Returns:
            Travel time between locations
        """
        if (from_location not in self.location_to_index or 
            to_location not in self.location_to_index):
            return float('inf')
        
        from_idx = self.location_to_index[from_location]
        to_idx = self.location_to_index[to_location]
        return self.time_matrix[from_idx][to_idx]
    
    def validate_route(self, route: List[int]) -> bool:
        """
        Validate if a route is feasible.
        
        Args:
            route: List of locations in the route
            
        Returns:
            True if route is valid, False otherwise
        """
        if not route:
            return False
        
        # Check if all locations in route are valid
        for location in route:
            if location not in self.location_to_index:
                return False
        
        return True
    
    def optimize_route_sequence(self, locations: List[int]) -> List[int]:
        """
        Optimize the sequence of locations for a given set of locations.
        
        Args:
            locations: List of locations to visit
            
        Returns:
            Optimized sequence of locations
        """
        if len(locations) <= 1:
            return locations
        
        # Simple nearest neighbor heuristic
        optimized = [locations[0]]
        remaining = locations[1:]
        
        while remaining:
            current = optimized[-1]
            nearest = min(remaining, key=lambda x: self.get_distance(current, x))
            optimized.append(nearest)
            remaining.remove(nearest)
        
        return optimized
    
    def get_route_statistics(self, route: List[int]) -> Dict:
        """
        Get statistics for a route.
        
        Args:
            route: List of locations in the route
            
        Returns:
            Dictionary with route statistics
        """
        if not route:
            return {
                'total_cost': 0.0,
                'total_time': 0.0,
                'distance': 0.0,
                'locations_visited': 0,
                'is_valid': False
            }
        
        total_cost = self.calculate_route_cost(route)
        total_time = 0.0
        
        for i in range(len(route) - 1):
            from_loc = route[i]
            to_loc = route[i + 1]
            total_time += self.get_time(from_loc, to_loc)
        
        return {
            'total_cost': total_cost,
            'total_time': total_time,
            'distance': total_cost,  # Assuming cost represents distance
            'locations_visited': len(route),
            'is_valid': self.validate_route(route)
        }
    
    def __str__(self) -> str:
        """String representation of the route optimizer."""
        return (f"RouteOptimizer(locations={len(self.all_locations)}, "
                f"vehicles={len(self.vehicle_capacities)}, "
                f"depots={len(self.depot_locations)})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the route optimizer."""
        return (f"RouteOptimizer(offsets={self.offsets.shape}, "
                f"edges={self.edges.shape}, weights={self.weights.shape}, "
                f"target_locations={self.target_locations}, "
                f"depot_locations={self.depot_locations}, "
                f"vehicle_capacities={self.vehicle_capacities})")
