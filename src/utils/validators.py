"""
Input validation utilities for the Optimus routing system.

This module provides comprehensive validation functions for all inputs
to ensure the system operates with valid data.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import warnings


class InputValidator:
    """
    Utility class for input validation.
    """
    
    @staticmethod
    def validate_waypoint_matrix(offsets: np.ndarray, edges: np.ndarray, 
                               weights: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate waypoint matrix inputs.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            weights: Edge weights
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check types
        if not isinstance(offsets, np.ndarray):
            errors.append("Offsets must be a numpy array")
        if not isinstance(edges, np.ndarray):
            errors.append("Edges must be a numpy array")
        if not isinstance(weights, np.ndarray):
            errors.append("Weights must be a numpy array")
        
        if errors:
            return False, errors
        
        # Check shapes and values
        if len(offsets) < 2:
            errors.append("Offsets must have at least 2 elements")
        
        if not np.all(offsets[1:] >= offsets[:-1]):
            errors.append("Offsets must be non-decreasing")
        
        if offsets[0] != 0:
            errors.append("First offset must be 0")
        
        if offsets[-1] != len(edges):
            errors.append("Last offset must equal length of edges")
        
        if len(edges) != len(weights):
            errors.append("Edges and weights must have same length")
        
        # Check edge indices
        num_nodes = len(offsets) - 1
        if np.any(edges >= num_nodes) or np.any(edges < 0):
            errors.append("Edge indices must be valid node indices")
        
        # Check weights
        if np.any(weights < 0):
            errors.append("Weights must be non-negative")
        
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            errors.append("Weights must be finite numbers")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_locations(locations: np.ndarray, num_nodes: int) -> Tuple[bool, List[str]]:
        """
        Validate location array.
        
        Args:
            locations: Array of location indices
            num_nodes: Total number of nodes in the graph
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(locations, np.ndarray):
            errors.append("Locations must be a numpy array")
            return False, errors
        
        if len(locations) == 0:
            errors.append("Locations cannot be empty")
        
        if np.any(locations >= num_nodes) or np.any(locations < 0):
            errors.append("Location indices must be valid node indices")
        
        if len(np.unique(locations)) != len(locations):
            errors.append("Locations must be unique")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_vehicle_capacities(capacities: List[int]) -> Tuple[bool, List[str]]:
        """
        Validate vehicle capacities.
        
        Args:
            capacities: List of vehicle capacities
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(capacities, list):
            errors.append("Vehicle capacities must be a list")
            return False, errors
        
        if len(capacities) == 0:
            errors.append("Vehicle capacities cannot be empty")
        
        if not all(isinstance(cap, int) for cap in capacities):
            errors.append("All capacities must be integers")
        
        if not all(cap > 0 for cap in capacities):
            errors.append("All capacities must be positive")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_stock_levels(stock: List[int], capacities: List[int]) -> Tuple[bool, List[str]]:
        """
        Validate stock levels against capacities.
        
        Args:
            stock: List of initial stock levels
            capacities: List of vehicle capacities
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(stock, list):
            errors.append("Stock levels must be a list")
            return False, errors
        
        if len(stock) != len(capacities):
            errors.append("Stock levels must match number of vehicles")
        
        if not all(isinstance(s, int) for s in stock):
            errors.append("All stock levels must be integers")
        
        if not all(0 <= s <= cap for s, cap in zip(stock, capacities)):
            errors.append("Stock levels must be between 0 and capacity")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_delivery_demands(demands: List[int]) -> Tuple[bool, List[str]]:
        """
        Validate delivery demands.
        
        Args:
            demands: List of delivery demands
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(demands, list):
            errors.append("Delivery demands must be a list")
            return False, errors
        
        if len(demands) == 0:
            errors.append("Delivery demands cannot be empty")
        
        if not all(isinstance(d, int) for d in demands):
            errors.append("All demands must be integers")
        
        if not all(d > 0 for d in demands):
            errors.append("All demands must be positive")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_time_array(time_array: np.ndarray, expected_length: int) -> Tuple[bool, List[str]]:
        """
        Validate time array.
        
        Args:
            time_array: Array of time values
            expected_length: Expected length of the array
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(time_array, np.ndarray):
            errors.append("Time array must be a numpy array")
            return False, errors
        
        if len(time_array) != expected_length:
            errors.append(f"Time array length must be {expected_length}")
        
        if np.any(time_array < 0):
            errors.append("All time values must be non-negative")
        
        if np.any(np.isnan(time_array)) or np.any(np.isinf(time_array)):
            errors.append("Time values must be finite numbers")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_problem_parameters(offsets: np.ndarray, edges: np.ndarray, 
                                  weights: np.ndarray, time_to_travel: np.ndarray,
                                  target_locations: np.ndarray, depot_locations: List[int],
                                  vehicle_capacities: List[int], initial_stock: List[int],
                                  delivery_demands: List[int]) -> Tuple[bool, List[str]]:
        """
        Validate all problem parameters together.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            weights: Edge weights
            time_to_travel: Travel times
            target_locations: Target locations
            depot_locations: Depot locations
            vehicle_capacities: Vehicle capacities
            initial_stock: Initial stock levels
            delivery_demands: Delivery demands
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors = []
        
        # Validate waypoint matrix
        is_valid, errors = InputValidator.validate_waypoint_matrix(offsets, edges, weights)
        if not is_valid:
            all_errors.extend([f"Waypoint matrix: {e}" for e in errors])
        
        # Validate time array
        is_valid, errors = InputValidator.validate_time_array(time_to_travel, len(edges))
        if not is_valid:
            all_errors.extend([f"Time array: {e}" for e in errors])
        
        # Validate locations
        num_nodes = len(offsets) - 1
        is_valid, errors = InputValidator.validate_locations(target_locations, num_nodes)
        if not is_valid:
            all_errors.extend([f"Target locations: {e}" for e in errors])
        
        is_valid, errors = InputValidator.validate_locations(np.array(depot_locations), num_nodes)
        if not is_valid:
            all_errors.extend([f"Depot locations: {e}" for e in errors])
        
        # Validate vehicle parameters
        is_valid, errors = InputValidator.validate_vehicle_capacities(vehicle_capacities)
        if not is_valid:
            all_errors.extend([f"Vehicle capacities: {e}" for e in errors])
        
        is_valid, errors = InputValidator.validate_stock_levels(initial_stock, vehicle_capacities)
        if not is_valid:
            all_errors.extend([f"Stock levels: {e}" for e in errors])
        
        # Validate demands
        is_valid, errors = InputValidator.validate_delivery_demands(delivery_demands)
        if not is_valid:
            all_errors.extend([f"Delivery demands: {e}" for e in errors])
        
        # Check consistency
        if len(target_locations) != len(delivery_demands):
            all_errors.append("Number of target locations must match number of delivery demands")
        
        if len(vehicle_capacities) != len(initial_stock):
            all_errors.append("Number of vehicles must match number of stock levels")
        
        return len(all_errors) == 0, all_errors
    
    @staticmethod
    def validate_solution(solution: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a solution dictionary.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        required_keys = ['total_cost', 'vehicles_used', 'vehicle_routes']
        for key in required_keys:
            if key not in solution:
                errors.append(f"Solution missing required key: {key}")
        
        if 'total_cost' in solution:
            if not isinstance(solution['total_cost'], (int, float)):
                errors.append("Total cost must be a number")
            elif solution['total_cost'] < 0:
                errors.append("Total cost must be non-negative")
        
        if 'vehicles_used' in solution:
            if not isinstance(solution['vehicles_used'], int):
                errors.append("Vehicles used must be an integer")
            elif solution['vehicles_used'] < 0:
                errors.append("Vehicles used must be non-negative")
        
        if 'vehicle_routes' in solution:
            if not isinstance(solution['vehicle_routes'], dict):
                errors.append("Vehicle routes must be a dictionary")
            else:
                for vehicle_id, route in solution['vehicle_routes'].items():
                    if not isinstance(route, list):
                        errors.append(f"Route for vehicle {vehicle_id} must be a list")
                    elif not all(isinstance(loc, int) for loc in route):
                        errors.append(f"Route for vehicle {vehicle_id} must contain integers")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def warn_about_potential_issues(offsets: np.ndarray, edges: np.ndarray, 
                                  weights: np.ndarray, vehicle_capacities: List[int],
                                  delivery_demands: List[int]) -> List[str]:
        """
        Warn about potential issues that might affect optimization.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            weights: Edge weights
            vehicle_capacities: Vehicle capacities
            delivery_demands: Delivery demands
            
        Returns:
            List of warning messages
        """
        warnings_list = []
        
        # Check if total demand exceeds total capacity
        total_demand = sum(delivery_demands)
        total_capacity = sum(vehicle_capacities)
        if total_demand > total_capacity:
            warnings_list.append(f"Total demand ({total_demand}) exceeds total capacity ({total_capacity})")
        
        # Check if any single demand exceeds any vehicle capacity
        max_demand = max(delivery_demands)
        max_capacity = max(vehicle_capacities)
        if max_demand > max_capacity:
            warnings_list.append(f"Maximum demand ({max_demand}) exceeds maximum vehicle capacity ({max_capacity})")
        
        # Check graph connectivity
        try:
            from .graph_utils import GraphUtils
            if not GraphUtils.validate_graph_connectivity(offsets, edges, weights):
                warnings_list.append("Graph is not connected - some locations may be unreachable")
        except ImportError:
            pass
        
        # Check for very large weights
        if np.any(weights > 1000):
            warnings_list.append("Some edge weights are very large - this may affect optimization")
        
        return warnings_list
