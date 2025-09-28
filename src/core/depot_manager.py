"""
Depot management for the Optimus routing system.

This module handles depot selection, distance calculations, and depot-related
operations to ensure vehicles always go to the closest depot for refilling.
"""

from typing import List, Tuple, Optional
import numpy as np


class DepotManager:
    """
    Manages depot operations including closest depot selection and distance calculations.
    
    This class ensures that vehicles always go to the closest depot for refilling
    operations, optimizing travel costs and time.
    """
    
    def __init__(self, depot_locations: List[int], cost_matrix: np.ndarray, 
                 location_to_index: dict):
        """
        Initialize the depot manager.
        
        Args:
            depot_locations: List of depot location IDs
            cost_matrix: Cost matrix for all locations
            location_to_index: Mapping from location ID to matrix index
        """
        self.depot_locations = depot_locations
        self.cost_matrix = cost_matrix
        self.location_to_index = location_to_index
        
        # Validate depot locations
        self._validate_depots()
    
    def _validate_depots(self) -> None:
        """Validate that all depot locations are accessible."""
        for depot in self.depot_locations:
            if depot not in self.location_to_index:
                raise ValueError(f"Depot {depot} not found in location mapping")
    
    def get_distance(self, from_location: int, to_location: int) -> float:
        """
        Get distance between two locations.
        
        Args:
            from_location: Source location ID
            to_location: Destination location ID
            
        Returns:
            Distance between locations, infinity if not accessible
        """
        if (from_location not in self.location_to_index or 
            to_location not in self.location_to_index):
            return float('inf')
        
        from_idx = self.location_to_index[from_location]
        to_idx = self.location_to_index[to_location]
        return self.cost_matrix[from_idx][to_idx]
    
    def find_nearest_depot(self, current_location: int) -> Tuple[int, float]:
        """
        Find the nearest depot to the current location.
        
        Args:
            current_location: Current location ID
            
        Returns:
            Tuple of (nearest_depot_id, distance_to_depot)
            
        Raises:
            ValueError: If no accessible depot is found
        """
        if current_location not in self.location_to_index:
            raise ValueError(f"Location {current_location} not found in location mapping")
        
        min_distance = float('inf')
        nearest_depot = None
        
        for depot in self.depot_locations:
            distance = self.get_distance(current_location, depot)
            if distance < min_distance:
                min_distance = distance
                nearest_depot = depot
        
        if nearest_depot is None:
            raise ValueError(f"No accessible depot found from location {current_location}")
        
        return nearest_depot, min_distance
    
    def find_nearest_depot_to_delivery(self, delivery_location: int) -> Tuple[int, float]:
        """
        Find the nearest depot to a delivery location (for return trips).
        
        Args:
            delivery_location: Delivery location ID
            
        Returns:
            Tuple of (nearest_depot_id, distance_to_depot)
        """
        return self.find_nearest_depot(delivery_location)
    
    def get_all_depot_distances(self, from_location: int) -> List[Tuple[int, float]]:
        """
        Get distances to all depots from a location.
        
        Args:
            from_location: Source location ID
            
        Returns:
            List of (depot_id, distance) tuples sorted by distance
        """
        distances = []
        for depot in self.depot_locations:
            distance = self.get_distance(from_location, depot)
            distances.append((depot, distance))
        
        return sorted(distances, key=lambda x: x[1])
    
    def is_depot(self, location: int) -> bool:
        """
        Check if a location is a depot.
        
        Args:
            location: Location ID to check
            
        Returns:
            True if the location is a depot, False otherwise
        """
        return location in self.depot_locations
    
    def get_depot_count(self) -> int:
        """
        Get the number of available depots.
        
        Returns:
            Number of depots
        """
        return len(self.depot_locations)
    
    def get_depot_locations(self) -> List[int]:
        """
        Get list of all depot locations.
        
        Returns:
            List of depot location IDs
        """
        return self.depot_locations.copy()
    
    def calculate_refill_cost(self, from_location: int, to_delivery: int) -> float:
        """
        Calculate total cost for refill operation.
        
        Args:
            from_location: Current vehicle location
            to_delivery: Delivery location
            
        Returns:
            Total cost for refill operation
        """
        # Cost to go to nearest depot
        nearest_depot, depot_distance = self.find_nearest_depot(from_location)
        
        # Cost from depot to delivery location
        delivery_cost = self.get_distance(nearest_depot, to_delivery)
        
        # Cost to return to nearest depot from delivery location
        return_depot, return_distance = self.find_nearest_depot_to_delivery(to_delivery)
        
        return depot_distance + delivery_cost + return_distance
    
    def get_optimal_depot_sequence(self, locations: List[int]) -> List[int]:
        """
        Get optimal depot sequence for visiting multiple locations.
        
        Args:
            locations: List of locations to visit
            
        Returns:
            List of depot IDs in optimal order
        """
        if not locations:
            return []
        
        # For now, return nearest depot for each location
        # This could be enhanced with more sophisticated routing
        depot_sequence = []
        for location in locations:
            nearest_depot, _ = self.find_nearest_depot(location)
            depot_sequence.append(nearest_depot)
        
        return depot_sequence
    
    def __str__(self) -> str:
        """String representation of the depot manager."""
        return f"DepotManager(depots={self.depot_locations}, count={len(self.depot_locations)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the depot manager."""
        return (f"DepotManager(depot_locations={self.depot_locations}, "
                f"cost_matrix_shape={self.cost_matrix.shape})")
