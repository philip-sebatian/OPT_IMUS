"""
Distance calculation utilities for the Optimus routing system.

This module provides utilities for calculating distances between locations,
including Euclidean distance, Manhattan distance, and custom distance functions.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
import math


class DistanceCalculator:
    """
    Utility class for distance calculations.
    """
    
    @staticmethod
    def euclidean_distance(point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def manhattan_distance(point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        Calculate Manhattan distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Manhattan distance
        """
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    
    @staticmethod
    def create_distance_matrix(coordinates: List[Tuple[float, float]], 
                             distance_func: Callable = None) -> np.ndarray:
        """
        Create a distance matrix from coordinates.
        
        Args:
            coordinates: List of (x, y) coordinates
            distance_func: Distance function to use (default: Euclidean)
            
        Returns:
            Distance matrix
        """
        if distance_func is None:
            distance_func = DistanceCalculator.euclidean_distance
        
        n = len(coordinates)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = distance_func(coordinates[i], coordinates[j])
        
        return matrix
    
    @staticmethod
    def calculate_route_distance(route: List[int], distance_matrix: np.ndarray) -> float:
        """
        Calculate total distance for a route.
        
        Args:
            route: List of location indices
            distance_matrix: Distance matrix
            
        Returns:
            Total route distance
        """
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            from_idx = route[i]
            to_idx = route[i + 1]
            total_distance += distance_matrix[from_idx][to_idx]
        
        return total_distance
    
    @staticmethod
    def find_nearest_location(target_location: int, candidate_locations: List[int],
                            distance_matrix: np.ndarray) -> Tuple[int, float]:
        """
        Find the nearest location from a list of candidates.
        
        Args:
            target_location: Target location index
            candidate_locations: List of candidate location indices
            distance_matrix: Distance matrix
            
        Returns:
            Tuple of (nearest_location, distance)
        """
        min_distance = float('inf')
        nearest_location = None
        
        for candidate in candidate_locations:
            distance = distance_matrix[target_location][candidate]
            if distance < min_distance:
                min_distance = distance
                nearest_location = candidate
        
        return nearest_location, min_distance
    
    @staticmethod
    def calculate_centroid(coordinates: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate the centroid of a set of coordinates.
        
        Args:
            coordinates: List of (x, y) coordinates
            
        Returns:
            Centroid coordinates (x, y)
        """
        if not coordinates:
            return (0.0, 0.0)
        
        x_sum = sum(coord[0] for coord in coordinates)
        y_sum = sum(coord[1] for coord in coordinates)
        n = len(coordinates)
        
        return (x_sum / n, y_sum / n)
    
    @staticmethod
    def calculate_route_centroid(route: List[int], coordinates: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate the centroid of a route.
        
        Args:
            route: List of location indices
            coordinates: List of (x, y) coordinates
            
        Returns:
            Centroid coordinates (x, y)
        """
        route_coordinates = [coordinates[i] for i in route if i < len(coordinates)]
        return DistanceCalculator.calculate_centroid(route_coordinates)
    
    @staticmethod
    def calculate_route_span(route: List[int], coordinates: List[Tuple[float, float]]) -> float:
        """
        Calculate the span (maximum distance between any two points) of a route.
        
        Args:
            route: List of location indices
            coordinates: List of (x, y) coordinates
            
        Returns:
            Route span
        """
        if len(route) < 2:
            return 0.0
        
        route_coordinates = [coordinates[i] for i in route if i < len(coordinates)]
        max_distance = 0.0
        
        for i in range(len(route_coordinates)):
            for j in range(i + 1, len(route_coordinates)):
                distance = DistanceCalculator.euclidean_distance(
                    route_coordinates[i], route_coordinates[j]
                )
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    @staticmethod
    def optimize_route_order(locations: List[int], distance_matrix: np.ndarray,
                           start_location: Optional[int] = None) -> List[int]:
        """
        Optimize the order of locations using nearest neighbor heuristic.
        
        Args:
            locations: List of location indices to visit
            distance_matrix: Distance matrix
            start_location: Starting location (if None, use first location)
            
        Returns:
            Optimized route order
        """
        if not locations:
            return []
        
        if start_location is None:
            start_location = locations[0]
        
        if start_location not in locations:
            locations = [start_location] + locations
        
        optimized = [start_location]
        remaining = [loc for loc in locations if loc != start_location]
        
        while remaining:
            current = optimized[-1]
            nearest = min(remaining, key=lambda x: distance_matrix[current][x])
            optimized.append(nearest)
            remaining.remove(nearest)
        
        return optimized
    
    @staticmethod
    def calculate_route_efficiency(route: List[int], distance_matrix: np.ndarray) -> float:
        """
        Calculate the efficiency of a route (actual distance / straight-line distance).
        
        Args:
            route: List of location indices
            distance_matrix: Distance matrix
            
        Returns:
            Route efficiency (0.0 to 1.0, higher is better)
        """
        if len(route) < 2:
            return 1.0
        
        # Calculate actual route distance
        actual_distance = DistanceCalculator.calculate_route_distance(route, distance_matrix)
        
        # Calculate straight-line distance from start to end
        start_idx = route[0]
        end_idx = route[-1]
        straight_distance = distance_matrix[start_idx][end_idx]
        
        if straight_distance == 0:
            return 1.0
        
        return straight_distance / actual_distance
    
    @staticmethod
    def validate_distance_matrix(matrix: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate a distance matrix for correctness.
        
        Args:
            matrix: Distance matrix
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            errors.append("Distance matrix must be square")
            return False, errors
        
        # Check if diagonal is zero
        if not np.allclose(np.diag(matrix), 0):
            errors.append("Diagonal elements must be zero")
        
        # Check if matrix is symmetric
        if not np.allclose(matrix, matrix.T):
            errors.append("Distance matrix must be symmetric")
        
        # Check if all elements are non-negative
        if np.any(matrix < 0):
            errors.append("All distances must be non-negative")
        
        return len(errors) == 0, errors
