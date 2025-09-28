"""
Cost calculation utilities for the Optimus routing system.

This module provides comprehensive cost calculation functions for all
delivery strategies including direct delivery, refill operations, and split deliveries.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from .vehicle import VehicleState
from .delivery_task import DeliveryTask
from .depot_manager import DepotManager


@dataclass
class DeliveryOption:
    """
    Represents a delivery option with cost analysis.
    
    Attributes:
        strategy: Delivery strategy ('direct', 'refill', 'new_vehicle', 'split')
        vehicles: List of vehicle IDs involved
        cost: Total cost for this option
        details: Additional details about the option
    """
    
    strategy: str
    vehicles: List[int]
    cost: float
    details: Dict[str, Any]


class CostCalculator:
    """
    Calculates costs for various delivery strategies and options.
    
    This class provides methods to calculate costs for different delivery
    strategies and compare them to find the optimal solution.
    """
    
    def __init__(self, depot_manager: DepotManager):
        """
        Initialize the cost calculator.
        
        Args:
            depot_manager: Depot manager instance for distance calculations
        """
        self.depot_manager = depot_manager
    
    def calculate_direct_delivery_cost(self, vehicle: VehicleState, task: DeliveryTask) -> float:
        """
        Calculate cost of direct delivery without refill.
        
        Args:
            vehicle: Vehicle performing the delivery
            task: Delivery task
            
        Returns:
            Total cost for direct delivery, infinity if not feasible
        """
        if not vehicle.can_deliver(task.demand):
            return float('inf')
        
        # Cost from current position to delivery location
        delivery_cost = self.depot_manager.get_distance(vehicle.position, task.location)
        
        # Cost to return to nearest depot
        return_depot, return_distance = self.depot_manager.find_nearest_depot_to_delivery(task.location)
        return_cost = return_distance
        
        return delivery_cost + return_cost
    
    def calculate_refill_cost(self, vehicle: VehicleState, task: DeliveryTask) -> float:
        """
        Calculate cost of refilling vehicle and delivering to task.
        
        Args:
            vehicle: Vehicle performing the delivery
            task: Delivery task
            
        Returns:
            Total cost for refill operation, infinity if not feasible
        """
        if not vehicle.can_carry(task.demand):
            return float('inf')
        
        return self.depot_manager.calculate_refill_cost(vehicle.position, task.location)
    
    def calculate_new_vehicle_cost(self, vehicles: List[VehicleState], task: DeliveryTask) -> float:
        """
        Calculate cost of using a fresh vehicle for delivery.
        
        Args:
            vehicles: List of available vehicles
            task: Delivery task
            
        Returns:
            Minimum cost for new vehicle delivery, infinity if not feasible
        """
        available_vehicles = [v for v in vehicles if v.can_deliver(task.demand)]
        
        if not available_vehicles:
            return float('inf')
        
        min_cost = float('inf')
        for vehicle in available_vehicles:
            # Cost from depot to delivery location
            delivery_cost = self.depot_manager.get_distance(vehicle.position, task.location)
            # Cost to return to nearest depot
            return_depot, return_distance = self.depot_manager.find_nearest_depot_to_delivery(task.location)
            return_cost = return_distance
            total_cost = delivery_cost + return_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
        
        return min_cost
    
    def calculate_split_delivery_cost(self, vehicles: List[VehicleState], 
                                    task: DeliveryTask) -> List[DeliveryOption]:
        """
        Calculate all possible split delivery options and their costs.
        
        Args:
            vehicles: List of available vehicles
            task: Delivery task
            
        Returns:
            List of valid split delivery options
        """
        split_options = []
        
        # Find all vehicles that can contribute to this delivery
        available_vehicles = [v for v in vehicles if v.capacity >= 1]
        
        if len(available_vehicles) < 2 or task.demand < 2:
            return []  # Need at least 2 vehicles and demand > 1 for split delivery
        
        # Generate all possible ways to split the demand
        demand = task.demand
        max_vehicles = min(len(available_vehicles), demand)
        
        for num_vehicles in range(2, max_vehicles + 1):
            # Generate all combinations of vehicles
            for vehicle_combo in self._generate_vehicle_combinations(available_vehicles, num_vehicles):
                # Generate all ways to split demand among these vehicles
                for split in self._generate_demand_splits(demand, num_vehicles):
                    if self._is_valid_split(vehicle_combo, split):
                        cost = self._calculate_split_cost(vehicle_combo, split, task)
                        if cost < float('inf'):
                            split_options.append(DeliveryOption(
                                strategy='split',
                                vehicles=[v.id for v in vehicle_combo],
                                cost=cost,
                                details={
                                    'vehicle_amounts': list(zip([v.id for v in vehicle_combo], split)),
                                    'total_vehicles': num_vehicles
                                }
                            ))
        
        return split_options
    
    def _generate_vehicle_combinations(self, vehicles: List[VehicleState], 
                                     num_vehicles: int) -> List[List[VehicleState]]:
        """Generate all combinations of vehicles."""
        import itertools
        return list(itertools.combinations(vehicles, num_vehicles))
    
    def _generate_demand_splits(self, demand: int, num_vehicles: int) -> List[List[int]]:
        """Generate all possible ways to split demand among vehicles."""
        if num_vehicles == 1:
            return [[demand]]
        
        splits = []
        for first_amount in range(1, demand - num_vehicles + 2):
            remaining = demand - first_amount
            if remaining > 0:
                for sub_split in self._generate_demand_splits(remaining, num_vehicles - 1):
                    splits.append([first_amount] + sub_split)
        
        return splits
    
    def _is_valid_split(self, vehicles: List[VehicleState], amounts: List[int]) -> bool:
        """Check if a split is valid (vehicles can handle their assigned amounts)."""
        for vehicle, amount in zip(vehicles, amounts):
            if amount > vehicle.capacity:
                return False
        return True
    
    def _calculate_split_cost(self, vehicles: List[VehicleState], amounts: List[int], 
                            task: DeliveryTask) -> float:
        """Calculate total cost for a split delivery."""
        total_cost = 0.0
        
        for vehicle, amount in zip(vehicles, amounts):
            # Check if vehicle needs refill
            if vehicle.current_stock < amount:
                # Need to refill
                nearest_depot, depot_distance = self.depot_manager.find_nearest_depot(vehicle.position)
                refill_cost = depot_distance
                
                # Cost from depot to delivery location
                delivery_cost = self.depot_manager.get_distance(nearest_depot, task.location)
                
                # Cost to return to nearest depot
                return_depot, return_distance = self.depot_manager.find_nearest_depot_to_delivery(task.location)
                return_cost = return_distance
                
                vehicle_cost = refill_cost + delivery_cost + return_cost
            else:
                # Direct delivery
                delivery_cost = self.depot_manager.get_distance(vehicle.position, task.location)
                return_depot, return_distance = self.depot_manager.find_nearest_depot_to_delivery(task.location)
                return_cost = return_distance
                
                vehicle_cost = delivery_cost + return_cost
            
            total_cost += vehicle_cost
        
        return total_cost
    
    def find_optimal_delivery_strategy(self, vehicles: List[VehicleState], 
                                     task: DeliveryTask) -> DeliveryOption:
        """
        Find the optimal delivery strategy considering all options.
        
        Args:
            vehicles: List of available vehicles
            task: Delivery task
            
        Returns:
            Optimal delivery option with minimal cost
            
        Raises:
            ValueError: If no feasible delivery option is found
        """
        all_options = []
        
        # Option 1: Direct delivery (if possible)
        for vehicle in vehicles:
            if vehicle.can_deliver(task.demand):
                cost = self.calculate_direct_delivery_cost(vehicle, task)
                if cost < float('inf'):
                    all_options.append(DeliveryOption(
                        strategy='direct',
                        vehicles=[vehicle.id],
                        cost=cost,
                        details={'vehicle_id': vehicle.id, 'amount': task.demand}
                    ))
        
        # Option 2: Refill and deliver
        for vehicle in vehicles:
            if vehicle.can_carry(task.demand):
                cost = self.calculate_refill_cost(vehicle, task)
                if cost < float('inf'):
                    all_options.append(DeliveryOption(
                        strategy='refill',
                        vehicles=[vehicle.id],
                        cost=cost,
                        details={'vehicle_id': vehicle.id, 'amount': task.demand}
                    ))
        
        # Option 3: New vehicle
        new_vehicle_cost = self.calculate_new_vehicle_cost(vehicles, task)
        if new_vehicle_cost < float('inf'):
            # Find the best available vehicle
            available_vehicles = [v for v in vehicles if v.can_deliver(task.demand)]
            if available_vehicles:
                all_options.append(DeliveryOption(
                    strategy='new_vehicle',
                    vehicles=[available_vehicles[0].id],
                    cost=new_vehicle_cost,
                    details={'vehicle_id': available_vehicles[0].id, 'amount': task.demand}
                ))
        
        # Option 4: Split delivery (if demand > 1)
        if task.demand > 1:
            split_options = self.calculate_split_delivery_cost(vehicles, task)
            all_options.extend(split_options)
        
        # Find the option with minimal cost
        if not all_options:
            raise ValueError(f"No feasible delivery option found for task at location {task.location}")
        
        best_option = min(all_options, key=lambda x: x.cost)
        return best_option
    
    def compare_strategies(self, vehicles: List[VehicleState], 
                          task: DeliveryTask) -> Dict[str, float]:
        """
        Compare costs of all delivery strategies for a task.
        
        Args:
            vehicles: List of available vehicles
            task: Delivery task
            
        Returns:
            Dictionary mapping strategy names to costs
        """
        costs = {}
        
        # Direct delivery costs
        direct_costs = []
        for vehicle in vehicles:
            if vehicle.can_deliver(task.demand):
                cost = self.calculate_direct_delivery_cost(vehicle, task)
                if cost < float('inf'):
                    direct_costs.append(cost)
        
        if direct_costs:
            costs['direct'] = min(direct_costs)
        
        # Refill costs
        refill_costs = []
        for vehicle in vehicles:
            if vehicle.can_carry(task.demand):
                cost = self.calculate_refill_cost(vehicle, task)
                if cost < float('inf'):
                    refill_costs.append(cost)
        
        if refill_costs:
            costs['refill'] = min(refill_costs)
        
        # New vehicle cost
        new_vehicle_cost = self.calculate_new_vehicle_cost(vehicles, task)
        if new_vehicle_cost < float('inf'):
            costs['new_vehicle'] = new_vehicle_cost
        
        # Split delivery costs
        if task.demand > 1:
            split_options = self.calculate_split_delivery_cost(vehicles, task)
            if split_options:
                costs['split'] = min(option.cost for option in split_options)
        
        return costs
