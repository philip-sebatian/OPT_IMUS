"""
Split delivery optimization algorithm for the Optimus routing system.

This module implements the EnhancedRefillSystem which includes split delivery
optimization to choose the minimal cost option for large deliveries.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import copy

from cuopt.distance_engine import WaypointMatrix
from cuopt.routing import DataModel, SolverSettings, Solve

from ..core.vehicle import VehicleState
from ..core.delivery_task import DeliveryTask
from ..core.depot_manager import DepotManager
from ..core.cost_calculator import CostCalculator, DeliveryOption


class EnhancedRefillSystem:
    """
    Enhanced refill system with closest depot selection and split delivery optimization.
    Considers all possible delivery strategies and chooses the minimal cost option.
    """
    
    def __init__(self, offsets: np.ndarray, edges: np.ndarray, weights: np.ndarray, 
                 time_to_travel: np.ndarray, target_locations: np.ndarray, 
                 depot_locations: List[int], vehicle_capacities: List[int], 
                 initial_stock: List[int], delivery_demands: List[int]):
        """
        Initialize the enhanced refill system.
        
        Args:
            offsets: Graph offsets for WaypointMatrix
            edges: Graph edges for WaypointMatrix  
            weights: Edge weights for WaypointMatrix
            time_to_travel: Travel times for edges
            target_locations: Delivery locations
            depot_locations: List of depot locations where vehicles can refill
            vehicle_capacities: Maximum capacity for each vehicle
            initial_stock: Initial stock level for each vehicle
            delivery_demands: Demand at each target location
        """
        self.offsets = offsets
        self.edges = edges
        self.weights = weights
        self.time_to_travel = time_to_travel
        self.target_locations = target_locations
        self.depot_locations = depot_locations
        self.vehicle_capacities = vehicle_capacities
        self.initial_stock = initial_stock
        self.delivery_demands = delivery_demands
        
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
        
        # Initialize core components
        self.depot_manager = DepotManager(self.depot_locations, self.cost_matrix, self.location_to_index)
        self.cost_calculator = CostCalculator(self.depot_manager)
        
        # Initialize vehicles
        self.vehicles = []
        for i, (capacity, stock) in enumerate(zip(vehicle_capacities, initial_stock)):
            self.vehicles.append(VehicleState(
                id=i, capacity=capacity, current_stock=stock, 
                position=depot_locations[0]  # Start at first depot
            ))
        
        # Initialize delivery tasks
        self.delivery_tasks = []
        for i, (location, demand) in enumerate(zip(target_locations, delivery_demands)):
            self.delivery_tasks.append(DeliveryTask(location=location, demand=demand))
    
    def get_distance(self, from_location: int, to_location: int) -> float:
        """Get distance between two locations"""
        return self.depot_manager.get_distance(from_location, to_location)
    
    def find_nearest_depot(self, current_location: int) -> Tuple[int, float]:
        """Find the nearest depot to the current location"""
        return self.depot_manager.find_nearest_depot(current_location)
    
    def find_optimal_delivery_strategy(self, task: DeliveryTask) -> DeliveryOption:
        """
        Find the optimal delivery strategy considering all options including split delivery.
        Returns the option with minimal cost.
        """
        return self.cost_calculator.find_optimal_delivery_strategy(self.vehicles, task)
    
    def execute_delivery(self, option: DeliveryOption, task: DeliveryTask) -> float:
        """Execute the delivery using the specified option"""
        total_cost = 0.0
        
        if option.strategy == 'split':
            # Handle split delivery
            for vehicle_id, amount in option.details['vehicle_amounts']:
                vehicle = self.vehicles[vehicle_id]
                
                if vehicle.current_stock < amount:
                    # Need to refill
                    nearest_depot, depot_distance = self.find_nearest_depot(vehicle.position)
                    vehicle.move_to(nearest_depot)
                    vehicle.refill()
                    vehicle.add_cost(depot_distance)
                    total_cost += depot_distance
                    
                    print(f"  🔄 Vehicle {vehicle_id} refilled at depot {nearest_depot} (cost: {depot_distance:.1f})")
                
                # Deliver partial amount
                delivery_cost = self.get_distance(vehicle.position, task.location)
                vehicle.deliver(amount)
                vehicle.move_to(task.location)
                vehicle.add_cost(delivery_cost)
                total_cost += delivery_cost
                
                print(f"  📦 Vehicle {vehicle_id} delivered {amount} units to {task.location} (cost: {delivery_cost:.1f})")
                
                # Return to nearest depot
                return_depot, return_distance = self.find_nearest_depot(task.location)
                vehicle.move_to(return_depot)
                vehicle.add_cost(return_distance)
                total_cost += return_distance
                
        else:
            # Handle single vehicle delivery
            vehicle_id = option.vehicles[0]
            vehicle = self.vehicles[vehicle_id]
            
            if option.strategy == 'direct':
                # Direct delivery
                delivery_cost = self.get_distance(vehicle.position, task.location)
                vehicle.deliver(task.demand)
                vehicle.move_to(task.location)
                vehicle.add_cost(delivery_cost)
                total_cost = delivery_cost
                
                print(f"  📦 Vehicle {vehicle_id} delivered directly to {task.location} (cost: {delivery_cost:.1f})")
                
            elif option.strategy == 'refill':
                # Refill and deliver
                nearest_depot, depot_distance = self.find_nearest_depot(vehicle.position)
                
                # Go to depot
                vehicle.move_to(nearest_depot)
                vehicle.refill()
                vehicle.add_cost(depot_distance)
                total_cost += depot_distance
                
                # Deliver to location
                delivery_cost = self.get_distance(nearest_depot, task.location)
                vehicle.deliver(task.demand)
                vehicle.move_to(task.location)
                vehicle.add_cost(delivery_cost)
                total_cost += delivery_cost
                
                print(f"  🔄 Vehicle {vehicle_id} refilled at depot {nearest_depot} (cost: {depot_distance:.1f})")
                print(f"  📦 Vehicle {vehicle_id} delivered to {task.location} (cost: {delivery_cost:.1f})")
                
            elif option.strategy == 'new_vehicle':
                # Use fresh vehicle
                delivery_cost = self.get_distance(vehicle.position, task.location)
                vehicle.deliver(task.demand)
                vehicle.move_to(task.location)
                vehicle.add_cost(delivery_cost)
                total_cost = delivery_cost
                
                print(f"  🚚 Vehicle {vehicle_id} (fresh) delivered to {task.location} (cost: {delivery_cost:.1f})")
        
        task.assigned_vehicle = option.vehicles[0] if len(option.vehicles) == 1 else option.vehicles
        task.completed = True
        return total_cost
    
    def solve_enhanced(self) -> Dict:
        """
        Solve the routing problem with enhanced optimization including split delivery.
        """
        print("🚀 Starting Enhanced PDP with Refill System")
        print(f"   Stores: {self.target_locations}")
        print(f"   Depots: {self.depot_locations}")
        print(f"   Vehicles: {len(self.vehicles)} with capacities {self.vehicle_capacities}")
        print(f"   Initial stock: {self.initial_stock}")
        print(f"   Demands: {self.delivery_demands}")
        
        total_cost = 0.0
        assignments = []
        
        # Process each delivery task
        for i, task in enumerate(self.delivery_tasks):
            print(f"\n🎯 Processing Task {i+1}: Deliver {task.demand} units to location {task.location}")
            
            # Find optimal delivery strategy
            try:
                optimal_option = self.find_optimal_delivery_strategy(task)
                print(f"  ✅ Best strategy: {optimal_option.strategy} using vehicles {optimal_option.vehicles} (cost: {optimal_option.cost:.1f})")
                
                # Execute the delivery
                actual_cost = self.execute_delivery(optimal_option, task)
                total_cost += actual_cost
                
                assignments.append({
                    'task_id': i,
                    'strategy': optimal_option.strategy,
                    'vehicles': optimal_option.vehicles,
                    'cost': actual_cost,
                    'location': task.location,
                    'demand': task.demand,
                    'details': optimal_option.details
                })
                
            except ValueError as e:
                print(f"  ❌ Error: {e}")
                continue
        
        # Calculate final statistics
        vehicles_used = set()
        for assignment in assignments:
            if isinstance(assignment['vehicles'], list):
                vehicles_used.update(assignment['vehicles'])
            else:
                vehicles_used.add(assignment['vehicles'])
        
        strategy_counts = {}
        for assignment in assignments:
            strategy = assignment['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        print(f"\n📊 ENHANCED OPTIMIZATION RESULTS")
        print(f"   💰 Total cost: {total_cost:.1f}")
        print(f"   🚚 Vehicles used: {len(vehicles_used)}")
        print(f"   📊 Strategy breakdown: {strategy_counts}")
        
        print(f"\n🚚 VEHICLE ROUTES:")
        for vehicle in self.vehicles:
            if vehicle.total_cost > 0:
                print(f"   Vehicle {vehicle.id}: {vehicle.route} (cost: {vehicle.total_cost:.1f}, stock: {vehicle.current_stock})")
        
        return {
            'total_cost': total_cost,
            'assignments': assignments,
            'vehicles_used': len(vehicles_used),
            'strategy_counts': strategy_counts,
            'vehicle_routes': {v.id: v.route for v in self.vehicles if v.total_cost > 0}
        }
