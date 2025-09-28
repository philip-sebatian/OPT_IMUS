"""
Refill optimization algorithm for the Optimus routing system.

This module implements the OptimizedRefillSystem which provides intelligent
refill planning and vehicle assignment for delivery tasks.
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


class OptimizedRefillSystem:
    """
    Optimized refill system that considers multiple vehicles, depots, and stores.
    Finds the cheapest possible solution by comparing refill vs new vehicle assignment.
    """
    
    def __init__(self, offsets: np.ndarray, edges: np.ndarray, weights: np.ndarray, 
                 time_to_travel: np.ndarray, target_locations: np.ndarray, 
                 depot_locations: List[int], vehicle_capacities: List[int], 
                 initial_stock: List[int], delivery_demands: List[int]):
        """
        Initialize the optimized refill system.
        
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
    
    def calculate_direct_delivery_cost(self, vehicle: VehicleState, task: DeliveryTask) -> float:
        """Calculate cost of direct delivery without refill"""
        return self.cost_calculator.calculate_direct_delivery_cost(vehicle, task)
    
    def calculate_refill_cost(self, vehicle: VehicleState, task: DeliveryTask) -> float:
        """Calculate cost of refilling vehicle and delivering to task"""
        return self.cost_calculator.calculate_refill_cost(vehicle, task)
    
    def calculate_new_vehicle_cost(self, task: DeliveryTask) -> float:
        """Calculate cost of using a fresh vehicle from depot"""
        return self.cost_calculator.calculate_new_vehicle_cost(self.vehicles, task)
    
    def assign_optimal_vehicle(self, task: DeliveryTask) -> Tuple[int, str, float]:
        """
        Find the optimal vehicle assignment for a task.
        Returns (vehicle_id, strategy, cost)
        Strategies: 'direct', 'refill', 'new_vehicle'
        """
        best_cost = float('inf')
        best_vehicle_id = None
        best_strategy = None
        
        # Strategy 1: Direct delivery (if possible)
        for vehicle in self.vehicles:
            if vehicle.can_deliver(task.demand):
                cost = self.calculate_direct_delivery_cost(vehicle, task)
                if cost < best_cost:
                    best_cost = cost
                    best_vehicle_id = vehicle.id
                    best_strategy = 'direct'
        
        # Strategy 2: Refill and deliver
        for vehicle in self.vehicles:
            if vehicle.can_carry(task.demand):
                cost = self.calculate_refill_cost(vehicle, task)
                if cost < best_cost:
                    best_cost = cost
                    best_vehicle_id = vehicle.id
                    best_strategy = 'refill'
        
        # Strategy 3: Use a new vehicle (if available)
        new_vehicle_cost = self.calculate_new_vehicle_cost(task)
        if new_vehicle_cost < best_cost:
            # Find the best available vehicle
            available_vehicles = [v for v in self.vehicles if v.can_deliver(task.demand)]
            if available_vehicles:
                best_vehicle_id = available_vehicles[0].id
                best_strategy = 'new_vehicle'
                best_cost = new_vehicle_cost
        
        return best_vehicle_id, best_strategy, best_cost
    
    def execute_delivery(self, vehicle_id: int, task: DeliveryTask, strategy: str) -> float:
        """Execute the delivery using the specified strategy"""
        vehicle = self.vehicles[vehicle_id]
        cost = 0.0
        
        if strategy == 'direct':
            # Direct delivery
            delivery_cost = self.get_distance(vehicle.position, task.location)
            vehicle.deliver(task.demand)
            vehicle.move_to(task.location)
            vehicle.add_cost(delivery_cost)
            cost = delivery_cost
            
            print(f"  📦 Vehicle {vehicle_id} delivered directly to {task.location} (cost: {delivery_cost:.1f})")
            
        elif strategy == 'refill':
            # Refill and deliver
            nearest_depot, depot_distance = self.find_nearest_depot(vehicle.position)
            
            # Go to depot
            vehicle.move_to(nearest_depot)
            vehicle.refill()
            vehicle.add_cost(depot_distance)
            cost += depot_distance
            
            # Deliver to location
            delivery_cost = self.get_distance(nearest_depot, task.location)
            vehicle.deliver(task.demand)
            vehicle.move_to(task.location)
            vehicle.add_cost(delivery_cost)
            cost += delivery_cost
            
            print(f"  🔄 Vehicle {vehicle_id} refilled at depot {nearest_depot} (cost: {depot_distance:.1f})")
            print(f"  📦 Vehicle {vehicle_id} delivered to {task.location} (cost: {delivery_cost:.1f})")
            
        elif strategy == 'new_vehicle':
            # Use fresh vehicle
            delivery_cost = self.get_distance(vehicle.position, task.location)
            vehicle.deliver(task.demand)
            vehicle.move_to(task.location)
            vehicle.add_cost(delivery_cost)
            cost = delivery_cost
            
            print(f"  🚚 Vehicle {vehicle_id} (fresh) delivered to {task.location} (cost: {delivery_cost:.1f})")
        
        task.assigned_vehicle = vehicle_id
        task.completed = True
        return cost
    
    def solve_optimized(self) -> Dict:
        """
        Solve the routing problem with optimal vehicle assignment and refill decisions.
        """
        print("🚀 Starting Optimized PDP with Refill System")
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
            
            # Find optimal assignment
            vehicle_id, strategy, cost = self.assign_optimal_vehicle(task)
            
            if vehicle_id is None:
                print(f"  ❌ No feasible assignment found for task {i+1}")
                continue
            
            print(f"  ✅ Best strategy: {strategy} using Vehicle {vehicle_id} (cost: {cost:.1f})")
            
            # Execute the delivery
            actual_cost = self.execute_delivery(vehicle_id, task, strategy)
            total_cost += actual_cost
            
            assignments.append({
                'task_id': i,
                'vehicle_id': vehicle_id,
                'strategy': strategy,
                'cost': actual_cost,
                'location': task.location,
                'demand': task.demand
            })
        
        # Calculate final statistics
        vehicles_used = set(assignment['vehicle_id'] for assignment in assignments)
        strategy_counts = {}
        for assignment in assignments:
            strategy = assignment['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        print(f"\n📊 OPTIMIZATION RESULTS")
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
