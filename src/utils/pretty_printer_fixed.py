"""
Fixed pretty printer module for Optimus routing system results.

This module provides functions to interpret and display the results of the
Optimus optimization systems in a human-readable format, focusing on accuracy
rather than step-by-step reconstruction.
"""

from typing import Dict, Any, List, Tuple
from ..core.vehicle import VehicleState
from ..core.delivery_task import DeliveryTask
from ..core.depot_manager import DepotManager

class PrettyPrinter:
    """
    A utility class to pretty print the results of the Optimus routing system.
    """

    def __init__(self, vehicles: List[VehicleState], delivery_tasks: List[DeliveryTask], depot_manager: DepotManager):
        self.vehicles = {v.id: v for v in vehicles}
        self.delivery_tasks = {i: task for i, task in enumerate(delivery_tasks)}
        self.depot_manager = depot_manager

    def print_vehicle_process(self, results: Dict[str, Any]) -> None:
        """
        Prints a detailed process for each vehicle involved in the optimization.

        Args:
            results: The dictionary containing the optimization results,
                     including 'assignments' and 'vehicle_routes'.
        """
        print("\n--- Vehicle Process Breakdown ---")
        
        vehicle_assignments: Dict[int, List[Dict[str, Any]]] = {}
        for assignment in results.get('assignments', []):
            vehicle_ids = assignment['vehicles'] if isinstance(assignment['vehicles'], list) else [assignment['vehicles']]
            for v_id in vehicle_ids:
                if v_id not in vehicle_assignments:
                    vehicle_assignments[v_id] = []
                vehicle_assignments[v_id].append(assignment)

        for vehicle_id, assignments in vehicle_assignments.items():
            vehicle = self.vehicles.get(vehicle_id)
            if not vehicle:
                print(f"Error: Vehicle {vehicle_id} not found.")
                continue

            print(f"\n--- Vehicle {vehicle.id} (Capacity: {vehicle.capacity}) ---")
            
            # Show the actual route taken
            if vehicle.route:
                print(f"  Route taken: {' -> '.join(map(str, vehicle.route))}")
            else:
                print(f"  Current position: {vehicle.position}")
            
            # Show assignments for this vehicle
            print(f"\n  Assignments for this vehicle:")
            for assignment in assignments:
                task_id = assignment['task_id']
                strategy = assignment['strategy']
                task_location = assignment['location']
                task_demand = assignment['demand']
                assignment_cost = assignment['cost']
                
                print(f"    Task {task_id}: {strategy} delivery of {task_demand} units to location {task_location}")
                
                if strategy == 'split':
                    # Find the specific amount delivered by this vehicle in the split
                    vehicle_amount_in_split = 0
                    for v_id, amount in assignment['details']['vehicle_amounts']:
                        if v_id == vehicle.id:
                            vehicle_amount_in_split = amount
                            break
                    
                    if vehicle_amount_in_split > 0:
                        print(f"      -> This vehicle delivered {vehicle_amount_in_split} units (part of split)")
                    else:
                        print(f"      -> Error: Vehicle {vehicle.id} not found in split details")
                else:
                    print(f"      -> This vehicle delivered {task_demand} units")
            
            # Show final state
            print(f"\n  Final state for Vehicle {vehicle.id}:")
            print(f"    Route: {vehicle.route}")
            print(f"    Total cost incurred: {vehicle.total_cost:.1f}")
            print(f"    Final stock: {vehicle.current_stock}")
            print(f"    Final position: {vehicle.position}")

    def print_overall_results(self, results: Dict[str, Any]) -> None:
        """
        Prints the overall optimization results.

        Args:
            results: The dictionary containing the optimization results.
        """
        print("\n--- Overall Optimization Results ---")
        print(f"Total cost: {results.get('total_cost', 0.0):.1f}")
        print(f"Vehicles used: {results.get('vehicles_used', 0)}")
        print(f"Strategy breakdown: {results.get('strategy_counts', {})}")
        print("------------------------------------")
