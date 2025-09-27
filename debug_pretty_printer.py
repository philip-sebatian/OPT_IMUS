#!/usr/bin/env python3
"""
Debug script to test pretty printer accuracy with simple, controlled data.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager
from src.utils.pretty_printer import PrettyPrinter

def create_simple_test():
    """Create a simple test case with known values for debugging."""
    print("=== Creating Simple Test Case ===")
    
    # Create simple location system
    depot_ids = [0, 1]  # Two depots
    delivery_ids = [2, 3]  # Two delivery locations
    
    # Create cost matrix (simple distances)
    cost_matrix = np.array([
        [0, 10, 5, 15],   # From depot 0
        [10, 0, 12, 8],   # From depot 1  
        [5, 12, 0, 20],   # From delivery 2
        [15, 8, 20, 0]    # From delivery 3
    ])
    
    location_to_index = {0: 0, 1: 1, 2: 2, 3: 3}
    
    depot_manager = DepotManager(depot_ids, cost_matrix, location_to_index)
    
    # Create one vehicle starting at depot 0
    vehicle = VehicleState(
        id=0,
        capacity=100,
        current_stock=50,
        position=0  # Start at depot 0
    )
    
    # Create one delivery task
    task = DeliveryTask(
        location=2,  # Delivery location 2
        demand=30
    )
    
    # Create simple assignment
    assignment = {
        'task_id': 0,
        'vehicles': [0],
        'location': 2,
        'demand': 30,
        'strategy': 'direct',
        'cost': 10.0,  # 5 to delivery + 5 back to depot
        'details': {}
    }
    
    # Update vehicle state as if the assignment was executed
    vehicle.route = [0, 2, 0]  # Start at depot 0, go to delivery 2, return to depot 0
    vehicle.total_cost = 10.0
    vehicle.current_stock = 20  # 50 - 30 = 20
    vehicle.position = 0  # Back at depot 0
    
    results = {
        'total_cost': 10.0,
        'vehicles_used': 1,
        'strategy_counts': {'direct': 1, 'refill': 0, 'split': 0, 'new_vehicle': 0},
        'assignments': [assignment],
        'vehicle_routes': {0: [0, 2, 0]}
    }
    
    print(f"Vehicle initial state: {vehicle}")
    print(f"Task: {task}")
    print(f"Assignment: {assignment}")
    print(f"Expected: Vehicle should have 20 stock, cost 10.0, position 0")
    
    # Test pretty printer
    printer = PrettyPrinter([vehicle], [task], depot_manager)
    printer.print_vehicle_process(results)
    printer.print_overall_results(results)
    
    print(f"\nVehicle final state: {vehicle}")
    print(f"Expected final state: stock=20, cost=10.0, position=0")

if __name__ == "__main__":
    create_simple_test()
