#!/usr/bin/env python3
"""
Simple test for the visualization system.
"""

import numpy as np
from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager
from visualization.static_visualizer import create_static_visualization

def simple_test():
    """Simple test of the visualization system."""
    print("Creating simple test data...")
    
    # Create simple location system
    depot_ids = [0, 1]
    delivery_ids = [2, 3]
    
    # Create cost matrix
    cost_matrix = np.array([
        [0, 10, 5, 15],
        [10, 0, 12, 8],
        [5, 12, 0, 20],
        [15, 8, 20, 0]
    ])
    
    location_to_index = {0: 0, 1: 1, 2: 2, 3: 3}
    depot_manager = DepotManager(depot_ids, cost_matrix, location_to_index)
    
    # Create vehicles
    vehicles = [
        VehicleState(id=0, capacity=100, current_stock=50, position=0),
        VehicleState(id=1, capacity=120, current_stock=40, position=1)
    ]
    
    # Create tasks
    tasks = [
        DeliveryTask(location=2, demand=25),
        DeliveryTask(location=3, demand=30)
    ]
    
    # Create results
    results = {
        'total_cost': 50.0,
        'vehicles_used': 2,
        'strategy_counts': {'direct': 2, 'refill': 0, 'split': 0, 'new_vehicle': 0},
        'assignments': [
            {'task_id': 0, 'vehicles': [0], 'location': 2, 'demand': 25, 'strategy': 'direct', 'cost': 25.0, 'details': {}},
            {'task_id': 1, 'vehicles': [1], 'location': 3, 'demand': 30, 'strategy': 'direct', 'cost': 25.0, 'details': {}}
        ],
        'vehicle_routes': {0: [0, 2, 0], 1: [1, 3, 1]}
    }
    
    # Update vehicle routes
    vehicles[0].route = [0, 2, 0]
    vehicles[0].total_cost = 25.0
    vehicles[0].current_stock = 25
    
    vehicles[1].route = [1, 3, 1]
    vehicles[1].total_cost = 25.0
    vehicles[1].current_stock = 10
    
    print("Generating visualization...")
    try:
        filename = create_static_visualization(vehicles, tasks, depot_manager, results, "simple_test.png")
        print(f"SUCCESS! Created: {filename}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
