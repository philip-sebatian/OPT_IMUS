#!/usr/bin/env python3
"""
Test script for distance-proportional visualization.
"""

import numpy as np
from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager
from visualization.static_visualizer import create_static_visualization

def create_distance_test_data():
    """Create test data with realistic distances for proportional visualization."""
    print("Creating test data with realistic distances...")
    
    # 1. Create location system with realistic coordinates
    depot_ids = [0, 1, 2]
    delivery_ids = [3, 4, 5, 6, 7]
    
    # Create realistic cost matrix based on actual coordinates
    # Let's simulate a city-like layout
    locations = {
        0: (0, 0),      # Depot-0: City center
        1: (50, 0),     # Depot-1: East side
        2: (25, 50),    # Depot-2: North side
        3: (10, 10),    # Store-3: Near center
        4: (40, 15),    # Store-4: East area
        5: (15, 30),    # Store-5: North area
        6: (45, 35),    # Store-6: Northeast area
        7: (5, 45),     # Store-7: Far north
    }
    
    # Create cost matrix based on Euclidean distances
    num_locations = len(locations)
    cost_matrix = np.zeros((num_locations, num_locations))
    location_to_index = {i: i for i in range(num_locations)}
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                pos1 = locations[i]
                pos2 = locations[j]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                cost_matrix[i][j] = distance
            else:
                cost_matrix[i][j] = 0.0
    
    depot_manager = DepotManager(depot_ids, cost_matrix, location_to_index)
    
    # 2. Create vehicles
    vehicles = [
        VehicleState(id=0, capacity=100, current_stock=60, position=0),
        VehicleState(id=1, capacity=120, current_stock=40, position=1),
        VehicleState(id=2, capacity=80, current_stock=50, position=2)
    ]
    
    # 3. Create delivery tasks
    tasks = [
        DeliveryTask(location=3, demand=25),
        DeliveryTask(location=4, demand=30),
        DeliveryTask(location=5, demand=20),
        DeliveryTask(location=6, demand=35),
        DeliveryTask(location=7, demand=15)
    ]
    
    # 4. Create results with realistic routes
    results = {
        'total_cost': 150.0,
        'vehicles_used': 3,
        'strategy_counts': {'direct': 4, 'refill': 1, 'split': 0, 'new_vehicle': 0},
        'assignments': [
            {'task_id': 0, 'vehicles': [0], 'location': 3, 'demand': 25, 'strategy': 'direct', 'cost': 30.0, 'details': {}},
            {'task_id': 1, 'vehicles': [1], 'location': 4, 'demand': 30, 'strategy': 'direct', 'cost': 25.0, 'details': {}},
            {'task_id': 2, 'vehicles': [2], 'location': 5, 'demand': 20, 'strategy': 'direct', 'cost': 35.0, 'details': {}},
            {'task_id': 3, 'vehicles': [0], 'location': 6, 'demand': 35, 'strategy': 'refill', 'cost': 40.0, 'details': {}},
            {'task_id': 4, 'vehicles': [1], 'location': 7, 'demand': 15, 'strategy': 'direct', 'cost': 20.0, 'details': {}}
        ],
        'vehicle_routes': {
            0: [0, 3, 0, 6, 0],  # Vehicle 0: Depot-0 -> Store-3 -> Depot-0 -> Store-6 -> Depot-0
            1: [1, 4, 1, 7, 1],  # Vehicle 1: Depot-1 -> Store-4 -> Depot-1 -> Store-7 -> Depot-1
            2: [2, 5, 2]         # Vehicle 2: Depot-2 -> Store-5 -> Depot-2
        }
    }
    
    # Update vehicle states
    vehicles[0].route = [0, 3, 0, 6, 0]
    vehicles[0].total_cost = 70.0
    vehicles[0].current_stock = 0  # Empty after deliveries
    
    vehicles[1].route = [1, 4, 1, 7, 1]
    vehicles[1].total_cost = 45.0
    vehicles[1].current_stock = 75  # 120 - 30 - 15 = 75
    
    vehicles[2].route = [2, 5, 2]
    vehicles[2].total_cost = 35.0
    vehicles[2].current_stock = 30  # 80 - 20 = 60, but let's say 30 for demo
    
    print(f"Created {len(vehicles)} vehicles and {len(tasks)} delivery tasks")
    print(f"Total cost: {results['total_cost']:.1f}")
    print(f"Strategy breakdown: {results['strategy_counts']}")
    
    # Print actual distances for verification
    print("\nActual distances between locations:")
    for i in range(num_locations):
        for j in range(i+1, num_locations):
            distance = cost_matrix[i][j]
            loc1_name = f"Depot-{i}" if i < 3 else f"Store-{i}"
            loc2_name = f"Depot-{j}" if j < 3 else f"Store-{j}"
            print(f"  {loc1_name} to {loc2_name}: {distance:.1f}")
    
    return vehicles, tasks, depot_manager, results

def test_distance_proportional_visualization():
    """Test the distance-proportional visualization."""
    print("="*80)
    print("TESTING DISTANCE-PROPORTIONAL VISUALIZATION")
    print("="*80)
    
    # Create test data
    vehicles, tasks, depot_manager, results = create_distance_test_data()
    
    # Generate visualization
    print("\nGenerating distance-proportional visualization...")
    try:
        filename = create_static_visualization(
            vehicles, tasks, depot_manager, results,
            "optimus_distance_proportional.png"
        )
        print(f"✅ SUCCESS! Distance-proportional visualization saved to: {filename}")
        
        print("\nGraph features:")
        print("- Edge lengths are proportional to actual distances")
        print("- Red squares: Depots (Depot-0, Depot-1, Depot-2)")
        print("- Blue circles: Stores (Store-3 to Store-7) with demand numbers")
        print("- Colored lines with arrows: Vehicle routes showing direction")
        print("- Route information boxes: Positioned outside graph area")
        print("- Distance-based layout: Closer locations appear closer in the graph")
        
        # List files to confirm creation
        import os
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"\nFile created successfully: {filename} ({file_size} bytes)")
        else:
            print(f"❌ Error: File {filename} was not created")
            
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_distance_proportional_visualization()
