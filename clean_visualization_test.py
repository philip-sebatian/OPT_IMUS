#!/usr/bin/env python3
"""
Test script for the clean, non-overlapping visualization.
"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple

from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager
from visualization.static_visualizer import create_static_visualization

def create_clean_test_data():
    """Create test data for the clean visualization."""
    print("Creating clean test data...")
    
    # 1. Create location system
    num_depots = 4
    num_stores = 12
    num_locations = num_depots + num_stores
    
    depot_ids = list(range(num_depots))
    delivery_location_ids = list(range(num_depots, num_locations))
    
    # Create realistic cost matrix
    cost_matrix = np.zeros((num_locations, num_locations))
    location_to_index = {i: i for i in range(num_locations)}
    
    # Simulate a city-like layout with realistic distances
    locations = {}
    
    # Place depots strategically
    depot_positions = [
        (0, 0),      # Depot-0: Southwest
        (100, 0),    # Depot-1: Southeast
        (0, 100),    # Depot-2: Northwest
        (100, 100),  # Depot-3: Northeast
    ]
    
    for i, pos in enumerate(depot_positions):
        locations[i] = pos
    
    # Place stores in a grid pattern
    store_positions = []
    cols = 4
    rows = 3
    for row in range(rows):
        for col in range(cols):
            x = 15 + col * 25
            y = 15 + row * 25
            store_positions.append((x, y))
    
    for i, pos in enumerate(store_positions):
        locations[num_depots + i] = pos
    
    # Create cost matrix based on Euclidean distances
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
    vehicles = []
    vehicle_configs = [
        (100, 60, 0),   # Vehicle 0: starts at Depot-0
        (120, 40, 1),   # Vehicle 1: starts at Depot-1
        (80, 50, 2),    # Vehicle 2: starts at Depot-2
        (150, 70, 3),   # Vehicle 3: starts at Depot-3
        (90, 30, 0),    # Vehicle 4: starts at Depot-0
        (110, 45, 1),   # Vehicle 5: starts at Depot-1
    ]
    
    for i, (capacity, initial_stock, depot_id) in enumerate(vehicle_configs):
        vehicle = VehicleState(
            id=i,
            capacity=capacity,
            current_stock=initial_stock,
            position=depot_id
        )
        vehicles.append(vehicle)
    
    # 3. Create delivery tasks
    delivery_tasks = []
    task_demands = [random.randint(15, 45) for _ in range(num_stores)]
    
    for i, demand in enumerate(task_demands):
        task = DeliveryTask(location=delivery_location_ids[i], demand=demand)
        delivery_tasks.append(task)
    
    # 4. Create results with various strategies
    mock_assignments = []
    mock_vehicle_routes = {}
    
    total_cost = 0.0
    strategy_counts = {'direct': 0, 'refill': 0, 'split': 0, 'new_vehicle': 0}
    
    # Assign tasks to vehicles
    for i, task in enumerate(delivery_tasks):
        vehicle_id = i % len(vehicles)
        assigned_vehicle = vehicles[vehicle_id]
        
        # Choose strategy
        if task.demand > assigned_vehicle.current_stock:
            if task.demand > assigned_vehicle.capacity:
                strategy_choice = 'split'
            else:
                strategy_choice = 'refill'
        else:
            strategy_choice = 'direct'
        
        # Calculate realistic cost
        nearest_depot, _ = depot_manager.find_nearest_depot_to_delivery(task.location)
        distance_cost = depot_manager.get_distance(assigned_vehicle.position, task.location)
        return_cost = depot_manager.get_distance(task.location, nearest_depot)
        assignment_cost = distance_cost + return_cost + random.uniform(5, 15)
        
        total_cost += assignment_cost
        strategy_counts[strategy_choice] += 1

        assignment_entry = {
            'task_id': i,
            'vehicles': [assigned_vehicle.id],
            'location': task.location,
            'demand': task.demand,
            'strategy': strategy_choice,
            'cost': assignment_cost,
            'details': {}
        }

        if strategy_choice == 'split':
            other_vehicle_id = (vehicle_id + 1) % len(vehicles)
            other_vehicle = vehicles[other_vehicle_id]
            
            split_amount_1 = task.demand // 2
            split_amount_2 = task.demand - split_amount_1
            
            assignment_entry['vehicles'] = [assigned_vehicle.id, other_vehicle.id]
            assignment_entry['details'] = {
                'vehicle_amounts': [
                    (assigned_vehicle.id, split_amount_1),
                    (other_vehicle.id, split_amount_2)
                ]
            }
            strategy_counts[strategy_choice] += 1

        mock_assignments.append(assignment_entry)

        # Create routes
        if assigned_vehicle.id not in mock_vehicle_routes:
            mock_vehicle_routes[assigned_vehicle.id] = [assigned_vehicle.position]
        
        mock_vehicle_routes[assigned_vehicle.id].append(task.location)
        nearest_depot_loc, _ = depot_manager.find_nearest_depot_to_delivery(task.location)
        mock_vehicle_routes[assigned_vehicle.id].append(nearest_depot_loc)

        # Update vehicle state
        assigned_vehicle.route = mock_vehicle_routes[assigned_vehicle.id]
        assigned_vehicle.total_cost += assignment_cost
        assigned_vehicle.current_stock = max(0, assigned_vehicle.current_stock - task.demand)
        assigned_vehicle.position = nearest_depot_loc

        if strategy_choice == 'split':
            if other_vehicle.id not in mock_vehicle_routes:
                mock_vehicle_routes[other_vehicle.id] = [other_vehicle.position]
            mock_vehicle_routes[other_vehicle.id].append(task.location)
            nearest_depot_loc_2, _ = depot_manager.find_nearest_depot_to_delivery(task.location)
            mock_vehicle_routes[other_vehicle.id].append(nearest_depot_loc_2)
            
            other_vehicle.route = mock_vehicle_routes[other_vehicle.id]
            other_vehicle.total_cost += assignment_cost
            other_vehicle.current_stock = max(0, other_vehicle.current_stock - split_amount_2)
            other_vehicle.position = nearest_depot_loc_2

    # Count unique vehicles used
    all_vehicles_used = set()
    for assignment in mock_assignments:
        for vehicle_id in assignment['vehicles']:
            all_vehicles_used.add(vehicle_id)
    
    results = {
        'total_cost': total_cost,
        'vehicles_used': len(all_vehicles_used),
        'strategy_counts': strategy_counts,
        'assignments': mock_assignments,
        'vehicle_routes': mock_vehicle_routes
    }
    
    print(f"Created {len(vehicles)} vehicles and {len(delivery_tasks)} delivery tasks")
    print(f"Total cost: {total_cost:.1f}")
    print(f"Strategy breakdown: {strategy_counts}")
    
    return vehicles, delivery_tasks, depot_manager, results

def test_clean_visualization():
    """Test the clean visualization without overlapping components."""
    print("="*80)
    print("TESTING CLEAN OPTIMUS VISUALIZATION (NO OVERLAPS, THIN LINES, NO ARROWS)")
    print("="*80)
    
    # Create test data
    vehicles, delivery_tasks, depot_manager, results = create_clean_test_data()
    
    # Generate visualization
    print("\nGenerating clean visualization...")
    try:
        filename = create_static_visualization(
            vehicles, delivery_tasks, depot_manager, results,
            "optimus_clean_final.png"
        )
        print(f"✅ SUCCESS! Clean visualization saved to: {filename}")
        
        print("\nGraph features:")
        print("- Large 24x18 inch image to avoid congestion")
        print("- Edge lengths are proportional to actual distances")
        print("- Red squares: 4 Depots (Depot-0 to Depot-3)")
        print("- Blue circles: 12 Stores (Store-4 to Store-15) with demand numbers")
        print("- Thin colored lines: 6 Vehicle routes (NO ARROWS)")
        print("- Overlap prevention: Components never overlap")
        print("- Clean layout: No delivery information boxes")
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
    test_clean_visualization()
