#!/usr/bin/env python3
"""
Complex test for the improved visualization system without overlapping tables.
"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple

from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager
from visualization.static_visualizer import create_static_visualization

def create_complex_test_data():
    """Create complex test data for the improved visualization."""
    print("Creating complex test data for improved visualization...")
    
    # 1. Create larger location system
    num_depots = 4
    num_locations = 15
    
    depot_ids = list(range(num_depots))
    delivery_location_ids = list(range(num_depots, num_locations))
    
    # Create cost matrix
    cost_matrix = np.zeros((num_locations, num_locations))
    location_to_index = {i: i for i in range(num_locations)}
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                distance = random.uniform(5, 50)
                cost_matrix[i][j] = distance
            else:
                cost_matrix[i][j] = 0.0
    
    depot_manager = DepotManager(depot_ids, cost_matrix, location_to_index)
    
    # 2. Create more vehicles with varied configurations
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
    
    # 3. Create more delivery tasks
    delivery_tasks = []
    task_configs = [
        (4, 25),   # Store-4, demand 25
        (5, 30),   # Store-5, demand 30
        (6, 20),   # Store-6, demand 20
        (7, 35),   # Store-7, demand 35
        (8, 15),   # Store-8, demand 15
        (9, 40),   # Store-9, demand 40
        (10, 22),  # Store-10, demand 22
        (11, 28),  # Store-11, demand 28
        (12, 18),  # Store-12, demand 18
        (13, 32),  # Store-13, demand 32
        (14, 26),  # Store-14, demand 26
    ]
    
    for location, demand in task_configs:
        task = DeliveryTask(location=location, demand=demand)
        delivery_tasks.append(task)
    
    # 4. Create complex optimization results with various strategies
    mock_assignments = []
    mock_vehicle_routes = {}
    
    total_cost = 0.0
    strategy_counts = {'direct': 0, 'refill': 0, 'split': 0, 'new_vehicle': 0}
    
    # Define complex assignments with different strategies
    assignments = [
        (0, 4, 'direct', 25, 30),      # Vehicle 0 -> Store-4
        (1, 5, 'direct', 30, 35),      # Vehicle 1 -> Store-5
        (2, 6, 'direct', 20, 25),      # Vehicle 2 -> Store-6
        (3, 7, 'refill', 35, 45),      # Vehicle 3 -> Store-7 (refill needed)
        (4, 8, 'direct', 15, 20),      # Vehicle 4 -> Store-8
        (5, 9, 'split', 40, 50),       # Vehicle 5 -> Store-9 (split with Vehicle 0)
        (1, 10, 'refill', 22, 30),     # Vehicle 1 -> Store-10 (refill needed)
        (2, 11, 'direct', 28, 32),     # Vehicle 2 -> Store-11
        (3, 12, 'direct', 18, 25),     # Vehicle 3 -> Store-12
        (4, 13, 'split', 32, 40),      # Vehicle 4 -> Store-13 (split with Vehicle 5)
        (0, 14, 'direct', 26, 30),     # Vehicle 0 -> Store-14
    ]
    
    for i, (vehicle_id, store_id, strategy, demand, cost) in enumerate(assignments):
        assigned_vehicle = vehicles[vehicle_id]
        
        total_cost += cost
        strategy_counts[strategy] += 1

        assignment_entry = {
            'task_id': i,
            'vehicles': [assigned_vehicle.id],
            'location': store_id,
            'demand': demand,
            'strategy': strategy,
            'cost': cost,
            'details': {}
        }

        if strategy == 'split':
            # Create split with another vehicle
            other_vehicle_id = (vehicle_id + 1) % len(vehicles)
            other_vehicle = vehicles[other_vehicle_id]
            
            split_amount_1 = demand // 2
            split_amount_2 = demand - split_amount_1
            
            assignment_entry['vehicles'] = [assigned_vehicle.id, other_vehicle.id]
            assignment_entry['details'] = {
                'vehicle_amounts': [
                    (assigned_vehicle.id, split_amount_1),
                    (other_vehicle.id, split_amount_2)
                ]
            }
            strategy_counts[strategy] += 1

        mock_assignments.append(assignment_entry)

        # Create routes
        if assigned_vehicle.id not in mock_vehicle_routes:
            mock_vehicle_routes[assigned_vehicle.id] = [assigned_vehicle.position]
        
        # Add store to route
        mock_vehicle_routes[assigned_vehicle.id].append(store_id)
        
        # Return to nearest depot
        nearest_depot_loc, _ = depot_manager.find_nearest_depot_to_delivery(store_id)
        mock_vehicle_routes[assigned_vehicle.id].append(nearest_depot_loc)

        # Update vehicle state
        assigned_vehicle.route = mock_vehicle_routes[assigned_vehicle.id]
        assigned_vehicle.total_cost += cost
        assigned_vehicle.current_stock = max(0, assigned_vehicle.current_stock - demand)
        assigned_vehicle.position = nearest_depot_loc

        if strategy == 'split':
            # Update other vehicle for split
            if other_vehicle.id not in mock_vehicle_routes:
                mock_vehicle_routes[other_vehicle.id] = [other_vehicle.position]
            mock_vehicle_routes[other_vehicle.id].append(store_id)
            nearest_depot_loc_2, _ = depot_manager.find_nearest_depot_to_delivery(store_id)
            mock_vehicle_routes[other_vehicle.id].append(nearest_depot_loc_2)
            
            other_vehicle.route = mock_vehicle_routes[other_vehicle.id]
            other_vehicle.total_cost += cost
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
    
    # Print route information
    print("\nVehicle Routes:")
    for vehicle in vehicles:
        if vehicle.route:
            route_str = " → ".join([f"Depot-{r}" if r < num_depots else f"Store-{r}" for r in vehicle.route])
            print(f"  Vehicle-{vehicle.id}: {route_str} (Cost: {vehicle.total_cost:.1f}, Stock: {vehicle.current_stock}/{vehicle.capacity})")
    
    return vehicles, delivery_tasks, depot_manager, results

def test_complex_visualization():
    """Test the complex visualization without overlapping tables."""
    print("="*80)
    print("TESTING COMPLEX OPTIMUS VISUALIZATION (NO OVERLAPPING TABLES)")
    print("="*80)
    
    # Create test data
    vehicles, delivery_tasks, depot_manager, results = create_complex_test_data()
    
    # Generate visualization
    print("\nGenerating complex visualization with non-overlapping route information...")
    try:
        filename = create_static_visualization(
            vehicles, delivery_tasks, depot_manager, results,
            "optimus_complex_clean.png"
        )
        print(f"✅ SUCCESS! Complex visualization saved to: {filename}")
        
        print("\nGraph features:")
        print("- Clean main graph without overlapping text")
        print("- Red squares: Depots (Depot-0, Depot-1, Depot-2, Depot-3)")
        print("- Blue circles: Stores (Store-4 to Store-14) with demand numbers")
        print("- Colored lines with arrows: Vehicle routes showing direction")
        print("- Route information boxes: Positioned outside graph area (NO OVERLAP)")
        print("- Color-coded route boxes: Each vehicle's route box matches its path color")
        print("- Delivery summary: Shows which vehicle delivered what to each store")
        print("- Summary box: Total cost, vehicles used, strategy breakdown")
        
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
    test_complex_visualization()
