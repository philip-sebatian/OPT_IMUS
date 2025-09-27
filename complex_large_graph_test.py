#!/usr/bin/env python3
"""
Test script for a very complex and large graph visualization.
"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple

from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager
from visualization.static_visualizer import create_static_visualization

def create_large_complex_test_data():
    """Create a very complex test data with many locations and vehicles."""
    print("Creating large complex test data...")
    
    # 1. Create a large location system
    num_depots = 6
    num_stores = 20
    num_locations = num_depots + num_stores
    
    depot_ids = list(range(num_depots))
    delivery_location_ids = list(range(num_depots, num_locations))
    
    # Create realistic cost matrix based on a city-like layout
    cost_matrix = np.zeros((num_locations, num_locations))
    location_to_index = {i: i for i in range(num_locations)}
    
    # Simulate a city grid with realistic distances
    locations = {}
    
    # Place depots strategically around the city
    depot_positions = [
        (0, 0),      # Depot-0: Southwest corner
        (100, 0),    # Depot-1: Southeast corner
        (0, 100),    # Depot-2: Northwest corner
        (100, 100),  # Depot-3: Northeast corner
        (50, 0),     # Depot-4: South center
        (50, 100),   # Depot-5: North center
    ]
    
    for i, pos in enumerate(depot_positions):
        locations[i] = pos
    
    # Place stores in a grid pattern across the city
    store_positions = []
    cols = 5
    rows = 4
    for row in range(rows):
        for col in range(cols):
            x = 10 + col * 20
            y = 10 + row * 20
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
    
    # 2. Create many vehicles with varied configurations
    vehicles = []
    vehicle_configs = [
        (100, 60, 0),   # Vehicle 0: starts at Depot-0
        (120, 40, 1),   # Vehicle 1: starts at Depot-1
        (80, 50, 2),    # Vehicle 2: starts at Depot-2
        (150, 70, 3),   # Vehicle 3: starts at Depot-3
        (90, 30, 4),    # Vehicle 4: starts at Depot-4
        (110, 45, 5),   # Vehicle 5: starts at Depot-5
        (130, 55, 0),   # Vehicle 6: starts at Depot-0
        (95, 35, 1),    # Vehicle 7: starts at Depot-1
        (140, 65, 2),   # Vehicle 8: starts at Depot-2
        (85, 25, 3),    # Vehicle 9: starts at Depot-3
    ]
    
    for i, (capacity, initial_stock, depot_id) in enumerate(vehicle_configs):
        vehicle = VehicleState(
            id=i,
            capacity=capacity,
            current_stock=initial_stock,
            position=depot_id
        )
        vehicles.append(vehicle)
    
    # 3. Create many delivery tasks
    delivery_tasks = []
    task_demands = [random.randint(10, 50) for _ in range(num_stores)]
    
    for i, demand in enumerate(task_demands):
        task = DeliveryTask(location=delivery_location_ids[i], demand=demand)
        delivery_tasks.append(task)
    
    # 4. Create complex optimization results with various strategies
    mock_assignments = []
    mock_vehicle_routes = {}
    
    total_cost = 0.0
    strategy_counts = {'direct': 0, 'refill': 0, 'split': 0, 'new_vehicle': 0}
    
    # Assign tasks to vehicles with different strategies
    for i, task in enumerate(delivery_tasks):
        vehicle_id = i % len(vehicles)
        assigned_vehicle = vehicles[vehicle_id]
        
        # Choose strategy based on realistic scenarios
        if task.demand > assigned_vehicle.current_stock:
            if task.demand > assigned_vehicle.capacity:
                strategy_choice = 'split'
            else:
                strategy_choice = 'refill'
        else:
            strategy_choice = 'direct'
        
        # Calculate realistic cost based on distance
        nearest_depot, _ = depot_manager.find_nearest_depot_to_delivery(task.location)
        distance_cost = depot_manager.get_distance(assigned_vehicle.position, task.location)
        return_cost = depot_manager.get_distance(task.location, nearest_depot)
        assignment_cost = distance_cost + return_cost + random.uniform(5, 20)
        
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
            # Create split with another vehicle
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
        
        # Add store to route
        mock_vehicle_routes[assigned_vehicle.id].append(task.location)
        
        # Return to nearest depot
        nearest_depot_loc, _ = depot_manager.find_nearest_depot_to_delivery(task.location)
        mock_vehicle_routes[assigned_vehicle.id].append(nearest_depot_loc)

        # Update vehicle state
        assigned_vehicle.route = mock_vehicle_routes[assigned_vehicle.id]
        assigned_vehicle.total_cost += assignment_cost
        assigned_vehicle.current_stock = max(0, assigned_vehicle.current_stock - task.demand)
        assigned_vehicle.position = nearest_depot_loc

        if strategy_choice == 'split':
            # Update other vehicle for split
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
    
    # Print some route information
    print("\nSample Vehicle Routes:")
    for i, vehicle in enumerate(vehicles[:5]):  # Show first 5 vehicles
        if vehicle.route:
            route_str = " → ".join([f"Depot-{r}" if r < num_depots else f"Store-{r}" for r in vehicle.route])
            print(f"  Vehicle-{vehicle.id}: {route_str} (Cost: {vehicle.total_cost:.1f}, Stock: {vehicle.current_stock}/{vehicle.capacity})")
    
    return vehicles, delivery_tasks, depot_manager, results

def test_large_complex_visualization():
    """Test the large complex visualization."""
    print("="*80)
    print("TESTING LARGE COMPLEX OPTIMUS VISUALIZATION")
    print("="*80)
    
    # Create test data
    vehicles, delivery_tasks, depot_manager, results = create_large_complex_test_data()
    
    # Generate visualization
    print("\nGenerating large complex visualization...")
    try:
        filename = create_static_visualization(
            vehicles, delivery_tasks, depot_manager, results,
            "optimus_large_complex.png"
        )
        print(f"✅ SUCCESS! Large complex visualization saved to: {filename}")
        
        print("\nGraph features:")
        print("- Large 24x18 inch image to avoid congestion")
        print("- Edge lengths are proportional to actual distances")
        print("- Red squares: 6 Depots (Depot-0 to Depot-5)")
        print("- Blue circles: 20 Stores (Store-6 to Store-25) with demand numbers")
        print("- Colored lines with arrows: 10 Vehicle routes showing direction")
        print("- Larger nodes and fonts for better visibility")
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
    test_large_complex_visualization()
