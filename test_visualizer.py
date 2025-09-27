#!/usr/bin/env python3
"""
Test script for the Optimus Visualizer module.
"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple

from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager
from src.utils.visualizer import create_visualization

def create_test_data():
    """Create test data for the visualizer."""
    print("Creating test data for visualizer...")
    
    # 1. Create location system
    num_depots = 3
    num_locations = 12  # Total locations including depots and delivery points
    
    # Create location IDs
    depot_ids = list(range(num_depots))  # depots: 0, 1, 2
    delivery_location_ids = list(range(num_depots, num_locations))  # delivery points: 3-11
    
    # Create cost matrix
    cost_matrix = np.zeros((num_locations, num_locations))
    location_to_index = {i: i for i in range(num_locations)}
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                # Create realistic distances
                distance = random.uniform(5, 50)
                cost_matrix[i][j] = distance
            else:
                cost_matrix[i][j] = 0.0
    
    depot_manager = DepotManager(depot_ids, cost_matrix, location_to_index)
    
    # 2. Create vehicles
    vehicles = []
    for i in range(5):
        initial_depot_id = depot_ids[i % num_depots]
        vehicle = VehicleState(
            id=i,
            capacity=random.randint(80, 150),
            current_stock=random.randint(30, 80),
            position=initial_depot_id
        )
        vehicles.append(vehicle)
    
    # 3. Create delivery tasks
    delivery_tasks = []
    for i in range(6):
        task = DeliveryTask(
            location=delivery_location_ids[i],
            demand=random.randint(15, 40)
        )
        delivery_tasks.append(task)
    
    # 4. Create realistic optimization results
    mock_assignments = []
    mock_vehicle_routes = {}
    total_cost = 0.0
    strategy_counts = {'direct': 0, 'refill': 0, 'split': 0, 'new_vehicle': 0}
    
    # Assign tasks to vehicles with realistic scenarios
    for i, task in enumerate(delivery_tasks):
        vehicle_id = i % len(vehicles)
        assigned_vehicle = vehicles[vehicle_id]
        
        # Choose strategy based on vehicle capacity and demand
        if task.demand > assigned_vehicle.current_stock:
            if task.demand > assigned_vehicle.capacity:
                strategy_choice = 'split'
            else:
                strategy_choice = 'refill'
        else:
            strategy_choice = 'direct'
        
        # Calculate realistic cost
        assignment_cost = random.uniform(15, 80)
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
            # Create split delivery
            vehicle_id_2 = (vehicle_id + 1) % len(vehicles)
            assigned_vehicle_2 = vehicles[vehicle_id_2]
            
            split_amount_1 = task.demand // 2
            split_amount_2 = task.demand - split_amount_1
            
            assignment_entry['vehicles'] = [assigned_vehicle.id, assigned_vehicle_2.id]
            assignment_entry['details'] = {
                'vehicle_amounts': [
                    (assigned_vehicle.id, split_amount_1),
                    (assigned_vehicle_2.id, split_amount_2)
                ]
            }
            strategy_counts[strategy_choice] += 1
        
        mock_assignments.append(assignment_entry)
        
        # Create realistic routes
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
            # Update second vehicle
            if assigned_vehicle_2.id not in mock_vehicle_routes:
                mock_vehicle_routes[assigned_vehicle_2.id] = [assigned_vehicle_2.position]
            mock_vehicle_routes[assigned_vehicle_2.id].append(task.location)
            nearest_depot_loc_2, _ = depot_manager.find_nearest_depot_to_delivery(task.location)
            mock_vehicle_routes[assigned_vehicle_2.id].append(nearest_depot_loc_2)
            
            assigned_vehicle_2.route = mock_vehicle_routes[assigned_vehicle_2.id]
            assigned_vehicle_2.total_cost += assignment_cost
            assigned_vehicle_2.current_stock = max(0, assigned_vehicle_2.current_stock - split_amount_2)
            assigned_vehicle_2.position = nearest_depot_loc_2
    
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

def test_visualizer():
    """Test the visualizer with sample data."""
    print("=== Testing Optimus Visualizer ===")
    
    # Create test data
    vehicles, delivery_tasks, depot_manager, results = create_test_data()
    
    # Create visualization
    print("Creating visualization...")
    visualizer = create_visualization(vehicles, delivery_tasks, depot_manager, results)
    
    # Show the visualization
    print("Displaying visualization...")
    print("Features:")
    print("- Red squares: Depots")
    print("- Blue circles: Stores (with demand numbers)")
    print("- Colored lines: Vehicle paths")
    print("- Yellow boxes: Delivery information per store")
    print("- Checkboxes: Toggle vehicle visibility")
    print("- Use the refresh button to update the display")
    
    try:
        visualizer.show()
    except Exception as e:
        print(f"Error displaying visualization: {e}")
        print("Saving to file instead...")
        visualizer.save("optimus_routing_visualization.png")

if __name__ == "__main__":
    test_visualizer()
