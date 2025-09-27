#!/usr/bin/env python3
"""
Comprehensive demo of the Optimus Visualizer system.
"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple

from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager
from src.utils.pretty_printer_fixed import PrettyPrinter
from src.utils.visualizer import create_visualization
from src.utils.static_visualizer import create_static_visualization

def create_demo_data():
    """Create comprehensive demo data for visualization."""
    print("Creating comprehensive demo data...")
    
    # 1. Create location system
    num_depots = 4
    num_locations = 15
    
    depot_ids = list(range(num_depots))
    delivery_location_ids = list(range(num_depots, num_locations))
    
    # Create realistic cost matrix
    cost_matrix = np.zeros((num_locations, num_locations))
    location_to_index = {i: i for i in range(num_locations)}
    
    # Create grid-like positions for more realistic visualization
    positions = {}
    for i in range(num_locations):
        if i < num_depots:  # Depots in corners
            if i == 0:
                positions[i] = (0, 0)
            elif i == 1:
                positions[i] = (100, 0)
            elif i == 2:
                positions[i] = (0, 100)
            else:
                positions[i] = (100, 100)
        else:  # Stores scattered
            positions[i] = (random.uniform(10, 90), random.uniform(10, 90))
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                pos1 = positions[i]
                pos2 = positions[j]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                cost_matrix[i][j] = distance
            else:
                cost_matrix[i][j] = 0.0
    
    depot_manager = DepotManager(depot_ids, cost_matrix, location_to_index)
    
    # 2. Create vehicles with varied capacities
    vehicles = []
    vehicle_configs = [
        (80, 60),   # Small vehicle
        (120, 40),  # Medium vehicle
        (150, 80),  # Large vehicle
        (100, 30),  # Medium vehicle
        (200, 100), # Extra large vehicle
        (90, 50),   # Small vehicle
    ]
    
    for i, (capacity, initial_stock) in enumerate(vehicle_configs):
        initial_depot_id = depot_ids[i % num_depots]
        vehicle = VehicleState(
            id=i,
            capacity=capacity,
            current_stock=initial_stock,
            position=initial_depot_id
        )
        vehicles.append(vehicle)
    
    # 3. Create delivery tasks with varied demands
    delivery_tasks = []
    task_demands = [25, 40, 15, 60, 30, 45, 20, 35, 50, 10, 55]
    
    for i, demand in enumerate(task_demands):
        task = DeliveryTask(
            location=delivery_location_ids[i],
            demand=demand
        )
        delivery_tasks.append(task)
    
    # 4. Create realistic optimization results
    mock_assignments = []
    mock_vehicle_routes = {}
    
    total_mock_cost = 0.0
    strategy_counts = {'direct': 0, 'refill': 0, 'split': 0, 'new_vehicle': 0}
    
    # Assign tasks with realistic strategies
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
        
        total_mock_cost += assignment_cost
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
        'total_cost': total_mock_cost,
        'vehicles_used': len(all_vehicles_used),
        'strategy_counts': strategy_counts,
        'assignments': mock_assignments,
        'vehicle_routes': mock_vehicle_routes
    }
    
    print(f"Created {len(vehicles)} vehicles and {len(delivery_tasks)} delivery tasks")
    print(f"Total cost: {total_mock_cost:.1f}")
    print(f"Strategy breakdown: {strategy_counts}")
    
    return vehicles, delivery_tasks, depot_manager, results

def run_comprehensive_demo():
    """Run comprehensive demo of the visualizer system."""
    print("="*80)
    print("OPTIMUS ROUTING SYSTEM - COMPREHENSIVE VISUALIZER DEMO")
    print("="*80)
    
    # Create demo data
    vehicles, delivery_tasks, depot_manager, results = create_demo_data()
    
    # 1. Pretty Printer Output
    print("\n" + "="*60)
    print("1. PRETTY PRINTER OUTPUT")
    print("="*60)
    printer = PrettyPrinter(vehicles, delivery_tasks, depot_manager)
    printer.print_vehicle_process(results)
    printer.print_overall_results(results)
    
    # 2. Static Visualizer
    print("\n" + "="*60)
    print("2. STATIC VISUALIZER")
    print("="*60)
    print("Creating static visualization...")
    try:
        static_filename = create_static_visualization(
            vehicles, delivery_tasks, depot_manager, results,
            "optimus_demo_static.png"
        )
        print(f"Static visualization saved to: {static_filename}")
    except Exception as e:
        print(f"Error creating static visualization: {e}")
    
    # 3. Interactive Visualizer
    print("\n" + "="*60)
    print("3. INTERACTIVE VISUALIZER")
    print("="*60)
    print("Creating interactive visualization...")
    print("Features:")
    print("- Red squares: Depots (D0, D1, D2, D3)")
    print("- Blue circles: Stores (S4-S14) with demand numbers")
    print("- Colored lines: Vehicle paths (different color per vehicle)")
    print("- Yellow boxes: Delivery information per store")
    print("- Checkboxes: Toggle vehicle visibility")
    print("- Refresh button: Update display")
    
    try:
        visualizer = create_visualization(vehicles, delivery_tasks, depot_manager, results)
        visualizer.show()
    except Exception as e:
        print(f"Error displaying interactive visualization: {e}")
        print("This is expected in headless environments.")
        print("Use the static visualizer instead.")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("Summary of features demonstrated:")
    print("✓ Pretty printer with accurate vehicle state representation")
    print("✓ Static visualization for headless environments")
    print("✓ Interactive visualization with vehicle path toggling")
    print("✓ Node differentiation (depots vs stores)")
    print("✓ Stock delivery information display")
    print("✓ Colored vehicle paths")
    print("✓ Comprehensive routing information")

if __name__ == "__main__":
    run_comprehensive_demo()
