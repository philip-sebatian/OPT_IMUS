#!/usr/bin/env python3
"""
Integrated test that combines pretty printer and visualizer.
"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple

from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager
from src.utils.pretty_printer_fixed import PrettyPrinter
from src.utils.visualizer import create_visualization

def run_integrated_test():
    """Run integrated test with both pretty printer and visualizer."""
    print("=== Integrated Pretty Printer + Visualizer Test ===")
    
    # 1. Create Depots and setup location system
    num_depots = 3
    num_locations = 10
    
    depot_ids = list(range(num_depots))
    delivery_location_ids = list(range(num_depots, num_locations))
    
    # Create cost matrix
    cost_matrix = np.zeros((num_locations, num_locations))
    location_to_index = {i: i for i in range(num_locations)}
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                coord1 = (i * 10, i * 8)  # Create grid-like positions
                coord2 = (j * 10, j * 8)
                distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                cost_matrix[i][j] = distance
            else:
                cost_matrix[i][j] = 0.0
    
    depot_manager = DepotManager(depot_ids, cost_matrix, location_to_index)
    print(f"Created {num_depots} depots with IDs: {depot_ids}")
    print(f"Created {len(delivery_location_ids)} delivery locations with IDs: {delivery_location_ids}")

    # 2. Create Vehicles
    num_vehicles = 6
    vehicles = []
    for i in range(num_vehicles):
        initial_depot_id = depot_ids[i % num_depots]
        vehicle = VehicleState(
            id=i,
            capacity=random.randint(80, 150),
            current_stock=random.randint(30, 100),
            position=initial_depot_id
        )
        vehicles.append(vehicle)
    print(f"Created {num_vehicles} vehicles.")

    # 3. Create Delivery Tasks
    num_stores = 5
    delivery_tasks = []
    for i in range(num_stores):
        task = DeliveryTask(
            location=delivery_location_ids[i],
            demand=random.randint(15, 45)
        )
        delivery_tasks.append(task)
    print(f"Created {num_stores} delivery tasks (stores).")

    # 4. Simulate Optimization Results
    mock_assignments = []
    mock_vehicle_routes = {}
    
    total_mock_cost = 0.0
    strategy_counts = {'direct': 0, 'refill': 0, 'split': 0, 'new_vehicle': 0}
    
    # Assign tasks to vehicles
    for i, task in enumerate(delivery_tasks):
        vehicle_id = i % num_vehicles
        assigned_vehicle = vehicles[vehicle_id]
        
        # Choose strategy based on realistic scenarios
        if task.demand > assigned_vehicle.current_stock:
            if task.demand > assigned_vehicle.capacity:
                strategy_choice = 'split'
            else:
                strategy_choice = 'refill'
        else:
            strategy_choice = 'direct'
        
        assignment_cost = random.uniform(20, 100)
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
            vehicle_id_2 = (vehicle_id + 1) % num_vehicles
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
    
    mock_results = {
        'total_cost': total_mock_cost,
        'vehicles_used': len(all_vehicles_used),
        'strategy_counts': strategy_counts,
        'assignments': mock_assignments,
        'vehicle_routes': mock_vehicle_routes
    }
    print("Simulated optimization results.")

    # 5. Use Pretty Printer
    print("\n" + "="*60)
    print("PRETTY PRINTER OUTPUT")
    print("="*60)
    printer = PrettyPrinter(vehicles, delivery_tasks, depot_manager)
    printer.print_vehicle_process(mock_results)
    printer.print_overall_results(mock_results)

    # 6. Use Visualizer
    print("\n" + "="*60)
    print("VISUALIZER OUTPUT")
    print("="*60)
    print("Creating interactive visualization...")
    print("Features:")
    print("- Red squares: Depots")
    print("- Blue circles: Stores (with demand numbers)")
    print("- Colored lines: Vehicle paths")
    print("- Yellow boxes: Delivery information per store")
    print("- Checkboxes: Toggle vehicle visibility")
    
    try:
        visualizer = create_visualization(vehicles, delivery_tasks, depot_manager, mock_results)
        visualizer.show()
    except Exception as e:
        print(f"Error displaying interactive visualization: {e}")
        print("Saving to file instead...")
        try:
            visualizer = create_visualization(vehicles, delivery_tasks, depot_manager, mock_results)
            visualizer.save("optimus_integrated_visualization.png")
            print("Visualization saved to 'optimus_integrated_visualization.png'")
        except Exception as e2:
            print(f"Error saving visualization: {e2}")

if __name__ == "__main__":
    run_integrated_test()
