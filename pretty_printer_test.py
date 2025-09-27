import random
import numpy as np
from typing import Dict, Any, List, Tuple

from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager
from src.utils.pretty_printer_fixed import PrettyPrinter

def run_pretty_printer_test():
    print("--- Setting up test data for PrettyPrinter ---")

    # 1. Create Depots and setup location system
    num_depots = 3
    num_locations = 10  # Total locations including depots and delivery points
    
    # Create location IDs (integers)
    depot_ids = list(range(num_depots))  # depots: 0, 1, 2
    delivery_location_ids = list(range(num_depots, num_locations))  # delivery points: 3, 4, 5, 6, 7, 8, 9
    
    # Create location coordinates for distance calculations
    location_coords = {}
    for i in range(num_locations):
        location_coords[i] = (random.uniform(0, 100), random.uniform(0, 100))
    
    # Create cost matrix (simplified - using Euclidean distance)
    cost_matrix = np.zeros((num_locations, num_locations))
    location_to_index = {i: i for i in range(num_locations)}
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                coord1 = location_coords[i]
                coord2 = location_coords[j]
                distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                cost_matrix[i][j] = distance
            else:
                cost_matrix[i][j] = 0.0
    
    depot_manager = DepotManager(depot_ids, cost_matrix, location_to_index)
    print(f"Created {num_depots} depots with IDs: {depot_ids}")
    print(f"Created {len(delivery_location_ids)} delivery locations with IDs: {delivery_location_ids}")

    # 2. Create Vehicles
    num_vehicles = 8
    vehicles = []
    for i in range(num_vehicles):
        initial_depot_id = depot_ids[i % num_depots]
        vehicle = VehicleState(
            id=i,
            capacity=random.randint(50, 200),
            current_stock=random.randint(20, 100),
            position=initial_depot_id  # Use depot ID as position
        )
        vehicles.append(vehicle)
    print(f"Created {num_vehicles} vehicles.")

    # 3. Create Delivery Tasks (Stores)
    num_stores = 6
    delivery_tasks = []
    for i in range(num_stores):
        task = DeliveryTask(
            location=delivery_location_ids[i],  # Use location ID
            demand=random.randint(10, 50)
        )
        delivery_tasks.append(task)
    print(f"Created {num_stores} delivery tasks (stores).")

    # 4. Simulate Optimization Results (simplified for demonstration)
    # This part would normally come from the actual optimization algorithm.
    # We'll create a mock 'assignments' and 'vehicle_routes' structure.
    
    mock_assignments: List[Dict[str, Any]] = []
    mock_vehicle_routes: Dict[int, List[int]] = {}
    
    total_mock_cost = 0.0
    strategy_counts = {'direct': 0, 'refill': 0, 'split': 0, 'new_vehicle': 0}
    
    # Assign tasks to vehicles
    for i, task in enumerate(delivery_tasks):
        vehicle_id = i % num_vehicles # Simple round-robin assignment
        assigned_vehicle = vehicles[vehicle_id]
        
        strategy_choice = random.choice(['direct', 'refill', 'split']) # Exclude new_vehicle for simplicity in mock
        
        assignment_cost = random.uniform(10, 100) # Mock cost
        total_mock_cost += assignment_cost
        strategy_counts[strategy_choice] += 1

        assignment_entry = {
            'task_id': i,  # Use task index as ID
            'vehicles': [assigned_vehicle.id],
            'location': task.location,
            'demand': task.demand,
            'strategy': strategy_choice,
            'cost': assignment_cost,
            'details': {}
        }

        if strategy_choice == 'split':
            # For split, simulate two vehicles delivering
            vehicle_id_2 = (i + 1) % num_vehicles
            if vehicle_id_2 == assigned_vehicle.id: # Ensure different vehicle for split
                vehicle_id_2 = (vehicle_id_2 + 1) % num_vehicles
            
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
            # Update strategy count for the second vehicle in split
            strategy_counts[strategy_choice] += 1

        mock_assignments.append(assignment_entry)

        # Simulate vehicle routes for each assigned vehicle
        # This is a very simplified route. In a real scenario, this would be complex.
        if assigned_vehicle.id not in mock_vehicle_routes:
            mock_vehicle_routes[assigned_vehicle.id] = [assigned_vehicle.position]
        
        mock_vehicle_routes[assigned_vehicle.id].append(task.location)
        # Simulate return to a depot
        nearest_depot_loc, _ = depot_manager.find_nearest_depot_to_delivery(task.location)
        mock_vehicle_routes[assigned_vehicle.id].append(nearest_depot_loc)

        # Update vehicle's internal state for pretty printer to use
        assigned_vehicle.route = mock_vehicle_routes[assigned_vehicle.id]
        assigned_vehicle.total_cost += assignment_cost # Accumulate mock cost
        assigned_vehicle.current_stock -= task.demand # Simple stock reduction
        assigned_vehicle.position = nearest_depot_loc # Final position after task

        if strategy_choice == 'split':
            # Also update the second vehicle in a split
            if 'vehicle_id_2' in locals(): # Check if a second vehicle was assigned
                if assigned_vehicle_2.id not in mock_vehicle_routes:
                    mock_vehicle_routes[assigned_vehicle_2.id] = [assigned_vehicle_2.position]
                mock_vehicle_routes[assigned_vehicle_2.id].append(task.location)
                nearest_depot_loc_2, _ = depot_manager.find_nearest_depot_to_delivery(task.location)
                mock_vehicle_routes[assigned_vehicle_2.id].append(nearest_depot_loc_2)
                
                assigned_vehicle_2.route = mock_vehicle_routes[assigned_vehicle_2.id]
                assigned_vehicle_2.total_cost += assignment_cost # Accumulate mock cost
                assigned_vehicle_2.current_stock -= split_amount_2
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
        'vehicle_routes': mock_vehicle_routes # This is not directly used by pretty_printer, but good to have
    }
    print("Simulated optimization results.")

    # 5. Instantiate and use PrettyPrinter
    printer = PrettyPrinter(vehicles, delivery_tasks, depot_manager)
    printer.print_vehicle_process(mock_results)
    printer.print_overall_results(mock_results)

if __name__ == "__main__":
    run_pretty_printer_test()