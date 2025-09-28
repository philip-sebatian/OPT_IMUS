#!/usr/bin/env python3
"""
Basic usage example for the Optimus routing system.

This example demonstrates the basic functionality of the Optimus system
with a simple multi-vehicle routing scenario.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, repo_root)

from workspace_optimus import CuOptEnhancedPlanner


def main():
    """Basic usage example."""
    print("🚀 Optimus Basic Usage Example")
    print("=" * 50)
    
    # Define the problem
    print("📋 Setting up the problem...")
    
    # Graph structure
    offsets = np.array([0, 3, 5, 7, 8, 9])
    edges = np.array([1, 2, 3, 0, 2, 0, 3, 4, 0])
    weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    time_to_travel = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    
    # Problem parameters
    target_locations = np.array([1, 3])        # 2 stores
    depot_locations = [0, 4]                   # 2 depots
    vehicle_capacities = [15, 12, 10, 8]       # 4 vehicles
    initial_stock = [5, 3, 2, 1]               # Initial stock levels
    delivery_demands = [3, 4]                  # Delivery requirements
    
    print(f"   Stores: {target_locations}")
    print(f"   Depots: {depot_locations}")
    print(f"   Vehicles: {len(vehicle_capacities)} with capacities {vehicle_capacities}")
    print(f"   Initial stock: {initial_stock}")
    print(f"   Demands: {delivery_demands}")
    
    # Create the system
    print("\n🔧 Creating CuOptEnhancedPlanner...")
    planner = CuOptEnhancedPlanner(
        offsets=offsets,
        edges=edges,
        weights=weights,
        time_to_travel=time_to_travel,
        target_locations=target_locations,
        depot_locations=depot_locations,
        vehicle_capacities=vehicle_capacities,
        initial_stock=initial_stock,
        delivery_demands=delivery_demands
    )
    
    # Solve the problem
    print("\n🎯 Solving the optimization problem...")
    planner_result = planner.solve()
    result = planner_result.as_dict()
    
    # Display results
    print("\n📊 RESULTS")
    print("=" * 30)
    
    if result:
        print(f"✅ Optimization completed successfully!")
        print(f"   💰 Total cost: {result.get('total_cost', 0.0):.1f}")
        print(f"   🚚 Vehicles used: {result.get('vehicles_used', 0)}")
        print(f"   📊 Strategy breakdown: {result.get('strategy_counts', {})}")
        
        print(f"\n🚚 Vehicle Routes:")
        for vehicle_id, route in result.get('vehicle_routes', {}).items():
            print(f"   Vehicle {vehicle_id}: {route}")

        # Show cost breakdown
        print(f"\n💰 Cost Analysis:")
        assignments = result.get('assignments', [])
        demand_by_task = {}
        vehicles_by_task = {}
        for entry in assignments:
            task_id = entry.get('task_id')
            demand_by_task.setdefault(task_id, 0)
            demand_by_task[task_id] += entry.get('demand', 0)
            vehicles_by_task.setdefault(task_id, set()).add(entry.get('vehicle_id'))

        total_direct = sum(
            demand for task_id, demand in demand_by_task.items()
            if len(vehicles_by_task.get(task_id, [])) <= 1
        )
        total_split = sum(
            demand for task_id, demand in demand_by_task.items()
            if len(vehicles_by_task.get(task_id, [])) > 1
        )

        print(f"   Direct deliveries demand served: {total_direct:.1f}")
        print(f"   Split deliveries demand served: {total_split:.1f}")

    else:
        print("❌ Optimization failed!")
    
    print(f"\n🎉 Basic usage example completed!")


if __name__ == "__main__":
    main()
