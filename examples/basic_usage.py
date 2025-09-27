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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimus import OptimizedRefillSystem


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
    print("\n🔧 Creating OptimizedRefillSystem...")
    system = OptimizedRefillSystem(
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
    result = system.solve_optimized()
    
    # Display results
    print("\n📊 RESULTS")
    print("=" * 30)
    
    if result:
        print(f"✅ Optimization completed successfully!")
        print(f"   💰 Total cost: {result['total_cost']:.1f}")
        print(f"   🚚 Vehicles used: {result['vehicles_used']}")
        print(f"   📊 Strategy breakdown: {result['strategy_counts']}")
        
        print(f"\n🚚 Vehicle Routes:")
        for vehicle_id, route in result['vehicle_routes'].items():
            print(f"   Vehicle {vehicle_id}: {route}")
        
        # Show cost breakdown
        print(f"\n💰 Cost Analysis:")
        total_direct = sum(assignment['cost'] for assignment in result['assignments'] 
                          if assignment['strategy'] == 'direct')
        total_refill = sum(assignment['cost'] for assignment in result['assignments'] 
                          if assignment['strategy'] == 'refill')
        total_new_vehicle = sum(assignment['cost'] for assignment in result['assignments'] 
                               if assignment['strategy'] == 'new_vehicle')
        
        print(f"   Direct deliveries: {total_direct:.1f}")
        print(f"   Refill deliveries: {total_refill:.1f}")
        print(f"   New vehicle deliveries: {total_new_vehicle:.1f}")
        
    else:
        print("❌ Optimization failed!")
    
    print(f"\n🎉 Basic usage example completed!")


if __name__ == "__main__":
    main()
