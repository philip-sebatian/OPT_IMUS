#!/usr/bin/env python3
"""
Split delivery demonstration for the Optimus routing system.

This example demonstrates the split delivery functionality where large
deliveries are distributed across multiple vehicles for cost optimization.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimus import EnhancedRefillSystem


def main():
    """Split delivery demonstration."""
    print("🎯 Optimus Split Delivery Demonstration")
    print("=" * 50)
    
    # Define a scenario where split delivery is beneficial
    print("📋 Setting up split delivery scenario...")
    
    # Graph structure
    offsets = np.array([0, 3, 5, 7, 8, 9])
    edges = np.array([1, 2, 3, 0, 2, 0, 3, 4, 0])
    weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    time_to_travel = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    
    # Scenario designed to benefit from split delivery
    target_locations = np.array([1, 3])
    depot_locations = [0, 4]
    vehicle_capacities = [6, 4, 3, 2]  # Small capacities
    initial_stock = [2, 1, 1, 0]       # Low stock levels
    delivery_demands = [3, 5]          # One demand that might benefit from splitting
    
    print(f"   Stores: {target_locations}")
    print(f"   Depots: {depot_locations}")
    print(f"   Vehicles: {len(vehicle_capacities)} with capacities {vehicle_capacities}")
    print(f"   Initial stock: {initial_stock}")
    print(f"   Demands: {delivery_demands}")
    print(f"   Note: Demand of 5 units might be split across vehicles")
    
    # Create the enhanced system
    print("\n🔧 Creating EnhancedRefillSystem...")
    system = EnhancedRefillSystem(
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
    
    # Solve with enhanced optimization
    print("\n🎯 Solving with enhanced optimization (including split delivery)...")
    result = system.solve_enhanced()
    
    # Display results
    print("\n📊 ENHANCED OPTIMIZATION RESULTS")
    print("=" * 40)
    
    if result:
        print(f"✅ Optimization completed successfully!")
        print(f"   💰 Total cost: {result['total_cost']:.1f}")
        print(f"   🚚 Vehicles used: {result['vehicles_used']}")
        print(f"   📊 Strategy breakdown: {result['strategy_counts']}")
        
        # Check if split delivery was used
        if 'split' in result['strategy_counts']:
            print(f"\n✨ Split delivery was used: {result['strategy_counts']['split']} times")
            
            # Show split delivery details
            for assignment in result['assignments']:
                if assignment['strategy'] == 'split':
                    print(f"   Task {assignment['task_id']}: {assignment['details']}")
        else:
            print(f"\nℹ️  No split delivery was needed for this scenario")
        
        print(f"\n🚚 Vehicle Routes:")
        for vehicle_id, route in result['vehicle_routes'].items():
            print(f"   Vehicle {vehicle_id}: {route}")
        
        # Show detailed cost breakdown
        print(f"\n💰 Detailed Cost Analysis:")
        for assignment in result['assignments']:
            strategy = assignment['strategy']
            cost = assignment['cost']
            location = assignment['location']
            demand = assignment['demand']
            print(f"   Location {location} (demand {demand}): {strategy} - {cost:.1f}")
        
    else:
        print("❌ Optimization failed!")
    
    print(f"\n🎉 Split delivery demonstration completed!")


def compare_with_basic_system():
    """Compare with basic system to show benefits of split delivery."""
    print("\n" + "=" * 60)
    print("🔍 COMPARISON: Enhanced vs Basic System")
    print("=" * 60)
    
    # Same problem parameters
    offsets = np.array([0, 3, 5, 7, 8, 9])
    edges = np.array([1, 2, 3, 0, 2, 0, 3, 4, 0])
    weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    time_to_travel = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    
    target_locations = np.array([1, 3])
    depot_locations = [0, 4]
    vehicle_capacities = [6, 4, 3, 2]
    initial_stock = [2, 1, 1, 0]
    delivery_demands = [3, 5]
    
    # Test enhanced system
    print("🚀 Testing Enhanced System (with split delivery)...")
    enhanced_system = EnhancedRefillSystem(
        offsets=offsets, edges=edges, weights=weights, time_to_travel=time_to_travel,
        target_locations=target_locations, depot_locations=depot_locations,
        vehicle_capacities=vehicle_capacities, initial_stock=initial_stock,
        delivery_demands=delivery_demands
    )
    
    enhanced_result = enhanced_system.solve_enhanced()
    
    # Test basic system
    print("\n🔄 Testing Basic System (without split delivery)...")
    from optimus import OptimizedRefillSystem
    
    basic_system = OptimizedRefillSystem(
        offsets=offsets, edges=edges, weights=weights, time_to_travel=time_to_travel,
        target_locations=target_locations, depot_locations=depot_locations,
        vehicle_capacities=vehicle_capacities, initial_stock=initial_stock,
        delivery_demands=delivery_demands
    )
    
    basic_result = basic_system.solve_optimized()
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS:")
    if enhanced_result and basic_result:
        print(f"   Enhanced cost: {enhanced_result['total_cost']:.1f}")
        print(f"   Basic cost: {basic_result['total_cost']:.1f}")
        
        if enhanced_result['total_cost'] < basic_result['total_cost']:
            improvement = ((basic_result['total_cost'] - enhanced_result['total_cost']) / basic_result['total_cost']) * 100
            print(f"   💰 Cost improvement: {improvement:.1f}%")
        else:
            print(f"   ℹ️  No significant improvement with split delivery")
        
        print(f"   Enhanced vehicles: {enhanced_result['vehicles_used']}")
        print(f"   Basic vehicles: {basic_result['vehicles_used']}")
    else:
        print("   ❌ Could not compare - one or both systems failed")


if __name__ == "__main__":
    main()
    compare_with_basic_system()
