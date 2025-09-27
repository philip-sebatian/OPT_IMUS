#!/usr/bin/env python3
"""
Comprehensive test script for the Optimus routing system.

This script tests all major components and functionality of the system.
"""

import sys
import os
import numpy as np

# Import from the installed optimus package
from optimus import OptimizedRefillSystem, EnhancedRefillSystem

# Import core components directly from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from core.vehicle import VehicleState
from core.delivery_task import DeliveryTask
from core.depot_manager import DepotManager
from core.cost_calculator import CostCalculator
from utils.validators import InputValidator


def test_basic_functionality():
    """Test basic system functionality."""
    print("🧪 Testing Basic Functionality")
    print("=" * 40)
    
    # Create a simple problem
    offsets = np.array([0, 3, 5, 7, 8, 9])
    edges = np.array([1, 2, 3, 0, 2, 0, 3, 4, 0])
    weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    time_to_travel = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    
    target_locations = np.array([1, 3])
    depot_locations = [0, 4]
    vehicle_capacities = [15, 12, 10]
    initial_stock = [5, 3, 2]
    delivery_demands = [3, 4]
    
    # Test input validation
    print("  ✓ Testing input validation...")
    is_valid, errors = InputValidator.validate_problem_parameters(
        offsets, edges, weights, time_to_travel, target_locations,
        depot_locations, vehicle_capacities, initial_stock, delivery_demands
    )
    assert is_valid, f"Input validation failed: {errors}"
    print("    Input validation passed")
    
    # Test basic system
    print("  ✓ Testing OptimizedRefillSystem...")
    system = OptimizedRefillSystem(
        offsets=offsets, edges=edges, weights=weights, time_to_travel=time_to_travel,
        target_locations=target_locations, depot_locations=depot_locations,
        vehicle_capacities=vehicle_capacities, initial_stock=initial_stock,
        delivery_demands=delivery_demands
    )
    
    result = system.solve_optimized()
    assert result is not None, "Basic system failed to find solution"
    assert result['total_cost'] > 0, "Solution cost should be positive"
    print(f"    Basic system solved: cost={result['total_cost']:.1f}")
    
    # Test enhanced system
    print("  ✓ Testing EnhancedRefillSystem...")
    enhanced_system = EnhancedRefillSystem(
        offsets=offsets, edges=edges, weights=weights, time_to_travel=time_to_travel,
        target_locations=target_locations, depot_locations=depot_locations,
        vehicle_capacities=vehicle_capacities, initial_stock=initial_stock,
        delivery_demands=delivery_demands
    )
    
    enhanced_result = enhanced_system.solve_enhanced()
    assert enhanced_result is not None, "Enhanced system failed to find solution"
    assert enhanced_result['total_cost'] > 0, "Enhanced solution cost should be positive"
    print(f"    Enhanced system solved: cost={enhanced_result['total_cost']:.1f}")
    
    print("  ✅ Basic functionality tests passed")


def test_vehicle_state():
    """Test VehicleState class."""
    print("\n🧪 Testing VehicleState Class")
    print("=" * 40)
    
    # Test initialization
    vehicle = VehicleState(id=0, capacity=10, current_stock=5, position=0)
    assert vehicle.id == 0
    assert vehicle.capacity == 10
    assert vehicle.current_stock == 5
    assert vehicle.position == 0
    print("  ✓ Initialization test passed")
    
    # Test methods
    assert vehicle.can_carry(5) == True
    assert vehicle.can_carry(10) == True
    assert vehicle.can_carry(11) == False
    print("  ✓ can_carry test passed")
    
    assert vehicle.has_stock(5) == True
    assert vehicle.has_stock(3) == True
    assert vehicle.has_stock(6) == False
    print("  ✓ has_stock test passed")
    
    assert vehicle.can_deliver(5) == True
    assert vehicle.can_deliver(3) == True
    assert vehicle.can_deliver(6) == False
    print("  ✓ can_deliver test passed")
    
    # Test refill
    vehicle.refill()
    assert vehicle.current_stock == 10
    assert vehicle.is_full() == True
    print("  ✓ refill test passed")
    
    # Test delivery
    success = vehicle.deliver(3)
    assert success == True
    assert vehicle.current_stock == 7
    print("  ✓ delivery test passed")
    
    # Test movement
    vehicle.move_to(5)
    assert vehicle.position == 5
    assert vehicle.route == [0, 5]
    print("  ✓ movement test passed")
    
    print("  ✅ VehicleState tests passed")


def test_delivery_task():
    """Test DeliveryTask class."""
    print("\n🧪 Testing DeliveryTask Class")
    print("=" * 40)
    
    # Test initialization
    task = DeliveryTask(location=1, demand=5)
    assert task.location == 1
    assert task.demand == 5
    assert task.completed == False
    print("  ✓ Initialization test passed")
    
    # Test split delivery
    task.add_split_delivery(0, 2)
    task.add_split_delivery(1, 3)
    assert task.is_split_delivery() == True
    assert task.get_total_assigned() == 5
    assert task.is_fully_assigned() == True
    print("  ✓ Split delivery test passed")
    
    # Test validation
    assert task.is_valid() == True
    print("  ✓ Validation test passed")
    
    print("  ✅ DeliveryTask tests passed")


def test_depot_manager():
    """Test DepotManager class."""
    print("\n🧪 Testing DepotManager Class")
    print("=" * 40)
    
    # Create test data
    depot_locations = [0, 4]
    cost_matrix = np.array([
        [0, 1, 2, 3, 4],
        [1, 0, 1, 2, 3],
        [2, 1, 0, 1, 2],
        [3, 2, 1, 0, 1],
        [4, 3, 2, 1, 0]
    ])
    location_to_index = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    
    depot_manager = DepotManager(depot_locations, cost_matrix, location_to_index)
    
    # Test distance calculation
    distance = depot_manager.get_distance(1, 3)
    assert distance == 2.0
    print("  ✓ Distance calculation test passed")
    
    # Test nearest depot finding
    nearest_depot, dist = depot_manager.find_nearest_depot(1)
    assert nearest_depot == 0  # Should be depot 0 (distance 1 vs 3)
    assert dist == 1.0
    print("  ✓ Nearest depot test passed")
    
    # Test depot validation
    assert depot_manager.is_depot(0) == True
    assert depot_manager.is_depot(1) == False
    print("  ✓ Depot validation test passed")
    
    print("  ✅ DepotManager tests passed")


def test_cost_calculator():
    """Test CostCalculator class."""
    print("\n🧪 Testing CostCalculator Class")
    print("=" * 40)
    
    # Create test data
    depot_locations = [0, 4]
    cost_matrix = np.array([
        [0, 1, 2, 3, 4],
        [1, 0, 1, 2, 3],
        [2, 1, 0, 1, 2],
        [3, 2, 1, 0, 1],
        [4, 3, 2, 1, 0]
    ])
    location_to_index = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    
    depot_manager = DepotManager(depot_locations, cost_matrix, location_to_index)
    cost_calculator = CostCalculator(depot_manager)
    
    # Create test vehicle and task
    vehicle = VehicleState(id=0, capacity=10, current_stock=5, position=0)
    task = DeliveryTask(location=3, demand=3)
    
    # Test direct delivery cost
    cost = cost_calculator.calculate_direct_delivery_cost(vehicle, task)
    assert cost > 0
    print("  ✓ Direct delivery cost test passed")
    
    # Test refill cost
    cost = cost_calculator.calculate_refill_cost(vehicle, task)
    assert cost > 0
    print("  ✓ Refill cost test passed")
    
    print("  ✅ CostCalculator tests passed")


def test_large_scale():
    """Test large-scale problem."""
    print("\n🧪 Testing Large-Scale Problem")
    print("=" * 40)
    
    # Create a larger problem
    num_locations = 20
    offsets = []
    edges = []
    weights = []
    time_to_travel = []
    
    edge_id = 0
    for i in range(num_locations):
        offsets.append(edge_id)
        # Connect to next 3 locations
        for j in range(1, 4):
            if i + j < num_locations:
                edges.append(i + j)
                weight = np.random.uniform(1, 10)
                weights.append(weight)
                time_to_travel.append(weight * 2)
                edge_id += 1
    
    offsets.append(edge_id)
    offsets = np.array(offsets)
    edges = np.array(edges)
    weights = np.array(weights)
    time_to_travel = np.array(time_to_travel)
    
    # Problem parameters
    target_locations = np.array(list(range(1, 11)))  # 10 stores
    depot_locations = [0, 10, 15]  # 3 depots
    vehicle_capacities = [20, 15, 12, 10, 8, 6, 5, 4]  # 8 vehicles
    initial_stock = [10, 8, 6, 5, 4, 3, 2, 1]  # Different stock levels
    delivery_demands = np.random.randint(1, 6, size=10)  # Random demands
    
    print(f"  Problem size: {len(target_locations)} stores, {len(depot_locations)} depots, {len(vehicle_capacities)} vehicles")
    
    # Test enhanced system
    system = EnhancedRefillSystem(
        offsets=offsets, edges=edges, weights=weights, time_to_travel=time_to_travel,
        target_locations=target_locations, depot_locations=depot_locations,
        vehicle_capacities=vehicle_capacities, initial_stock=initial_stock,
        delivery_demands=delivery_demands
    )
    
    result = system.solve_enhanced()
    assert result is not None, "Large-scale system failed to find solution"
    assert result['total_cost'] > 0, "Large-scale solution cost should be positive"
    print(f"  Large-scale system solved: cost={result['total_cost']:.1f}, vehicles={result['vehicles_used']}")
    
    print("  ✅ Large-scale test passed")


def main():
    """Run all tests."""
    print("🚀 Optimus Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_vehicle_state()
        test_delivery_task()
        test_depot_manager()
        test_cost_calculator()
        test_large_scale()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The Optimus routing system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
