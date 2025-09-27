#!/usr/bin/env python3
"""
Simple test to verify the Optimus system works.
"""

import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports."""
    print("🧪 Testing imports...")
    
    try:
        from optimus import OptimizedRefillSystem
        print("✓ OptimizedRefillSystem imported")
    except Exception as e:
        print(f"✗ Error importing OptimizedRefillSystem: {e}")
        return False
    
    try:
        from optimus import EnhancedRefillSystem
        print("✓ EnhancedRefillSystem imported")
    except Exception as e:
        print(f"✗ Error importing EnhancedRefillSystem: {e}")
        return False
    
    try:
        from optimus.core.vehicle import VehicleState
        print("✓ VehicleState imported")
    except Exception as e:
        print(f"✗ Error importing VehicleState: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from optimus import OptimizedRefillSystem
        
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
        
        # Create system
        system = OptimizedRefillSystem(
            offsets=offsets, edges=edges, weights=weights, time_to_travel=time_to_travel,
            target_locations=target_locations, depot_locations=depot_locations,
            vehicle_capacities=vehicle_capacities, initial_stock=initial_stock,
            delivery_demands=delivery_demands
        )
        
        print("✓ System created successfully")
        
        # Test solving
        result = system.solve_optimized()
        
        if result:
            print(f"✓ System solved successfully: cost={result['total_cost']:.1f}")
            return True
        else:
            print("✗ System failed to solve")
            return False
            
    except Exception as e:
        print(f"✗ Error in basic functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple tests."""
    print("🚀 Optimus Simple Test")
    print("=" * 30)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed!")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed!")
        return False
    
    print("\n✅ All tests passed!")
    print("The Optimus system is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
