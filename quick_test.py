#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

print("🚀 OPTIMUS QUICK TEST")
print("=" * 25)

# Test imports
print("Testing imports...")
try:
    from optimus import OptimizedRefillSystem
    print("✅ OptimizedRefillSystem imported")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Test basic functionality
print("Testing basic functionality...")
import numpy as np

offsets = np.array([0, 3, 5, 7, 8, 9])
edges = np.array([1, 2, 3, 0, 2, 0, 3, 4, 0])
weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
time_to_travel = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

target_locations = np.array([1, 3])
depot_locations = [0, 4]
vehicle_capacities = [15, 12, 10]
initial_stock = [5, 3, 2]
delivery_demands = [3, 4]

try:
    system = OptimizedRefillSystem(
        offsets=offsets, edges=edges, weights=weights, time_to_travel=time_to_travel,
        target_locations=target_locations, depot_locations=depot_locations,
        vehicle_capacities=vehicle_capacities, initial_stock=initial_stock,
        delivery_demands=delivery_demands
    )
    print("✅ System created successfully")
    
    result = system.solve_optimized()
    if result:
        print(f"✅ Optimization successful! Cost: {result['total_cost']:.1f}")
    else:
        print("❌ Optimization failed")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed!")

