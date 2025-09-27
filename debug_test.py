#!/usr/bin/env python3
"""
Debug script to identify issues with the Optimus system.
"""

import sys
import os

print("🔍 DEBUGGING OPTIMUS SYSTEM")
print("=" * 40)

# Check current directory
print(f"Current directory: {os.getcwd()}")
print(f"Python version: {sys.version}")

# Check if src directory exists
src_path = os.path.join(os.getcwd(), 'src')
print(f"Source path: {src_path}")
print(f"Source exists: {os.path.exists(src_path)}")

if os.path.exists(src_path):
    print("Contents of src directory:")
    for item in os.listdir(src_path):
        print(f"  - {item}")

# Add src to path
sys.path.insert(0, src_path)
print(f"Python path: {sys.path[:3]}")

# Test basic imports
print("\nTesting basic imports...")

try:
    print("Importing sys, os, numpy...")
    import numpy as np
    print("✅ numpy imported")
except Exception as e:
    print(f"❌ numpy import failed: {e}")

try:
    print("Importing cuopt...")
    import cuopt
    print("✅ cuopt imported")
except Exception as e:
    print(f"❌ cuopt import failed: {e}")

try:
    print("Importing from src.core...")
    from src.core.vehicle import VehicleState
    print("✅ VehicleState imported")
except Exception as e:
    print(f"❌ VehicleState import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Importing from src.algorithms...")
    from src.algorithms.refill_optimizer import OptimizedRefillSystem
    print("✅ OptimizedRefillSystem imported")
except Exception as e:
    print(f"❌ OptimizedRefillSystem import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug completed!")

