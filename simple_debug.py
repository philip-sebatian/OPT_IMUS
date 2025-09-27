#!/usr/bin/env python3
print("Starting simple debug test...")

import sys
import os

print(f"Current directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")

# Check if we can import basic modules
try:
    import numpy as np
    print("✅ NumPy imported successfully")
except Exception as e:
    print(f"❌ NumPy import failed: {e}")

# Check if cuOpt is available
try:
    import cuopt
    print("✅ cuOpt imported successfully")
except Exception as e:
    print(f"❌ cuOpt import failed: {e}")

# Check file structure
print("\nChecking file structure...")
src_dir = "src"
if os.path.exists(src_dir):
    print(f"✅ {src_dir} directory exists")
    for item in os.listdir(src_dir):
        print(f"  - {item}")
else:
    print(f"❌ {src_dir} directory not found")

print("Simple debug completed!")

