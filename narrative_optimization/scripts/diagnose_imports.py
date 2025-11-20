#!/usr/bin/env python3
"""
Diagnostic script to identify which import is causing the bus error.
Run this to isolate the problematic module.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Platform: {sys.platform}")
print()

# Test imports one by one
imports_to_test = [
    ("sys", "sys"),
    ("os", "os"),
    ("json", "json"),
    ("pathlib", "Path"),
    ("time", "time"),
    ("numpy", "np"),
    ("sklearn", "sklearn"),
    ("pandas", "pd"),
    ("yaml", "yaml"),
]

print("=" * 80)
print("TESTING IMPORTS ONE BY ONE")
print("=" * 80)
print()

failed_imports = []

for module_name, import_as in imports_to_test:
    try:
        print(f"Testing import: {module_name}...", end=" ", flush=True)
        if import_as:
            exec(f"import {module_name} as {import_as}")
        else:
            exec(f"import {module_name}")
        print("✓ OK")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        failed_imports.append((module_name, str(e)))
        if "bus error" in str(e).lower() or isinstance(e, SystemError):
            print(f"  ⚠ BUS ERROR DETECTED!")
            break

print()
print("=" * 80)
if failed_imports:
    print("FAILED IMPORTS:")
    for module, error in failed_imports:
        print(f"  - {module}: {error}")
else:
    print("ALL IMPORTS SUCCESSFUL")
print("=" * 80)

# Test numpy specifically (common culprit)
print()
print("Testing numpy operations...")
try:
    import numpy as np
    arr = np.array([1, 2, 3])
    print(f"✓ NumPy array creation: OK")
    print(f"  NumPy version: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy operations failed: {e}")

# Test sklearn specifically
print()
print("Testing sklearn operations...")
try:
    from sklearn.pipeline import Pipeline
    print(f"✓ sklearn Pipeline import: OK")
except Exception as e:
    print(f"✗ sklearn import failed: {e}")

