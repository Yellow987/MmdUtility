#!/usr/bin/env python3
"""Debug script to test pytransform3d import."""

import sys

print("Python path:", sys.path)
print("Python version:", sys.version)

try:
    import pytransform3d
    print("✓ pytransform3d module imported successfully")
    print(f"  Version: {pytransform3d.__version__}")
    print(f"  Location: {pytransform3d.__file__}")
except ImportError as e:
    print(f"✗ Failed to import pytransform3d: {e}")
    sys.exit(1)

# Check what's available in transformations module
try:
    import pytransform3d.transformations as pt
    print("✓ pytransform3d.transformations module imported")
    print("Available items:", [x for x in dir(pt) if not x.startswith('_')])
except ImportError as e:
    print(f"✗ Failed to import transformations: {e}")

# Try different import paths
try:
    from pytransform3d.transform_manager import TransformManager
    print("✓ TransformManager found in transform_manager module")
    
    # Test creating an instance
    tm = TransformManager()
    print("✓ TransformManager instance created")
    
except ImportError as e:
    print(f"✗ Failed to import TransformManager from transform_manager: {e}")
    
    try:
        from pytransform3d import TransformManager
        print("✓ TransformManager found in main module")
        
        tm = TransformManager()
        print("✓ TransformManager instance created")
        
    except ImportError as e2:
        print(f"✗ Failed to import TransformManager from main module: {e2}")
        
        # List all available classes
        try:
            import pytransform3d
            print("All pytransform3d modules:")
            for attr in dir(pytransform3d):
                if not attr.startswith('_'):
                    print(f"  {attr}")
        except Exception as e3:
            print(f"Error exploring pytransform3d: {e3}")

print("✓ Import exploration complete")