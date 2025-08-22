#!/usr/bin/env python
# coding: utf-8
"""
Test script to verify pytransform3d concat behavior
"""

import numpy as np
from pytransform3d.transformations import transform_from, concat

def test_pytransform3d_concat():
    """Test pytransform3d concat function with multiple matrices"""
    
    print("=== Testing pytransform3d concat function ===\n")
    
    # Create three simple transformation matrices
    # 1. Rest: translate by (1, 0, 0)
    rest_transform = transform_from(R=np.eye(3), p=np.array([1.0, 0.0, 0.0]))
    
    # 2. Rotation: identity (no rotation)
    rotation_transform = transform_from(R=np.eye(3), p=np.array([0.0, 0.0, 0.0]))
    
    # 3. Animation: translate by (2, 3, 4)
    anim_transform = transform_from(R=np.eye(3), p=np.array([2.0, 3.0, 4.0]))
    
    print("Input matrices:")
    print("Rest transform (translate by 1,0,0):")
    print(rest_transform)
    print("\nRotation transform (identity):")
    print(rotation_transform)
    print("\nAnimation transform (translate by 2,3,4):")
    print(anim_transform)
    
    # Test 1: Sequential concatenation (what works in debug)
    print("\n=== Method 1: Sequential concatenation ===")
    step1 = concat(rest_transform, rotation_transform)
    step2 = concat(step1, anim_transform)
    
    print("Step 1 (rest * rotation):")
    print(f"Translation: ({step1[0,3]:.3f}, {step1[1,3]:.3f}, {step1[2,3]:.3f})")
    print("Step 2 (step1 * animation):")
    print(f"Translation: ({step2[0,3]:.3f}, {step2[1,3]:.3f}, {step2[2,3]:.3f})")
    
    # Test 2: Three-matrix concatenation (what's broken)
    print("\n=== Method 2: Three-matrix concatenation ===")
    try:
        three_matrix = concat(rest_transform, rotation_transform, anim_transform)
        print("Three-matrix result:")
        print(f"Translation: ({three_matrix[0,3]:.3f}, {three_matrix[1,3]:.3f}, {three_matrix[2,3]:.3f})")
        print("Full matrix:")
        print(three_matrix)
    except Exception as e:
        print(f"Error with three-matrix concat: {e}")
    
    # Test 3: Manual matrix multiplication
    print("\n=== Method 3: Manual numpy matrix multiplication ===")
    manual = rest_transform @ rotation_transform @ anim_transform
    print("Manual result:")
    print(f"Translation: ({manual[0,3]:.3f}, {manual[1,3]:.3f}, {manual[2,3]:.3f})")
    
    # Test 4: Check concat function signature
    print("\n=== Method 4: Check concat function documentation ===")
    try:
        import inspect
        print("concat function signature:")
        print(inspect.signature(concat))
        print("\nconcat function docstring:")
        print(concat.__doc__)
    except Exception as e:
        print(f"Could not inspect concat function: {e}")

if __name__ == "__main__":
    test_pytransform3d_concat()