#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the new pymeshio animation functionality.

This script demonstrates how to use the AnimatedModel class to query
bone world positions from PMX and VMD files.
"""

import sys
import os

# Add current directory to Python path for pymeshio import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import torch
    print(f"PyTorch available: {torch.__version__}")
except ImportError:
    print("PyTorch not available - animation functionality requires PyTorch")
    sys.exit(1)

try:
    # Import the new animation functionality
    from pymeshio.animation import AnimatedModel, create_animated_model
    print("Successfully imported pymeshio animation module")
except ImportError as e:
    print(f"Failed to import animation module: {e}")
    sys.exit(1)


def test_animation_basic():
    """Test basic animation functionality."""
    print("\n" + "=" * 60)
    print("TESTING PYMESHIO ANIMATION FUNCTIONALITY")
    print("=" * 60)
    
    # Use test files if they exist
    pmx_path = "test/pdtt.pmx"
    vmd_path = "test/dan_alivef_01.imo.vmd"
    
    if not os.path.exists(pmx_path):
        print(f"PMX test file not found: {pmx_path}")
        return False
    
    if not os.path.exists(vmd_path):
        print(f"VMD test file not found: {vmd_path}")
        return False
    
    print(f"Loading PMX: {pmx_path}")
    print(f"Loading VMD: {vmd_path}")
    
    try:
        # Create animated model using convenience function
        animated_model = create_animated_model(pmx_path, vmd_path, device='cpu')
        print(f"✓ Successfully created AnimatedModel")
        
        # Get basic info
        bone_names = animated_model.get_bone_names()
        available_frames = animated_model.get_available_frames()
        
        print(f"✓ Model contains {len(bone_names)} bones")
        print(f"✓ Animation contains {len(available_frames)} frames")
        
        if available_frames:
            # Test querying a specific frame
            test_frame = available_frames[0] if len(available_frames) > 0 else 0
            print(f"\nTesting frame {test_frame}:")
            
            # Get world positions for all bones
            world_positions = animated_model.get_world_positions(test_frame)
            print(f"✓ Retrieved world positions for {len(world_positions)} bones")
            
            # Show some example bone positions
            sample_bones = list(world_positions.keys())[:5]  # First 5 bones
            for bone_name in sample_bones:
                pos = world_positions[bone_name]
                rest_pos = animated_model.get_rest_position(bone_name)
                print(f"  {bone_name}:")
                print(f"    Rest: [{rest_pos[0]:.3f}, {rest_pos[1]:.3f}, {rest_pos[2]:.3f}]")
                print(f"    World: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            # Test single bone query
            if sample_bones:
                bone_name = sample_bones[0]
                single_pos = animated_model.get_bone_world_position(bone_name, test_frame)
                print(f"\n✓ Single bone query for '{bone_name}': [{single_pos[0]:.3f}, {single_pos[1]:.3f}, {single_pos[2]:.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during animation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_animation_performance():
    """Test animation performance with caching."""
    print("\n" + "=" * 60)
    print("TESTING ANIMATION PERFORMANCE")
    print("=" * 60)
    
    pmx_path = "test/pdtt.pmx"
    vmd_path = "test/dan_alivef_01.imo.vmd"
    
    if not (os.path.exists(pmx_path) and os.path.exists(vmd_path)):
        print("Test files not available, skipping performance test")
        return True
    
    try:
        import time
        
        animated_model = create_animated_model(pmx_path, vmd_path, device='cpu')
        available_frames = animated_model.get_available_frames()
        
        if len(available_frames) < 3:
            print("Not enough frames for performance test")
            return True
        
        test_frames = available_frames[:3]  # Test first 3 frames
        
        # Test without cache
        print("Testing without cache...")
        start_time = time.time()
        for frame in test_frames:
            positions = animated_model.get_world_positions(frame, use_cache=False)
        no_cache_time = time.time() - start_time
        
        # Test with cache
        print("Testing with cache...")
        animated_model.clear_cache()  # Clear any existing cache
        start_time = time.time()
        for frame in test_frames:
            positions = animated_model.get_world_positions(frame, use_cache=True)
        # Second pass should be faster due to caching
        for frame in test_frames:
            positions = animated_model.get_world_positions(frame, use_cache=True)
        cache_time = time.time() - start_time
        
        print(f"✓ Without cache: {no_cache_time:.3f}s")
        print(f"✓ With cache: {cache_time:.3f}s")
        print(f"✓ Speedup: {no_cache_time/cache_time:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def main():
    """Run all animation tests."""
    print("PyTorch Animation Module Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Basic functionality
    if test_animation_basic():
        tests_passed += 1
    
    # Test 2: Performance
    if test_animation_performance():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Animation module is working correctly.")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)