#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance test script for BatchAnimatedModel vs original AnimatedModel.

Tests both correctness and performance improvements of the batch processing approach.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except ImportError:
    print("ERROR: PyTorch not available")
    sys.exit(1)

try:
    from pymeshio.animation import AnimatedModel, BatchAnimatedModel, create_animated_model, create_batch_animated_model
    print("✓ Animation modules loaded successfully")
except ImportError as e:
    print(f"ERROR: Failed to import animation modules: {e}")
    sys.exit(1)


def find_test_files():
    """Find test PMX and VMD files."""
    # Look for test files in common locations
    test_pmx = None
    test_vmd = None
    
    # Check if we have test files in the current directory
    current_path = Path(current_dir)
    
    # Look for PMX files
    for pmx_pattern in ['*.pmx', 'test/*.pmx', '../**/defaultModel.pmx']:
        pmx_files = list(current_path.glob(pmx_pattern))
        if pmx_files:
            test_pmx = str(pmx_files[0])
            break
    
    # Look for VMD files  
    for vmd_pattern in ['*.vmd', 'test/*.vmd', '../**/*.vmd']:
        vmd_files = list(current_path.glob(vmd_pattern))
        if vmd_files:
            test_vmd = str(vmd_files[0])
            break
    
    return test_pmx, test_vmd


def test_single_vs_batch_correctness(pmx_path: str, vmd_path: str, test_bones: list = None):
    """Test that batch processing produces the same results as single processing."""
    print(f"\n{'='*60}")
    print("CORRECTNESS TEST: Single vs Batch Processing")
    print(f"{'='*60}")
    
    if test_bones is None:
        test_bones = ["センター", "腰", "上半身", "頭"]  # A few core bones
    
    # Test single processing
    print("Testing single processing...")
    single_model = create_animated_model(pmx_path, vmd_path, device='cpu')
    frames = single_model.get_available_frames()[:10]  # Test first 10 frames
    
    single_results = {}
    for frame in frames:
        positions = single_model.get_world_positions(frame)
        single_results[frame] = {bone: positions.get(bone, np.zeros(3)) for bone in test_bones}
    
    # Test batch processing
    print("Testing batch processing...")
    batch_model = create_batch_animated_model(pmx_path, device='cpu', chunk_size=5)
    batch_data = batch_model.process_vmd_file(vmd_path, filter_bones=test_bones)
    
    if batch_data is None:
        print("ERROR: Batch processing failed")
        return False
    
    # Compare results
    print("Comparing results...")
    max_diff = 0.0
    total_comparisons = 0
    
    for i, frame in enumerate(frames):
        if frame in single_results and i < len(batch_data["frame_numbers"]):
            for j, bone in enumerate(test_bones):
                single_pos = single_results[frame][bone]
                batch_pos = batch_data["positions"][i, j]
                
                diff = np.linalg.norm(single_pos - batch_pos)
                max_diff = max(max_diff, diff)
                total_comparisons += 1
    
    print(f"✓ Compared {total_comparisons} bone positions")
    print(f"✓ Maximum difference: {max_diff:.6f}")
    
    # Results are correct if difference is very small (floating point precision)
    success = max_diff < 1e-4
    if success:
        print("✅ CORRECTNESS TEST PASSED - Results are identical!")
    else:
        print("❌ CORRECTNESS TEST FAILED - Results differ significantly!")
    
    return success


def test_performance_comparison(pmx_path: str, vmd_path: str, test_bones: list = None):
    """Compare performance between single and batch processing."""
    print(f"\n{'='*60}")
    print("PERFORMANCE TEST: Single vs Batch Processing")
    print(f"{'='*60}")
    
    if test_bones is None:
        test_bones = ["センター", "腰", "上半身", "頭", "左肩", "右肩", "左腕", "右腕"]
    
    # Test single processing performance
    print("Benchmarking single processing...")
    start_time = time.time()
    
    single_model = create_animated_model(pmx_path, vmd_path, device='cpu')
    frames = single_model.get_available_frames()[:100]  # Test first 100 frames
    
    single_results = []
    for frame in frames:
        positions = single_model.get_world_positions(frame)
        frame_data = {bone: positions.get(bone, np.zeros(3)) for bone in test_bones}
        single_results.append(frame_data)
    
    single_time = time.time() - start_time
    single_fps = len(frames) / single_time if single_time > 0 else 0
    
    # Test batch processing performance  
    print("Benchmarking batch processing...")
    start_time = time.time()
    
    batch_model = create_batch_animated_model(pmx_path, device='cpu', chunk_size=50)
    batch_data = batch_model.process_vmd_file(vmd_path, filter_bones=test_bones)
    
    batch_time = time.time() - start_time
    batch_frames = len(batch_data["frame_numbers"]) if batch_data else 0
    batch_fps = batch_frames / batch_time if batch_time > 0 else 0
    
    # Calculate speedup
    speedup = batch_fps / single_fps if single_fps > 0 else 0
    
    print(f"\n📊 PERFORMANCE RESULTS:")
    print(f"  Single Processing:")
    print(f"    - Frames processed: {len(frames)}")
    print(f"    - Time taken: {single_time:.3f}s")
    print(f"    - Speed: {single_fps:.1f} fps")
    print(f"  Batch Processing:")
    print(f"    - Frames processed: {batch_frames}")
    print(f"    - Time taken: {batch_time:.3f}s") 
    print(f"    - Speed: {batch_fps:.1f} fps")
    print(f"  🚀 Speedup: {speedup:.1f}x faster")
    
    if speedup > 2.0:
        print("✅ PERFORMANCE TEST PASSED - Significant speedup achieved!")
        return True
    else:
        print("⚠️ PERFORMANCE TEST WARNING - Expected higher speedup")
        return False


def main():
    """Run all tests."""
    print("VMD Batch Processing Performance Test")
    print("=====================================")
    
    # Find test files
    pmx_path, vmd_path = find_test_files()
    
    if not pmx_path or not vmd_path:
        print("ERROR: Could not find test PMX and VMD files")
        print("Please ensure you have test files available, or run from the mmd-dance directory")
        return False
    
    print(f"Using test files:")
    print(f"  PMX: {pmx_path}")
    print(f"  VMD: {vmd_path}")
    
    # Run tests
    correctness_passed = test_single_vs_batch_correctness(pmx_path, vmd_path)
    performance_passed = test_performance_comparison(pmx_path, vmd_path)
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Correctness Test: {'✅ PASSED' if correctness_passed else '❌ FAILED'}")
    print(f"Performance Test: {'✅ PASSED' if performance_passed else '⚠️ WARNING'}")
    
    overall_success = correctness_passed and performance_passed
    print(f"\nOverall: {'✅ SUCCESS' if overall_success else '❌ NEEDS ATTENTION'}")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)