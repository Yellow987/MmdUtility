#!/usr/bin/env python3
"""
Test script to validate the corrected MMD bone transform system.
Compares the new corrected method with the legacy method to demonstrate
the bone length preservation fix.
"""

import os
import sys

def test_corrected_transforms():
    """Test the corrected transform system and validate bone length preservation."""
    print("="*80)
    print("CORRECTED MMD BONE TRANSFORM VALIDATION TEST")
    print("="*80)
    
    # Define test file paths
    pmx_path = "test/pdtt.pmx"
    vmd_path = "test/dan_alivef_01.imo.vmd"
    
    # Check if files exist
    if not os.path.exists(pmx_path) or not os.path.exists(vmd_path):
        print(f"✗ Test files not found: {pmx_path}, {vmd_path}")
        return False
    
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        
        # Load files
        print("Loading PMX and VMD files...")
        extractor.load_pmx(pmx_path)
        extractor.load_vmd(vmd_path)
        
        target_frame = 3000
        
        print(f"\n" + "="*80)
        print(f"TESTING CORRECTED VS LEGACY METHODS FOR FRAME {target_frame}")
        print("="*80)
        
        # Test 1: Get rest pose positions for reference
        print("\n1. Getting rest pose positions for validation...")
        rest_positions = extractor.get_rest_pose_positions()
        rest_bone_lengths = extractor.calculate_rest_bone_lengths()
        
        print(f"   Rest pose contains {len(rest_positions)} bones")
        print(f"   Calculated {len(rest_bone_lengths)} bone lengths")
        
        # Test 2: Apply corrected method with validation
        print(f"\n2. Using CORRECTED method with bone length validation:")
        try:
            world_positions_corrected = extractor.apply_frame_transforms_to_skeleton(
                target_frame, use_interpolation=True, validate_lengths=True
            )
            corrected_success = True
            print(f"   ✓ Corrected method completed successfully")
        except Exception as e:
            print(f"   ✗ Corrected method failed: {e}")
            corrected_success = False
            world_positions_corrected = {}
        
        # Test 3: Apply legacy method for comparison
        print(f"\n3. Using LEGACY method for comparison:")
        try:
            world_positions_legacy = extractor.apply_frame_transforms_to_skeleton_legacy(
                target_frame, use_interpolation=True
            )
            legacy_success = True
            print(f"   ✓ Legacy method completed successfully")
        except Exception as e:
            print(f"   ✗ Legacy method failed: {e}")
            legacy_success = False
            world_positions_legacy = {}
        
        if not corrected_success and not legacy_success:
            print("✗ Both methods failed!")
            return False
        
        # Test 4: Detailed comparison
        print(f"\n" + "="*80)
        print("DETAILED COMPARISON RESULTS")
        print("="*80)
        
        print(f"{'Bone Name':<15} | {'Rest Pose':<20} | {'Corrected':<20} | {'Legacy':<20} | {'Rest Length':<12} | {'Corrected Len':<12} | {'Legacy Len':<12}")
        print("-" * 130)
        
        # Get all bones that appear in any result
        all_bones = set()
        if world_positions_corrected:
            all_bones.update(world_positions_corrected.keys())
        if world_positions_legacy:
            all_bones.update(world_positions_legacy.keys())
        all_bones.update(rest_positions.keys())
        
        bone_length_errors_corrected = []
        bone_length_errors_legacy = []
        
        for bone_name in sorted(all_bones):
            if bone_name in rest_positions:
                rest_pos = rest_positions[bone_name]
                corrected_pos = world_positions_corrected.get(bone_name, rest_pos)
                legacy_pos = world_positions_legacy.get(bone_name, rest_pos)
                
                rest_str = f"({rest_pos[0]:.2f},{rest_pos[1]:.2f},{rest_pos[2]:.2f})"
                corrected_str = f"({corrected_pos[0]:.2f},{corrected_pos[1]:.2f},{corrected_pos[2]:.2f})"
                legacy_str = f"({legacy_pos[0]:.2f},{legacy_pos[1]:.2f},{legacy_pos[2]:.2f})"
                
                # Calculate bone length validation for child bones
                rest_length = rest_bone_lengths.get(bone_name, 0.0)
                corrected_length = "N/A"
                legacy_length = "N/A"
                
                # Find parent bone to calculate actual lengths
                for bone_index, bone_info in extractor.bone_hierarchy.items():
                    if bone_info['name'] == bone_name:
                        parent_idx = bone_info['parent_index']
                        if parent_idx >= 0 and parent_idx in extractor.bone_hierarchy:
                            parent_name = extractor.bone_hierarchy[parent_idx]['name']
                            
                            if parent_name in world_positions_corrected and bone_name in world_positions_corrected:
                                parent_pos_corrected = world_positions_corrected[parent_name]
                                child_pos_corrected = world_positions_corrected[bone_name]
                                corrected_actual_length = ((child_pos_corrected[0] - parent_pos_corrected[0])**2 + 
                                                         (child_pos_corrected[1] - parent_pos_corrected[1])**2 + 
                                                         (child_pos_corrected[2] - parent_pos_corrected[2])**2)**0.5
                                corrected_length = f"{corrected_actual_length:.3f}"
                                
                                # Track length errors
                                if rest_length > 0.01:  # Only track non-zero lengths
                                    length_error = abs(corrected_actual_length - rest_length)
                                    bone_length_errors_corrected.append((bone_name, length_error, rest_length))
                            
                            if parent_name in world_positions_legacy and bone_name in world_positions_legacy:
                                parent_pos_legacy = world_positions_legacy[parent_name]
                                child_pos_legacy = world_positions_legacy[bone_name]
                                legacy_actual_length = ((child_pos_legacy[0] - parent_pos_legacy[0])**2 + 
                                                       (child_pos_legacy[1] - parent_pos_legacy[1])**2 + 
                                                       (child_pos_legacy[2] - parent_pos_legacy[2])**2)**0.5
                                legacy_length = f"{legacy_actual_length:.3f}"
                                
                                # Track length errors
                                if rest_length > 0.01:  # Only track non-zero lengths
                                    length_error = abs(legacy_actual_length - rest_length)
                                    bone_length_errors_legacy.append((bone_name, length_error, rest_length))
                        break
                
                rest_length_str = f"{rest_length:.3f}" if rest_length > 0.01 else "root"
                
                print(f"{bone_name:<15} | {rest_str:<20} | {corrected_str:<20} | {legacy_str:<20} | {rest_length_str:<12} | {corrected_length:<12} | {legacy_length:<12}")
        
        # Test 5: Bone length validation summary
        print(f"\n" + "="*80)
        print("BONE LENGTH VALIDATION SUMMARY")
        print("="*80)
        
        if bone_length_errors_corrected:
            corrected_max_error = max(error for _, error, _ in bone_length_errors_corrected)
            corrected_avg_error = sum(error for _, error, _ in bone_length_errors_corrected) / len(bone_length_errors_corrected)
            corrected_valid_count = sum(1 for _, error, _ in bone_length_errors_corrected if error < 0.01)
            
            print(f"CORRECTED METHOD:")
            print(f"  Total bones with length validation: {len(bone_length_errors_corrected)}")
            print(f"  Bones with valid lengths (error < 0.01): {corrected_valid_count}")
            print(f"  Maximum length error: {corrected_max_error:.6f}")
            print(f"  Average length error: {corrected_avg_error:.6f}")
            
            # Show worst errors
            worst_corrected = sorted(bone_length_errors_corrected, key=lambda x: x[1], reverse=True)[:5]
            print(f"  Worst length errors:")
            for bone_name, error, rest_length in worst_corrected:
                print(f"    {bone_name}: error={error:.6f}, expected={rest_length:.3f}")
        
        if bone_length_errors_legacy:
            legacy_max_error = max(error for _, error, _ in bone_length_errors_legacy)
            legacy_avg_error = sum(error for _, error, _ in bone_length_errors_legacy) / len(bone_length_errors_legacy)
            legacy_valid_count = sum(1 for _, error, _ in bone_length_errors_legacy if error < 0.01)
            
            print(f"\nLEGACY METHOD:")
            print(f"  Total bones with length validation: {len(bone_length_errors_legacy)}")
            print(f"  Bones with valid lengths (error < 0.01): {legacy_valid_count}")
            print(f"  Maximum length error: {legacy_max_error:.6f}")
            print(f"  Average length error: {legacy_avg_error:.6f}")
            
            # Show worst errors
            worst_legacy = sorted(bone_length_errors_legacy, key=lambda x: x[1], reverse=True)[:5]
            print(f"  Worst length errors:")
            for bone_name, error, rest_length in worst_legacy:
                print(f"    {bone_name}: error={error:.6f}, expected={rest_length:.3f}")
        
        # Test 6: Overall assessment
        print(f"\n" + "="*80)
        print("OVERALL ASSESSMENT")
        print("="*80)
        
        improvement_detected = False
        
        if bone_length_errors_corrected and bone_length_errors_legacy:
            if corrected_avg_error < legacy_avg_error:
                improvement_ratio = legacy_avg_error / corrected_avg_error if corrected_avg_error > 0 else float('inf')
                print(f"✓ IMPROVEMENT DETECTED!")
                print(f"  Average length error reduced by {improvement_ratio:.2f}x")
                print(f"  From {legacy_avg_error:.6f} (legacy) to {corrected_avg_error:.6f} (corrected)")
                improvement_detected = True
            else:
                print(f"⚠ No significant improvement in average error")
                print(f"  Corrected: {corrected_avg_error:.6f}, Legacy: {legacy_avg_error:.6f}")
            
            if corrected_valid_count > legacy_valid_count:
                print(f"✓ More bones with valid lengths: {corrected_valid_count} vs {legacy_valid_count}")
                improvement_detected = True
        
        if corrected_success and not legacy_success:
            print(f"✓ Corrected method succeeded while legacy failed")
            improvement_detected = True
        
        if improvement_detected:
            print(f"\n🎉 Corrected transform system shows measurable improvements!")
        else:
            print(f"\n⚠ No clear improvement detected - further investigation needed")
        
        return improvement_detected
        
    except Exception as e:
        print(f"✗ Error during corrected transform test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run corrected transform validation test."""
    print("Starting corrected MMD transform validation...")
    
    success = test_corrected_transforms()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 Transform correction validation completed successfully!")
        print("The corrected bone transform system shows improvements!")
    else:
        print("❌ Transform correction validation failed or showed no improvement!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)