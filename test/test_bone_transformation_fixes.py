#!/usr/bin/env python
# coding: utf-8
"""
Comprehensive Test Suite for Bone Transformation Fixes
=====================================================

This script tests the corrected bone transformation logic against the issues
identified in the mathematical analysis:

1. Animation data misapplication 
2. Coordinate system confusion (underground feet)
3. Incorrect transformation order
4. Matrix accumulation direction issues

Usage:
    python test_bone_transformation_fixes.py

Expected Results:
- Foot bones should be near ground level (Y ≈ 0-2)
- Left and right feet should have different Y coordinates for dancing
- センター bone should get its own animation data
- Transformation chain should accumulate correctly
"""

import os
import sys
import numpy as np
from typing import Tuple, Dict, List, Optional

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pymeshio.pmx.reader as pmx_reader
import pymeshio.vmd.reader as vmd_reader
from pymeshio import common

# Import existing implementations for comparison
from bone_animation import get_bone_world_position, BoneHierarchyWalker
from bone_animation_pytransform3d import get_bone_world_position_pt3d
from bone_animation_corrected import (
    get_bone_world_position_corrected,
    get_bone_world_position_corrected_v2,
    get_bone_world_position_corrected_v3
)


class BoneTransformationTester:
    """Comprehensive tester for bone transformation fixes."""
    
    def __init__(self, pmx_model, vmd_motion):
        self.pmx_model = pmx_model
        self.vmd_motion = vmd_motion
        self.test_frame = 3000
        
        # Define test bones for comprehensive validation
        self.test_bones = {
            'center': ['センター', 'center'],
            'hip': ['腰', 'waist'], 
            'left_foot': ['左足', '左足首', '左つま先'],
            'right_foot': ['右足', '右足首', '右つま先'],
            'left_leg': ['左足', '左ひざ', '左足首'],
            'right_leg': ['右足', '右ひざ', '右足首']
        }
        
        self.results = {}
    
    def find_available_bone(self, bone_names: List[str]) -> Optional[str]:
        """Find the first available bone from a list of potential names."""
        for name in bone_names:
            result = BoneHierarchyWalker.find_bone_by_name(self.pmx_model.bones, name)
            if result is not None:
                return name
        return None
    
    def test_ground_level_validation(self) -> Dict:
        """Test 1: Validate that foot bones are near ground level."""
        print("=" * 60)
        print("TEST 1: Ground Level Validation (All Implementations)")
        print("=" * 60)
        
        results = {
            'passed': True,
            'details': {},
            'errors': []
        }
        
        foot_bone_types = ['left_foot', 'right_foot']
        
        # All implementations to test
        implementations = {
            'Original': get_bone_world_position,
            'Pytransform3d': get_bone_world_position_pt3d,
            'Corrected_v1': get_bone_world_position_corrected,
            'Corrected_v2': get_bone_world_position_corrected_v2,
            'Corrected_v3': get_bone_world_position_corrected_v3
        }
        
        for foot_type in foot_bone_types:
            bone_name = self.find_available_bone(self.test_bones[foot_type])
            if not bone_name:
                results['errors'].append(f"No {foot_type} bones found in model")
                results['passed'] = False
                continue
            
            print(f"\n{foot_type} ({bone_name}):")
            
            foot_results = {}
            
            for impl_name, impl_func in implementations.items():
                try:
                    pos = impl_func(self.pmx_model, self.vmd_motion, self.test_frame, bone_name)
                    ground_level_ok = -5.0 <= pos[1] <= 20.0
                    
                    foot_results[impl_name] = {
                        'position': pos,
                        'ground_level_ok': ground_level_ok
                    }
                    
                    status = "✅" if ground_level_ok else "❌"
                    print(f"  {impl_name:12}: ({pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}) {status}")
                    
                    if not ground_level_ok:
                        results['passed'] = False
                        
                except Exception as e:
                    print(f"  {impl_name:12}: ❌ Error: {e}")
                    results['errors'].append(f"Error in {impl_name} for {foot_type}: {e}")
                    results['passed'] = False
                    foot_results[impl_name] = {'error': str(e)}
            
            results['details'][f'{foot_type}_{bone_name}'] = foot_results
        
        return results
    
    def test_left_right_foot_differences(self) -> Dict:
        """Test 2: Validate that left and right feet have different positions (dancing motion)."""
        print("\n" + "=" * 60)
        print("TEST 2: Left/Right Foot Position Differences")
        print("=" * 60)
        
        results = {
            'passed': True,
            'details': {},
            'errors': []
        }
        
        left_bone = self.find_available_bone(self.test_bones['left_foot'])
        right_bone = self.find_available_bone(self.test_bones['right_foot'])
        
        if not left_bone or not right_bone:
            results['errors'].append("Could not find both left and right foot bones")
            results['passed'] = False
            print("❌ Could not find both left and right foot bones")
            return results
        
        try:
            # Test original implementation
            left_pos_orig = get_bone_world_position(self.pmx_model, self.vmd_motion, 
                                                  self.test_frame, left_bone)
            right_pos_orig = get_bone_world_position(self.pmx_model, self.vmd_motion,
                                                   self.test_frame, right_bone)
            
            # Test pytransform3d implementation  
            left_pos_pt3d = get_bone_world_position_pt3d(self.pmx_model, self.vmd_motion,
                                                       self.test_frame, left_bone)
            right_pos_pt3d = get_bone_world_position_pt3d(self.pmx_model, self.vmd_motion,
                                                        self.test_frame, right_bone)
            
            # Calculate differences
            y_diff_orig = abs(left_pos_orig[1] - right_pos_orig[1])
            y_diff_pt3d = abs(left_pos_pt3d[1] - right_pos_pt3d[1])
            
            results['details'] = {
                'left_original': left_pos_orig,
                'right_original': right_pos_orig,
                'left_pt3d': left_pos_pt3d,
                'right_pt3d': right_pos_pt3d,
                'y_diff_original': y_diff_orig,
                'y_diff_pt3d': y_diff_pt3d
            }
            
            print(f"Left foot ({left_bone}):")
            print(f"  Original:     ({left_pos_orig[0]:8.3f}, {left_pos_orig[1]:8.3f}, {left_pos_orig[2]:8.3f})")
            print(f"  Pytransform3d: ({left_pos_pt3d[0]:8.3f}, {left_pos_pt3d[1]:8.3f}, {left_pos_pt3d[2]:8.3f})")
            
            print(f"Right foot ({right_bone}):")
            print(f"  Original:     ({right_pos_orig[0]:8.3f}, {right_pos_orig[1]:8.3f}, {right_pos_orig[2]:8.3f})")
            print(f"  Pytransform3d: ({right_pos_pt3d[0]:8.3f}, {right_pos_pt3d[1]:8.3f}, {right_pos_pt3d[2]:8.3f})")
            
            print(f"\nY-coordinate differences:")
            print(f"  Original implementation: {y_diff_orig:.3f}")
            print(f"  Pytransform3d implementation: {y_diff_pt3d:.3f}")
            
            # Validate that feet have different positions (should be > 0.1 for dancing)
            if y_diff_orig < 0.1:
                print(f"  ❌ Original Y difference {y_diff_orig:.3f} too small (feet too similar)")
                results['passed'] = False
            else:
                print(f"  ✅ Original Y difference {y_diff_orig:.3f} indicates different positions")
            
            if y_diff_pt3d < 0.1:
                print(f"  ❌ Pytransform3d Y difference {y_diff_pt3d:.3f} too small (feet too similar)")
                results['passed'] = False
            else:
                print(f"  ✅ Pytransform3d Y difference {y_diff_pt3d:.3f} indicates different positions")
                
        except Exception as e:
            results['errors'].append(f"Error comparing left/right feet: {e}")
            results['passed'] = False
            print(f"❌ Error: {e}")
        
        return results
    
    def test_bone_chain_accumulation(self) -> Dict:
        """Test 3: Validate that bone chain transformations accumulate correctly."""
        print("\n" + "=" * 60)
        print("TEST 3: Bone Chain Transformation Accumulation (All Implementations)")
        print("=" * 60)
        
        results = {
            'passed': True,
            'details': {},
            'errors': []
        }
        
        # Test right leg chain: 右足 -> 右ひざ -> 右足首 -> 右つま先
        leg_bones = ['右足', '右ひざ', '右足首', '右つま先']
        available_bones = []
        
        for bone_name in leg_bones:
            if self.find_available_bone([bone_name]):
                available_bones.append(bone_name)
        
        if len(available_bones) < 2:
            results['errors'].append("Not enough leg bones available for chain testing")
            results['passed'] = False
            print("❌ Not enough leg bones available for chain testing")
            return results
        
        print(f"Testing bone chain: {' -> '.join(available_bones)}")
        
        # All implementations to test
        implementations = {
            'Original': get_bone_world_position,
            'Pytransform3d': get_bone_world_position_pt3d,
            'Corrected_v1': get_bone_world_position_corrected,
            'Corrected_v2': get_bone_world_position_corrected_v2,
            'Corrected_v3': get_bone_world_position_corrected_v3
        }
        
        try:
            chain_positions = {}
            
            for bone_name in available_bones:
                print(f"\n{bone_name:10}:")
                bone_results = {}
                
                for impl_name, impl_func in implementations.items():
                    try:
                        pos = impl_func(self.pmx_model, self.vmd_motion, self.test_frame, bone_name)
                        bone_results[impl_name] = pos
                        
                        underground = pos[1] < -10
                        status = "❌" if underground else "✅"
                        print(f"  {impl_name:12}: ({pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}) {status}")
                        
                        if underground:
                            results['passed'] = False
                            
                    except Exception as e:
                        print(f"  {impl_name:12}: ❌ Error: {e}")
                        results['errors'].append(f"Error in {impl_name} for {bone_name}: {e}")
                        results['passed'] = False
                        bone_results[impl_name] = {'error': str(e)}
                
                chain_positions[bone_name] = bone_results
            
            results['details']['chain_positions'] = chain_positions
            
            # Summary check for each implementation
            print(f"\n🎯 Implementation Summary:")
            for impl_name in implementations.keys():
                underground_count = 0
                total_count = 0
                
                for bone_name in available_bones:
                    if impl_name in chain_positions[bone_name]:
                        if isinstance(chain_positions[bone_name][impl_name], tuple):
                            total_count += 1
                            if chain_positions[bone_name][impl_name][1] < -10:
                                underground_count += 1
                
                if total_count > 0:
                    status = "✅" if underground_count == 0 else f"❌ {underground_count}/{total_count} underground"
                    print(f"  {impl_name:12}: {status}")
                    
        except Exception as e:
            results['errors'].append(f"Error testing bone chain: {e}")
            results['passed'] = False
            print(f"❌ Error: {e}")
        
        return results
    
    def test_center_bone_animation(self) -> Dict:
        """Test 4: Validate that センター bone gets its own animation data correctly."""
        print("\n" + "=" * 60)  
        print("TEST 4: センター Bone Animation Data Application")
        print("=" * 60)
        
        results = {
            'passed': True,
            'details': {},
            'errors': []
        }
        
        center_bone = self.find_available_bone(self.test_bones['center'])
        if not center_bone:
            results['errors'].append("センター bone not found")
            results['passed'] = False
            print("❌ センター bone not found")
            return results
        
        try:
            # Get センター bone position
            center_pos_orig = get_bone_world_position(self.pmx_model, self.vmd_motion,
                                                    self.test_frame, center_bone)
            center_pos_pt3d = get_bone_world_position_pt3d(self.pmx_model, self.vmd_motion,
                                                         self.test_frame, center_bone)
            
            # Also get the root bone position (index 0) for comparison
            root_bones = ['', 'Root']  # Common root bone names
            root_bone = None
            for name in root_bones:
                if BoneHierarchyWalker.find_bone_by_name(self.pmx_model.bones, name):
                    root_bone = name
                    break
            
            if root_bone is not None:
                try:
                    root_pos_orig = get_bone_world_position(self.pmx_model, self.vmd_motion,
                                                          self.test_frame, root_bone)
                    root_pos_pt3d = get_bone_world_position_pt3d(self.pmx_model, self.vmd_motion,
                                                               self.test_frame, root_bone)
                    
                    results['details']['root_positions'] = {
                        'original': root_pos_orig,
                        'pytransform3d': root_pos_pt3d
                    }
                    
                    print(f"Root bone ('{root_bone}'):")
                    print(f"  Original:     ({root_pos_orig[0]:8.3f}, {root_pos_orig[1]:8.3f}, {root_pos_orig[2]:8.3f})")
                    print(f"  Pytransform3d: ({root_pos_pt3d[0]:8.3f}, {root_pos_pt3d[1]:8.3f}, {root_pos_pt3d[2]:8.3f})")
                    
                except Exception as e:
                    print(f"Warning: Could not get root bone position: {e}")
            
            results['details']['center_positions'] = {
                'original': center_pos_orig,
                'pytransform3d': center_pos_pt3d
            }
            
            print(f"Center bone ('{center_bone}'):")
            print(f"  Original:     ({center_pos_orig[0]:8.3f}, {center_pos_orig[1]:8.3f}, {center_pos_orig[2]:8.3f})")
            print(f"  Pytransform3d: ({center_pos_pt3d[0]:8.3f}, {center_pos_pt3d[1]:8.3f}, {center_pos_pt3d[2]:8.3f})")
            
            # Based on the debug output, センター should have significant animation offset
            # The debug showed (-19.095, 0.000, -17.193) animation for センター
            center_distance_orig = np.linalg.norm(center_pos_orig)
            center_distance_pt3d = np.linalg.norm(center_pos_pt3d)
            
            print(f"\nセンター bone distance from origin:")
            print(f"  Original: {center_distance_orig:.3f}")
            print(f"  Pytransform3d: {center_distance_pt3d:.3f}")
            
            # センター should have substantial movement for dancing (based on debug output)
            if center_distance_orig < 5.0:
                print(f"  ⚠️  Original センター distance seems small for dancing motion")
            else:
                print(f"  ✅ Original センター distance indicates movement")
                
            if center_distance_pt3d < 5.0:
                print(f"  ⚠️  Pytransform3d センター distance seems small for dancing motion")
            else:
                print(f"  ✅ Pytransform3d センター distance indicates movement")
                
        except Exception as e:
            results['errors'].append(f"Error testing センター bone: {e}")
            results['passed'] = False
            print(f"❌ Error: {e}")
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Run all comprehensive tests and return combined results."""
        print("🧪 Running Comprehensive Bone Transformation Tests")
        print("=" * 80)
        
        all_results = {
            'overall_passed': True,
            'test_results': {},
            'summary': {}
        }
        
        # Run all tests
        tests = [
            ('ground_level', self.test_ground_level_validation),
            ('left_right_differences', self.test_left_right_foot_differences),
            ('bone_chain', self.test_bone_chain_accumulation),
            ('center_animation', self.test_center_bone_animation)
        ]
        
        passed_count = 0
        total_count = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                all_results['test_results'][test_name] = result
                
                if result['passed']:
                    passed_count += 1
                else:
                    all_results['overall_passed'] = False
                    
            except Exception as e:
                print(f"❌ Test {test_name} failed with exception: {e}")
                all_results['test_results'][test_name] = {
                    'passed': False,
                    'errors': [f"Test exception: {e}"],
                    'details': {}
                }
                all_results['overall_passed'] = False
        
        # Generate summary
        all_results['summary'] = {
            'tests_passed': passed_count,
            'tests_total': total_count,
            'success_rate': passed_count / total_count if total_count > 0 else 0.0
        }
        
        print("\n" + "=" * 80)
        print("🎯 TEST SUMMARY")
        print("=" * 80)
        print(f"Tests passed: {passed_count}/{total_count} ({all_results['summary']['success_rate']*100:.1f}%)")
        
        if all_results['overall_passed']:
            print("✅ ALL TESTS PASSED - Bone transformations appear to be working correctly!")
        else:
            print("❌ SOME TESTS FAILED - Bone transformation issues detected")
            print("\nFailed tests:")
            for test_name, result in all_results['test_results'].items():
                if not result['passed']:
                    print(f"  - {test_name}: {len(result['errors'])} errors")
                    for error in result['errors']:
                        print(f"    • {error}")
        
        return all_results


def main():
    """Main test execution function."""
    # Load test files
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pmx_file = os.path.join(test_dir, "pdtt.pmx")
    vmd_file = os.path.join(test_dir, "dan_alivef_01.imo.vmd")
    
    if not os.path.exists(pmx_file):
        print(f"❌ PMX file not found: {pmx_file}")
        return False
    
    if not os.path.exists(vmd_file):
        print(f"❌ VMD file not found: {vmd_file}")
        return False
    
    print(f"📁 Loading test files:")
    print(f"   PMX: {os.path.basename(pmx_file)}")
    print(f"   VMD: {os.path.basename(vmd_file)}")
    
    try:
        # Load PMX model
        with open(pmx_file, 'rb') as f:
            pmx_model = pmx_reader.read(f)
        print("✅ PMX loaded successfully")
        
        # Load VMD motion
        with open(vmd_file, 'rb') as f:
            vmd_motion = vmd_reader.read(f)
        print("✅ VMD loaded successfully")
        
        # Run comprehensive tests
        tester = BoneTransformationTester(pmx_model, vmd_motion)
        results = tester.run_all_tests()
        
        return results['overall_passed']
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)