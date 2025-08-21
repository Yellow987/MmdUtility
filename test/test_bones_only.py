#!/usr/bin/env python
# coding: utf-8
"""
Simplified Bone Animation Test
==============================

This test focuses only on bone hierarchy and animation,
bypassing problematic vertex/mesh data.
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pymeshio.vmd.reader as vmd_reader
from pymeshio import common, pmx
from bone_animation import (
    get_bone_world_position, 
    get_foot_positions, 
    BoneNotFoundError,
    COMMON_FOOT_BONE_NAMES
)


def create_simple_test_model():
    """Create a simple test PMX model with basic bone hierarchy."""
    model = pmx.Model()
    
    # Create a simple bone hierarchy for testing
    # Root bone
    root_bone = pmx.Bone(
        name="センター", english_name="Center",
        position=common.Vector3(0, 0, 0),
        parent_index=-1, layer=0, flag=0
    )
    
    # Lower body bone
    lower_body = pmx.Bone(
        name="下半身", english_name="LowerBody", 
        position=common.Vector3(0, 5, 0),
        parent_index=0, layer=0, flag=0
    )
    
    # Left leg bones
    left_leg = pmx.Bone(
        name="左足", english_name="LeftLeg",
        position=common.Vector3(-1, 5, 0),
        parent_index=1, layer=0, flag=0
    )
    
    left_knee = pmx.Bone(
        name="左ひざ", english_name="LeftKnee",
        position=common.Vector3(-1, 2, 0), 
        parent_index=2, layer=0, flag=0
    )
    
    left_foot = pmx.Bone(
        name="左足首", english_name="LeftAnkle",
        position=common.Vector3(-1, 0, 0),
        parent_index=3, layer=0, flag=0
    )
    
    # Right leg bones  
    right_leg = pmx.Bone(
        name="右足", english_name="RightLeg",
        position=common.Vector3(1, 5, 0),
        parent_index=1, layer=0, flag=0
    )
    
    right_knee = pmx.Bone(
        name="右ひざ", english_name="RightKnee",
        position=common.Vector3(1, 2, 0),
        parent_index=5, layer=0, flag=0
    )
    
    right_foot = pmx.Bone(
        name="右足首", english_name="RightAnkle", 
        position=common.Vector3(1, 0, 0),
        parent_index=6, layer=0, flag=0
    )
    
    model.bones = [root_bone, lower_body, left_leg, left_knee, left_foot, right_leg, right_knee, right_foot]
    
    # Set bone indices
    for i, bone in enumerate(model.bones):
        bone.index = i
    
    return model


def load_vmd_file():
    """Load VMD file from test directory."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    vmd_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.vmd')]
    
    if not vmd_files:
        print("No VMD files found in test directory")
        return None
    
    vmd_path = os.path.join(test_dir, vmd_files[0])
    print(f"Loading VMD: {vmd_files[0]}")
    
    try:
        return vmd_reader.read_from_file(vmd_path)
    except Exception as e:
        print(f"Error loading VMD: {e}")
        return None


def test_bone_animation():
    """Test bone animation functionality with simple model."""
    print("Bone Animation Test")
    print("=" * 50)
    
    # Create simple test model
    model = create_simple_test_model()
    print(f"Created test model with {len(model.bones)} bones")
    
    # Load VMD animation
    vmd_motion = load_vmd_file()
    if vmd_motion is None:
        return
    
    print(f"VMD Motion: {vmd_motion.model_name}")
    print(f"Total keyframes: {len(vmd_motion.motions)}")
    
    # List some available bones for reference
    print(f"\nAvailable motion bones (first 10):")
    motion_bones = set()
    for motion in vmd_motion.motions[:100]:  # Check first 100 frames
        bone_name = motion.name
        if isinstance(bone_name, bytes):
            try:
                bone_name = bone_name.decode('utf-8')
            except:
                bone_name = str(bone_name)
        motion_bones.add(bone_name)
        if len(motion_bones) >= 10:
            break
    
    for bone_name in list(motion_bones)[:10]:
        print(f"  - {bone_name}")
    
    # Test frame 3000
    frame = 3000
    print(f"\nTesting bone positions at frame {frame}:")
    print("-" * 40)
    
    # Test our simple model bones
    test_bones = ["センター", "左足", "右足", "左足首", "右足首"]
    
    for bone_name in test_bones:
        try:
            pos = get_bone_world_position(model, vmd_motion, frame, bone_name)
            print(f"{bone_name:>10}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
        except BoneNotFoundError:
            print(f"{bone_name:>10}: Not found in animation")
        except Exception as e:
            print(f"{bone_name:>10}: Error - {e}")
    
    # Try some common VMD bone names
    print(f"\nTesting common VMD bone names at frame {frame}:")
    print("-" * 50)
    
    vmd_bone_names = [
        "センター", "上半身", "下半身", 
        "左足", "右足", "左ひざ", "右ひざ",
        "左足首", "右足首", "左つま先", "右つま先"
    ]
    
    for bone_name in vmd_bone_names:
        try:
            pos = get_bone_world_position(model, vmd_motion, frame, bone_name)
            y_pos = pos[1]
            ground_status = "ON GROUND" if y_pos <= 0.5 else "IN AIR"
            print(f"{bone_name:>10}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}) - {ground_status}")
        except BoneNotFoundError:
            continue
        except Exception as e:
            print(f"{bone_name:>10}: Error - {e}")
    
    print("\nTest completed!")
    print("This demonstrates the bone world position calculation functionality")
    print("for machine learning foot contact detection.")


if __name__ == "__main__":
    test_bone_animation()