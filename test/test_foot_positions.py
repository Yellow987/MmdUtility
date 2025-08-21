#!/usr/bin/env python
# coding: utf-8
"""
Test Script for Foot Position Calculation
==========================================

This script loads PMX model and VMD motion files from the test folder
and outputs the world space positions of foot bones at frame 3000.

Usage:
1. Place your PMX model file as 'test/model.pmx' 
2. Place your VMD motion file as 'test/motion.vmd'
3. Run this script: python test_foot_positions.py

The script will output foot bone positions and indicate potential ground contact.
"""

import os
import sys

# Add parent directory to path to import bone_animation module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import pymeshio.pmx.reader as pmx_reader
    import pymeshio.vmd.reader as vmd_reader
    import bone_animation
    from bone_animation import (
        get_bone_world_position,
        get_foot_positions,
        is_foot_on_ground,
        BoneNotFoundError,
        InvalidFrameError,
        COMMON_FOOT_BONE_NAMES
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the MmdUtility directory")
    print("Current working directory:", os.getcwd())
    sys.exit(1)


def load_test_files():
    """Load PMX and VMD files from the test directory."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for PMX files
    pmx_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.pmx')]
    vmd_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.vmd')]
    
    if not pmx_files:
        print(f"No PMX files found in {test_dir}")
        print("Please place a PMX model file in the test directory")
        return None, None
    
    if not vmd_files:
        print(f"No VMD files found in {test_dir}")
        print("Please place a VMD motion file in the test directory")
        return None, None
    
    pmx_path = os.path.join(test_dir, pmx_files[0])
    vmd_path = os.path.join(test_dir, vmd_files[0])
    
    print(f"Loading PMX: {pmx_files[0]}")
    print(f"Loading VMD: {vmd_files[0]}")
    
    try:
        print("Attempting to load PMX...")
        pmx_model = pmx_reader.read_from_file(pmx_path)
        print("PMX loaded successfully!")
        
        print("Attempting to load VMD...")
        vmd_motion = vmd_reader.read_from_file(vmd_path)
        print("VMD loaded successfully!")
        
        return pmx_model, vmd_motion
    except Exception as e:
        print(f"Error loading files: {e}")
        # If PMX fails but VMD works, try with VMD only
        try:
            print("Trying VMD only...")
            vmd_motion = vmd_reader.read_from_file(vmd_path)
            print("VMD loaded successfully! PMX failed but we can still test with animation data.")
            return None, vmd_motion
        except Exception as e2:
            print(f"VMD also failed: {e2}")
            return None, None


def list_available_bones(pmx_model):
    """List all available bones in the model for reference."""
    print("\nAvailable bones in model:")
    print("=" * 50)
    
    for i, bone in enumerate(pmx_model.bones):
        name = bone.name
        english_name = bone.english_name
        
        # Convert bytes to string if needed
        if isinstance(name, bytes):
            try:
                name = name.decode('utf-8')
            except UnicodeDecodeError:
                name = str(name)
        
        if isinstance(english_name, bytes):
            try:
                english_name = english_name.decode('utf-8')
            except UnicodeDecodeError:
                english_name = str(english_name)
        
        print(f"  {i:3d}: {name} ({english_name})")


def test_foot_positions_at_frame(pmx_model, vmd_motion, frame_number=3000):
    """Test foot bone positions at the specified frame."""
    print(f"\nTesting foot positions at frame {frame_number}")
    print("=" * 60)
    
    # Try to get all common foot bone positions
    foot_positions = get_foot_positions(pmx_model, vmd_motion, frame_number)
    
    ground_threshold = 0.5  # Adjust based on your model scale
    
    for foot_type, position in foot_positions.items():
        if position is not None:
            x, y, z = position
            is_on_ground = y <= ground_threshold
            ground_status = "ON GROUND" if is_on_ground else "IN AIR"
            
            print(f"{foot_type.upper():>12}: ({x:8.3f}, {y:8.3f}, {z:8.3f}) - {ground_status}")
        else:
            print(f"{foot_type.upper():>12}: Not found")
    
    # Try specific bone names that might exist
    additional_bone_names = [
        '左足首', '右足首',  # Japanese ankle
        '左足先', '右足先',  # Japanese foot tip
        'ankle_L', 'ankle_R',  # English ankle
        'foot.L', 'foot.R',   # Rigify style
    ]
    
    print(f"\nAdditional foot-related bones at frame {frame_number}:")
    print("-" * 60)
    
    for bone_name in additional_bone_names:
        try:
            pos = get_bone_world_position(pmx_model, vmd_motion, frame_number, bone_name)
            x, y, z = pos
            is_on_ground = y <= ground_threshold
            ground_status = "ON GROUND" if is_on_ground else "IN AIR"
            print(f"{bone_name:>12}: ({x:8.3f}, {y:8.3f}, {z:8.3f}) - {ground_status}")
        except BoneNotFoundError:
            continue  # Skip bones that don't exist


def analyze_ground_contact_over_time(pmx_model, vmd_motion, bone_name, 
                                   start_frame=2990, end_frame=3010):
    """Analyze ground contact for a bone over a range of frames."""
    print(f"\nGround contact analysis for '{bone_name}' (frames {start_frame}-{end_frame}):")
    print("-" * 70)
    
    ground_threshold = 0.5
    
    try:
        for frame in range(start_frame, end_frame + 1):
            try:
                pos = get_bone_world_position(pmx_model, vmd_motion, frame, bone_name)
                y_pos = pos[1]
                is_on_ground = y_pos <= ground_threshold
                contact_char = "█" if is_on_ground else "░"
                
                print(f"Frame {frame:4d}: Y={y_pos:7.3f} {contact_char}")
                
            except InvalidFrameError:
                print(f"Frame {frame:4d}: Invalid frame")
                
    except BoneNotFoundError:
        print(f"Bone '{bone_name}' not found in model")


def main():
    """Main test function."""
    print("MMD Foot Position Test")
    print("=" * 50)
    
    # Load test files
    pmx_model, vmd_motion = load_test_files()
    if pmx_model is None or vmd_motion is None:
        return
    
    print(f"\nModel: {pmx_model.name}")
    print(f"Motion: {vmd_motion.model_name}")
    print(f"Total bones: {len(pmx_model.bones)}")
    print(f"Total motion frames: {len(vmd_motion.motions)}")
    
    # Test foot positions at frame 3000
    test_foot_positions_at_frame(pmx_model, vmd_motion, 3000)
    
    # List some bones for reference
    list_available_bones(pmx_model)
    
    # If we found any foot bones, analyze them over time
    foot_positions = get_foot_positions(pmx_model, vmd_motion, 3000)
    
    # Find the first available foot bone for temporal analysis
    for foot_type, bone_names in COMMON_FOOT_BONE_NAMES.items():
        for bone_name in bone_names:
            try:
                # Test if bone exists by trying to get its position
                get_bone_world_position(pmx_model, vmd_motion, 3000, bone_name)
                # If successful, analyze this bone over time
                analyze_ground_contact_over_time(pmx_model, vmd_motion, bone_name)
                break
            except BoneNotFoundError:
                continue
        else:
            continue
        break  # Exit outer loop if we found and analyzed a bone
    
    print("\nTest completed!")
    print("\nTips for using this data in ML:")
    print("- Y coordinate near 0 typically indicates ground contact")
    print("- Sudden changes in Y velocity can indicate foot strikes")
    print("- Consider the model's scale when setting ground thresholds")
    print("- Foot contact patterns are crucial for realistic dance generation")


if __name__ == "__main__":
    main()