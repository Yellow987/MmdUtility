#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to compare AnimatedModel vs analyze_bone_chain.py results
"""

import sys
import os
import numpy as np

# Add current directory to Python path for pymeshio import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import torch
    from pymeshio.animation import create_animated_model
    from analyze_bone_chain import read_vmd_bone_data, load_pmx_model
    print("Successfully imported required modules")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def debug_vmd_loading():
    """Compare VMD loading between AnimatedModel and analyze_bone_chain.py"""
    print("\n" + "=" * 80)
    print("DEBUGGING VMD DATA LOADING")
    print("=" * 80)
    
    # Use the same files as analyze_bone_chain.py
    pmx_path = "test/pdtt.pmx"
    vmd_path = "test/dan_alivef_01.imo.vmd"
    
    if not (os.path.exists(pmx_path) and os.path.exists(vmd_path)):
        print(f"Test files not found: {pmx_path}, {vmd_path}")
        return
    
    print("1. Loading VMD data using analyze_bone_chain.py method...")
    bone_data_manual = read_vmd_bone_data(vmd_path)
    
    print("2. Loading VMD data using AnimatedModel...")
    try:
        animated_model = create_animated_model(pmx_path, vmd_path, device='cpu')
        vmd_data_animated = animated_model.vmd_data_by_frame
    except Exception as e:
        print(f"Error creating AnimatedModel: {e}")
        return
    
    # Check frame 6 data for groove bone
    target_frame = 6
    groove_bone_names = ["グルーブ", "groove"]
    
    print(f"\n3. Comparing frame {target_frame} data for groove bone:")
    
    # Check manual VMD loading
    print("\n--- analyze_bone_chain.py VMD data ---")
    for bone_name in groove_bone_names:
        if bone_name in bone_data_manual:
            for frame_data in bone_data_manual[bone_name]:
                if frame_data["frame"] == target_frame:
                    pos = frame_data["position"]
                    quat = frame_data["quaternion"]
                    print(f"  {bone_name} at frame {target_frame}:")
                    print(f"    Position: {pos}")
                    print(f"    Quaternion: {quat}")
                    break
            else:
                print(f"  {bone_name}: No data at frame {target_frame}")
        else:
            print(f"  {bone_name}: Not found in VMD data")
    
    # Check AnimatedModel VMD loading
    print("\n--- AnimatedModel VMD data ---")
    if target_frame in vmd_data_animated:
        frame_data = vmd_data_animated[target_frame]
        for bone_name in groove_bone_names:
            if bone_name in frame_data:
                data = frame_data[bone_name]
                print(f"  {bone_name} at frame {target_frame}:")
                print(f"    Position: {data['position']}")
                print(f"    Quaternion: {data['quaternion']}")
            else:
                print(f"  {bone_name}: Not found in frame {target_frame}")
    else:
        print(f"  Frame {target_frame}: Not found in AnimatedModel VMD data")
    
    # Check all available bones in frame 6
    print(f"\n4. All bones available at frame {target_frame}:")
    if target_frame in vmd_data_animated:
        print(f"  AnimatedModel has {len(vmd_data_animated[target_frame])} bones at frame {target_frame}:")
        for bone_name in sorted(vmd_data_animated[target_frame].keys()):
            print(f"    - {bone_name}")
    
    # Check available frames
    available_frames = sorted(list(vmd_data_animated.keys()))
    print(f"\n5. Available frames: {available_frames[:10]}..." if len(available_frames) > 10 else f"Available frames: {available_frames}")


def debug_fk_calculations():
    """Compare FK calculations between both methods"""
    print("\n" + "=" * 80)
    print("DEBUGGING FK CALCULATIONS")
    print("=" * 80)
    
    pmx_path = "test/pdtt.pmx"
    vmd_path = "test/dan_alivef_01.imo.vmd"
    
    if not (os.path.exists(pmx_path) and os.path.exists(vmd_path)):
        return
    
    try:
        # Get AnimatedModel result
        animated_model = create_animated_model(pmx_path, vmd_path, device='cpu')
        world_positions = animated_model.get_world_positions(6)
        
        if "グルーブ" in world_positions:
            groove_pos = world_positions["グルーブ"]
            print(f"AnimatedModel groove position: {groove_pos}")
        else:
            print("AnimatedModel: グルーブ bone not found")
        
        # Get rest position for comparison
        groove_rest = animated_model.get_rest_position("グルーブ")
        if groove_rest is not None:
            print(f"Groove rest position: {groove_rest}")
        else:
            print("Groove rest position not found")
        
    except Exception as e:
        print(f"Error in FK calculations: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run debug analysis"""
    print("Animation Debug Script - Comparing VMD Loading and FK Calculations")
    debug_vmd_loading()
    debug_fk_calculations()


if __name__ == "__main__":
    main()