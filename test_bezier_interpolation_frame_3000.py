#!/usr/bin/env python3
"""
Test script to implement bezier interpolation for VMD animation at frame 3000.
Finds keyframes before and after frame 3000 and interpolates the values.
"""

import os
import sys
import numpy as np
import bezier

def lerp(a, b, t):
    """Linear interpolation between a and b with factor t."""
    return a + (b - a) * t

def slerp(q1, q2, t):
    """Spherical linear interpolation between quaternions q1 and q2."""
    # Convert to numpy arrays
    q1 = np.array([q1.x, q1.y, q1.z, q1.w])
    q2 = np.array([q2.x, q2.y, q2.z, q2.w])
    
    # Compute dot product
    dot = np.dot(q1, q2)
    
    # If dot product is negative, use -q2 to take shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        result = result / np.linalg.norm(result)
        return result
    
    # Calculate angle between quaternions
    theta_0 = np.arccos(abs(dot))
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q1) + (s1 * q2)

def parse_vmd_bezier_params(complement_hex):
    """
    Parse VMD bezier interpolation parameters from complement field.
    
    VMD stores 4 sets of bezier control points for X, Y, Z, Rotation in 64 bytes.
    Each channel has 4 control point values (x1, y1, x2, y2) for the bezier curve.
    """
    try:
        if isinstance(complement_hex, str):
            # Convert hex string to bytes
            complement_bytes = bytes.fromhex(complement_hex)
        else:
            # Already bytes or other format
            return None
            
        if len(complement_bytes) < 64:
            return None
            
        # VMD bezier parameters format:
        # Each channel (X, Y, Z, R) has 4 control point bytes
        # We'll extract the first channel (X position) for now
        x1 = complement_bytes[0] / 127.0  # Normalize to 0-1
        x2 = complement_bytes[4] / 127.0
        y1 = complement_bytes[8] / 127.0
        y2 = complement_bytes[12] / 127.0
        
        return {
            'x_pos': (x1, y1, x2, y2),
            'y_pos': (complement_bytes[1]/127.0, complement_bytes[9]/127.0,
                     complement_bytes[5]/127.0, complement_bytes[13]/127.0),
            'z_pos': (complement_bytes[2]/127.0, complement_bytes[10]/127.0,
                     complement_bytes[6]/127.0, complement_bytes[14]/127.0),
            'rotation': (complement_bytes[3]/127.0, complement_bytes[11]/127.0,
                        complement_bytes[7]/127.0, complement_bytes[15]/127.0)
        }
    except:
        return None

def bezier_interpolate_value(t, p1, p2, control_points):
    """
    Perform bezier interpolation between two values using control points.
    
    Args:
        t: Time parameter (0-1)
        p1: Start value
        p2: End value
        control_points: Tuple of (x1, y1, x2, y2) bezier control points
    
    Returns:
        Interpolated value
    """
    if control_points is None:
        # Fallback to linear interpolation
        return lerp(p1, p2, t)
    
    try:
        x1, y1, x2, y2 = control_points
        
        # Create bezier curve using the bezier library
        # Points: (0,0), (x1,y1), (x2,y2), (1,1)
        control_points_array = np.array([
            [0.0, x1, x2, 1.0],  # X coordinates
            [0.0, y1, y2, 1.0]   # Y coordinates
        ])
        
        curve = bezier.Curve(control_points_array, degree=3)
        
        # Find the parameter s such that curve(s).x = t
        # For simplicity, we'll approximate by evaluating the curve at t directly
        point = curve.evaluate(t)
        bezier_t = float(point[1, 0])  # Y coordinate gives us the interpolation factor
        
        # Clamp to [0, 1]
        bezier_t = max(0.0, min(1.0, bezier_t))
        
        # Interpolate between p1 and p2 using the bezier-adjusted factor
        return lerp(p1, p2, bezier_t)
        
    except Exception as e:
        # Fallback to linear interpolation
        return lerp(p1, p2, t)

def bezier_interpolate_vmd(frame, prev_frame, next_frame, prev_data, next_data, complement_data=None):
    """
    Perform bezier interpolation for VMD animation data using actual bezier curves.
    
    Args:
        frame: Target frame number
        prev_frame: Previous keyframe number
        next_frame: Next keyframe number
        prev_data: Data at previous keyframe
        next_data: Data at next keyframe
        complement_data: VMD interpolation control points (hex string)
    
    Returns:
        Interpolated data
    """
    if prev_frame == next_frame:
        return prev_data
    
    # Calculate time parameter
    t = (frame - prev_frame) / (next_frame - prev_frame)
    
    # Parse bezier parameters from VMD complement data
    bezier_params = parse_vmd_bezier_params(complement_data)
    
    if hasattr(prev_data, 'x'):  # It's a quaternion
        # Use SLERP for quaternions (bezier curves on quaternions are complex)
        return slerp(prev_data, next_data, t)
    elif isinstance(prev_data, (tuple, list)):  # It's a position (x, y, z)
        if bezier_params:
            # Use bezier interpolation for each axis
            x = bezier_interpolate_value(t, prev_data[0], next_data[0], bezier_params['x_pos'])
            y = bezier_interpolate_value(t, prev_data[1], next_data[1], bezier_params['y_pos'])
            z = bezier_interpolate_value(t, prev_data[2], next_data[2], bezier_params['z_pos'])
            return (x, y, z)
        else:
            # Fallback to linear interpolation
            return tuple(lerp(prev_data[i], next_data[i], t) for i in range(len(prev_data)))
    else:
        # Single value - use bezier if available
        if bezier_params:
            return bezier_interpolate_value(t, prev_data, next_data, bezier_params['x_pos'])
        else:
            return lerp(prev_data, next_data, t)

def find_surrounding_keyframes(motions, bone_name, target_frame):
    """Find keyframes before and after target frame for a specific bone."""
    bone_keyframes = []
    
    for motion in motions:
        motion_bone_name = motion.name
        if isinstance(motion_bone_name, bytes):
            # Use CP932 encoding for Japanese bone names
            motion_bone_name = motion_bone_name.decode('cp932', errors='ignore')
        
        if motion_bone_name == bone_name:
            bone_keyframes.append(motion)
    
    # Sort by frame number
    bone_keyframes.sort(key=lambda m: m.frame)
    
    # Find prev and next keyframes
    prev_motion = None
    next_motion = None
    
    for motion in bone_keyframes:
        if motion.frame <= target_frame:
            prev_motion = motion
        elif motion.frame > target_frame and next_motion is None:
            next_motion = motion
            break
    
    return prev_motion, next_motion

def interpolate_frame_3000():
    """Interpolate animation data for frame 3000 using bezier curves."""
    print("Interpolating VMD Animation Data for Frame 3000...")
    
    # Define test file paths
    pmx_path = "test/pdtt.pmx"
    vmd_path = "test/dan_alivef_01.imo.vmd"
    
    # Check if files exist
    if not os.path.exists(pmx_path):
        print(f"✗ PMX test file not found: {pmx_path}")
        return False
    
    if not os.path.exists(vmd_path):
        print(f"✗ VMD test file not found: {vmd_path}")
        return False
    
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        
        # Load files
        print("Loading PMX model...")
        extractor.load_pmx(pmx_path)
        
        print("Loading VMD motion...")
        extractor.load_vmd(vmd_path)
        
        # Define core bones
        core_bone_names = {
            "", "センター", "グルーブ", "腰", "上半身", "上半身2",
            "首", "頭",
            "左肩", "右肩", "左腕", "右腕", "左ひじ", "右ひじ", "左手首", "右手首",
            "左足", "右足", "左ひざ", "右ひざ", "左足首", "右足首", "左つま先", "右つま先",
        }
        
        target_frame = 3000
        interpolated_data = {}
        
        print(f"\n" + "="*80)
        print(f"BEZIER INTERPOLATION FOR FRAME {target_frame}")
        print("="*80)
        
        print(f"{'Bone Name':<15} | {'Prev Frame':<10} | {'Next Frame':<10} | {'Position':<25} | {'Quaternion':<30}")
        print("-" * 100)
        
        for bone_name in core_bone_names:
            if not bone_name:  # Skip empty bone name for display
                display_name = "(root)"
            else:
                display_name = bone_name
                
            # Find surrounding keyframes for this bone
            prev_motion, next_motion = find_surrounding_keyframes(
                extractor.vmd_motion.motions, bone_name, target_frame
            )
            
            if prev_motion and next_motion:
                # Interpolate position
                prev_pos = (prev_motion.pos.x, prev_motion.pos.y, prev_motion.pos.z)
                next_pos = (next_motion.pos.x, next_motion.pos.y, next_motion.pos.z)
                interp_pos = bezier_interpolate_vmd(
                    target_frame, prev_motion.frame, next_motion.frame,
                    prev_pos, next_pos, prev_motion.complement
                )
                
                # Interpolate quaternion
                interp_quat = bezier_interpolate_vmd(
                    target_frame, prev_motion.frame, next_motion.frame,
                    prev_motion.q, next_motion.q, prev_motion.complement
                )
                
                # Store interpolated data
                interpolated_data[bone_name] = {
                    'position': interp_pos,
                    'quaternion': interp_quat,
                    'prev_frame': prev_motion.frame,
                    'next_frame': next_motion.frame
                }
                
                # Display results
                pos_str = f"({interp_pos[0]:.3f},{interp_pos[1]:.3f},{interp_pos[2]:.3f})"
                if hasattr(interp_quat, '__len__'):  # numpy array
                    quat_str = f"({interp_quat[0]:.3f},{interp_quat[1]:.3f},{interp_quat[2]:.3f},{interp_quat[3]:.3f})"
                else:  # quaternion object
                    quat_str = f"({interp_quat.x:.3f},{interp_quat.y:.3f},{interp_quat.z:.3f},{interp_quat.w:.3f})"
                
                print(f"{display_name:<15} | {prev_motion.frame:<10} | {next_motion.frame:<10} | {pos_str:<25} | {quat_str:<30}")
                
            elif prev_motion:
                # Only previous keyframe exists, use its values
                prev_pos = (prev_motion.pos.x, prev_motion.pos.y, prev_motion.pos.z)
                interpolated_data[bone_name] = {
                    'position': prev_pos,
                    'quaternion': prev_motion.q,
                    'prev_frame': prev_motion.frame,
                    'next_frame': None
                }
                
                pos_str = f"({prev_pos[0]:.3f},{prev_pos[1]:.3f},{prev_pos[2]:.3f})"
                quat_str = f"({prev_motion.q.x:.3f},{prev_motion.q.y:.3f},{prev_motion.q.z:.3f},{prev_motion.q.w:.3f})"
                
                print(f"{display_name:<15} | {prev_motion.frame:<10} | {'None':<10} | {pos_str:<25} | {quat_str:<30}")
                
            elif next_motion:
                # Only next keyframe exists, use its values  
                next_pos = (next_motion.pos.x, next_motion.pos.y, next_motion.pos.z)
                interpolated_data[bone_name] = {
                    'position': next_pos,
                    'quaternion': next_motion.q,
                    'prev_frame': None,
                    'next_frame': next_motion.frame
                }
                
                pos_str = f"({next_pos[0]:.3f},{next_pos[1]:.3f},{next_pos[2]:.3f})"
                quat_str = f"({next_motion.q.x:.3f},{next_motion.q.y:.3f},{next_motion.q.z:.3f},{next_motion.q.w:.3f})"
                
                print(f"{display_name:<15} | {'None':<10} | {next_motion.frame:<10} | {pos_str:<25} | {quat_str:<30}")
                
            else:
                print(f"{display_name:<15} | {'No keyframes found':<60}")
        
        print(f"\n✓ Interpolated animation data for {len(interpolated_data)} core bones at frame {target_frame}")
        
        # Compare with current get_frame_transforms method
        print(f"\n" + "="*60)
        print("COMPARISON WITH CURRENT METHOD")
        print("="*60)
        
        current_transforms = extractor.get_frame_transforms(target_frame)
        
        # Decode current transform bone names using CP932
        decoded_current_transforms = {}
        for raw_name, transform_data in current_transforms.items():
            if isinstance(raw_name, bytes):
                decoded_name = raw_name.decode('cp932', errors='ignore')
            else:
                decoded_name = raw_name
            decoded_current_transforms[decoded_name] = transform_data
        
        print(f"Current method found: {len(decoded_current_transforms)} transforms")
        print(f"Interpolation found: {len(interpolated_data)} transforms")
        
        # Show differences for bones that have both
        common_bones = set(decoded_current_transforms.keys()) & set(interpolated_data.keys())
        if common_bones:
            print(f"\nComparison for {len(common_bones)} common bones:")
            print(f"{'Bone':<13} | {'Current Pos':<25} | {'Interpolated Pos':<25} | {'Pos Diff':<8} | {'Quat Diff':<8}")
            print("-" * 90)
            
            for bone_name in sorted(common_bones):
                current_pos, current_quat = decoded_current_transforms[bone_name]
                interp_pos = interpolated_data[bone_name]['position']
                interp_quat = interpolated_data[bone_name]['quaternion']
                
                # Calculate position difference
                pos_diff = sum((current_pos[i] - interp_pos[i])**2 for i in range(3))**0.5
                
                # Calculate quaternion difference
                if hasattr(interp_quat, '__len__'):  # numpy array
                    quat_diff = sum((current_quat.x - interp_quat[0])**2 +
                                   (current_quat.y - interp_quat[1])**2 +
                                   (current_quat.z - interp_quat[2])**2 +
                                   (current_quat.w - interp_quat[3])**2)**0.5
                else:  # quaternion object
                    quat_diff = sum((current_quat.x - interp_quat.x)**2 +
                                   (current_quat.y - interp_quat.y)**2 +
                                   (current_quat.z - interp_quat.z)**2 +
                                   (current_quat.w - interp_quat.w)**2)**0.5
                
                curr_pos_str = f"({current_pos[0]:.3f},{current_pos[1]:.3f},{current_pos[2]:.3f})"
                interp_pos_str = f"({interp_pos[0]:.3f},{interp_pos[1]:.3f},{interp_pos[2]:.3f})"
                
                print(f"{bone_name:<13} | {curr_pos_str:<25} | {interp_pos_str:<25} | {pos_diff:<8.3f} | {quat_diff:<8.3f}")
        
        # Show bones only in current method
        current_only = set(decoded_current_transforms.keys()) - set(interpolated_data.keys())
        if current_only:
            print(f"\nBones only in current method ({len(current_only)}):")
            for bone_name in sorted(current_only):
                print(f"  {bone_name}")
        
        # Show bones only in interpolation
        interp_only = set(interpolated_data.keys()) - set(decoded_current_transforms.keys())
        if interp_only:
            print(f"\nBones only in interpolation ({len(interp_only)}):")
            for bone_name in sorted(interp_only):
                print(f"  {bone_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during bezier interpolation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run bezier interpolation test."""
    print("=" * 80)
    print("VMD Bezier Interpolation Test - Frame 3000")
    print("=" * 80)
    
    success = interpolate_frame_3000()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 Bezier interpolation test completed!")
    else:
        print("❌ Bezier interpolation test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)