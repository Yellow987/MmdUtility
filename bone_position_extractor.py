"""
Bone Position Extractor Module

This module provides functionality to extract bone positions for every frame
from PMX model files and VMD motion files.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
try:
    # Try relative import first (when used as part of the package)
    from .pymeshio.pmx import reader as pmx_reader
    from .pymeshio.vmd import reader as vmd_reader
except ImportError:
    # Fallback to absolute import (when run directly)
    from pymeshio.pmx import reader as pmx_reader
    from pymeshio.vmd import reader as vmd_reader

# Pytransform3d integration
try:
    from pytransform3d.transform_manager import TransformManager
    HAS_PYTRANSFORM3D = True
except ImportError as e:
    HAS_PYTRANSFORM3D = False
    TransformManager = None
    print(f"Warning: pytransform3d not available: {e}")

# Bezier interpolation support
try:
    import bezier
    HAS_BEZIER = True
except ImportError:
    HAS_BEZIER = False
    print("Warning: bezier library not available. Install with: pip install bezier")

# Centralized definition of core bones for MMD models
# This includes ALL necessary parent bones to maintain proper FK chain
CORE_BONE_NAMES = {
    # Root bones
    "",                    # root bone (empty name)
    "全ての親",            # all parent (if exists)
    "センター",            # center
    "グルーブ",            # groove
    
    # Lower body hierarchy
    "下半身",              # lower body (critical parent bone often missing!)
    "腰",                 # waist
    "左足",               # leg_L
    "右足",               # leg_R
    "左ひざ",             # knee_L
    "右ひざ",             # knee_R
    "左足首",             # ankle_L
    "右足首",             # ankle_R
    "左つま先",           # toe_L
    "右つま先",           # toe_R
    
    # Upper body hierarchy
    "上半身",              # upper body
    "上半身2",             # upper body2
    "首",                 # neck
    "KUBI",               # neck (alternative naming)
    "頭",                 # head
    
    # Arms - left side
    "左肩",               # shoulder_L
    "左腕",               # arm_L
    "左ひじ",             # elbow_L
    "左手首",             # wrist_L
    
    # Arms - right side
    "右肩",               # shoulder_R
    "右腕",               # arm_R
    "右ひじ",             # elbow_R
    "右手首",             # wrist_R
    
    # Additional common parent bones that may be needed
    "センター先",          # center tip
    "下半身先",            # lower body tip
}


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
    """Parse VMD bezier interpolation parameters from complement field."""
    try:
        if isinstance(complement_hex, str):
            # Convert hex string to bytes
            complement_bytes = bytes.fromhex(complement_hex)
        else:
            # Already bytes or other format
            return None
            
        if len(complement_bytes) < 64:
            return None
            
        # VMD bezier parameters format: X, Y, Z, R channels
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
    """Perform bezier interpolation between two values using control points."""
    if control_points is None or not HAS_BEZIER:
        # Fallback to linear interpolation
        return lerp(p1, p2, t)
    
    try:
        x1, y1, x2, y2 = control_points
        
        # VMD bezier interpolation: solve for Y given X=t
        # Use iterative method to find parameter s where curve_x(s) = t
        def solve_bezier_x(target_x, max_iterations=20, tolerance=1e-6):
            """Find parameter s where the X coordinate of bezier curve equals target_x"""
            # Binary search approach
            s_low, s_high = 0.0, 1.0
            
            for _ in range(max_iterations):
                s_mid = (s_low + s_high) * 0.5
                
                # Evaluate bezier curve at s_mid: B(s) = (1-s)³P₀ + 3(1-s)²sP₁ + 3(1-s)s²P₂ + s³P₃
                # For X: P₀=(0,0), P₁=(x1,y1), P₂=(x2,y2), P₃=(1,1)
                curve_x = 3 * (1 - s_mid)**2 * s_mid * x1 + 3 * (1 - s_mid) * s_mid**2 * x2 + s_mid**3
                
                if abs(curve_x - target_x) < tolerance:
                    # Found it, now calculate Y
                    curve_y = 3 * (1 - s_mid)**2 * s_mid * y1 + 3 * (1 - s_mid) * s_mid**2 * y2 + s_mid**3
                    return curve_y
                
                if curve_x < target_x:
                    s_low = s_mid
                else:
                    s_high = s_mid
            
            # Fallback: linear interpolation if convergence fails
            return target_x
        
        bezier_t = solve_bezier_x(t)
        
        # Clamp to [0, 1]
        bezier_t = max(0.0, min(1.0, bezier_t))
        
        # Interpolate between p1 and p2 using the bezier-adjusted factor
        return lerp(p1, p2, bezier_t)
        
    except Exception as e:
        # Fallback to linear interpolation
        return lerp(p1, p2, t)


class BonePositionExtractor:
    """
    Extracts bone positions for every frame given a PMX model and VMD motion.
    
    Usage:
        extractor = BonePositionExtractor()
        positions = extractor.extract_bone_positions(pmx_path, vmd_path)
    """
    
    def __init__(self):
        self.pmx_model = None
        self.vmd_motion = None
        self.bone_hierarchy = {}
        self.bone_name_to_index = {}
        self.transform_manager = None
        
    def load_pmx(self, pmx_path: str):
        """
        Load PMX model file.
        
        Args:
            pmx_path: Path to the PMX file
        """
        try:
            self.pmx_model = pmx_reader.read_from_file(pmx_path)
            if self.pmx_model is None:
                raise ValueError(f"Failed to load PMX file: {pmx_path}")
            
            # Build bone hierarchy and name mapping
            self._build_bone_hierarchy()
            print(f"Successfully loaded PMX: {self.pmx_model.name}")
            print(f"Bones loaded: {len(self.pmx_model.bones)}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading PMX file {pmx_path}: {str(e)}")
    
    def load_vmd(self, vmd_path: str):
        """
        Load VMD motion file.
        
        Args:
            vmd_path: Path to the VMD file
        """
        try:
            self.vmd_motion = vmd_reader.read_from_file(vmd_path)
            if self.vmd_motion is None:
                raise ValueError(f"Failed to load VMD file: {vmd_path}")
                
            print(f"Successfully loaded VMD: {self.vmd_motion.model_name}")
            print(f"Bone motions loaded: {len(self.vmd_motion.motions)}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading VMD file {vmd_path}: {str(e)}")
    
    def _build_bone_hierarchy(self):
        """
        Build bone hierarchy from PMX model for efficient lookups.
        """
        if not self.pmx_model or not self.pmx_model.bones:
            return
            
        self.bone_hierarchy = {}
        self.bone_name_to_index = {}
        
        for i, bone in enumerate(self.pmx_model.bones):
            # Map bone name to index
            bone_name = bone.name
            if isinstance(bone_name, bytes):
                bone_name = bone_name.decode('utf-8', errors='ignore')
            
            self.bone_name_to_index[bone_name] = i
            
            # Store bone hierarchy info
            self.bone_hierarchy[i] = {
                'name': bone_name,
                'parent_index': bone.parent_index,
                'position': (bone.position.x, bone.position.y, bone.position.z),
                'children': []
            }
        
        # Build parent-child relationships
        for i, bone_info in self.bone_hierarchy.items():
            parent_idx = bone_info['parent_index']
            if parent_idx != -1 and parent_idx in self.bone_hierarchy:
                self.bone_hierarchy[parent_idx]['children'].append(i)
    
    def get_rest_pose_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get the world positions of all bones in rest pose (T-pose).
        
        Returns:
            Dictionary mapping bone names to their (x, y, z) world positions
        """
        if not self.pmx_model or not self.bone_hierarchy:
            raise ValueError("PMX model must be loaded first")
        
        # Calculate world positions for all bones
        world_positions = {}
        
        def calculate_world_position(bone_index: int, parent_world_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
            """
            Recursively calculate world position for a bone and its children.
            """
            if bone_index not in self.bone_hierarchy:
                return parent_world_pos
            
            bone_info = self.bone_hierarchy[bone_index]
            
            # PMX bone.position is absolute model-space position - convert to local offset
            if bone_info['parent_index'] == -1:
                # Root bone: use absolute position as-is
                local_offset = bone_info['position']
            else:
                # Child bone: local_offset = child.position - parent.position
                parent_info = self.bone_hierarchy[bone_info['parent_index']]
                parent_abs_pos = parent_info['position']
                child_abs_pos = bone_info['position']
                
                local_offset = (
                    child_abs_pos[0] - parent_abs_pos[0],
                    child_abs_pos[1] - parent_abs_pos[1],
                    child_abs_pos[2] - parent_abs_pos[2]
                )
            
            # Calculate world position by adding local offset to parent's world position
            world_pos = (
                parent_world_pos[0] + local_offset[0],
                parent_world_pos[1] + local_offset[1],
                parent_world_pos[2] + local_offset[2]
            )
            
            # Store the world position, handling duplicate names
            bone_name = bone_info['name']
            if bone_name in world_positions:
                # Handle duplicate bone names by appending index
                unique_name = f"{bone_name}_{bone_index}"
                world_positions[unique_name] = world_pos
            else:
                world_positions[bone_name] = world_pos
            
            # Process children
            for child_index in bone_info['children']:
                calculate_world_position(child_index, world_pos)
            
            return world_pos
        
        # Start with root bones (bones with no parent, parent_index = -1)
        root_bones = [i for i, info in self.bone_hierarchy.items() if info['parent_index'] == -1]
        
        for root_bone_index in root_bones:
            calculate_world_position(root_bone_index)
        
        # Handle any orphaned bones that weren't processed (bones with invalid parent references)
        processed_bones = set(world_positions.keys())
        all_bone_names = {info['name'] for info in self.bone_hierarchy.values()}
        orphaned_bones = all_bone_names - processed_bones
        
        if orphaned_bones:
            print(f"Debug: Found {len(orphaned_bones)} unprocessed bones: {list(orphaned_bones)}")
            # Process orphaned bones as if they were root bones
            for bone_index, bone_info in self.bone_hierarchy.items():
                if bone_info['name'] in orphaned_bones:
                    calculate_world_position(bone_index)
        
        return world_positions
    
    def _initialize_transform_manager(self):
        """
        Initialize TransformManager for skeleton transformations.
        """
        if not HAS_PYTRANSFORM3D:
            raise ImportError(
                "pytransform3d is required for skeleton transforms. "
                "Install with: pip install pytransform3d"
            )
        self.transform_manager = TransformManager()
    
    def _load_skeleton_to_transform_manager(self):
        """
        Load PMX skeleton into TransformManager with rest pose transforms.
        
        This creates a coordinate frame for each bone with the proper parent-child
        relationships and rest pose positions. Only loads core skeletal bones.
        """
        if not self.pmx_model or not self.bone_hierarchy:
            raise ValueError("PMX model must be loaded first")
        
        self._initialize_transform_manager()
        
        # Add world coordinate frame
        self.transform_manager.add_transform("world", "world", np.eye(4))
        
        # Use centralized core bone definition
        core_bone_names = CORE_BONE_NAMES
        
        # Find bone indices for core bones
        core_bone_indices = set()
        for bone_index, bone_info in self.bone_hierarchy.items():
            bone_name = bone_info['name']
            if bone_name in core_bone_names:
                core_bone_indices.add(bone_index)
        
        print(f"Found {len(core_bone_indices)} core bones out of {len(self.bone_hierarchy)} total bones")
        
        # Keep track of processed bones to handle dependencies
        processed_bones = set()
        
        def process_bone(bone_index):
            """Recursively process bone and its dependencies."""
            if bone_index in processed_bones or bone_index not in core_bone_indices:
                return
                
            bone_info = self.bone_hierarchy[bone_index]
            bone_name = f"bone_{bone_index}"  # Unique frame name
            parent_idx = bone_info['parent_index']
            
            # Ensure parent is processed first (if it's also a core bone)
            if parent_idx != -1 and parent_idx in core_bone_indices and parent_idx not in processed_bones:
                process_bone(parent_idx)
            
            # Create transformation matrix for this bone using local offset
            # PMX bone.position is absolute - convert to local offset
            if bone_info['parent_index'] == -1:
                # Root bone: use absolute position as-is
                local_offset = bone_info['position']
            else:
                # Child bone: local_offset = child.position - parent.position
                parent_info = self.bone_hierarchy[bone_info['parent_index']]
                parent_abs_pos = parent_info['position']
                child_abs_pos = bone_info['position']
                
                local_offset = (
                    child_abs_pos[0] - parent_abs_pos[0],
                    child_abs_pos[1] - parent_abs_pos[1],
                    child_abs_pos[2] - parent_abs_pos[2]
                )
            
            transform_matrix = np.eye(4)
            transform_matrix[:3, 3] = local_offset  # Set translation using local offset
            
            # Add transform to manager
            if parent_idx == -1:
                parent_frame = "world"
            elif parent_idx in core_bone_indices:
                parent_frame = f"bone_{parent_idx}"
            else:
                parent_bone_name = self.bone_hierarchy.get(parent_idx, {}).get('name', f'index_{parent_idx}')
                raise ValueError(f"Core bone '{bone_name}' (index {bone_index}) has parent '{parent_bone_name}' (index {parent_idx}) that is not in CORE_BONE_NAMES. "
                               f"Add '{parent_bone_name}' to CORE_BONE_NAMES to maintain proper FK chain.")
            
            self.transform_manager.add_transform(
                parent_frame, bone_name, transform_matrix
            )
            
            processed_bones.add(bone_index)
        
        # Process only core bones
        for bone_index in core_bone_indices:
            process_bone(bone_index)
        
        print(f"Loaded {len(processed_bones)} core bones into TransformManager")
    
    def get_rest_pose_positions_pytransform3d(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get rest pose positions using TransformManager.
        Only returns positions for core bones that were loaded into TransformManager.
        
        Returns:
            Dictionary mapping bone names to their (x, y, z) world positions
        """
        if not self.transform_manager:
            self._load_skeleton_to_transform_manager()
        
        # Use centralized core bone definition
        core_bone_names = CORE_BONE_NAMES
        
        positions = {}
        
        # Only try to get transforms for core bones
        for bone_index, bone_info in self.bone_hierarchy.items():
            bone_name = bone_info['name']
            
            # Skip non-core bones
            if bone_name not in core_bone_names:
                continue
                
            frame_name = f"bone_{bone_index}"
            
            try:
                # Get world transform for this bone
                world_transform = self.transform_manager.get_transform(
                    "world", frame_name
                )
                
                # Extract position (translation part of transformation matrix)
                world_pos = world_transform[:3, 3]
                positions[bone_name] = tuple(world_pos)
                
            except Exception as e:
                print(f"Warning: Could not get transform for core bone {bone_name}: {e}")
                # Skip this bone rather than providing a fallback
                continue
        
        return positions
    
    def _find_surrounding_keyframes(self, bone_name: str, target_frame: int):
        """Find keyframes before and after target frame for a specific bone."""
        bone_keyframes = []
        
        for motion in self.vmd_motion.motions:
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

    def _bezier_interpolate_vmd(self, frame, prev_frame, next_frame, prev_data, next_data, complement_data=None):
        """Perform bezier interpolation for VMD animation data using actual bezier curves."""
        if prev_frame == next_frame:
            return prev_data
        
        # Calculate time parameter
        t = (frame - prev_frame) / (next_frame - prev_frame)
        
        # Parse bezier parameters from VMD complement data
        bezier_params = parse_vmd_bezier_params(complement_data)
        
        if hasattr(prev_data, 'x'):  # It's a quaternion
            # Use SLERP for quaternions
            return slerp(prev_data, next_data, t)
        elif isinstance(prev_data, (tuple, list)):  # It's a position (x, y, z)
            if bezier_params and HAS_BEZIER:
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
            if bezier_params and HAS_BEZIER:
                return bezier_interpolate_value(t, prev_data, next_data, bezier_params['x_pos'])
            else:
                return lerp(prev_data, next_data, t)

    def get_frame_transforms(self, frame_number: int) -> Dict[str, Tuple]:
        """
        Extract position offsets and quaternions for a specific frame from VMD.
        
        Args:
            frame_number: The frame number to extract transforms from
            
        Returns:
            Dictionary mapping bone names to (position_offset, quaternion) tuples
        """
        if not self.vmd_motion:
            raise ValueError("VMD motion must be loaded first")
        
        frame_transforms = {}
        
        # Extract transforms for the specific frame from VMD motions
        for motion in self.vmd_motion.motions:
            if motion.frame == frame_number:
                bone_name = motion.name
                if isinstance(bone_name, bytes):
                    bone_name = bone_name.decode('utf-8', errors='ignore')
                
                # Store position offset and quaternion
                pos_offset = (motion.pos.x, motion.pos.y, motion.pos.z)
                quaternion = motion.q  # This is the Quaternion object
                
                frame_transforms[bone_name] = (pos_offset, quaternion)
        
        return frame_transforms

    def get_interpolated_frame_transforms(self, frame_number: int, core_bones_only: bool = True) -> Dict[str, Tuple]:
        """
        Extract interpolated position offsets and quaternions for a specific frame from VMD.
        Uses bezier interpolation between surrounding keyframes when exact keyframes don't exist.
        
        Args:
            frame_number: The frame number to extract transforms from
            core_bones_only: If True, only process core skeletal bones
            
        Returns:
            Dictionary mapping bone names to (position_offset, quaternion) tuples
        """
        if not self.vmd_motion:
            raise ValueError("VMD motion must be loaded first")

        # Define core bones if filtering is requested
        if core_bones_only:
            target_bone_names = {
                "グルーブ", "センター", "上半身", "上半身2",
                "右つま先", "右ひざ", "右ひじ", "右手首", "右肩", "右腕", "右足", "右足首",
                "左つま先", "左ひざ", "左ひじ", "左手首", "左肩", "左腕", "左足", "左足首",
                "腰", "頭", "首"
            }
        else:
            # Get all unique bone names from VMD
            target_bone_names = set()
            for motion in self.vmd_motion.motions:
                bone_name = motion.name
                if isinstance(bone_name, bytes):
                    bone_name = bone_name.decode('cp932', errors='ignore')
                target_bone_names.add(bone_name)
        
        interpolated_transforms = {}
        
        # Process each target bone
        for bone_name in target_bone_names:
            # Find surrounding keyframes for this bone
            prev_motion, next_motion = self._find_surrounding_keyframes(bone_name, frame_number)
            
            if prev_motion and next_motion:
                # Interpolate position
                prev_pos = (prev_motion.pos.x, prev_motion.pos.y, prev_motion.pos.z)
                next_pos = (next_motion.pos.x, next_motion.pos.y, next_motion.pos.z)
                interp_pos = self._bezier_interpolate_vmd(
                    frame_number, prev_motion.frame, next_motion.frame,
                    prev_pos, next_pos, prev_motion.complement
                )
                
                # Interpolate quaternion using SLERP
                interp_quat = self._bezier_interpolate_vmd(
                    frame_number, prev_motion.frame, next_motion.frame,
                    prev_motion.q, next_motion.q, prev_motion.complement
                )
                
                interpolated_transforms[bone_name] = (interp_pos, interp_quat)
                
            elif prev_motion:
                # Only previous keyframe exists, use its values
                prev_pos = (prev_motion.pos.x, prev_motion.pos.y, prev_motion.pos.z)
                interpolated_transforms[bone_name] = (prev_pos, prev_motion.q)
                
            elif next_motion:
                # Only next keyframe exists, use its values
                next_pos = (next_motion.pos.x, next_motion.pos.y, next_motion.pos.z)
                interpolated_transforms[bone_name] = (next_pos, next_motion.q)
        
        return interpolated_transforms
    
    def apply_frame_transforms_to_skeleton(self, frame_number: int, use_interpolation: bool = True) -> Dict[str, Tuple[float, float, float]]:
        """
        Apply VMD transforms from specific frame to core bones using pytransform3d.
        
        Args:
            frame_number: The frame number to extract and apply transforms from
            use_interpolation: If True, use bezier interpolation between keyframes
            
        Returns:
            Dictionary mapping core bone names to their (x, y, z) world positions after applying transforms
        """
        if not self.pmx_model or not self.vmd_motion or not self.bone_hierarchy:
            raise ValueError("Both PMX model and VMD motion must be loaded first")
        
        # Ensure skeleton is loaded into TransformManager
        if not self.transform_manager:
            self._load_skeleton_to_transform_manager()
        
        # Get frame transforms from VMD (with or without interpolation)
        if use_interpolation:
            frame_transforms = self.get_interpolated_frame_transforms(frame_number, core_bones_only=True)
            print(f"Found {len(frame_transforms)} interpolated bone transforms for frame {frame_number}")
        else:
            frame_transforms = self.get_frame_transforms(frame_number)
            print(f"Found {len(frame_transforms)} exact bone transforms in frame {frame_number}")
        
        # Use centralized core bone definition
        core_bone_names = CORE_BONE_NAMES
        
        # Apply VMD transforms to the TransformManager
        updated_bones = 0
        for bone_index, bone_info in self.bone_hierarchy.items():
            bone_name = bone_info['name']
            
            # Skip non-core bones
            if bone_name not in core_bone_names:
                continue
                
            frame_name = f"bone_{bone_index}"
            parent_idx = bone_info['parent_index']
            
            # Get rest position (as local offset) and VMD transforms
            # Calculate local offset for this bone
            if bone_info['parent_index'] == -1:
                # Root bone: use absolute position as-is
                rest_local_offset = bone_info['position']
            else:
                # Child bone: local_offset = child.position - parent.position
                parent_info = self.bone_hierarchy[bone_info['parent_index']]
                parent_abs_pos = parent_info['position']
                child_abs_pos = bone_info['position']
                
                rest_local_offset = (
                    child_abs_pos[0] - parent_abs_pos[0],
                    child_abs_pos[1] - parent_abs_pos[1],
                    child_abs_pos[2] - parent_abs_pos[2]
                )
            
            vmd_offset = (0.0, 0.0, 0.0)
            quaternion = None
            
            # Look for bone transforms (frame_transforms already contains properly decoded names)
            if bone_name in frame_transforms:
                vmd_offset, quaternion = frame_transforms[bone_name]
                updated_bones += 1
            
            # Create combined transform matrix: Rest_Local_Offset + VMD_Offset + Quaternion_Rotation
            transform_matrix = np.eye(4)
            
            # Apply rest local offset + VMD offset
            combined_position = (
                rest_local_offset[0] + vmd_offset[0],
                rest_local_offset[1] + vmd_offset[1],
                rest_local_offset[2] + vmd_offset[2]
            )
            transform_matrix[:3, 3] = combined_position
            
            # Apply quaternion rotation if available
            if quaternion is not None:
                # Handle different quaternion types
                if hasattr(quaternion, 'getMatrix'):
                    # VMD quaternion object
                    rotation_matrix = quaternion.getMatrix()
                    transform_matrix[:3, :3] = rotation_matrix[:3, :3]
                elif hasattr(quaternion, '__len__') and len(quaternion) == 4:
                    # Numpy array quaternion (x, y, z, w) from SLERP
                    try:
                        from scipy.spatial.transform import Rotation
                        # Convert to scipy format (x, y, z, w)
                        r = Rotation.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
                        rotation_matrix = r.as_matrix()
                        transform_matrix[:3, :3] = rotation_matrix
                    except ImportError:
                        # Fallback: convert to rotation matrix manually
                        x, y, z, w = quaternion
                        transform_matrix[:3, :3] = np.array([
                            [1-2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                            [2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)],
                            [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)]
                        ])
            
            # Update the transform in TransformManager
            if parent_idx == -1:
                parent_frame = "world"
            else:
                # Check if parent is in our core bone set
                parent_in_core = any(i == parent_idx and info['name'] in core_bone_names
                                   for i, info in self.bone_hierarchy.items())
                if parent_in_core:
                    parent_frame = f"bone_{parent_idx}"
                else:
                    parent_bone_name = self.bone_hierarchy.get(parent_idx, {}).get('name', f'index_{parent_idx}')
                    raise ValueError(f"Core bone '{bone_name}' has parent '{parent_bone_name}' (index {parent_idx}) that is not in CORE_BONE_NAMES. "
                                   f"Add '{parent_bone_name}' to CORE_BONE_NAMES to maintain proper FK chain.")
            
            try:
                # Remove existing transform and add updated one
                if frame_name in self.transform_manager.nodes:
                    self.transform_manager.remove_transform(parent_frame, frame_name)
                
                self.transform_manager.add_transform(parent_frame, frame_name, transform_matrix)
                
            except Exception as e:
                print(f"Warning: Could not update transform for {bone_name}: {e}")
        
        print(f"Updated {updated_bones} bones with VMD transforms")
        
        # Get world positions using TransformManager
        positions = {}
        for bone_index, bone_info in self.bone_hierarchy.items():
            bone_name = bone_info['name']
            
            # Skip non-core bones
            if bone_name not in core_bone_names:
                continue
                
            frame_name = f"bone_{bone_index}"
            
            try:
                # Get world transform from TransformManager
                world_transform = self.transform_manager.get_transform("world", frame_name)
                world_pos = tuple(world_transform[:3, 3])
                positions[bone_name] = world_pos
                
            except Exception as e:
                print(f"Warning: Could not get world position for {bone_name}: {e}")
        
        return positions
    
    def get_rest_pose_positions_array(self) -> np.ndarray:
        """
        Get rest pose positions as a numpy array.
        
        Returns:
            numpy array of shape (num_bones, 3) with XYZ positions
        """
        positions_dict = self.get_rest_pose_positions()
        
        # Create array in bone index order
        positions_array = np.zeros((len(self.bone_hierarchy), 3))
        
        for bone_index, bone_info in self.bone_hierarchy.items():
            bone_name = bone_info['name']
            if bone_name in positions_dict:
                positions_array[bone_index] = positions_dict[bone_name]
        
        return positions_array
    
    def extract_bone_positions(self, pmx_path: str, vmd_path: str) -> List[Dict[str, Tuple[float, float, float]]]:
        """
        Extract bone positions for every frame.
        
        Args:
            pmx_path: Path to the PMX model file
            vmd_path: Path to the VMD motion file
            
        Returns:
            List of dictionaries, where each dictionary maps bone names to their
            (x, y, z) world positions for that frame
        """
        # Load files
        self.load_pmx(pmx_path)
        self.load_vmd(vmd_path)
        
        if not self.pmx_model or not self.vmd_motion:
            raise ValueError("Both PMX model and VMD motion must be loaded")
        
        # TODO: Implement actual bone position calculation
        # For now, return empty structure showing the intended format
        print("Note: Bone position calculation not yet implemented in this draft")
        print("This module currently only loads the PMX and VMD files")
        
        # Placeholder return structure
        frame_positions = []
        
        # Show what the final structure will look like
        sample_frame = {}
        for bone_name in self.bone_name_to_index.keys():
            sample_frame[bone_name] = (0.0, 0.0, 0.0)  # Placeholder positions
        
        frame_positions.append(sample_frame)
        
        return frame_positions
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.pmx_model:
            return {"error": "No PMX model loaded"}
        
        return {
            "model_name": self.pmx_model.name,
            "english_name": getattr(self.pmx_model, 'english_name', ''),
            "bone_count": len(self.pmx_model.bones),
            "vertex_count": len(self.pmx_model.vertices),
            "material_count": len(self.pmx_model.materials)
        }
    
    def get_motion_info(self) -> Dict:
        """
        Get information about the loaded motion.
        
        Returns:
            Dictionary containing motion information
        """
        if not self.vmd_motion:
            return {"error": "No VMD motion loaded"}
        
        # Find frame range
        max_frame = 0
        if self.vmd_motion.motions:
            max_frame = max(motion.frame for motion in self.vmd_motion.motions)
        
        return {
            "model_name": self.vmd_motion.model_name,
            "bone_motion_count": len(self.vmd_motion.motions),
            "morph_count": len(self.vmd_motion.shapes),
            "camera_frame_count": len(self.vmd_motion.cameras),
            "max_frame": max_frame
        }


# Convenience functions for direct usage
def extract_bone_positions(pmx_path: str, vmd_path: str) -> List[Dict[str, Tuple[float, float, float]]]:
    """
    Convenience function to extract bone positions directly.
    
    Args:
        pmx_path: Path to the PMX model file
        vmd_path: Path to the VMD motion file
        
    Returns:
        List of dictionaries mapping bone names to (x, y, z) positions per frame
    """
    extractor = BonePositionExtractor()
    return extractor.extract_bone_positions(pmx_path, vmd_path)


def get_rest_pose_positions(pmx_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Convenience function to get rest pose positions directly.
    
    Args:
        pmx_path: Path to the PMX model file
        
    Returns:
        Dictionary mapping bone names to (x, y, z) rest positions
    """
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    return extractor.get_rest_pose_positions()


def get_rest_pose_positions_array(pmx_path: str) -> np.ndarray:
    """
    Convenience function to get rest pose positions as numpy array.
    
    Args:
        pmx_path: Path to the PMX model file
        
    Returns:
        numpy array of shape (num_bones, 3) with XYZ positions
    """
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    return extractor.get_rest_pose_positions_array()


def get_file_info(pmx_path: str, vmd_path: str) -> Tuple[Dict, Dict]:
    """
    Get information about PMX and VMD files.
    
    Args:
        pmx_path: Path to the PMX model file
        vmd_path: Path to the VMD motion file
        
    Returns:
        Tuple of (model_info, motion_info) dictionaries
    """
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    extractor.load_vmd(vmd_path)
    return extractor.get_model_info(), extractor.get_motion_info()