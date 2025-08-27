#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze VMD bone chain from right ankle to center bone
"""

import numpy as np
import struct
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
import quaternion
import math
import torch

# Set UTF-8 encoding for stdout on Windows
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add current directory to Python path for pymeshio import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import PMX reader (using local pymeshio in MmdUtility)
try:
    import pymeshio.pmx.reader as pmx_reader
    import pymeshio.pmx as pmx
    print("Successfully imported pymeshio from MmdUtility directory")
except ImportError as e:
    print(f"Error: Could not import pymeshio: {e}")
    print("Please ensure pymeshio is available in the MmdUtility directory")
    pmx_reader = None
    pmx = None


def create_bone_name_mapping() -> Dict[str, str]:
    """Create mapping from skeleton bone names to VMD bone names"""
    return {
        # Right side mappings (Blender .R to Japanese 右)
        "足首.R": "右足首",  # ankle.R -> right ankle
        "ひざ.R": "右ひざ",  # knee.R -> right knee
        "足.R": "右足",  # leg.R -> right leg
        # Left side mappings (Blender .L to Japanese 左)
        "足首.L": "左足首",  # ankle.L -> left ankle
        "ひざ.L": "左ひざ",  # knee.L -> left knee
        "足.L": "左足",  # leg.L -> left leg
        # Common bones that might have different names
        "腰": "腰",  # waist -> waist (same)
        "センター": "センター",  # center -> center (same)
        "グルーブ": "グルーブ",  # groove -> groove (same)
        # Add more mappings as needed
    }


def load_skeleton(
    skeleton_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load skeleton data from .npz file"""
    try:
        data = np.load(skeleton_path, allow_pickle=True)
        names = data["names"]
        parents = data["parents"]
        offsets = data["offsets"]
        meta = data["meta"].item()
        return names, parents, offsets, meta
    except Exception as e:
        print(f"Error loading skeleton: {e}")
        return None, None, None, None


def find_bone_index(names: np.ndarray, bone_name: str) -> int:
    """Find the index of a bone by name"""
    for i, name in enumerate(names):
        if name == bone_name:
            return i
    return -1


def get_bone_chain_to_root(
    names: np.ndarray, parents: np.ndarray, start_bone_idx: int
) -> List[int]:
    """Get the bone chain from start bone to root"""
    chain = []
    current_idx = start_bone_idx

    while current_idx != -1:
        chain.append(current_idx)
        current_idx = parents[current_idx]

    return chain


def convert_vmd_position_to_blender(vmd_pos: List[float]) -> List[float]:
    """
    Convert VMD position to Blender coordinates
    VMD: Right-handed, +Y up, +Z forward, +X left
    Blender: Right-handed, +Z up, +Y forward, +X right
    Scale: VMD uses 0.08m per unit, Blender uses 1m per unit
    """
    x_vmd, y_vmd, z_vmd = vmd_pos

    # Apply coordinate transformation and scaling
    x_b = -x_vmd * 0.08  # Flip X and scale
    y_b = y_vmd * 0.08  # Keep Y as Y and scale
    z_b = -z_vmd * 0.08  # Flip Z and scale

    return [x_b, y_b, z_b]



def load_pmx_model(pmx_path: str) -> Any:
    """Load PMX model using pymeshio - throws exception if failed"""
    if pmx_reader is None:
        raise ImportError("PMX reader not available - pymeshio not properly imported")
    
    if not os.path.exists(pmx_path):
        raise FileNotFoundError(f"PMX file not found: {pmx_path}")
    
    try:
        model = pmx_reader.read_from_file(pmx_path)
        if not model:
            raise ValueError("PMX model loaded but is empty")
        
        print(f"Successfully loaded PMX model with {len(model.bones)} bones")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load PMX model from {pmx_path}: {e}")

def convert_pmx_coord(pos, scale=1.0):
    """
    Convert PMX coordinates to Blender coordinates
    Left handed y-up to Right handed z-up (same as import_pmx.py)
    """
    return (pos.x * scale, pos.z * scale, pos.y * scale)

def create_vector(x, y, z):
    """Create a 3D vector as numpy array"""
    return np.array([x, y, z])

def calculate_bone_rest_matrix(bone, pmx_bones, scale=1.0):
    """
    Calculate the rest matrix for a PMX bone
    Returns 4x4 transformation matrix representing the bone's rest pose
    """
    # Get bone head position (converted to Blender coordinates)
    head = np.array(convert_pmx_coord(bone.position, scale))
    
    # Calculate tail position
    if hasattr(bone, 'getConnectionFlag') and bone.getConnectionFlag() and bone.tail_index != -1:
        # Connected bone: tail connects to child bone's head
        child_bone = pmx_bones[bone.tail_index]
        tail = np.array(convert_pmx_coord(child_bone.position, scale))
    else:
        # Offset bone: use tail_position
        if hasattr(bone, 'tail_position') and bone.tail_position:
            tail = head + np.array(convert_pmx_coord(bone.tail_position, scale))
        else:
            # Default tail direction
            tail = head + np.array([0, 0.01, 0])
    
    # Calculate bone direction vector
    bone_dir = tail - head
    bone_length = np.linalg.norm(bone_dir)
    
    if bone_length > 1e-6:
        bone_dir = bone_dir / bone_length
    else:
        bone_dir = np.array([0, 0, 1])  # Default up direction
    
    # Create transformation matrix
    # For now, create a simple translation matrix
    # More sophisticated roll calculation could be added later
    matrix = np.eye(4)
    matrix[:3, 3] = head  # Set translation
    
    # Create rotation matrix from bone direction
    # Use bone direction as local Y axis (Blender bone convention)
    local_y = bone_dir
    
    # Choose perpendicular vectors for X and Z axes
    if abs(local_y[2]) < 0.9:
        local_z = np.array([0, 0, 1])
    else:
        local_z = np.array([1, 0, 0])
    
    local_x = np.cross(local_y, local_z)
    local_x = local_x / np.linalg.norm(local_x)
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)
    
    # Set rotation part of matrix
    matrix[:3, 0] = local_x
    matrix[:3, 1] = local_y
    matrix[:3, 2] = local_z
    
    return matrix

def calculate_blender_bone_rest_matrix(bone_idx, names, parents, offsets):
    """
    Calculate the rest matrix for a Blender bone from skeleton.npz data
    """
    bone_name = names[bone_idx]
    offset = offsets[bone_idx]
    parent_idx = parents[bone_idx]
    
    # Start with identity matrix
    matrix = np.eye(4)
    
    # Set translation from offset
    matrix[:3, 3] = offset
    
    # For parent bones, accumulate transforms
    if parent_idx != -1:
        parent_matrix = calculate_blender_bone_rest_matrix(parent_idx, names, parents, offsets)
        # Accumulate parent transformation
        matrix[:3, 3] = parent_matrix[:3, 3] + offset
    
    return matrix

def calculate_bone_conversion_matrix(pmx_rest_matrix, blender_rest_matrix):
    """
    Calculate conversion matrix: C_bone = RestBlender @ RestPMX^-1
    """
    try:
        pmx_inv = np.linalg.inv(pmx_rest_matrix)
        conversion_matrix = blender_rest_matrix @ pmx_inv
        return conversion_matrix
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix encountered, using identity")
        return np.eye(4)

def convert_quaternion_with_matrix(vmd_quat: List[float], conversion_matrix: np.ndarray) -> List[float]:
    """
    Convert VMD quaternion using the per-bone conversion matrix
    """
    # Convert to numpy quaternion (w, x, y, z format)
    x_vmd, y_vmd, z_vmd, w_vmd = vmd_quat
    quat_vmd = np.quaternion(w_vmd, x_vmd, y_vmd, z_vmd)
    
    # Extract rotation part of conversion matrix
    rot_matrix = conversion_matrix[:3, :3]
    
    # Convert rotation matrix to quaternion
    conversion_quat = quaternion.from_rotation_matrix(rot_matrix)
    
    # Apply conversion: q_blender = conversion_quat * q_vmd * conversion_quat.conjugate()
    quat_converted = conversion_quat * quat_vmd * conversion_quat.conjugate()
    
    # Return as [x, y, z, w] list
    return [quat_converted.x, quat_converted.y, quat_converted.z, quat_converted.w]


class ForwardKinematics:
    """PyTorch-based forward kinematics calculator for bone chains."""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
    
    def compute_world_positions(self,
                              bone_offsets: torch.Tensor,
                              quaternions: torch.Tensor,
                              parent_indices: List[int],
                              local_translations: torch.Tensor = None) -> torch.Tensor:
        """
        Compute world positions using forward kinematics.
        
        Args:
            bone_offsets: [N, 3] tensor of bone offsets from parent (bind pose)
            quaternions: [N, 4] tensor of local rotations (x,y,z,w) - normalized
            parent_indices: List of parent bone indices (-1 for root)
            local_translations: [N, 3] tensor of per-frame local translations (optional)
            
        Returns:
            world_positions: [N, 3] tensor of world positions
        """
        n_bones = bone_offsets.shape[0]
        world_positions = torch.zeros_like(bone_offsets)
        world_rotations = quaternions.clone()  # Start with normalized quaternions
        
        if local_translations is None:
            local_translations = torch.zeros_like(bone_offsets)
        
        # Get traversal order (parents before children)
        traversal_order = self._get_traversal_order(parent_indices)
        
        for j in traversal_order:
            p = parent_indices[j]
            
            if p == -1:  # Root bone
                world_positions[j] = local_translations[j]
                world_rotations[j] = quaternions[j]
            else:  # Child bone
                parent_rot = world_rotations[p]
                world_rotations[j] = self._quaternion_multiply(parent_rot, quaternions[j])
                step = bone_offsets[j] + local_translations[j]
                world_positions[j] = world_positions[p] + self._rotate_vector_by_quaternion(step, parent_rot)
        
        return world_positions
    
    def _get_traversal_order(self, parent_indices: List[int]) -> List[int]:
        """Get bone traversal order (topological sort - parents before children)."""
        n_bones = len(parent_indices)
        visited = [False] * n_bones
        order = []
        
        def visit(bone_idx):
            if visited[bone_idx]:
                return
            
            parent_idx = parent_indices[bone_idx]
            if parent_idx != -1:
                visit(parent_idx)  # Visit parent first
            
            visited[bone_idx] = True
            order.append(bone_idx)
        
        for i in range(n_bones):
            visit(i)
        
        return order
    
    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions: q1 * q2."""
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
        
        result = torch.empty_like(q1)
        result[0] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        result[1] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        result[2] = w1*z2 + x1*y2 - y1*x2 + z1*w2
        result[3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        return result
    
    def _rotate_vector_by_quaternion(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q using efficient method."""
        q_vec = q[:3]  # x, y, z components
        q_w = q[3]     # w component
        
        # First cross product: cross(q_vec, v) + q_w * v
        cross1 = torch.linalg.cross(q_vec, v) + q_w * v
        
        # Second cross product: cross(q_vec, cross1)
        cross2 = torch.linalg.cross(q_vec, cross1)
        
        # Final result: v + 2 * cross2
        return v + 2.0 * cross2


def extract_pmx_rest_positions(pmx_model: Any, bone_indices: List[int]) -> Dict[int, np.ndarray]:
    """
    Extract rest positions from PMX model for specified bone indices.
    
    Args:
        pmx_model: PMX model object
        bone_indices: List of bone indices to extract positions for
        
    Returns:
        Dictionary mapping bone_index -> rest_position (as numpy array in MMD coordinates)
    """
    rest_positions = {}
    
    for bone_idx in bone_indices:
        if bone_idx < len(pmx_model.bones):
            pmx_bone = pmx_model.bones[bone_idx]
            # Keep PMX coordinates in MMD coordinate system for FK calculations
            rest_pos = [pmx_bone.position.x, pmx_bone.position.y, pmx_bone.position.z]
            rest_positions[bone_idx] = np.array(rest_pos)
        else:
            # Fallback if bone index is out of range
            rest_positions[bone_idx] = np.array([0.0, 0.0, 0.0])
    
    return rest_positions


def read_vmd_bone_data(vmd_path: str) -> Dict[str, List[Dict]]:
    """Read VMD file and extract bone animation data"""
    bone_data = {}

    try:
        with open(vmd_path, "rb") as f:
            # Read VMD header
            header = f.read(30)
            if not header.startswith(b"Vocaloid Motion Data 0002"):
                print("Not a valid VMD file")
                return {}

            # Read model name (20 bytes)
            model_name = f.read(20)
            decoded_name = model_name.decode("shift_jis", errors="ignore").rstrip(
                "\x00"
            )
            print(f"Model name: {decoded_name}")

            # Read bone frame count
            bone_count = struct.unpack("<L", f.read(4))[0]
            print(f"Bone frame count: {bone_count}")

            # Read bone frames
            for i in range(bone_count):
                # Read bone name (15 bytes)
                bone_name_bytes = f.read(15)
                bone_name = bone_name_bytes.decode("shift_jis", errors="ignore").rstrip(
                    "\x00"
                )

                # Read frame number
                frame_num = struct.unpack("<L", f.read(4))[0]

                # Read position (3 floats)
                pos_x, pos_y, pos_z = struct.unpack("<fff", f.read(12))

                # Read quaternion (4 floats)
                quat_x, quat_y, quat_z, quat_w = struct.unpack("<ffff", f.read(16))

                # Read interpolation data (64 bytes)
                interpolation = f.read(64)

                # Store bone data
                if bone_name not in bone_data:
                    bone_data[bone_name] = []

                bone_data[bone_name].append(
                    {
                        "frame": frame_num,
                        "position": [pos_x, pos_y, pos_z],
                        "quaternion": [quat_x, quat_y, quat_z, quat_w],
                    }
                )

                # Progress indicator for large files
                if i % 50000 == 0:
                    print(f"Processing frame {i+1}/{bone_count}...")

    except Exception as e:
        print(f"Error reading VMD file: {e}")
        return {}

    return bone_data


def print_bone_chain_animation(
    names: np.ndarray,
    bone_chain: List[int],
    bone_data: Dict[str, List[Dict]],
    target_frame: int = 6,
    pmx_model: Any = None,
    parents: np.ndarray = None,
    offsets: np.ndarray = None,
):
    """Print animation data for the entire bone chain at specific frame with PyTorch FK calculations"""
    print("\n" + "=" * 80)
    print("BONE CHAIN ANIMATION DATA WITH PYTORCH FORWARD KINEMATICS")
    print("=" * 80)

    # Create bone name mapping
    bone_mapping = create_bone_name_mapping()
    
    # Initialize PyTorch FK calculator
    fk_calculator = ForwardKinematics(device='cpu')
    
    # Extract PMX rest positions for bones in the chain
    pmx_rest_positions = {}
    if pmx_model:
        # Create PMX bone name to index mapping
        pmx_bone_map = {bone.name: i for i, bone in enumerate(pmx_model.bones)}
        
        # Map skeleton bone indices to PMX bone indices
        skeleton_to_pmx_map = {}
        for bone_idx in bone_chain:
            skeleton_bone_name = names[bone_idx]
            vmd_bone_name = bone_mapping.get(skeleton_bone_name, skeleton_bone_name)
            
            if skeleton_bone_name in pmx_bone_map:
                pmx_bone_idx = pmx_bone_map[skeleton_bone_name]
                skeleton_to_pmx_map[bone_idx] = pmx_bone_idx
            elif vmd_bone_name in pmx_bone_map:
                pmx_bone_idx = pmx_bone_map[vmd_bone_name]
                skeleton_to_pmx_map[bone_idx] = pmx_bone_idx
        
        # Extract PMX rest positions
        pmx_rest_positions = extract_pmx_rest_positions(pmx_model, list(skeleton_to_pmx_map.values()))
    
    # Calculate conversion matrices for all bones in the chain if PMX model is available
    conversion_matrices = {}
    if pmx_model and parents is not None and offsets is not None:
        print("\nCalculating per-bone conversion matrices...")
        
        for bone_idx in bone_chain:
            skeleton_bone_name = names[bone_idx]
            vmd_bone_name = bone_mapping.get(skeleton_bone_name, skeleton_bone_name)
            
            # Find corresponding PMX bone
            pmx_bone = None
            pmx_bone_idx = None
            
            # Try skeleton name first, then VMD name
            if skeleton_bone_name in pmx_bone_map:
                pmx_bone_idx = pmx_bone_map[skeleton_bone_name]
                pmx_bone = pmx_model.bones[pmx_bone_idx]
            elif vmd_bone_name in pmx_bone_map:
                pmx_bone_idx = pmx_bone_map[vmd_bone_name]
                pmx_bone = pmx_model.bones[pmx_bone_idx]
                
            if pmx_bone:
                # Calculate PMX rest matrix
                pmx_rest_matrix = calculate_bone_rest_matrix(pmx_bone, pmx_model.bones)
                
                # Calculate Blender rest matrix
                blender_rest_matrix = calculate_blender_bone_rest_matrix(bone_idx, names, parents, offsets)
                
                # Calculate conversion matrix
                conv_matrix = calculate_bone_conversion_matrix(pmx_rest_matrix, blender_rest_matrix)
                conversion_matrices[bone_idx] = {
                    'matrix': conv_matrix,
                    'pmx_rest': pmx_rest_matrix,
                    'blender_rest': blender_rest_matrix,
                    'pmx_bone_name': pmx_bone.name
                }
                
                print(f"  {skeleton_bone_name} -> PMX: {pmx_bone.name}")
            else:
                print(f"  {skeleton_bone_name} -> No PMX match found")
    else:
        raise ValueError("PMX model is required for per-bone conversion matrices")

    # Get all frame numbers that exist for any bone in the chain
    all_frames = set()
    chain_bone_names = [names[idx] for idx in bone_chain]

    for bone_idx in bone_chain:
        skeleton_bone_name = names[bone_idx]
        # Try to find VMD bone name using mapping, otherwise use skeleton name
        vmd_bone_name = bone_mapping.get(skeleton_bone_name, skeleton_bone_name)

        # Check both skeleton name and mapped VMD name for animation data
        if skeleton_bone_name in bone_data:
            for frame_data in bone_data[skeleton_bone_name]:
                all_frames.add(frame_data["frame"])
        elif vmd_bone_name in bone_data:
            for frame_data in bone_data[vmd_bone_name]:
                all_frames.add(frame_data["frame"])

    # Use only the target frame if it exists
    if target_frame in all_frames:
        sorted_frames = [target_frame]
    else:
        print(f"Frame {target_frame} not found in animation data")
        sorted_frames = sorted(list(all_frames))[:5]  # fallback to first 5 frames

    print(f"\nBone chain ({len(bone_chain)} bones):")
    for i, bone_idx in enumerate(bone_chain):
        bone_name = names[bone_idx]
        print(f"  {i}: {bone_name} (index: {bone_idx})")

    print(f"\nShowing animation data for first {len(sorted_frames)} frames:")
    print(f"Available frames: {sorted_frames}")

    for frame_num in sorted_frames:
        print("\n" + "-" * 60)
        print(f"FRAME {frame_num}")
        print("-" * 60)
        
        # Create bone index to chain position mapping first
        bone_idx_to_chain_pos = {bone_idx: pos for pos, bone_idx in enumerate(bone_chain)}
        
        # Prepare PyTorch tensors for FK calculations
        # Calculate proper relative bone offsets from PMX rest positions
        bone_offsets_mmd = []
        for i, bone_idx in enumerate(bone_chain):
            parent_idx = parents[bone_idx]
            
            if pmx_model and bone_idx in skeleton_to_pmx_map:
                pmx_bone_idx = skeleton_to_pmx_map[bone_idx]
                if pmx_bone_idx in pmx_rest_positions:
                    current_rest_pos = pmx_rest_positions[pmx_bone_idx]
                    
                    # Find parent's rest position
                    if parent_idx in bone_idx_to_chain_pos and parent_idx in skeleton_to_pmx_map:
                        parent_pmx_idx = skeleton_to_pmx_map[parent_idx]
                        if parent_pmx_idx in pmx_rest_positions:
                            parent_rest_pos = pmx_rest_positions[parent_pmx_idx]
                            # Calculate relative offset: child_pos - parent_pos
                            relative_offset = current_rest_pos - parent_rest_pos
                            bone_offsets_mmd.append(relative_offset)
                        else:
                            # Parent not in PMX, use absolute position
                            bone_offsets_mmd.append(current_rest_pos)
                    else:
                        # No parent in chain, use absolute position (for root bones)
                        bone_offsets_mmd.append(current_rest_pos)
                else:
                    # Fallback to skeleton offset converted to MMD coordinates
                    blender_offset = offsets[bone_idx]
                    mmd_offset = [-blender_offset[0]*12.5, blender_offset[1]*12.5, -blender_offset[2]*12.5]
                    bone_offsets_mmd.append(np.array(mmd_offset))
            else:
                # Fallback to skeleton offset converted to MMD coordinates
                blender_offset = offsets[bone_idx]
                mmd_offset = [-blender_offset[0]*12.5, blender_offset[1]*12.5, -blender_offset[2]*12.5]
                bone_offsets_mmd.append(np.array(mmd_offset))
        
        bone_offsets_tensor = torch.tensor(np.array(bone_offsets_mmd), dtype=torch.float32)
        
        # Debug output for bone offsets
        print(f"\n   Calculated relative bone offsets (MMD):")
        for i, (bone_idx, offset) in enumerate(zip(bone_chain, bone_offsets_mmd)):
            bone_name = names[bone_idx]
            print(f"     {bone_name}: X={offset[0]:.6f}, Y={offset[1]:.6f}, Z={offset[2]:.6f}")
        
        # Create parent indices mapping for bone chain (map to chain positions, not full skeleton indices)
        parent_indices_chain = []
        for bone_idx in bone_chain:
            parent_idx = parents[bone_idx]
            if parent_idx in bone_idx_to_chain_pos:
                # Parent is in the chain, map to chain position
                parent_indices_chain.append(bone_idx_to_chain_pos[parent_idx])
            else:
                # Parent is not in chain, treat as root
                parent_indices_chain.append(-1)
        quaternions_frame = torch.zeros(len(bone_chain), 4, dtype=torch.float32)
        quaternions_frame[:, 3] = 1.0  # Initialize with identity quaternions (w=1)
        local_translations_frame = torch.zeros(len(bone_chain), 3, dtype=torch.float32)
        
        # Collect quaternions and translations for this frame
        for chain_pos, bone_idx in enumerate(bone_chain):
            skeleton_bone_name = names[bone_idx]
            vmd_bone_name = bone_mapping.get(skeleton_bone_name, skeleton_bone_name)
            
            # Find data for this bone at this frame
            frame_data = None
            if skeleton_bone_name in bone_data:
                for data in bone_data[skeleton_bone_name]:
                    if data["frame"] == frame_num:
                        frame_data = data
                        break
            elif vmd_bone_name in bone_data and vmd_bone_name != skeleton_bone_name:
                for data in bone_data[vmd_bone_name]:
                    if data["frame"] == frame_num:
                        frame_data = data
                        break
            
            if frame_data:
                # Convert VMD quaternion (x,y,z,w) to tensor
                quat = frame_data["quaternion"]
                quaternions_frame[chain_pos] = torch.tensor([quat[0], quat[1], quat[2], quat[3]], dtype=torch.float32)
                
                # Keep VMD position in MMD coordinates for FK calculations
                pos = frame_data["position"]
                local_translations_frame[chain_pos] = torch.tensor([pos[0], pos[1], pos[2]], dtype=torch.float32)
        
        # Normalize quaternions
        quaternions_frame = quaternions_frame / torch.norm(quaternions_frame, dim=1, keepdim=True)
        
        # Calculate world positions using PyTorch FK
        try:
            world_positions_tensor = fk_calculator.compute_world_positions(
                bone_offsets_tensor,
                quaternions_frame,
                parent_indices_chain,
                local_translations_frame
            )
            world_positions_np = world_positions_tensor.numpy()
            fk_success = True
        except Exception as e:
            print(f"PyTorch FK calculation failed: {e}")
            world_positions_np = None
            fk_success = False

        for i, bone_idx in enumerate(bone_chain):
            skeleton_bone_name = names[bone_idx]
            # Try to find VMD bone name using mapping
            vmd_bone_name = bone_mapping.get(skeleton_bone_name, skeleton_bone_name)

            # Find data for this bone at this frame
            frame_data = None
            data_source = None

            # First try skeleton bone name
            if skeleton_bone_name in bone_data:
                for data in bone_data[skeleton_bone_name]:
                    if data["frame"] == frame_num:
                        frame_data = data
                        data_source = skeleton_bone_name
                        break

            # If not found, try mapped VMD bone name
            if (
                not frame_data
                and vmd_bone_name in bone_data
                and vmd_bone_name != skeleton_bone_name
            ):
                for data in bone_data[vmd_bone_name]:
                    if data["frame"] == frame_num:
                        frame_data = data
                        data_source = vmd_bone_name
                        break

            print(f"\n{i+1}. {skeleton_bone_name} (index: {bone_idx})")
            if vmd_bone_name != skeleton_bone_name:
                print(f"   VMD bone name: {vmd_bone_name}")
            
            # Display PMX rest position if available (in MMD coordinates)
            if pmx_model and bone_idx in skeleton_to_pmx_map:
                pmx_bone_idx = skeleton_to_pmx_map[bone_idx]
                if pmx_bone_idx in pmx_rest_positions:
                    rest_pos = pmx_rest_positions[pmx_bone_idx]
                    print(f"   PMX Rest Position (MMD): X={rest_pos[0]:.6f}, Y={rest_pos[1]:.6f}, Z={rest_pos[2]:.6f}")
            
            # Display PyTorch FK calculated world position (in MMD coordinates)
            if fk_success and world_positions_np is not None:
                world_pos = world_positions_np[i]
                print(f"   PyTorch FK World Position (MMD): X={world_pos[0]:.6f}, Y={world_pos[1]:.6f}, Z={world_pos[2]:.6f}")
            
            if frame_data:
                pos = frame_data["position"]
                quat = frame_data["quaternion"]

                # Display original VMD data
                print(
                    f"   VMD Position: X={pos[0]:.6f}, Y={pos[1]:.6f}, Z={pos[2]:.6f}"
                )
                print(
                    f"   VMD Quaternion: X={quat[0]:.6f}, Y={quat[1]:.6f}, Z={quat[2]:.6f}, W={quat[3]:.6f}"
                )

                # Display rest matrices if available
                if bone_idx in conversion_matrices:
                    conv_info = conversion_matrices[bone_idx]
                    print(f"   PMX Bone: {conv_info['pmx_bone_name']}")
                    print(f"   PMX Rest Matrix:")
                    for row_idx, row in enumerate(conv_info['pmx_rest']):
                        print(f"     [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")
                    print(f"   Blender Rest Matrix:")
                    for row_idx, row in enumerate(conv_info['blender_rest']):
                        print(f"     [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")
                    print(f"   Conversion Matrix:")
                    for row_idx, row in enumerate(conv_info['matrix']):
                        print(f"     [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

                # Convert to Blender format and display
                try:
                    blender_pos = convert_vmd_position_to_blender(pos)
                    
                    # Use per-bone conversion matrix (required)
                    if bone_idx in conversion_matrices:
                        conv_matrix = conversion_matrices[bone_idx]['matrix']
                        blender_quat = convert_quaternion_with_matrix(quat, conv_matrix)
                        print(f"   Using per-bone conversion matrix")
                    else:
                        raise ValueError(f"No conversion matrix available for bone {skeleton_bone_name}")
                    
                    print(
                        f"   Blender Position: X={blender_pos[0]:.6f}, Y={blender_pos[1]:.6f}, Z={blender_pos[2]:.6f}"
                    )
                    print(
                        f"   Blender Quaternion: X={blender_quat[0]:.6f}, Y={blender_quat[1]:.6f}, Z={blender_quat[2]:.6f}, W={blender_quat[3]:.6f}"
                    )
                except Exception as e:
                    print(f"   Error converting to Blender format: {e}")

                if data_source != skeleton_bone_name:
                    print(f"   (Data from: {data_source})")
            else:
                print("   No animation data for this frame")


def main():
    # Paths (updated for MmdUtility structure)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    skeleton_path = os.path.join(base_dir, "test", "skeleton.npz")
    pmx_path = os.path.join(base_dir, "test", "pdtt.pmx")
    vmd_path = os.path.join(base_dir, "test", "dan_alivef_01.imo.vmd")

    print("Loading skeleton data...")
    names, parents, offsets, meta = load_skeleton(skeleton_path)

    if names is None:
        print("Failed to load skeleton data")
        return

    print(f"Loaded skeleton with {len(names)} bones")
    print(f"Meta info: {meta}")

    # Load PMX model - required for per-bone conversion
    print(f"\nLoading PMX model from: {pmx_path}")
    try:
        pmx_model = load_pmx_model(pmx_path)
        print(f"PMX bones sample: {[bone.name for bone in pmx_model.bones[:5]]}")
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        print("Per-bone conversion requires PMX model - cannot continue without it")
        return

    # Find the target bones
    center_bone_name = "センター"
    right_ankle_name = "右足首"  # Right ankle bone name (Japanese convention)

    # Also try alternative ankle names (prioritizing Japanese naming)
    alternative_ankle_names = [
        "右足首",
        "右足ＩＫ",
        "右足",
        "右足先",
        "足首.R",
        "foot_R",
        "ankle_R",
        "R_ankle",
    ]

    center_idx = find_bone_index(names, center_bone_name)
    ankle_idx = find_bone_index(names, right_ankle_name)

    # If primary ankle name not found, try alternatives
    if ankle_idx == -1:
        for alt_name in alternative_ankle_names:
            ankle_idx = find_bone_index(names, alt_name)
            if ankle_idx != -1:
                right_ankle_name = alt_name
                print(f"Found ankle bone with alternative name: {alt_name}")
                break

    print("\nBone indices:")
    print(f"  {center_bone_name}: {center_idx}")
    print(f"  {right_ankle_name}: {ankle_idx}")

    if center_idx == -1:
        print(f"Could not find {center_bone_name} bone")
        return
    if ankle_idx == -1:
        print(
            f"Could not find right ankle bone with any of these names: {alternative_ankle_names}"
        )
        return

    # Get bone chain from ankle to root (which should include center)
    bone_chain = get_bone_chain_to_root(names, parents, ankle_idx)
    print(f"\nBone chain from {right_ankle_name} to root:")
    for i, bone_idx in enumerate(bone_chain):
        bone_name = names[bone_idx]
        parent_idx = parents[bone_idx]
        parent_name = names[parent_idx] if parent_idx != -1 else "ROOT"
        print(f"  {i}: {bone_name} -> {parent_name}")

    # Check if center bone is in the chain
    if center_idx in bone_chain:
        print(
            f"\n✓ {center_bone_name} is in the bone chain at position {bone_chain.index(center_idx)}"
        )
    else:
        print(f"\n✗ {center_bone_name} is NOT in the bone chain")
        # Let's see the chain to center too
        center_chain = get_bone_chain_to_root(names, parents, center_idx)
        print(f"\nBone chain from {center_bone_name} to root:")
        for i, bone_idx in enumerate(center_chain):
            bone_name = names[bone_idx]
            parent_idx = parents[bone_idx]
            parent_name = names[parent_idx] if parent_idx != -1 else "ROOT"
            print(f"  {i}: {bone_name} -> {parent_name}")

    # Load VMD data
    print(f"\nLoading VMD animation data from: {vmd_path}")
    bone_data = read_vmd_bone_data(vmd_path)

    if not bone_data:
        print("No bone data loaded from VMD file")
        return

    print(f"Loaded animation data for {len(bone_data)} bones")

    # Print bone chain animation data for frame 6 specifically
    print_bone_chain_animation(
        names, bone_chain, bone_data, target_frame=6,
        pmx_model=pmx_model, parents=parents, offsets=offsets
    )

    # Also print some summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    bone_mapping = create_bone_name_mapping()

    for i, bone_idx in enumerate(bone_chain):
        skeleton_bone_name = names[bone_idx]
        vmd_bone_name = bone_mapping.get(skeleton_bone_name, skeleton_bone_name)

        # Check both skeleton name and VMD name for data
        found_data = False
        if skeleton_bone_name in bone_data:
            frame_count = len(bone_data[skeleton_bone_name])
            frames = [data["frame"] for data in bone_data[skeleton_bone_name]]
            min_frame = min(frames)
            max_frame = max(frames)
            print(
                f"{i+1}. {skeleton_bone_name}: {frame_count} keyframes (frames {min_frame}-{max_frame})"
            )
            found_data = True
        elif vmd_bone_name in bone_data and vmd_bone_name != skeleton_bone_name:
            frame_count = len(bone_data[vmd_bone_name])
            frames = [data["frame"] for data in bone_data[vmd_bone_name]]
            min_frame = min(frames)
            max_frame = max(frames)
            print(
                f"{i+1}. {skeleton_bone_name} (VMD: {vmd_bone_name}): {frame_count} keyframes (frames {min_frame}-{max_frame})"
            )
            found_data = True

        if not found_data:
            vmd_suffix = (
                f" (VMD: {vmd_bone_name})"
                if vmd_bone_name != skeleton_bone_name
                else ""
            )
            print(f"{i+1}. {skeleton_bone_name}{vmd_suffix}: No animation data")


if __name__ == "__main__":
    main()
