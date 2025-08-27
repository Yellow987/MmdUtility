# Skeleton Quaternion Validation Script Design

## Overview

Create a standalone script to load a PMX model, extract quaternions from JSON data, perform forward kinematics using PyTorch, and validate derived positions against JSON positions.

## Requirements Analysis

### Input Files

1. **PMX File**: `MmdUtility/test/pdtt.pmx` (binary MMD model)
2. **JSON File**: `MmdUtility/test/dan_alivef_01.imo_bone_positions_quaternions.json` (animation data)

### Validation Parameters

- **Target Frame**: Frame 0 (rest pose validation)
- **Tolerance**: 0.001 unit tolerance for position matching
- **Output**: Summary report with accuracy metrics

## Technical Architecture

### Data Structures

#### PMX Bone Structure

```python
{
    'name': str,                    # Japanese bone name
    'english_name': str,            # English bone name
    'position': (x, y, z),          # Local rest position
    'parent_index': int,            # Parent bone index (-1 for root)
    'index': int                    # Bone index in hierarchy
}
```

#### JSON Bone Structure (Frame 0)

```python
{
    'position': {'x': float, 'y': float, 'z': float},      # World position
    'quaternion': {'x': float, 'y': float, 'z': float, 'w': float},  # Local rotation
    'boneIndex': int                                        # Bone index
}
```

### Core Components

#### 1. PMX Loader (`load_pmx_model`)

- Use `MmdUtility.pymeshio.pmx.reader.read_from_file()`
- Extract bone hierarchy with parent relationships
- Apply coordinate conversion: Left-handed Y-up → Right-handed Z-up
- Build bone index mapping

#### 2. JSON Parser (`parse_quaternion_data`)

- Load JSON and extract frame 0 data
- Map bone names to quaternions and positions
- Handle Japanese character encoding properly

#### 3. Forward Kinematics Engine (`pytorch_forward_kinematics`)

- Use PyTorch tensors for efficient computation
- Apply quaternion rotations using `torch.quaternion_multiply`
- Traverse bone hierarchy depth-first
- Accumulate transformations from root to leaf

#### 4. Validation Engine (`validate_positions`)

- Compare derived vs JSON positions
- Calculate per-bone error metrics
- Generate summary statistics

## Implementation Details

### Coordinate System Handling

The PMX format uses a different coordinate system than the JSON data:

```python
def convert_coord(pos, scale=1.0):
    """Left handed y-up to Right handed z-up"""
    return (pos.x * scale, pos.z * scale, pos.y * scale)
```

### Forward Kinematics Algorithm

```python
def compute_world_positions(rest_poses, quaternions, hierarchy):
    """
    Args:
        rest_poses: [N, 3] tensor of local rest positions
        quaternions: [N, 4] tensor of local rotations (x,y,z,w)
        hierarchy: List of parent indices
    Returns:
        world_positions: [N, 3] tensor of world positions
    """
    world_positions = torch.zeros_like(rest_poses)
    world_rotations = torch.zeros_like(quaternions)

    for bone_idx in traversal_order(hierarchy):
        parent_idx = hierarchy[bone_idx]

        if parent_idx == -1:  # Root bone
            world_rotations[bone_idx] = quaternions[bone_idx]
            world_positions[bone_idx] = rest_poses[bone_idx]
        else:  # Child bone
            # Accumulate rotation
            parent_rot = world_rotations[parent_idx]
            local_rot = quaternions[bone_idx]
            world_rotations[bone_idx] = quaternion_multiply(parent_rot, local_rot)

            # Transform position
            rotated_offset = rotate_vector(rest_poses[bone_idx], parent_rot)
            world_positions[bone_idx] = world_positions[parent_idx] + rotated_offset

    return world_positions
```

### Error Metrics

```python
def calculate_validation_metrics(derived_pos, json_pos, tolerance=0.001):
    """
    Returns:
        - per_bone_errors: Individual bone position errors
        - mean_error: Average position error
        - max_error: Maximum position error
        - within_tolerance: Percentage within tolerance
        - failed_bones: List of bones exceeding tolerance
    """
```

## Expected Challenges

### 1. Bone Name Mapping

- JSON uses Japanese names: "センター", "グルーブ", "上半身"
- PMX may have different name encoding
- **Solution**: Build robust name mapping with fallback to bone indices

### 2. Coordinate System Differences

- PMX: Left-handed Y-up coordinate system
- JSON: May use different coordinate conventions
- **Solution**: Apply consistent coordinate transformations

### 3. Quaternion Format

- JSON format: `{x, y, z, w}`
- PyTorch format: `[x, y, z, w]` tensor
- **Solution**: Ensure proper quaternion normalization and format conversion

### 4. Bone Hierarchy Traversal

- Must process bones in proper parent→child order
- Root bones before child bones
- **Solution**: Topological sort of bone hierarchy

## File Structure

```
validate_skeleton_quaternions.py       # Main script
├── PMXLoader class                    # PMX file handling
├── QuaternionParser class             # JSON data parsing
├── ForwardKinematics class            # PyTorch FK engine
├── ValidationEngine class             # Position validation
└── main() function                    # Orchestration
```

## Dependencies

```python
import torch                           # PyTorch for FK computation
import numpy as np                     # Numerical operations
import json                            # JSON parsing
import sys                             # Path handling
import os                              # File operations
from pathlib import Path               # Path utilities

# MmdUtility imports
sys.path.append('MmdUtility')
from pymeshio.pmx import reader        # PMX file reading
```

## Success Criteria

1. Successfully load PMX model with complete bone hierarchy
2. Parse JSON quaternion data for frame 0 (51 core bones)
3. Implement working forward kinematics in PyTorch
4. Achieve position validation within 0.001 unit tolerance
5. Generate comprehensive validation report

## Output Format

```
=== Skeleton Quaternion Validation Report ===
PMX File: MmdUtility/test/pdtt.pmx
JSON File: MmdUtility/test/dan_alivef_01.imo_bone_positions_quaternions.json
Frame: 0 (Rest Pose)
Tolerance: 0.001 units

=== Summary Statistics ===
Total Bones Validated: 51
Mean Position Error: 0.000234 units
Max Position Error: 0.000891 units
Bones Within Tolerance: 51/51 (100.0%)

=== Validation Result ===
✓ PASSED - All bones within tolerance threshold

=== Per-Bone Details ===
センター: Error=0.000012, Status=PASS
グルーブ: Error=0.000034, Status=PASS
上半身: Error=0.000056, Status=PASS
[... additional bones ...]
```

This design provides a complete blueprint for implementing the validation script with proper error handling, coordinate system management, and comprehensive validation reporting.
