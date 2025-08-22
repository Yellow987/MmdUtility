# Pytransform3d Implementation Plan

## Overview

Create an alternative bone animation implementation using `pytransform3d` library to validate the existing numpy-based implementation in `bone_animation.py`.

## Key Pytransform3d Classes to Use

### 1. Transform Class (`pytransform3d.transformations`)

- `Transform` - Main class for representing 4x4 transformation matrices
- `concat()` - Method for combining transformations
- `transform_points()` - Apply transformation to points

### 2. Quaternion Handling (`pytransform3d.rotations`)

- `matrix_from_quaternion()` - Convert quaternion to rotation matrix
- `quaternion_slerp()` - Spherical linear interpolation for quaternions
- `concatenate_quaternions()` - Combine quaternions

### 3. Matrix Operations (`pytransform3d.transformations`)

- `transform_from()` - Create transformation matrix from rotation and translation
- `pq_from_transform()` - Extract position and quaternion from transformation
- `invert_transform()` - Matrix inversion operations

## Implementation Structure

### File: `bone_animation_pytransform3d.py`

```python
from pytransform3d.transformations import Transform, transform_from, concat
from pytransform3d.rotations import matrix_from_quaternion, quaternion_slerp
import numpy as np
from typing import Tuple, Union
from pymeshio import common, pmx, vmd
```

### Core Functions to Implement

#### 1. `get_bone_world_position_pt3d()`

**Purpose**: Alternative to `get_bone_world_position()` using pytransform3d

**Key differences from numpy version**:

- Use `Transform` objects instead of raw numpy matrices
- Use `concat()` instead of `np.dot()` for matrix multiplication
- Use `transform_points()` to get final world position
- Leverage pytransform3d's built-in validation and optimization

**Algorithm**:

```python
def get_bone_world_position_pt3d(pmx_model, vmd_motion, frame_number, bone_name):
    # 1. Find target bone (same as current)
    bone_result = BoneHierarchyWalker.find_bone_by_name(pmx_model.bones, bone_name)

    # 2. Get bone chain (same as current)
    bone_chain = BoneHierarchyWalker.get_bone_chain_to_root(pmx_model.bones, bone_index)

    # 3. Initialize with identity Transform
    world_transform = Transform()  # Identity transform

    # 4. Apply transformations using pytransform3d
    for chain_bone_index in bone_chain:
        # Calculate local position (same logic)
        local_pos = calculate_local_position(bone, parent_bone)

        # Get animation data (reuse existing function)
        anim_pos, anim_quat = get_bone_animation_data(vmd_motion, bone.name, frame_number)

        # Create transformations using pytransform3d
        rest_transform = transform_from(
            R=np.eye(3),  # No rotation for rest position
            p=np.array([local_pos.x, local_pos.y, local_pos.z])
        )

        rotation_matrix = matrix_from_quaternion([anim_quat.w, anim_quat.x, anim_quat.y, anim_quat.z])
        rotation_transform = Transform(matrix=np.eye(4))
        rotation_transform.matrix[:3, :3] = rotation_matrix

        anim_transform = transform_from(
            R=np.eye(3),
            p=np.array([anim_pos.x, anim_pos.y, anim_pos.z])
        )

        # Combine: Rest * Rotation * Animation
        bone_transform = concat(rest_transform, rotation_transform, anim_transform)

        # Accumulate transformation
        world_transform = concat(world_transform, bone_transform)

    # 5. Extract world position
    world_pos = world_transform.transform_points(np.array([[0, 0, 0]]))[0]
    return tuple(world_pos)
```

#### 2. Enhanced Interpolation with pytransform3d

**Purpose**: Improve quaternion interpolation using SLERP

```python
def interpolate_bone_frame_pt3d(frame1, frame2, target_frame):
    # Calculate interpolation parameter
    t = (target_frame - frame1.frame) / (frame2.frame - frame1.frame)

    # Position interpolation (same as before with Bezier)
    pos = interpolate_position_with_bezier(frame1.pos, frame2.pos, t, bezier_data)

    # Quaternion interpolation using pytransform3d SLERP
    q1 = np.array([frame1.q.w, frame1.q.x, frame1.q.y, frame1.q.z])
    q2 = np.array([frame2.q.w, frame2.q.x, frame2.q.y, frame2.q.z])

    # Apply Bezier interpolation to t parameter
    t_rot = VMDInterpolator.bezier_interpolate(t, bezier_data['rot'])

    # Use pytransform3d's SLERP
    interpolated_q = quaternion_slerp(q1, q2, t_rot)

    # Convert back to common.Quaternion
    quat = common.Quaternion(interpolated_q[1], interpolated_q[2], interpolated_q[3], interpolated_q[0])

    return pos, quat
```

### Testing and Validation

#### 1. Comparison Function

```python
def compare_implementations(pmx_model, vmd_motion, frame_number, bone_name):
    """Compare numpy vs pytransform3d implementations"""

    # Get results from both methods
    numpy_pos = get_bone_world_position(pmx_model, vmd_motion, frame_number, bone_name)
    pt3d_pos = get_bone_world_position_pt3d(pmx_model, vmd_motion, frame_number, bone_name)

    # Calculate differences
    diff = np.array(numpy_pos) - np.array(pt3d_pos)
    distance_diff = np.linalg.norm(diff)

    return {
        'numpy_result': numpy_pos,
        'pytransform3d_result': pt3d_pos,
        'difference': diff,
        'distance_difference': distance_diff,
        'tolerance_check': distance_diff < 1e-6
    }
```

#### 2. Enhanced Test Script

Modify `debug_bone_step_by_step.py` to:

- Import both implementations
- Run both methods for same bone/frame
- Display side-by-side comparison
- Highlight any significant differences
- Test multiple bones and frames

```python
def debug_bone_both_methods(pmx_model, vmd_motion, frame, bone_name):
    print(f"\n=== COMPARING BOTH IMPLEMENTATIONS FOR '{bone_name}' AT FRAME {frame} ===")

    # Test both implementations
    try:
        numpy_result = get_bone_world_position(pmx_model, vmd_motion, frame, bone_name)
        print(f"NumPy result:        ({numpy_result[0]:.6f}, {numpy_result[1]:.6f}, {numpy_result[2]:.6f})")
    except Exception as e:
        print(f"NumPy implementation failed: {e}")
        return

    try:
        pt3d_result = get_bone_world_position_pt3d(pmx_model, vmd_motion, frame, bone_name)
        print(f"Pytransform3d result: ({pt3d_result[0]:.6f}, {pt3d_result[1]:.6f}, {pt3d_result[2]:.6f})")
    except Exception as e:
        print(f"Pytransform3d implementation failed: {e}")
        return

    # Compare results
    comparison = compare_implementations(pmx_model, vmd_motion, frame, bone_name)
    print(f"Difference:          ({comparison['difference'][0]:.6f}, {comparison['difference'][1]:.6f}, {comparison['difference'][2]:.6f})")
    print(f"Distance difference: {comparison['distance_difference']:.10f}")

    if comparison['tolerance_check']:
        print("✅ Results match within tolerance!")
    else:
        print("⚠️  Results differ beyond tolerance!")
```

## Expected Benefits of Pytransform3d

1. **Better Numerical Stability**: Built-in handling of edge cases
2. **Optimized Operations**: Library optimizations for transformation operations
3. **Validation**: Built-in checks for valid transformations
4. **Cleaner Code**: Higher-level abstractions
5. **Better Quaternion Handling**: Proper SLERP implementation

## Testing Strategy

1. **Unit Tests**: Test individual transformation operations
2. **Integration Tests**: Full bone hierarchy calculations
3. **Regression Tests**: Compare with known good results
4. **Performance Tests**: Compare speed of both implementations
5. **Edge Case Tests**: Test with extreme values and edge cases

## Implementation Priority

1. **Core Function**: `get_bone_world_position_pt3d()`
2. **Enhanced Interpolation**: Improved quaternion SLERP
3. **Testing Integration**: Modify debug script
4. **Validation Suite**: Comprehensive comparison tests
5. **Documentation**: Usage examples and API docs

## Potential Issues to Address

1. **Quaternion Conventions**: Ensure consistent w,x,y,z vs x,y,z,w ordering
2. **Matrix Multiplication Order**: Verify transformation concatenation order
3. **Coordinate System**: Ensure consistent left/right-handed systems
4. **Performance**: Pytransform3d might be slower due to additional validation
5. **Dependencies**: Additional library dependency vs pure numpy

This plan provides a systematic approach to implementing and validating the pytransform3d alternative while maintaining compatibility with existing code.
