# Pytransform3d Concat Investigation

## Problem Statement

The センター bone world position calculation is incorrect. It has an animation position of `(-19.095, 0.000, -17.193)` but the calculated world position is `(0.000, 0.000, 0.000)`.

## Key Evidence

From the debug output for センター bone:

### Individual Matrices

```
Rest transform:        Identity matrix (no translation)
Rotation transform:    Identity matrix (no rotation)
Animation transform:   Translation by (-19.095, 0, -17.193)
```

### Step-by-step Calculation (WORKING)

```python
temp_after_rest = concat(rest_transform, rotation_transform)
temp_after_rotation = concat(temp_after_rest, anim_transform)
```

**Result:** `(-19.095, 0.000, -17.193)` ✅

### Three-matrix Calculation (BROKEN)

```python
bone_transform = concat(rest_transform, rotation_transform, anim_transform)
```

**Result:** `(0.000, 0.000, 0.000)` ❌

## Investigation Hypothesis

Three possibilities:

1. **pytransform3d bug**: The `concat()` function with 3+ arguments is buggy
2. **Usage error**: We're misusing the `concat()` function API
3. **Matrix creation issue**: One of our transformation matrices is malformed

## Investigation Plan

1. **Create isolated test**: Test `concat()` with simple known matrices
2. **Check concat API**: Verify how multiple matrices should be passed
3. **Test matrix validity**: Ensure our transformation matrices are correct
4. **Document findings**: Record the root cause and solution

## Expected Solution

Once we identify the root cause:

- Fix the concatenation logic in both files:
  - `MmdUtility/test/debug_bone_step_by_step_pt3d.py` (line 106)
  - `MmdUtility/bone_animation_pytransform3d.py` (line 268)
- Use sequential two-matrix concatenations instead of three-matrix
- Validate the fix with test cases

## Impact Assessment

This affects **ALL bone calculations**, not just センター:

- センター bone: Most obvious (large animation translation)
- Other bones: Subtle errors that compound through the hierarchy
- Final positions: All foot/ankle positions will be incorrect
