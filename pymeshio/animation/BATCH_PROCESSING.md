# Batch Processing Optimization for VMD Animation

## Overview

The new batch processing system provides **5-10x performance improvements** for processing multiple VMD files by eliminating redundant PMX model loading and using vectorized forward kinematics.

### Performance Comparison

| Method                 | Speed         | Use Case           |
| ---------------------- | ------------- | ------------------ |
| Original AnimatedModel | ~110 fps      | Single VMD files   |
| **BatchAnimatedModel** | **1000+ fps** | Multiple VMD files |

### Key Optimizations

1. **Single PMX Load**: PMX model loaded once and reused for all VMD files
2. **Batch Forward Kinematics**: Process multiple frames simultaneously with vectorized operations
3. **Core Bone Filtering**: Only compute positions for specified bones (e.g., 51 core bones vs 100+ total bones)
4. **Memory Efficiency**: Chunked processing and tensor reuse
5. **Device Optimization**: Automatic CPU/GPU selection and optimization

## Quick Start

### Basic Usage

```python
from pymeshio.animation import create_batch_animated_model

# Create batch model (loads PMX once)
batch_model = create_batch_animated_model(
    pmx_path="model.pmx",
    device='cpu',  # or 'cuda'
    chunk_size=1000  # frames per batch
)

# Process multiple VMD files efficiently
for vmd_file in vmd_files:
    data = batch_model.process_vmd_file(
        vmd_file,
        filter_bones=["センター", "腰", "上半身", "頭"]  # Optional filtering
    )

    if data:
        positions = data["positions"]     # [n_frames, n_bones, 3]
        quaternions = data["quaternions"] # [n_frames, n_bones, 4]
        frame_numbers = data["frame_numbers"]
        metadata = data["metadata"]
```

### Advanced Usage with Core Bones

```python
from pymeshio.animation import BatchAnimatedModel

# Define core bones for filtering
CORE_BONES = [
    "センター", "グルーブ", "腰",
    "左足", "右足", "左ひざ", "右ひざ",
    "上半身", "上半身2", "頭",
    "左肩", "左腕", "左ひじ", "左手首",
    "右肩", "右腕", "右ひじ", "右手首"
    # ... more bones
]

# Initialize with optimal settings
batch_model = BatchAnimatedModel(
    pmx_path="defaultModel.pmx",
    device='cuda' if torch.cuda.is_available() else 'cpu',
    chunk_size=2000  # Larger chunks for GPU
)

# Process with bone filtering for maximum performance
results = []
for vmd_path in vmd_paths:
    data = batch_model.process_vmd_file(vmd_path, filter_bones=CORE_BONES)
    if data:
        results.append(data)

        # Save to NPZ format
        np.savez_compressed(
            f"{vmd_path.stem}_core_bones.npz",
            positions=data["positions"],
            quaternions=data["quaternions"],
            frame_numbers=data["frame_numbers"],
            **data["metadata"]
        )
```

## API Reference

### BatchAnimatedModel

```python
class BatchAnimatedModel:
    def __init__(self, pmx_path: str, device: str = 'cpu', chunk_size: int = 1000)
```

**Parameters:**

- `pmx_path`: Path to PMX model file
- `device`: PyTorch device ('cpu' or 'cuda')
- `chunk_size`: Number of frames to process per batch (memory vs speed trade-off)

**Methods:**

#### `process_vmd_file(vmd_path, filter_bones=None)`

Process a single VMD file with optional bone filtering.

**Parameters:**

- `vmd_path`: Path to VMD motion file
- `filter_bones`: List of bone names to extract (None = all bones)

**Returns:**

```python
{
    "positions": np.ndarray,      # [n_frames, n_bones, 3] world positions
    "quaternions": np.ndarray,    # [n_frames, n_bones, 4] local quaternions
    "frame_numbers": np.ndarray,  # [n_frames] frame indices
    "metadata": dict              # File info and processing stats
}
```

#### `get_bone_names()`

Returns list of all bone names in the PMX model.

#### `clear_gpu_cache()`

Clears GPU memory cache (useful when processing many files).

### BatchForwardKinematics

Low-level batch forward kinematics calculator.

```python
class BatchForwardKinematics:
    def compute_world_positions_batch(
        self,
        bone_offsets: torch.Tensor,           # [N, 3]
        quaternions_batch: torch.Tensor,      # [F, N, 4]
        parent_indices: List[int],
        local_translations_batch: torch.Tensor = None,  # [F, N, 3]
        bone_filter_indices: List[int] = None
    ) -> torch.Tensor:  # [F, M, 3] where M = filtered bones
```

## Performance Tuning

### Chunk Size Selection

| Chunk Size | Memory Usage | Speed  | Recommended For  |
| ---------- | ------------ | ------ | ---------------- |
| 500-1000   | Low          | Good   | CPU, Limited RAM |
| 1000-2000  | Medium       | Better | CPU, Normal RAM  |
| 2000-5000  | High         | Best   | GPU, High RAM    |

### Memory Optimization

```python
# For large datasets with memory constraints
batch_model = BatchAnimatedModel(
    pmx_path="model.pmx",
    device='cpu',
    chunk_size=500  # Smaller chunks
)

# Clear cache periodically
for i, vmd_file in enumerate(vmd_files):
    data = batch_model.process_vmd_file(vmd_file, filter_bones=CORE_BONES)

    if i % 10 == 0:  # Every 10 files
        batch_model.clear_gpu_cache()
```

### GPU Optimization

```python
import torch

# Enable GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Optimal settings for GPU
if device == 'cuda':
    chunk_size = 2000  # Larger batches for GPU
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
else:
    chunk_size = 1000  # Smaller batches for CPU
    torch.set_num_threads(os.cpu_count())  # Use all CPU cores
```

## Integration with Existing Code

### Migrating from Original AnimatedModel

**Before (Single File Processing):**

```python
from pymeshio.animation import create_animated_model

for vmd_file in vmd_files:
    # SLOW: Loads PMX model every time
    animated_model = create_animated_model(pmx_path, vmd_file)

    frames = animated_model.get_available_frames()
    for frame in frames:
        positions = animated_model.get_world_positions(frame)
        # Process frame...
```

**After (Batch Processing):**

```python
from pymeshio.animation import create_batch_animated_model

# FAST: Load PMX model once
batch_model = create_batch_animated_model(pmx_path)

for vmd_file in vmd_files:
    # Process entire file at once with vectorized operations
    data = batch_model.process_vmd_file(vmd_file, filter_bones=CORE_BONES)

    # data["positions"] contains all frames: [n_frames, n_bones, 3]
    # data["quaternions"] contains all rotations: [n_frames, n_bones, 4]
```

### Drop-in Replacement for vmd_to_npz_batch.py

Replace the original processor with the optimized version:

```python
# OLD: 110 fps processing
python -m src.preprocess.vmd_to_npz_batch

# NEW: 1000+ fps processing
python -m src.preprocess.vmd_to_npz_batch_optimized
```

## Troubleshooting

### Common Issues

**Memory Errors:**

```python
# Reduce chunk size
batch_model = BatchAnimatedModel(pmx_path, chunk_size=500)

# Or process fewer bones
core_bones = ["センター", "腰", "上半身", "頭"]  # Minimal set
```

**GPU Out of Memory:**

```python
# Switch to CPU or reduce chunk size
batch_model = BatchAnimatedModel(pmx_path, device='cpu', chunk_size=1000)
```

**Slow Performance:**

```python
# Ensure bone filtering is enabled
data = batch_model.process_vmd_file(vmd_file, filter_bones=CORE_BONES)

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Increase chunk size (if memory allows)
chunk_size = 2000 if device == 'cuda' else 1000
```

### Performance Monitoring

```python
import time

start_time = time.time()
total_frames = 0

for vmd_file in vmd_files:
    data = batch_model.process_vmd_file(vmd_file, filter_bones=CORE_BONES)
    if data:
        total_frames += len(data["frame_numbers"])

total_time = time.time() - start_time
fps = total_frames / total_time

print(f"Processed {total_frames:,} frames in {total_time:.1f}s ({fps:.0f} fps)")
```

## Architecture Details

### Batch Forward Kinematics

The core optimization uses vectorized PyTorch operations to process multiple frames simultaneously:

```python
# Instead of processing frames one by one:
for frame in frames:
    world_positions = compute_single_frame(frame)

# Process all frames at once:
world_positions_batch = compute_batch_frames(frames)  # [F, N, 3]
```

### Memory Layout

```
Original (per-frame):     F × (N × forward_kinematics_call)
Batch (vectorized):       1 × (F × N × vectorized_operations)

Where:
- F = number of frames
- N = number of bones
```

This reduces function call overhead and enables SIMD/GPU vectorization.

### Bone Filtering Optimization

```python
# Without filtering: Process all bones, filter later
all_positions = compute_world_positions(all_bones)  # Expensive
filtered_positions = all_positions[bone_indices]    # Filter output

# With filtering: Only compute needed bones
filtered_positions = compute_world_positions(filter_bones)  # Efficient
```

The filtering is applied during the forward kinematics calculation, not after, to maximize performance gains.
