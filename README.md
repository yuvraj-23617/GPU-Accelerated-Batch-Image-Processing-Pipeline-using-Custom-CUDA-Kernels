# CUDA Image Processing Pipeline

**CUDA at Scale for the Enterprise — Capstone Project**

A high-performance, GPU-accelerated batch image processing pipeline that processes **200 images** through four custom CUDA kernels, demonstrating **~28x speedup** over CPU-only processing.

## Project Description

### Motivation

Image processing is one of the most naturally parallelizable workloads in computing. Each pixel can be processed independently, making it an ideal candidate for GPU acceleration. This project demonstrates how a multi-stage image processing pipeline — commonly used in computer vision, autonomous driving, and medical imaging — achieves dramatic speedups when offloaded to NVIDIA GPUs using custom CUDA kernels.

### What It Does

The pipeline processes each image through four sequential stages:

| Stage | Kernel | Operation | Reads/Thread |
|-------|--------|-----------|-------------|
| 1 | `GrayscaleKernel` | RGB → Grayscale via luminance formula `Y = 0.299R + 0.587G + 0.114B` | 3 bytes |
| 2 | `GaussianBlurKernel` | 5×5 Gaussian convolution (σ ≈ 1.0) for noise reduction | 25 pixels |
| 3 | `SobelEdgeKernel` | 3×3 Sobel gradient magnitude for edge detection | 9 pixels |
| 4 | `ThresholdKernel` | Binary thresholding to produce clean edge map | 1 pixel |

### GPU Parallelism Details

Each CUDA kernel maps **one thread per pixel**. For a 256×256 image:
- **Grid**: 16×16 blocks
- **Block**: 16×16 threads (256 threads per block)
- **Total**: 65,536 threads per kernel launch
- **Pipeline**: 4 kernels × 200 images = **800 kernel launches** per run
- **Data**: 200 images × 256×256 × 3 channels = **37.5 MB** processed

## Code Structure

```
cuda-image-pipeline/
├── src/
│   └── gpu_image_processing.cu    # CUDA kernels + CPU reference + main
├── sample_output/                 # Proof-of-execution artifacts
│   ├── input_rgb.ppm              # Original synthetic test image
│   ├── stage1_grayscale.pgm       # After grayscale conversion
│   ├── stage2_gaussian_blur.pgm   # After Gaussian smoothing
│   ├── stage3_sobel_edges.pgm     # After Sobel edge detection
│   ├── stage4_threshold.pgm       # Final binary edge map
│   ├── benchmark_results.csv      # Timing data (GPU vs CPU)
│   └── execution_log.txt          # Full console output
├── Makefile                       # Build automation
├── run.sh                         # One-command build + run + log capture
├── INSTALL                        # Installation instructions (multi-OS)
├── LICENSE                        # GPL-3.0
└── README.md                      # This file
```

## Building and Running

### Prerequisites
- NVIDIA GPU with CUDA support (Compute Capability ≥ 3.5)
- CUDA Toolkit ≥ 11.0 (`nvcc` compiler)
- See `INSTALL` file for detailed setup instructions

### Quick Start

```bash
# Option 1: One-command build + run + generate all artifacts
chmod +x run.sh
./run.sh

# Option 2: Use Makefile
make
make run

# Option 3: Manual compilation
nvcc -O2 -o imgpipeline src/gpu_image_processing.cu
./imgpipeline --num_images 200 --width 256 --height 256
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--num_images N` | 200 | Number of images to batch-process |
| `--width W` | 256 | Image width in pixels |
| `--height H` | 256 | Image height in pixels |
| `--threshold T` | 50 | Binary threshold value (0–255) |
| `--output DIR` | sample_output | Directory for output artifacts |
| `--verbose` | off | Print per-image timing |
| `--help` | — | Show usage information |

### Usage Examples

```bash
# Default: 200 images at 256×256
./imgpipeline --num_images 200 --width 256 --height 256

# 20 large images for higher speedup
./imgpipeline --num_images 20 --width 1024 --height 1024

# Custom threshold with verbose per-image output
./imgpipeline --num_images 100 --threshold 80 --verbose

# Run all benchmarks (standard + large)
make run_all
```

### Google Colab (No Local GPU Required)

```python
# Cell 1: Upload and compile
%%writefile gpu_image_processing.cu
# (paste contents of src/gpu_image_processing.cu)

# Cell 2: Build and run
!mkdir -p sample_output
!nvcc -O2 -o imgpipeline gpu_image_processing.cu
!./imgpipeline --num_images 200 --width 256 --height 256

# Cell 3: Visualize results
from PIL import Image
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
files = ['sample_output/input_rgb.ppm',
         'sample_output/stage1_grayscale.pgm',
         'sample_output/stage2_gaussian_blur.pgm',
         'sample_output/stage3_sobel_edges.pgm',
         'sample_output/stage4_threshold.pgm']
titles = ['Input RGB', 'Grayscale', 'Gaussian Blur', 'Sobel Edges', 'Threshold']
for ax, f, t in zip(axes, files, titles):
    ax.imshow(Image.open(f), cmap='gray')
    ax.set_title(t, fontsize=12); ax.axis('off')
plt.suptitle('CUDA Image Processing Pipeline — Stage Outputs', fontsize=14)
plt.tight_layout(); plt.savefig('sample_output/pipeline_visualization.png', dpi=150)
plt.show()
```

## Proof of Execution Artifacts

All artifacts are generated in `sample_output/` by running `./run.sh` or `make run`.

### Output Images

| File | Format | Description |
|------|--------|-------------|
| `input_rgb.ppm` | PPM (color) | Synthetic test image with geometric shapes |
| `stage1_grayscale.pgm` | PGM (gray) | After luminance conversion |
| `stage2_gaussian_blur.pgm` | PGM (gray) | After 5×5 Gaussian smoothing |
| `stage3_sobel_edges.pgm` | PGM (gray) | After Sobel gradient computation |
| `stage4_threshold.pgm` | PGM (gray) | Final binary edge map |

### Benchmark CSV (`benchmark_results.csv`)

```csv
metric,value
gpu_name,Tesla T4
num_images,200
image_width,256
image_height,256
pixels_per_image,65536
total_kernels,800
gpu_total_ms,312.45
cpu_total_ms,8934.21
speedup_x,28.6
```

### Execution Log (`execution_log.txt`)

Full console output is captured automatically by `run.sh` and `make run`.

## Sample Console Output

```
==========================================================
  CUDA Image Processing Pipeline
==========================================================
  GPU            : Tesla T4
  SMs            : 40
  Images         : 200  (256 x 256 RGB)
  Threshold      : 50
  Total data     : 37.5 MB
  Pipeline       : Grayscale -> GaussBlur -> Sobel -> Threshold
==========================================================

[GPU] Processing 200 images through 4-kernel pipeline...
  [GPU] Image   1 / 200 done
  [GPU] Image  50 / 200 done
  [GPU] Image 100 / 200 done
  [GPU] Image 150 / 200 done
  [GPU] Image 200 / 200 done
  [GPU] Complete: 312.45 ms total (1.56 ms/image)

[CPU] Processing 200 images (reference)...
  [CPU] Complete: 8934.21 ms total (44.67 ms/image)

==========================================================
  Results Summary
==========================================================
  Images processed : 200  (256 x 256)
  Kernels per image: 4  (Grayscale, Blur, Sobel, Threshold)
  Total kernels    : 800
  Threads per image: 65536  (16x16 grid of 16x16 blocks)
----------------------------------------------------------
  GPU total time   :     312.45 ms
  CPU total time   :    8934.21 ms
  GPU per image    :       1.56 ms
  CPU per image    :      44.67 ms
  Speedup          : 28.6x
==========================================================
```

## Algorithm Details

### Kernel 1: RGB to Grayscale
Converts each RGB pixel to a single intensity value using the ITU-R BT.601 luminance formula. Each CUDA thread reads 3 bytes (R, G, B) and writes 1 byte. This is a **memory-bound** kernel with minimal arithmetic.

### Kernel 2: Gaussian Blur (5×5)
Applies a 5×5 convolution with a discrete Gaussian kernel (σ ≈ 1.0, sum = 273). Each thread reads a 5×5 neighborhood using **clamped boundary conditions** (edge pixels are repeated). This kernel has a high arithmetic intensity due to the 25 multiply-accumulate operations per pixel.

### Kernel 3: Sobel Edge Detection
Computes gradient magnitude using two 3×3 Sobel operators (horizontal Gx and vertical Gy). The magnitude `√(Gx² + Gy²)` is clamped to [0, 255]. Border pixels are set to 0. This kernel effectively highlights edges and contours in the image.

### Kernel 4: Binary Threshold
Simple per-pixel comparison: pixels ≥ threshold → 255 (white), else → 0 (black). This produces a clean binary edge map suitable for downstream processing (contour detection, object segmentation, etc.).

### Data Flow

```
RGB Image (W×H×3 bytes)
    │
    ▼ [Kernel 1: GrayscaleKernel]
Grayscale (W×H bytes)
    │
    ▼ [Kernel 2: GaussianBlurKernel]
Blurred (W×H bytes)
    │
    ▼ [Kernel 3: SobelEdgeKernel]
Edge Map (W×H bytes)
    │
    ▼ [Kernel 4: ThresholdKernel]
Binary Edges (W×H bytes)
```

## Key Technical Decisions

1. **Custom CUDA Kernels vs NPP**: Chose to implement all kernels from scratch rather than using NVIDIA Performance Primitives (NPP) to demonstrate understanding of CUDA programming model.

2. **Synthetic Test Images**: Generated 200 unique test images with randomized geometric shapes on gradient backgrounds. This provides varied input data for meaningful edge detection while avoiding external data dependencies.

3. **CPU Reference Implementation**: Included sequential CPU implementations of all four algorithms to provide accurate speedup measurements using `clock()` wall-time.

4. **CUDA Error Checking**: All CUDA API calls are wrapped in `CUDA_CHECK()` macro that prints file/line info and aborts on failure.

5. **CSV Export**: Benchmark results are automatically exported to CSV format for programmatic analysis and reproducibility.

## Lessons Learned

- **Memory Transfer Overhead**: For small images (256×256), the `cudaMemcpy` H2D/D2H transfers can dominate kernel execution time. Using **pinned memory** (`cudaMallocHost`) or **CUDA streams** for overlapping transfers with compute would improve throughput.

- **Kernel Launch Overhead**: With 800 kernel launches, the per-launch overhead (~5-10μs each) accumulates. Fusing kernels or using **CUDA graphs** would amortize this cost.

- **Shared Memory Opportunity**: The Gaussian blur kernel reads overlapping 5×5 neighborhoods — using shared memory tiling would eliminate redundant global memory reads and improve performance by ~2-3x for this kernel.

- **Speedup Scales with Problem Size**: Larger images (1024×1024) show higher speedups because the GPU's massive parallelism is better utilized. Small images under-utilize the GPU's thousands of cores.

- **Occupancy**: The 16×16 block size (256 threads) achieves good occupancy on most architectures. Using `cudaOccupancyMaxPotentialBlockSize()` could auto-tune this.

## Future Work

- Implement **CUDA streams** for pipelined transfers + compute
- Add **shared memory tiling** for convolution kernels
- Support loading real images (JPEG/PNG via stb_image)
- Add more filters (bilateral, median, morphological operations)
- Implement **multi-GPU** processing for very large batches

## Author

Yuvraj Verma — CUDA at Scale for the Enterprise Specialization, Coursera
