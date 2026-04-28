#!/bin/bash
# run.sh - Build and run the CUDA Image Processing Pipeline
# Generates all proof-of-execution artifacts (images, CSV, log)
#
# Usage:
#   ./run.sh                     # Default: 200 images, 256x256
#   ./run.sh 50 512 512          # Custom: 50 images, 512x512

set -e

NUM_IMAGES=${1:-200}
WIDTH=${2:-256}
HEIGHT=${3:-256}
LOG_FILE="sample_output/execution_log.txt"

echo "============================================"
echo "  Building CUDA Image Processing Pipeline"
echo "============================================"

mkdir -p sample_output

nvcc -O2 -o imgpipeline src/gpu_image_processing.cu
echo "Build successful."
echo ""

echo "============================================"
echo "  Running: $NUM_IMAGES images (${WIDTH}x${HEIGHT})"
echo "============================================"

# Run and tee output to both console and log file
./imgpipeline --num_images $NUM_IMAGES --width $WIDTH --height $HEIGHT --threshold 50 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================"
echo "  Artifacts Generated"
echo "============================================"
echo "  Images:    sample_output/*.pgm, *.ppm"
echo "  Benchmark: sample_output/benchmark_results.csv"
echo "  Log:       $LOG_FILE"
echo ""
echo "Commit sample_output/ to your repository as proof of execution."
