NVCC = nvcc
NVCC_FLAGS = -O2
TARGET = imgpipeline
SRC = src/gpu_image_processing.cu
OUTPUT_DIR = sample_output

.PHONY: all run run_large run_verbose clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Default run: 200 images at 256x256, captures log
run: $(TARGET) $(OUTPUT_DIR)
	./$(TARGET) --num_images 200 --width 256 --height 256 --threshold 50 \
		2>&1 | tee $(OUTPUT_DIR)/execution_log.txt

# Large images for higher speedup demonstration
run_large: $(TARGET) $(OUTPUT_DIR)
	./$(TARGET) --num_images 20 --width 1024 --height 1024 --threshold 60 \
		2>&1 | tee $(OUTPUT_DIR)/execution_log_large.txt

# Verbose per-image output
run_verbose: $(TARGET) $(OUTPUT_DIR)
	./$(TARGET) --num_images 200 --width 256 --height 256 --verbose \
		2>&1 | tee $(OUTPUT_DIR)/execution_log_verbose.txt

# Run all benchmark configurations
run_all: $(TARGET) $(OUTPUT_DIR)
	@echo "=== Standard benchmark (200 x 256x256) ==="
	./$(TARGET) --num_images 200 --width 256 --height 256 \
		2>&1 | tee $(OUTPUT_DIR)/execution_log.txt
	@echo ""
	@echo "=== Large-image benchmark (20 x 1024x1024) ==="
	./$(TARGET) --num_images 20 --width 1024 --height 1024 \
		2>&1 | tee $(OUTPUT_DIR)/execution_log_large.txt

clean:
	rm -f $(TARGET)
	rm -f $(OUTPUT_DIR)/*.pgm $(OUTPUT_DIR)/*.ppm
	rm -f $(OUTPUT_DIR)/*.csv $(OUTPUT_DIR)/*.txt

help:
	@echo "Targets:"
	@echo "  all         - Build the CUDA image pipeline"
	@echo "  run         - Process 200 images (256x256) + save log"
	@echo "  run_large   - Process 20 large images (1024x1024) + save log"
	@echo "  run_verbose - Process 200 images with per-image output"
	@echo "  run_all     - Run standard + large benchmarks"
	@echo "  clean       - Remove binary and output files"
	@echo "  help        - Show this message"
