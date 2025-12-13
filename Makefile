# Makefile for llvc.cpp
# CMake wrapper for convenient building

BUILD_DIR := build
CMAKE := cmake
MAKE_CMD := make
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Release
CMAKE_FLAGS_DEBUG := -DCMAKE_BUILD_TYPE=Debug

# Number of parallel jobs (use all available cores)
JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.PHONY: all release debug clean rebuild test infer infer-streaming help

# Default target
all: release

# Release build
release:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) $(CMAKE_FLAGS) .. && $(MAKE_CMD) -j$(JOBS)
	@echo "Build complete: $(BUILD_DIR)/llvc_infer"

# Debug build
debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) $(CMAKE_FLAGS_DEBUG) .. && $(MAKE_CMD) -j$(JOBS)
	@echo "Debug build complete: $(BUILD_DIR)/llvc_infer"

# Clean build directory
clean:
	@rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned"

# Clean and rebuild
rebuild: clean release

# Run the inference executable (requires model file)
test: release
	@if [ -f $(BUILD_DIR)/llvc_infer ]; then \
		echo "Running llvc_infer..."; \
		./$(BUILD_DIR)/llvc_infer; \
	else \
		echo "Error: llvc_infer not found. Build first with 'make release'"; \
		exit 1; \
	fi

# Run inference on test_wavs/ with default settings (output to converted_out/)
infer: release
	@if [ -f $(BUILD_DIR)/llvc_infer ]; then \
		./$(BUILD_DIR)/llvc_infer; \
	else \
		echo "Error: llvc_infer not found. Build first with 'make release'"; \
		exit 1; \
	fi

# Run streaming inference on test_wavs/ (output to converted_out/)
infer-streaming: release
	@if [ -f $(BUILD_DIR)/llvc_infer ]; then \
		./$(BUILD_DIR)/llvc_infer -s; \
	else \
		echo "Error: llvc_infer not found. Build first with 'make release'"; \
		exit 1; \
	fi

# Show help
help:
	@echo "llvc.cpp Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make                - Build release version (default)"
	@echo "  make release        - Build release version"
	@echo "  make debug          - Build debug version"
	@echo "  make clean          - Remove build directory"
	@echo "  make rebuild        - Clean and rebuild"
	@echo "  make test           - Run the inference executable"
	@echo "  make infer          - Run inference on test_wavs/ -> converted_out/"
	@echo "  make infer-streaming - Run streaming inference on test_wavs/"
	@echo "  make help           - Show this help message"
