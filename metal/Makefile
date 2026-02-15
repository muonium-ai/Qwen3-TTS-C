# Qwen3-TTS Metal - GPU-accelerated build (from metal/ directory)
#
# Usage:
#   cd metal
#   make          # optimized build with Metal GPU
#   make cpu      # optimized build, CPU-only (no Metal)
#   make debug    # debug + sanitizers
#   make clean
#
# Requirements:
#   - macOS with Xcode Command Line Tools
#   - Apple Silicon (M1+) for optimal BF16 support

CC      ?= cc
CFLAGS  ?= -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
LDFLAGS ?=

# C source files (same as c/ directory)
C_SRCS = main.c qwen_tts.c qwen_tts_kernels.c qwen_tts_talker.c \
         qwen_tts_codec.c qwen_tts_audio.c qwen_tts_safetensors.c

# Metal Objective-C source
OBJC_SRCS = qwen_tts_metal.m

# Metal shader
METAL_SHADER = qwen_tts_metal_kernels.metal
METALLIB = qwen_tts_metal_kernels.metallib

BIN = qwen-tts-metal

UNAME_S := $(shell uname -s)
OPENMP ?= 1
FAST_MATH ?= 0
OPT_LEVEL ?= -O3

# ---- Metal configuration (macOS only) ----
ifeq ($(UNAME_S),Darwin)
  USE_METAL ?= 1
else
  USE_METAL = 0
endif

# ---- BLAS configuration ----
ifeq ($(UNAME_S),Darwin)
  CFLAGS  += -DUSE_BLAS -DACCELERATE_NEW_LAPACK
  LDFLAGS += -framework Accelerate
else ifeq ($(UNAME_S),Linux)
  OPENBLAS_CHECK := $(shell pkg-config --exists openblas 2>/dev/null && echo yes)
  ifeq ($(OPENBLAS_CHECK),yes)
    CFLAGS  += -DUSE_BLAS -DUSE_OPENBLAS $(shell pkg-config --cflags openblas)
    LDFLAGS += $(shell pkg-config --libs openblas)
  else
    ifneq ($(wildcard /usr/include/openblas/cblas.h),)
      CFLAGS  += -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
      LDFLAGS += -lopenblas
    else ifneq ($(wildcard /usr/local/include/openblas/cblas.h),)
      CFLAGS  += -DUSE_BLAS -DUSE_OPENBLAS -I/usr/local/include/openblas
      LDFLAGS += -L/usr/local/lib -lopenblas
    else
      $(warning OpenBLAS not found -- building without BLAS.)
    endif
  endif
endif

# ---- OpenMP configuration ----
ifeq ($(OPENMP),1)
  ifeq ($(UNAME_S),Darwin)
    LIBOMP_CFLAGS := $(shell pkg-config --cflags libomp 2>/dev/null)
    LIBOMP_LIBS := $(shell pkg-config --libs libomp 2>/dev/null)
    ifneq ($(strip $(LIBOMP_LIBS)),)
      CFLAGS  += -DUSE_OPENMP -Xpreprocessor -fopenmp $(LIBOMP_CFLAGS)
      LDFLAGS += $(LIBOMP_LIBS)
    else ifneq ($(wildcard /opt/homebrew/opt/libomp/include/omp.h),)
      CFLAGS  += -DUSE_OPENMP -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
      LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
    else ifneq ($(wildcard /usr/local/opt/libomp/include/omp.h),)
      CFLAGS  += -DUSE_OPENMP -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include
      LDFLAGS += -L/usr/local/opt/libomp/lib -lomp
    else
      $(warning libomp not found -- building without OpenMP.)
    endif
  else
    OPENMP_CHECK := $(shell $(CC) -fopenmp -dM -E - </dev/null >/dev/null 2>&1 && echo yes)
    ifeq ($(OPENMP_CHECK),yes)
      CFLAGS  += -DUSE_OPENMP -fopenmp
      LDFLAGS += -fopenmp
    else
      $(warning OpenMP not supported by compiler -- building without OpenMP.)
    endif
  endif
endif

LDFLAGS += -lm

ifeq ($(FAST_MATH),1)
  OPT_LEVEL := -Ofast
  CFLAGS += -ffast-math -fno-math-errno
endif

# ---- Metal flags ----
ifeq ($(USE_METAL),1)
  CFLAGS  += -DUSE_METAL
  LDFLAGS += -framework Metal -framework Foundation -framework MetalPerformanceShaders
  ALL_SRCS = $(C_SRCS) $(OBJC_SRCS)
  OBJCFLAGS = -fobjc-arc
else
  ALL_SRCS = $(C_SRCS)
  OBJCFLAGS =
endif

# ---- Build targets ----

.PHONY: all
all: CFLAGS += $(OPT_LEVEL) -march=native -DNDEBUG
all: $(BIN)

.PHONY: cpu
cpu: USE_METAL = 0
cpu: CFLAGS += $(OPT_LEVEL) -march=native -DNDEBUG
cpu: BIN = qwen-tts-cpu
cpu:
	$(CC) $(CFLAGS) -o qwen-tts-cpu $(C_SRCS) $(LDFLAGS)

.PHONY: debug
debug: CFLAGS += -O0 -g -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: $(BIN)

.PHONY: sanitize
sanitize: CFLAGS += -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
sanitize: LDFLAGS += -fsanitize=address,undefined
sanitize: $(BIN)

# Pre-compile Metal shaders (optional - runtime compilation is the fallback)
.PHONY: metallib
metallib: $(METALLIB)

$(METALLIB): $(METAL_SHADER)
	@if xcrun -sdk macosx metal -c $(METAL_SHADER) -o qwen_tts_metal_kernels.air 2>/dev/null; then \
		xcrun -sdk macosx metallib qwen_tts_metal_kernels.air -o $(METALLIB); \
		rm -f qwen_tts_metal_kernels.air; \
		echo "Metal shaders pre-compiled to $(METALLIB)"; \
	else \
		echo "Note: Metal toolchain not available. Shaders will be compiled at runtime."; \
	fi

# Build binary
ifeq ($(USE_METAL),1)
# With Metal: compile .c files + .m file, link with Metal frameworks
# Shaders are compiled at runtime from .metal source if no .metallib exists
$(BIN): $(C_SRCS) $(OBJC_SRCS) qwen_tts.h qwen_tts_kernels.h qwen_tts_safetensors.h qwen_tts_metal.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -o $@ $(ALL_SRCS) $(LDFLAGS)
	@# Try to pre-compile metallib (optional, falls back to runtime compile)
	@$(MAKE) -s metallib 2>/dev/null || true
else
$(BIN): $(C_SRCS) qwen_tts.h qwen_tts_kernels.h qwen_tts_safetensors.h
	$(CC) $(CFLAGS) -o $@ $(C_SRCS) $(LDFLAGS)
endif

.PHONY: clean
clean:
	rm -f $(BIN) qwen-tts-cpu $(METALLIB) *.air

.PHONY: help
help:
	@echo "Qwen3-TTS Metal Build System (metal/)"
	@echo ""
	@echo "Targets:"
	@echo "  make          Build with Metal GPU acceleration (default)"
	@echo "  make cpu      Build CPU-only (no Metal)"
	@echo "  make debug    Build with debug symbols + AddressSanitizer"
	@echo "  make sanitize Build sanitizer-focused binary"
	@echo "  make clean    Remove build artifacts"
	@echo "  make help     Show this help"
	@echo ""
	@echo "Detected platform: $(UNAME_S)"
	@echo "Metal enabled: $(USE_METAL)"
