# Qwen3-TTS C - local build (from c/ directory)
#
# Usage:
#   cd c
#   make          # optimized build
#   make debug    # debug + sanitizers
#   make sanitize # CI sanitizer build
#   make clean

CC      ?= cc
CFLAGS  ?= -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
LDFLAGS ?=

SRCS = main.c qwen_tts.c qwen_tts_kernels.c qwen_tts_talker.c \
       qwen_tts_codec.c qwen_tts_audio.c qwen_tts_safetensors.c

BIN = qwen-tts

UNAME_S := $(shell uname -s)
OPENMP ?= 1

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
      $(warning OpenBLAS not found -- building without BLAS. Performance will be poor.)
    endif
  endif
endif

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

.PHONY: all
all: CFLAGS += -O3 -march=native -DNDEBUG
all: $(BIN)

.PHONY: debug
debug: CFLAGS += -O0 -g -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: $(BIN)

.PHONY: sanitize
sanitize: CFLAGS += -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
sanitize: LDFLAGS += -fsanitize=address,undefined
sanitize: $(BIN)

$(BIN): $(SRCS) qwen_tts.h qwen_tts_kernels.h qwen_tts_safetensors.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(BIN)

.PHONY: help
help:
	@echo "Qwen3-TTS C Local Build System (c/)"
	@echo ""
	@echo "Targets:"
	@echo "  make          Build optimized binary (default)"
	@echo "  make debug    Build with debug symbols + AddressSanitizer"
	@echo "  make sanitize Build sanitizer-focused binary"
	@echo "  make clean    Remove build artifacts"
	@echo "  make help     Show this help"
	@echo ""
	@echo "Detected platform: $(UNAME_S)"
