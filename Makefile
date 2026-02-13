# Qwen3-TTS C - Pure C TTS inference engine
#
# Targets:
#   make          Build optimized binary
#   make debug    Build with debug symbols
#   make clean    Remove build artifacts

CC      = cc
CFLAGS  = -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
LDFLAGS =

# Source files
SRCS = c/main.c c/qwen_tts.c c/qwen_tts_kernels.c c/qwen_tts_talker.c \
       c/qwen_tts_codec.c c/qwen_tts_audio.c c/qwen_tts_safetensors.c

# Output binary
BIN = qwen-tts

# ---- Platform detection ----
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
  # macOS: use Accelerate framework for BLAS
  CFLAGS  += -DUSE_BLAS -DACCELERATE_NEW_LAPACK
  LDFLAGS += -framework Accelerate
else ifeq ($(UNAME_S),Linux)
  # Linux: try OpenBLAS
  OPENBLAS_CHECK := $(shell pkg-config --exists openblas 2>/dev/null && echo yes)
  ifeq ($(OPENBLAS_CHECK),yes)
    CFLAGS  += -DUSE_BLAS -DUSE_OPENBLAS $(shell pkg-config --cflags openblas)
    LDFLAGS += $(shell pkg-config --libs openblas)
  else
    # Try finding openblas headers/libs in common paths
    ifneq ($(wildcard /usr/include/openblas/cblas.h),)
      CFLAGS  += -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
      LDFLAGS += -lopenblas
    else ifneq ($(wildcard /usr/local/include/openblas/cblas.h),)
      CFLAGS  += -DUSE_BLAS -DUSE_OPENBLAS -I/usr/local/include/openblas
      LDFLAGS += -L/usr/local/lib -lopenblas
    else
      $(warning OpenBLAS not found — building without BLAS. Performance will be poor.)
    endif
  endif
endif

LDFLAGS += -lm

# ---- Build modes ----

# Default: optimized build
.PHONY: all
all: CFLAGS += -O3 -march=native -DNDEBUG
all: $(BIN)

# Debug build
.PHONY: debug
debug: CFLAGS += -O0 -g -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: $(BIN)

# Sanitizer build (for CI/testing)
.PHONY: sanitize
sanitize: CFLAGS += -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
sanitize: LDFLAGS += -fsanitize=address,undefined
sanitize: $(BIN)

$(BIN): $(SRCS) c/qwen_tts.h c/qwen_tts_kernels.h c/qwen_tts_safetensors.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(BIN)

.PHONY: help
help:
	@echo "Qwen3-TTS C Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make          Build optimized binary (default)"
	@echo "  make debug    Build with debug symbols + AddressSanitizer"
	@echo "  make clean    Remove build artifacts"
	@echo "  make help     Show this help"
	@echo ""
	@echo "Detected platform: $(UNAME_S)"
