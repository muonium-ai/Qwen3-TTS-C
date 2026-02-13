# Qwen3-TTS C - Pure C TTS inference engine
#
# Targets:
#   make          Build optimized binary
#   make debug    Build with debug symbols
#   make benchmark Run Python vs C benchmark
#   make clean    Remove build artifacts

CC      = cc
CFLAGS  = -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
LDFLAGS =

# Source files
SRCS = c/main.c c/qwen_tts.c c/qwen_tts_kernels.c c/qwen_tts_talker.c \
       c/qwen_tts_codec.c c/qwen_tts_audio.c c/qwen_tts_safetensors.c

# Output binary
BIN = qwen-tts

# Benchmark configuration (override in CI/environment)
PYTHON           ?= python3
BENCH_SCRIPT     ?= scripts/benchmark_py_vs_c.py
BENCH_OUTPUT_DIR ?= benchmark_output
BENCH_TEXT       ?= Hello from Qwen3-TTS benchmark.
BENCH_LANGUAGE   ?= English
BENCH_SPEAKER    ?=
BENCH_RUNS       ?= 3
BENCH_WARMUP     ?= 1
BENCH_DEVICE     ?= cpu
BENCH_DTYPE      ?= float32
BENCH_ATTN_IMPL  ?=
BENCH_MAX_TOKENS ?= 512
BENCH_TOP_K      ?= 50
BENCH_TOP_P      ?= 1.0
BENCH_TEMP       ?= 0.9
BENCH_REP_PEN    ?= 1.05
BENCH_SUB_TEMP   ?= 0.9
BENCH_SUB_TOP_K  ?= 50
MODEL_DIR        ?=
PYTHON_MODEL     ?= $(MODEL_DIR)
C_MODEL_DIR      ?= $(MODEL_DIR)
TOKENIZER_PATH   ?= $(PYTHON_MODEL)

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

.PHONY: benchmark
benchmark: all
	@test -n "$(PYTHON_MODEL)" || (echo "Error: set PYTHON_MODEL or MODEL_DIR"; exit 1)
	@test -n "$(C_MODEL_DIR)" || (echo "Error: set C_MODEL_DIR or MODEL_DIR"; exit 1)
	$(PYTHON) $(BENCH_SCRIPT) \
		--python-model "$(PYTHON_MODEL)" \
		--c-model-dir "$(C_MODEL_DIR)" \
		--tokenizer "$(TOKENIZER_PATH)" \
		--c-bin "./$(BIN)" \
		--text "$(BENCH_TEXT)" \
		--language "$(BENCH_LANGUAGE)" \
		--speaker "$(BENCH_SPEAKER)" \
		--warmup-runs "$(BENCH_WARMUP)" \
		--runs "$(BENCH_RUNS)" \
		--python-device "$(BENCH_DEVICE)" \
		--python-dtype "$(BENCH_DTYPE)" \
		--attn-implementation "$(BENCH_ATTN_IMPL)" \
		--max-new-tokens "$(BENCH_MAX_TOKENS)" \
		--top-k "$(BENCH_TOP_K)" \
		--top-p "$(BENCH_TOP_P)" \
		--temperature "$(BENCH_TEMP)" \
		--repetition-penalty "$(BENCH_REP_PEN)" \
		--subtalker-temperature "$(BENCH_SUB_TEMP)" \
		--subtalker-top-k "$(BENCH_SUB_TOP_K)" \
		--output-dir "$(BENCH_OUTPUT_DIR)"

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
	@echo "  make benchmark Run Python vs C benchmark (set MODEL_DIR or PYTHON_MODEL/C_MODEL_DIR)"
	@echo "  make clean    Remove build artifacts"
	@echo "  make help     Show this help"
	@echo ""
	@echo "Detected platform: $(UNAME_S)"
