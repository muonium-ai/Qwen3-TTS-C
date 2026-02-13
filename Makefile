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
# Prefer python3.11 when available (commonly where torch is installed).
PYTHON           ?= $(shell command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || echo python3)
BENCH_SCRIPT     ?= scripts/benchmark_py_vs_c.py
EOS_PARITY_SCRIPT ?= scripts/validate_eos_parity.py
BENCH_OUTPUT_DIR ?= benchmark_output
BENCH_TEXT       ?= Hello from Qwen3-TTS benchmark. porting done by Muonium AI Studios
EOS_TEXT         ?= With great power comes great responsibility.
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
BENCH_SUB_TOP_P  ?= 1.0
EOS_TEST_SCRIPT  ?= test/test_eos_regression.py
EOS_MAX_TOKENS   ?= 256
# Default local model path (downloaded by this repo workflow)
MODEL_DIR        ?= tmp/model
PYTHON_MODEL     ?= $(MODEL_DIR)
C_MODEL_DIR      ?= $(MODEL_DIR)
TOKENIZER_PATH   ?= $(PYTHON_MODEL)

# ---- Platform detection ----
UNAME_S := $(shell uname -s)
OPENMP ?= 1

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
	@test -d "$(PYTHON_MODEL)" || (echo "Error: model directory not found: $(PYTHON_MODEL)"; exit 1)
	@test -d "$(C_MODEL_DIR)" || (echo "Error: model directory not found: $(C_MODEL_DIR)"; exit 1)
	@$(PYTHON) -c "import torch, transformers" >/dev/null 2>&1 || \
		(echo "Error: $(PYTHON) is missing benchmark dependencies (torch/transformers)."; \
		 echo "Run: make setup-benchmark PYTHON=$(PYTHON)"; exit 1)
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
		--subtalker-top-p "$(BENCH_SUB_TOP_P)" \
		--output-dir "$(BENCH_OUTPUT_DIR)"

.PHONY: setup-benchmark
setup-benchmark:
	$(PYTHON) -m pip install -e .

.PHONY: validate-eos
validate-eos: all
	@test -n "$(PYTHON_MODEL)" || (echo "Error: set PYTHON_MODEL or MODEL_DIR"; exit 1)
	@test -n "$(C_MODEL_DIR)" || (echo "Error: set C_MODEL_DIR or MODEL_DIR"; exit 1)
	@test -d "$(PYTHON_MODEL)" || (echo "Error: model directory not found: $(PYTHON_MODEL)"; exit 1)
	@test -d "$(C_MODEL_DIR)" || (echo "Error: model directory not found: $(C_MODEL_DIR)"; exit 1)
	@$(PYTHON) -c "import torch, transformers" >/dev/null 2>&1 || \
		(echo "Error: $(PYTHON) is missing validation dependencies (torch/transformers)."; \
		 echo "Run: make setup-benchmark PYTHON=$(PYTHON)"; exit 1)
	$(PYTHON) $(EOS_PARITY_SCRIPT) \
		--python-model "$(PYTHON_MODEL)" \
		--c-model-dir "$(C_MODEL_DIR)" \
		--tokenizer "$(TOKENIZER_PATH)" \
		--c-bin "./$(BIN)" \
		--text "$(EOS_TEXT)" \
		--language "$(BENCH_LANGUAGE)" \
		--speaker "$(BENCH_SPEAKER)" \
		--python-device "$(BENCH_DEVICE)" \
		--python-dtype "$(BENCH_DTYPE)" \
		--max-new-tokens "$(EOS_MAX_TOKENS)" \
		--top-k "1" \
		--top-p "1.0" \
		--temperature "1.0" \
		--repetition-penalty "1.0" \
		--subtalker-top-k "1" \
		--subtalker-top-p "1.0" \
		--subtalker-temperature "1.0"

.PHONY: test-eos-regression
test-eos-regression: all
	@test -n "$(C_MODEL_DIR)" || (echo "Error: set C_MODEL_DIR or MODEL_DIR"; exit 1)
	@test -d "$(C_MODEL_DIR)" || (echo "Error: model directory not found: $(C_MODEL_DIR)"; exit 1)
	$(PYTHON) $(EOS_TEST_SCRIPT) \
		--c-bin "./$(BIN)" \
		--model-dir "$(C_MODEL_DIR)" \
		--language "$(BENCH_LANGUAGE)" \
		--speaker "$(BENCH_SPEAKER)" \
		--max-tokens "$(EOS_MAX_TOKENS)"

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
	@echo "  make setup-benchmark Install Python benchmark dependencies into $(PYTHON)"
	@echo "  make benchmark Run Python vs C benchmark (set MODEL_DIR or PYTHON_MODEL/C_MODEL_DIR)"
	@echo "  make validate-eos Validate Python/C EOS stop parity (deterministic decode)"
	@echo "  make test-eos-regression Assert C stops before max_tokens on standard prompt"
	@echo "  make clean    Remove build artifacts"
	@echo "  make help     Show this help"
	@echo ""
	@echo "Detected platform: $(UNAME_S)"
