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
EMSDK_DIR ?= vendor/emsdk
EMSDK_VERSION ?= latest
EMSDK_EMCC ?= $(EMSDK_DIR)/upstream/emscripten/emcc
EMCC ?= $(if $(wildcard $(EMSDK_EMCC)),$(EMSDK_EMCC),emcc)
EMSDK_PYTHON ?= $(firstword $(wildcard $(EMSDK_DIR)/python/*_64bit/bin/python3) \
	$(wildcard $(EMSDK_DIR)/python/*_64bit/bin/python3.*))
EMSDK_ENV_VARS = EMSDK=$(abspath $(EMSDK_DIR)) EM_CONFIG=$(abspath $(EMSDK_DIR))/.emscripten \
	EMSDK_PYTHON=$(EMSDK_PYTHON)
WASM_OUT_DIR ?= dist/wasm
WASM_BASENAME ?= qwen-tts
WASM_MODEL_DIR ?= tmp/model
WASM_JS := $(WASM_OUT_DIR)/$(WASM_BASENAME).js
WASM_WASM := $(WASM_OUT_DIR)/$(WASM_BASENAME).wasm
WASM_WASI_BASENAME ?= qwen-tts-wasi
WASM_WASI := $(WASM_OUT_DIR)/$(WASM_WASI_BASENAME).wasm
WASM_BUILD_ID ?= $(shell date +%s)
WASM_WEB_SRC_DIR ?= web/wasm
WASM_WEB_FILES := $(WASM_WEB_SRC_DIR)/index.html $(WASM_WEB_SRC_DIR)/app.js
WASM_MODEL_MAX_MIB ?= 1400
WASM_COMMON_FLAGS = -O3 -DNDEBUG -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME=QwenTTSModule \
	-s ENVIRONMENT=web,worker -s ALLOW_MEMORY_GROWTH=1 -s FORCE_FILESYSTEM=1 \
	-s EXPORTED_FUNCTIONS='["_main","_malloc","_free"]' \
	-s EXPORTED_RUNTIME_METHODS='["FS","callMain","ccall","cwrap"]'
WASM_WASI_FLAGS = -O3 -DNDEBUG -s STANDALONE_WASM=1 -s PURE_WASI=1 -s WASMFS=1 \
	-s INITIAL_MEMORY=268435456

# Benchmark configuration (override in CI/environment)
# Prefer python3.11 when available (commonly where torch is installed).
PYTHON           ?= $(shell command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || echo python3)
BENCH_SCRIPT     ?= scripts/benchmark_py_vs_c.py
BENCH_ALL_SCRIPT ?= scripts/benchmark_all.py
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
BENCH_PERSISTENT ?= 1
BENCH_EQUAL_TOKEN_BUDGET ?= 0
BENCH_GATE_MAX_C_OVER_PY_MS_PER_TOKEN ?= 0
BENCH_GATE_MAX_C_OVER_PY_MS_PER_AUDIO_SEC ?= 0
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

.PHONY: wasm
wasm: wasm-prepare-tokenizer
	@command -v $(EMCC) >/dev/null 2>&1 || \
		(echo "Error: emcc not found. Install and activate Emscripten SDK first."; \
		 echo "This repo vendors emsdk as a submodule under $(EMSDK_DIR)."; \
		 echo "Run: make wasm-setup"; \
		 echo "Then: make wasm"; \
		 exit 1)
	@test -n "$(EMSDK_PYTHON)" || \
		(echo "Error: vendored emsdk Python not found under $(EMSDK_DIR)/python."; \
		 echo "Run: make wasm-setup"; exit 1)
	@mkdir -p "$(WASM_OUT_DIR)"
	$(EMSDK_ENV_VARS) $(EMCC) $(WASM_COMMON_FLAGS) -o "$(WASM_JS)" $(SRCS)
	@for f in $(WASM_WEB_FILES); do cp "$$f" "$(WASM_OUT_DIR)/"; done
	@perl -0pi -e 's/__WASM_BUILD_ID__/$(WASM_BUILD_ID)/g' "$(WASM_OUT_DIR)/index.html"
	@echo "WASM build complete:"
	@echo "  $(WASM_JS)"
	@echo "  $(WASM_WASM)"
	@echo "  $(WASM_OUT_DIR)/index.html"

.PHONY: wasm-wasi
wasm-wasi:
	@command -v $(EMCC) >/dev/null 2>&1 || \
		(echo "Error: emcc not found. Run: make wasm-setup"; exit 1)
	@test -n "$(EMSDK_PYTHON)" || \
		(echo "Error: vendored emsdk Python not found under $(EMSDK_DIR)/python."; \
		 echo "Run: make wasm-setup"; exit 1)
	@mkdir -p "$(WASM_OUT_DIR)"
	$(EMSDK_ENV_VARS) $(EMCC) $(WASM_WASI_FLAGS) -o "$(WASM_WASI)" $(SRCS)
	@echo "WASI wasm build complete: $(WASM_WASI)"

.PHONY: wasm-runtime-smoke
wasm-runtime-smoke: wasm-wasi
	@command -v wasmtime >/dev/null 2>&1 || (echo "Error: wasmtime not found"; exit 1)
	@command -v wasmer >/dev/null 2>&1 || (echo "Error: wasmer not found"; exit 1)
	@out="$$(wasmtime run $(WASM_WASI) 2>&1 || true)"; \
		echo "$$out"; \
		echo "$$out" | rg -q "model directory required"
	@out="$$(wasmer run $(WASM_WASI) 2>&1 || true)"; \
		echo "$$out"; \
		echo "$$out" | rg -q "model directory required"
	@echo "WASM runtime smoke test passed for wasmtime + wasmer"

.PHONY: wasm-check-model
wasm-check-model:
	@test -d "$(WASM_MODEL_DIR)" || \
		(echo "Error: model directory not found: $(WASM_MODEL_DIR)"; exit 1)
	@$(PYTHON) scripts/wasm_check_model.py \
		--model-dir "$(WASM_MODEL_DIR)" \
		--max-mib "$(WASM_MODEL_MAX_MIB)"

.PHONY: wasm-prepare-tokenizer
wasm-prepare-tokenizer:
	@if [ ! -d "$(WASM_MODEL_DIR)" ]; then \
		echo "Skipping tokenizer prep: $(WASM_MODEL_DIR) not found"; \
	elif [ -f "$(WASM_MODEL_DIR)/tokenizer.json" ]; then \
		echo "Tokenizer already prepared: $(WASM_MODEL_DIR)/tokenizer.json"; \
	else \
		echo "Preparing tokenizer.json in $(WASM_MODEL_DIR)"; \
		$(PYTHON) -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$(WASM_MODEL_DIR)', fix_mistral_regex=True); t.save_pretrained('$(WASM_MODEL_DIR)'); print('Generated tokenizer.json')" || { \
			echo "Warning: failed to generate tokenizer.json automatically."; \
			echo "Run manually after installing dependencies:"; \
			echo "  $(PYTHON) -c \"from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$(WASM_MODEL_DIR)', fix_mistral_regex=True); t.save_pretrained('$(WASM_MODEL_DIR)')\""; \
			exit 0; \
		}; \
	fi

.PHONY: wasm-setup
wasm-setup:
	@test -d "$(EMSDK_DIR)" || \
		(echo "Error: $(EMSDK_DIR) missing. Run: git submodule update --init --recursive"; exit 1)
	cd "$(EMSDK_DIR)" && ./emsdk install "$(EMSDK_VERSION)" && ./emsdk activate "$(EMSDK_VERSION)"
	@echo "Emscripten installed via $(EMSDK_DIR)"
	@echo "You can now run:"
	@echo "  make wasm"

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
		--equal-token-budget "$(BENCH_EQUAL_TOKEN_BUDGET)" \
		--gate-max-c-over-python-ms-per-token "$(BENCH_GATE_MAX_C_OVER_PY_MS_PER_TOKEN)" \
		--gate-max-c-over-python-ms-per-audio-sec "$(BENCH_GATE_MAX_C_OVER_PY_MS_PER_AUDIO_SEC)" \
		--output-dir "$(BENCH_OUTPUT_DIR)"

.PHONY: benchmark-all
benchmark-all: all
	@test -n "$(MODEL_DIR)" || (echo "Error: set MODEL_DIR"; exit 1)
	@test -d "$(MODEL_DIR)" || (echo "Error: model directory not found: $(MODEL_DIR)"; exit 1)
	$(MAKE) -C metal all
	QWEN_TTS_ENABLE_METAL=1 \
	QWEN_TTS_METAL_TALKER=1 \
	QWEN_TTS_METAL_SUBTALKER=1 \
	$(PYTHON) $(BENCH_ALL_SCRIPT) \
		--model-dir "$(MODEL_DIR)" \
		--text "$(BENCH_TEXT)" \
		--language "$(BENCH_LANGUAGE)" \
		--speaker "$(if $(BENCH_SPEAKER),$(BENCH_SPEAKER),aiden)" \
		--runs "$(BENCH_RUNS)" \
		--warmup "$(BENCH_WARMUP)" \
		--max-new-tokens "$(BENCH_MAX_TOKENS)" \
		--temperature "$(BENCH_TEMP)" \
		--top-k "$(BENCH_TOP_K)" \
		--top-p "$(BENCH_TOP_P)" \
		--repetition-penalty "$(BENCH_REP_PEN)" \
		--subtalker-temperature "$(BENCH_SUB_TEMP)" \
		--subtalker-top-k "$(BENCH_SUB_TOP_K)" \
		--subtalker-top-p "$(BENCH_SUB_TOP_P)" \
		$(if $(filter 1 true yes,$(BENCH_PERSISTENT)),--persistent,) \
		--output-dir "$(BENCH_OUTPUT_DIR)"

.PHONY: benchmark-gate
benchmark-gate: BENCH_EQUAL_TOKEN_BUDGET = 128
benchmark-gate: BENCH_RUNS = 1
benchmark-gate: BENCH_WARMUP = 0
benchmark-gate: BENCH_TOP_K = 1
benchmark-gate: BENCH_TOP_P = 1.0
benchmark-gate: BENCH_TEMP = 1.0
benchmark-gate: BENCH_REP_PEN = 1.0
benchmark-gate: BENCH_SUB_TOP_K = 1
benchmark-gate: BENCH_SUB_TOP_P = 1.0
benchmark-gate: BENCH_SUB_TEMP = 1.0
benchmark-gate: BENCH_GATE_MAX_C_OVER_PY_MS_PER_TOKEN = 2.0
benchmark-gate: BENCH_GATE_MAX_C_OVER_PY_MS_PER_AUDIO_SEC = 2.0
benchmark-gate: benchmark

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
	rm -rf $(WASM_OUT_DIR)

.PHONY: help
help:
	@echo "Qwen3-TTS C Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make          Build optimized binary (default)"
	@echo "  make debug    Build with debug symbols + AddressSanitizer"
	@echo "  make wasm     Build browser-loadable WASM artifacts (Emscripten required)"
	@echo "  make wasm-prepare-tokenizer Ensure $(WASM_MODEL_DIR)/tokenizer.json for browser tokenizer"
	@echo "  make wasm-setup Install/activate vendored emsdk toolchain ($(EMSDK_VERSION))"
	@echo "  make setup-benchmark Install Python benchmark dependencies into $(PYTHON)"
	@echo "  make benchmark Run Python vs C benchmark (set MODEL_DIR or PYTHON_MODEL/C_MODEL_DIR)"
	@echo "  make benchmark-all Run Python vs C vs Metal benchmark with default settings"
	@echo "  make benchmark-gate Run benchmark with normalized-metric quality gates (CI-friendly)"
	@echo "  make validate-eos Validate Python/C EOS stop parity (deterministic decode)"
	@echo "  make test-eos-regression Assert C stops before max_tokens on standard prompt"
	@echo "  make clean    Remove build artifacts"
	@echo "  make help     Show this help"
	@echo ""
	@echo "Detected platform: $(UNAME_S)"
