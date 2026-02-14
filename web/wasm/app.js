let modulePromise = null;
let tokenizerPromise = null;
let loadedModelBase = null;
let audioUrl = null;

const el = {
  modelBase: document.getElementById("modelBase"),
  tokenizerBase: document.getElementById("tokenizerBase"),
  speaker: document.getElementById("speaker"),
  language: document.getElementById("language"),
  maxTokens: document.getElementById("maxTokens"),
  text: document.getElementById("text"),
  logs: document.getElementById("logs"),
  audio: document.getElementById("audio"),
  download: document.getElementById("download"),
  loadMetaBtn: document.getElementById("loadMetaBtn"),
  generateBtn: document.getElementById("generateBtn"),
};

function log(msg) {
  el.logs.textContent += `${msg}\n`;
  el.logs.scrollTop = el.logs.scrollHeight;
}

function clearLogs() {
  el.logs.textContent = "";
}

function toBaseUrl(input) {
  return input.replace(/\/+$/, "");
}

function isLocalLikePath(v) {
  return (
    v.startsWith(".") ||
    v.startsWith("/") ||
    v.startsWith("http://") ||
    v.startsWith("https://")
  );
}

async function canFetchConfig(modelBase) {
  try {
    const res = await fetch(`${modelBase}/config.json`, { method: "GET" });
    return res.ok;
  } catch (_) {
    return false;
  }
}

async function canFetchFile(base, fileName) {
  try {
    const res = await fetch(`${base}/${fileName}`, { method: "GET" });
    return res.ok;
  } catch (_) {
    return false;
  }
}

async function resolveModelBase(baseInput) {
  const candidates = [];
  const push = (v) => {
    if (!v) return;
    if (!candidates.includes(v)) candidates.push(v);
  };
  push(toBaseUrl(baseInput || ""));
  push("../../tmp/model");
  push("/tmp/model");
  push("./model");

  for (const c of candidates) {
    if (await canFetchConfig(c)) return c;
  }
  throw new Error(`Failed to fetch model config from candidates: ${candidates.join(", ")}`);
}

function toAbsoluteBaseUrl(base) {
  return new URL(`${toBaseUrl(base)}/`, window.location.href).href.replace(/\/+$/, "");
}

function toTokenizerModelId(base) {
  const b = toBaseUrl(base);
  if (b.startsWith("http://") || b.startsWith("https://")) {
    try {
      const u = new URL(b, window.location.href);
      if (u.origin !== window.location.origin) return b;
      return u.pathname.replace(/^\/+/, "").replace(/\/+$/, "");
    } catch (_) {
      return b;
    }
  }
  if (b.startsWith(".") || b.startsWith("/")) {
    const u = new URL(`${b}/`, window.location.href);
    return u.pathname.replace(/^\/+/, "").replace(/\/+$/, "");
  }
  return b;
}

async function resolveTokenizerBase(baseInput, modelBase) {
  const candidates = [];
  const push = (v) => {
    if (!v) return;
    const t = toBaseUrl(v);
    if (!candidates.includes(t)) candidates.push(t);
  };
  push(baseInput || "");
  push(modelBase);
  push("../../tmp/model");
  push("/tmp/model");
  push("./model");
  push("Qwen/Qwen3-TTS");

  for (const c of candidates) {
    if (isLocalLikePath(c)) {
      if (await canFetchFile(c, "tokenizer.json")) return toTokenizerModelId(c);
      continue;
    }
    return c;
  }
  throw new Error(`Failed to fetch tokenizer.json from candidates: ${candidates.join(", ")}`);
}

function setOptions(selectEl, options, fallback = "") {
  selectEl.innerHTML = "";
  for (const option of options) {
    const node = document.createElement("option");
    node.value = option;
    node.textContent = option;
    selectEl.appendChild(node);
  }
  if (fallback && options.includes(fallback)) selectEl.value = fallback;
}

async function getModule() {
  if (modulePromise) return modulePromise;
  if (typeof QwenTTSModule !== "function") {
    throw new Error("qwen-tts.js not loaded. Run make wasm first.");
  }

  modulePromise = QwenTTSModule({
    noInitialRun: true,
    print: (text) => log(text),
    printErr: (text) => log(text),
  });
  return modulePromise;
}

function ensureDir(FS, path) {
  const parts = path.split("/").filter(Boolean);
  let cur = "";
  for (const part of parts) {
    cur += `/${part}`;
    try { FS.mkdir(cur); } catch (_) {}
  }
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.json();
}

async function fetchBinary(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function discoverModelFiles(modelBase) {
  const files = new Set([
    "config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
  ]);

  try {
    const indexJson = await fetchJson(`${modelBase}/model.safetensors.index.json`);
    files.add("model.safetensors.index.json");
    const weightMap = indexJson.weight_map || {};
    for (const shard of Object.values(weightMap)) files.add(shard);
  } catch (_) {
    files.add("model.safetensors");
  }

  return Array.from(files);
}

async function preloadModelToFS(module, modelBase) {
  if (loadedModelBase === modelBase) return;

  const { FS } = module;
  ensureDir(FS, "/model");
  ensureDir(FS, "/model/speech_tokenizer");

  const files = await discoverModelFiles(modelBase);
  log(`Discovered ${files.length} model files to preload`);

  for (let i = 0; i < files.length; i++) {
    const rel = files[i];
    const src = `${modelBase}/${rel}`;
    const dst = `/model/${rel}`;
    log(`[${i + 1}/${files.length}] downloading ${rel}`);
    const bytes = await fetchBinary(src);
    FS.writeFile(dst, bytes);
  }

  loadedModelBase = modelBase;
  log("Model files loaded into WASM FS");
}

async function loadSpeakersAndLanguages(modelBase) {
  const resolvedModelBase = await resolveModelBase(modelBase);
  if (resolvedModelBase !== modelBase) {
    el.modelBase.value = resolvedModelBase;
    log(`Model base resolved to ${resolvedModelBase}`);
  }

  const cfg = await fetchJson(`${resolvedModelBase}/config.json`);
  const tcfg = cfg.talker_config || {};
  const spkMap = tcfg.spk_id || {};
  const langMap = tcfg.codec_language_id || {};

  const speakers = Object.keys(spkMap);
  const languages = ["auto", ...Object.keys(langMap)];

  setOptions(el.speaker, speakers, speakers[0] || "");
  setOptions(el.language, languages, "auto");

  log(`Loaded ${speakers.length} speakers and ${languages.length} languages`);
}

async function getTokenizer(tokenizerBase) {
  if (!tokenizerPromise || tokenizerPromise.__base !== tokenizerBase) {
    log("Loading browser tokenizer (@huggingface/transformers)...");
    tokenizerPromise = (async () => {
      const lib = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1/+esm");
      // Enable local model lookup under current origin (e.g. /tmp/model/tokenizer.json).
      if (lib.env) {
        lib.env.allowLocalModels = true;
        lib.env.allowRemoteModels = true;
        lib.env.localModelPath = `${window.location.origin}/`;
      }
      try {
        const tokenizer = await lib.AutoTokenizer.from_pretrained(tokenizerBase);
        return tokenizer;
      } catch (err) {
        if (tokenizerBase !== "Qwen/Qwen3-TTS") {
          log(`Tokenizer load failed for ${tokenizerBase}`);
          log("Falling back to Hugging Face tokenizer: Qwen/Qwen3-TTS");
          el.tokenizerBase.value = "Qwen/Qwen3-TTS";
          return lib.AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS");
        }
        throw err;
      }
    })();
    tokenizerPromise.__base = tokenizerBase;
  }
  return tokenizerPromise;
}

async function tokenizePrompt(text, tokenizerBase) {
  const tokenizer = await getTokenizer(tokenizerBase);
  const chat = `<|im_start|>assistant\n${text}<|im_end|>\n<|im_start|>assistant\n`;
  const encoded = await tokenizer(chat, { add_special_tokens: false });
  const raw = Array.isArray(encoded.input_ids?.[0]) ? encoded.input_ids[0] : encoded.input_ids;
  if (!raw) throw new Error("Tokenizer returned no input_ids");
  return Array.from(raw, (v) => Number(v));
}

async function generate() {
  clearLogs();
  if (audioUrl) {
    URL.revokeObjectURL(audioUrl);
    audioUrl = null;
  }

  const requestedModelBase = toBaseUrl(el.modelBase.value.trim());
  const modelBase = await resolveModelBase(requestedModelBase);
  const tokenizerBase = await resolveTokenizerBase(
    toBaseUrl(el.tokenizerBase.value.trim() || modelBase),
    modelBase,
  );
  if (modelBase !== requestedModelBase) {
    el.modelBase.value = modelBase;
    log(`Model base resolved to ${modelBase}`);
  }
  if (el.tokenizerBase.value.trim() !== tokenizerBase) {
    el.tokenizerBase.value = tokenizerBase;
    log(`Tokenizer base resolved to ${tokenizerBase}`);
  }
  const text = el.text.value.trim();
  const speaker = el.speaker.value || "";
  const language = el.language.value || "auto";
  const maxTokens = String(Math.max(1, Number(el.maxTokens.value) || 128));

  if (!text) throw new Error("Please enter text");

  const module = await getModule();
  await preloadModelToFS(module, modelBase);

  log("Tokenizing input text...");
  const tokenIds = await tokenizePrompt(text, tokenizerBase);
  log(`Tokenized ${tokenIds.length} prompt tokens`);

  const outPath = "/tmp/output.wav";
  try { module.FS.unlink(outPath); } catch (_) {}
  ensureDir(module.FS, "/tmp");

  const args = [
    "-d", "/model",
    "-t", tokenIds.join(","),
    "-o", outPath,
    "-s", speaker,
    "-l", language,
    "--max-tokens", maxTokens,
  ];

  log("Running WASM inference...");
  try {
    module.callMain(args);
  } catch (err) {
    const msg = String(err && err.message ? err.message : err);
    if (!msg.includes("ExitStatus")) throw err;
  }

  try {
    module.FS.stat(outPath);
  } catch (_) {
    throw new Error(
      "Inference failed before WAV output. Check logs above (often missing speech_tokenizer weights or browser memory limits).",
    );
  }

  const wav = module.FS.readFile(outPath, { encoding: "binary" });
  const blob = new Blob([wav], { type: "audio/wav" });
  audioUrl = URL.createObjectURL(blob);

  el.audio.src = audioUrl;
  el.download.href = audioUrl;
  el.audio.play().catch(() => {});
  log("Done");
}

el.loadMetaBtn.addEventListener("click", async () => {
  clearLogs();
  try {
    await loadSpeakersAndLanguages(toBaseUrl(el.modelBase.value.trim()));
  } catch (err) {
    log(`Error: ${err.message || err}`);
  }
});

el.generateBtn.addEventListener("click", async () => {
  try {
    await generate();
  } catch (err) {
    log(`Error: ${err.message || err}`);
  }
});

// Best-effort initial metadata load.
loadSpeakersAndLanguages(toBaseUrl(el.modelBase.value.trim())).catch((err) => {
  log(`Info: could not auto-load config (${err.message || err})`);
});
