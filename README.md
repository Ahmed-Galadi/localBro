# LocalBro

A fully offline, terminal-based AI assistant powered by a local GGUF model (Gemma by default). No cloud, no API keys, no data leaving your machine.

---

## Features

- **100% local** — runs entirely on your hardware via `llama-cpp-python`
- **Streaming output** — responses render in real-time with Markdown formatting
- **Automatic memory compression** — after every 3 exchanges, conversation history is summarized and compacted so the context window never explodes
- **GPU support** — CUDA is auto-detected during install for hardware-accelerated inference
- **Single command launch** — type `localbro` from anywhere after install

---

## Requirements

- Python 3.10+
- A `.gguf` model file (see [Model](#model) section below)
- `sudo` access (only needed once, to register the `localbro` command)
- *(Optional)* CUDA toolkit if you want GPU acceleration

---

## Installation

```bash
git clone https://github.com/your-username/localbro.git
cd localbro
chmod +x install.sh
./install.sh
```

The installer will:

1. Check your Python version
2. Create a `venv/` virtual environment inside the project folder
3. Validate compiler settings (`CC`/`CXX`) and auto-fallback to available system compilers
4. Detect CUDA and build `llama-cpp-python` with GPU support if available
5. Install all dependencies from `requirements.txt`
6. Create a `models/` directory if one doesn't exist
7. Register `localbro` as a system-wide command at `/usr/local/bin/localbro`

> **GPU override:** If CUDA isn't auto-detected but you want GPU support, run:
> ```bash
> LLAMA_BUILD_FLAGS="CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1" ./install.sh
> ```

---

## Models

LocalBro uses two models with specific roles:

| Model | Size | Role |
|---|---|---|
| Gemma 4 E4B Instruct Q4_K_M | 4.7 GB | Main chat — all conversations |
| Qwen 2.5 0.5B Instruct Q4_K_M | 380 MB | Memory compression only |

Both models are downloaded automatically when you run `install.sh`. No HuggingFace account or token required.

---

## Usage

```bash
localbro
```

Once running:

- Type your message and press **Enter** to send
- Type `exit` or `quit` to close
- Press **Ctrl+C** to force quit at any time

---

## Project Structure

```
localbro/
├── main.py          # Entry point — REPL loop, rendering, memory trigger
├── engine.py        # ChatEngine — prompt formatting and llama.cpp inference
├── summarizer.py    # Memory compression — keeps context lean
├── run.sh           # Dev launcher (activates venv, runs main.py directly)
├── install.sh       # One-time setup and `localbro` command registration
├── requirements.txt # Python dependencies
└── models/          # Drop your .gguf file here
```

---

## How Memory Works

After every **3 exchanges**, the conversation is compressed:

- All **user questions** from the session are extracted and kept
- **Assistant responses** are discarded (they're verbose and reconstructable)
- The summary is prepended to the next session as a `[Memory]` block, injected as established fact — not as a hallucinated model turn
- Memory is capped at **20 entries** to stay within context limits

This means LocalBro maintains continuity across a long conversation without blowing up the context window.

---

## Configuration

Key settings are at the top of `main.py`:

| Variable         | Default                              | Description                        |
|------------------|--------------------------------------|------------------------------------|
| `CHAT_MODEL`     | `./models/gemma-4-E4B-it-Q4_K_M.gguf` | Path to your GGUF model          |
| `COMPRESS_AFTER` | `3`                                  | Exchanges before memory compresses |

Engine settings are in `engine.py` inside the `Llama(...)` constructor:

| Parameter    | Default | Description                        |
|--------------|---------|------------------------------------|
| `n_ctx`      | `4096`  | Context window size (tokens)       |
| `n_threads`  | `4`     | CPU threads for inference          |
| `temperature`| `0.2`   | Lower = more focused/deterministic |
| `max_tokens` | `1024`  | Max tokens per response            |

---

## Dependencies

- [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) — local LLM inference
- [`rich`](https://github.com/Textualize/rich) — terminal Markdown rendering and live streaming

---

## License

MIT
