#!/bin/bash
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}❯ LocalBro Installer (Multi‑GPU Auto‑Detection)${NC}"
echo -e "${CYAN}────────────────────────────────────────────${NC}"

# ── 1. Python check ──────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}✗ python3 not found. Install Python 3.10+ and retry.${NC}"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}✓ Python ${PY_VER} found${NC}"

# ── 2. Virtual environment ────────────────────────────────────────────────────
if [ ! -d "$INSTALL_DIR/venv" ]; then
    echo -e "${CYAN}❯ Creating virtual environment...${NC}"
    python3 -m venv "$INSTALL_DIR/venv"
    echo -e "${GREEN}✓ venv created${NC}"
else
    echo -e "${GREEN}✓ venv already exists — skipping${NC}"
fi

source "$INSTALL_DIR/venv/bin/activate"
pip install --upgrade pip --quiet

# ── 3. OS & GPU Detection ─────────────────────────────────────────────────────
OS_TYPE=$(uname -s)
LLAMA_BUILD_FLAGS=""
BACKEND_NAME="CPU"

if [[ "$OS_TYPE" == "Darwin" ]]; then
    # macOS – always enable Metal (works on Apple Silicon and Intel + AMD GPU Macs)
    echo -e "${YELLOW} macOS detected — enabling Metal support${NC}"
    LLAMA_BUILD_FLAGS="CMAKE_ARGS='-DGGML_METAL=on' FORCE_CMAKE=1"
    BACKEND_NAME="Metal"
else
    # Linux (and other Unix‑like)
    # Check for NVIDIA GPU + CUDA toolkit
    if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
        if command -v nvcc &>/dev/null; then
            echo -e "${YELLOW}⚡ NVIDIA GPU + CUDA toolkit detected — building with CUDA${NC}"
            LLAMA_BUILD_FLAGS="CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1"
            BACKEND_NAME="CUDA"
        else
            echo -e "${YELLOW}⚠ NVIDIA GPU found but CUDA toolkit (nvcc) missing${NC}"
            echo -e "${YELLOW}   → Falling back to Vulkan if possible${NC}"
        fi
    fi

    # If no CUDA, try Vulkan (works on AMD and NVIDIA)
    if [[ -z "$LLAMA_BUILD_FLAGS" ]]; then
        if command -v vulkaninfo &>/dev/null && vulkaninfo --summary 2>/dev/null | grep -q "deviceName"; then
            # Check for glslc (shader compiler) – required for Vulkan build
            if command -v glslc &>/dev/null && [ -f /usr/include/vulkan/vulkan.h ]; then
                echo -e "${YELLOW}⚡ Vulkan‑capable GPU + glslc detected — building with Vulkan${NC}"
                LLAMA_BUILD_FLAGS="CMAKE_ARGS='-DGGML_VULKAN=on' FORCE_CMAKE=1"
                BACKEND_NAME="Vulkan"
            else
                echo -e "${YELLOW}⚠ Vulkan runtime found but glslc or vulkan headers are missing${NC}"
                echo -e "${YELLOW}   → Building CPU‑only (safe)${NC}"
                echo -e "${YELLOW}   To enable Vulkan later, run:${NC}"
                echo -e "${YELLOW}   sudo apt install glslc libvulkan-dev   # Debian/Ubuntu${NC}"
            fi
        fi
    fi
fi

if [[ -z "$LLAMA_BUILD_FLAGS" ]]; then
    echo -e "${YELLOW}⚠ No compatible GPU acceleration detected — CPU‑only build${NC}"
    BACKEND_NAME="CPU"
fi

# ── 4. Compiler selection (Linux only) ───────────────────────────────────────
unset CC CXX
if [[ "$OS_TYPE" != "Darwin" ]]; then
    for candidate in gcc-13 gcc-12 gcc-11 gcc; do
        if command -v "$candidate" &>/dev/null; then
            export CC="$candidate"
            break
        fi
    done
    for candidate in g++-13 g++-12 g++-11 g++; do
        if command -v "$candidate" &>/dev/null; then
            export CXX="$candidate"
            break
        fi
    done
    echo -e "${GREEN}✓ Using compiler: ${CC:-default} / ${CXX:-default}${NC}"
fi

# ── 5. Install dependencies ───────────────────────────────────────────────────
echo -e "${CYAN}❯ Installing dependencies (backend: ${BACKEND_NAME})...${NC}"

# Try pre‑built wheel first (may already have desired backend)
if pip install llama-cpp-python --only-binary=:all: --quiet 2>/dev/null; then
    echo -e "${GREEN}✓ Installed pre‑built llama-cpp-python (no compilation)${NC}"
    # Note: pre‑built wheels often have Metal on macOS, but CUDA/Vulkan usually not.
else
    echo -e "${YELLOW}⚠ No suitable pre‑built wheel — building from source with ${BACKEND_NAME} support...${NC}"
    if [[ -n "$LLAMA_BUILD_FLAGS" ]]; then
        eval "$LLAMA_BUILD_FLAGS pip install llama-cpp-python --upgrade --no-cache-dir"
    else
        pip install llama-cpp-python --upgrade --no-cache-dir
    fi
fi

pip install rich --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

# ── 6. Download models ───────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR/models"

GEMMA="$INSTALL_DIR/models/gemma-4-E4B-it-Q4_K_M.gguf"
QWEN="$INSTALL_DIR/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

download_model() {
    local url="$1"
    local dest="$2"
    local name="$(basename $dest)"
    local size="$3"

    if [ -f "$dest" ]; then
        echo -e "${GREEN}✓ $name already exists — skipping${NC}"
        return
    fi

    echo -e "${CYAN}❯ Downloading $name ($size)...${NC}"
    curl -L --progress-bar --retry 3 --retry-delay 2 -o "$dest" "$url"
    echo -e "${GREEN}✓ $name downloaded${NC}"
}

download_model \
    "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf" \
    "$GEMMA" "4.7GB"

download_model \
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
    "$QWEN" "380MB"

# ── 7. Register command (optional) ───────────────────────────────────────────
LAUNCHER="/usr/local/bin/localbro"
echo -e "${CYAN}❯ Register global command? (optional)${NC}"
read -p "Install 'localbro' globally? [y/N]: " choice
if [[ "$choice" =~ ^[Yy]$ ]]; then
    sudo tee "$LAUNCHER" > /dev/null << EOF
#!/bin/bash
source "$INSTALL_DIR/venv/bin/activate"
cd "$INSTALL_DIR"
exec python3 "$INSTALL_DIR/main.py" "\$@"
EOF
    sudo chmod +x "$LAUNCHER"
    echo -e "${GREEN}✓ 'localbro' registered globally${NC}"
else
    echo -e "${YELLOW}⚠ Skipped global install. Use ./run.sh instead.${NC}"
fi

# ── 8. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}✓ Installation complete!${NC}"
echo -e "  GPU backend used: ${CYAN}${BACKEND_NAME}${NC}"
echo -e "  Run ${CYAN}./run.sh${NC} or ${CYAN}localbro${NC} (if installed globally)"
echo -e "  Models are in: ${CYAN}$INSTALL_DIR/models/${NC}"