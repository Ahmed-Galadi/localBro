#!/bin/bash
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}❯ LocalBro Installer${NC}"
echo -e "${CYAN}──────────────────────${NC}"

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

# ── 3. Pip upgrade ────────────────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── 3.1 Compiler sanity checks ─────────────────────────────────────────────────
# Some environments export CC/CXX values that don't exist on the host.
# llama-cpp-python falls back to a source build when needed, so ensure a working
# compiler command is available before pip install.
resolve_compiler() {
    local var_name="$1"
    shift
    local current_value="${!var_name}"

    if [ -n "$current_value" ] && command -v "$current_value" &>/dev/null; then
        return
    fi

    if [ -n "$current_value" ]; then
        echo -e "${YELLOW}⚠ $var_name='$current_value' not found — selecting a fallback${NC}"
    fi

    for candidate in "$@"; do
        if command -v "$candidate" &>/dev/null; then
            export "$var_name=$candidate"
            return
        fi
    done

    echo -e "${RED}✗ No usable compiler found for $var_name. Install build tools and retry.${NC}"
    exit 1
}

resolve_compiler CC gcc cc clang
resolve_compiler CXX g++ c++ clang++

# ── 4. llama-cpp-python — GPU hint ───────────────────────────────────────────
# Detect CUDA so we can offer a GPU-accelerated build.
# You can override by setting LLAMA_BUILD_FLAGS before running this script.
# Example: LLAMA_BUILD_FLAGS="CMAKE_ARGS='-DGGML_CUDA=on'" ./install.sh
if [ -z "$LLAMA_BUILD_FLAGS" ] && command -v nvcc &>/dev/null; then
    echo -e "${YELLOW}⚡ CUDA detected — building llama-cpp-python with GPU support${NC}"
    LLAMA_BUILD_FLAGS="CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1"
fi

# ── 5. Install dependencies ───────────────────────────────────────────────────
echo -e "${CYAN}❯ Installing dependencies...${NC}"
if [ -n "$LLAMA_BUILD_FLAGS" ]; then
    eval "$LLAMA_BUILD_FLAGS pip install llama-cpp-python --upgrade"
    pip install rich --quiet
else
    pip install -r "$INSTALL_DIR/requirements.txt" --quiet
fi
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

# ── 7. Register `localbro` command ────────────────────────────────────────────
LAUNCHER="/usr/local/bin/localbro"

echo -e "${CYAN}❯ Registering 'localbro' command (may ask for sudo)...${NC}"

sudo tee "$LAUNCHER" > /dev/null << EOF
#!/bin/bash
source "$INSTALL_DIR/venv/bin/activate"
cd "$INSTALL_DIR"
exec python3 "$INSTALL_DIR/main.py" "\$@"
EOF

sudo chmod +x "$LAUNCHER"
echo -e "${GREEN}✓ 'localbro' registered at $LAUNCHER${NC}"

# ── 8. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}✓ Installation complete!${NC}"
echo -e "  Run ${CYAN}localbro${NC} from any terminal to start."
echo -e "  Models are ready in: ${CYAN}$INSTALL_DIR/models/${NC}"
