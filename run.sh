#!/bin/bash

# ============================================================================
# artgate-gen: set up the environment and generate item icons.
#
# Installs everything it needs (uv, the venv, the Python deps, and the SDXL
# checkpoint + LoRA) and then generates. Safe to re-run: each step is skipped
# when already satisfied, so the common case is a fast no-op plus generation.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly SCRIPT_NAME="${0##*/}"
readonly VENV="$SCRIPT_DIR/venv"
readonly PY="$VENV/bin/python"
readonly ARTGATE_REPO="${ARTGATE_REPO:-$HOME/testsAndMisc}"

# The finisher lives in testsAndMisc; generate.py imports it from there.
readonly FINISH_MODULE="$ARTGATE_REPO/python_pkg/artgate/finish.py"

OUT_DIR=""
SIZE=64
CANDIDATES=1
SETUP_ONLY=0
SUBJECTS=()

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] <outdir> "name=description" ["name=description" ...]

Sets up the environment if needed, then generates gate-passing item icons.

Options:
  -s, --size N          Icon edge in pixels: 64 (default) or 32
  -c, --candidates N    Renders per subject, each with its own seed (default 1)
      --setup-only      Install dependencies and the model, then exit
  -h, --help            Show this help

Examples:
  ./$SCRIPT_NAME --setup-only
  ./$SCRIPT_NAME ./icons "lantern=a brass lantern with a lit wick"
  ./$SCRIPT_NAME -c 3 ./icons "crown=a golden crown with red jewels"

Prompt tips (measured): name the object once and add two or three concrete
attributes. Negation does not work ("no tree" pushes toward trees) and
repeating the noun multiplies the object. Change the seed before the prompt.
EOF
    exit 0
}

log() { printf '==> %s\n' "$1"; }

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        return
    fi
    log "Installing uv"
    if command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --needed --noconfirm uv
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
}

create_venv() {
    if [[ -x "$PY" ]]; then
        return
    fi
    log "Creating venv at $VENV"
    uv venv "$VENV"
}

install_deps() {
    # torch is the expensive one; if it imports, everything else already did.
    if "$PY" -c 'import torch, diffusers, mcp' 2>/dev/null; then
        return
    fi
    log "Installing Python dependencies (this downloads several GB)"
    VIRTUAL_ENV="$VENV" uv pip install -r "$SCRIPT_DIR/requirements.txt"
}

fetch_model() {
    # snapshot_download is a no-op once the HF cache holds the weights.
    log "Ensuring SDXL base + pixel-art LoRA are cached"
    "$PY" "$SCRIPT_DIR/fetch_model.py"
}

validate_requirements() {
    if [[ ! -f "$FINISH_MODULE" ]]; then
        echo "Error: artgate finisher not found at $FINISH_MODULE" >&2
        echo "       Clone github.com/kuhyx/testsAndMisc, or set ARTGATE_REPO." >&2
        exit 1
    fi
    if ! "$PY" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
        echo "Warning: CUDA is not available; generation will be very slow." >&2
    fi
}

main() {
    install_uv
    create_venv
    install_deps
    fetch_model
    validate_requirements

    if [[ $SETUP_ONLY -eq 1 ]]; then
        log "Setup complete. Run again with an output dir and subjects to generate."
        return 0
    fi

    if [[ -z "$OUT_DIR" || ${#SUBJECTS[@]} -eq 0 ]]; then
        echo "Error: an output directory and at least one subject are required" >&2
        echo "Run '$SCRIPT_NAME --help' for usage." >&2
        exit 1
    fi

    log "Generating ${#SUBJECTS[@]} subject(s) at ${SIZE}px"
    "$PY" "$SCRIPT_DIR/generate.py" "$OUT_DIR" "${SUBJECTS[@]}" \
        --size "$SIZE" --candidates "$CANDIDATES"

    log "Gating output"
    local spec="python_pkg/artgate/styles/pixel_icon_64.toml"
    [[ "$SIZE" == "32" ]] && spec="python_pkg/artgate/styles/pixel_icon.toml"
    local passed=0 total=0
    for icon in "$OUT_DIR"/*.png; do
        [[ -f "$icon" ]] || continue
        total=$((total + 1))
        if (cd "$ARTGATE_REPO" && python -m python_pkg.artgate.cli "$icon" \
            --spec "$spec" --json 2>/dev/null | grep -q '"exit": 0'); then
            passed=$((passed + 1))
        else
            echo "    gate FAILED: $icon"
        fi
    done
    log "$passed/$total passed the ${SIZE}px gate"
    log "Gates check canvas/palette/alpha/margin/contrast, NOT whether the art"
    log "looks good. Look at the icons before calling them done."
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--size)       SIZE="$2"; shift 2 ;;
        -c|--candidates) CANDIDATES="$2"; shift 2 ;;
        --setup-only)    SETUP_ONLY=1; shift ;;
        -h|--help)       usage ;;
        -*)              echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$OUT_DIR" ]]; then
                OUT_DIR="$1"
            else
                SUBJECTS+=("$1")
            fi
            shift
            ;;
    esac
done

main
