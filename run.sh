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
PROFILE=0
SUBJECTS=()

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] <outdir> "name=description" ["name=description" ...]

Sets up the environment if needed, then generates gate-passing item icons.

Options:
  -s, --size N          Icon edge in pixels: 64 (default) or 32
  -c, --candidates N    Renders per subject, each with its own seed (default 1)
      --setup-only      Install dependencies and the model, then exit
      --profile         Report where the time went, plus peak GPU/RAM/load
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
    # fetch_model.py probes the cache first and prints its own one-line status,
    # so there is deliberately no log line here -- an unconditional "Ensuring
    # ..." banner is exactly the noise this step was asked to stop emitting.
    "$PY" "$SCRIPT_DIR/fetch_model.py"
}

# Set by gate_one: the CLI's combined output and its exit status, for the
# caller to branch on and report.
GATE_OUTPUT=""
GATE_STATUS=0

gate_one() {
    local icon="$1" spec="$2" icon_abs
    # MUST be absolute. The gate runs with cwd=$ARTGATE_REPO, so a relative path
    # from the caller's directory (e.g. ./icons/cat.png) resolves against the
    # wrong root and the CLI reports "cannot read image" -- which the old code
    # then counted as a failed gate. That bug reported 0/1 on a passing icon.
    icon_abs="$(realpath "$icon")"
    # Branch on the process exit code, which is already meaningful (0 pass,
    # 1 gate failure, 2 harness error), instead of grepping the JSON for a
    # substring. Uses "$PY" so the gate does not depend on whatever `python`
    # happens to be on PATH having numpy and Pillow.
    GATE_OUTPUT="$(cd "$ARTGATE_REPO" && "$PY" -m python_pkg.artgate.cli \
        "$icon_abs" --spec "$spec" --json 2>&1)"
    GATE_STATUS=$?
    return $GATE_STATUS
}

# Set by repair_one to the repairer's output, so a crash is reportable.
REPAIR_OUTPUT=""

repair_one() {
    local icon="$1" spec="$2" icon_abs
    # Only exit status 1 is an art verdict. 2 is a harness error (unreadable
    # file, bad spec) which re-encoding cannot fix, so do not try -- and read
    # the status, not the JSON text, for the same reason gate_one does.
    if [[ $GATE_STATUS -ne 1 ]]; then
        return 1
    fi
    icon_abs="$(realpath "$icon")"
    # Capture rather than discard: silently swallowing a repairer traceback
    # would hide a broken repair behind the gate's own verdict, which is the
    # failure mode this whole loop was rewritten to eliminate.
    REPAIR_OUTPUT="$("$PY" "$SCRIPT_DIR/repair_cli.py" "$icon_abs" \
        --spec "$ARTGATE_REPO/$spec" --codes "$GATE_OUTPUT" 2>&1)"
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
    local gen_args=(--size "$SIZE" --candidates "$CANDIDATES")
    [[ $PROFILE -eq 1 ]] && gen_args+=(--profile)
    "$PY" "$SCRIPT_DIR/generate.py" "$OUT_DIR" "${SUBJECTS[@]}" "${gen_args[@]}"

    log "Gating output"
    local spec="python_pkg/artgate/styles/pixel_icon_64.toml"
    [[ "$SIZE" == "32" ]] && spec="python_pkg/artgate/styles/pixel_icon.toml"
    local passed=0 total=0 repaired=0
    for icon in "$OUT_DIR"/*.png; do
        [[ -f "$icon" ]] || continue
        total=$((total + 1))
        REPAIR_OUTPUT=""  # per-icon, or a stale message follows the next failure
        if gate_one "$icon" "$spec"; then
            passed=$((passed + 1))
        elif repair_one "$icon" "$spec" && gate_one "$icon" "$spec"; then
            passed=$((passed + 1))
            repaired=$((repaired + 1))
            echo "    repaired: $icon"
        else
            printf '    gate FAILED: %s\n' "$icon"
            # Print the verdict rather than discarding it. The previous version
            # sent stderr to /dev/null, which hid "cannot read image" and made a
            # path bug look like an art failure for every icon.
            printf '      %s\n' "$GATE_OUTPUT"
            # A repairer that crashed is a bug in this repo, not a verdict on
            # the art -- say so instead of letting it look like an art failure.
            if [[ -n "$REPAIR_OUTPUT" ]]; then
                printf '      repair did not help: %s\n' "$REPAIR_OUTPUT"
            fi
        fi
    done
    if [[ $repaired -gt 0 ]]; then
        log "$passed/$total passed the ${SIZE}px gate ($repaired after auto-repair)"
    else
        log "$passed/$total passed the ${SIZE}px gate"
    fi
    log "Gates check canvas/palette/alpha/margin/contrast, NOT whether the art"
    log "looks good. Look at the icons before calling them done."
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--size)       SIZE="$2"; shift 2 ;;
        -c|--candidates) CANDIDATES="$2"; shift 2 ;;
        --setup-only)    SETUP_ONLY=1; shift ;;
        --profile)       PROFILE=1; shift ;;
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
