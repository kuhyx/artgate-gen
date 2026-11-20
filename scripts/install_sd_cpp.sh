#!/bin/bash

# ============================================================================
# Install stable-diffusion.cpp's `sd-cli` (MIT, Vulkan backend), which renders
# every image, then check that it sees a Vulkan GPU. Safe to re-run: an
# installed sd-cli makes the install step a no-op. Called by run.sh.
# ============================================================================

set -euo pipefail

log() { printf '==> %s\n' "$1"; }

install_sd_cpp() {
    if command -v sd-cli >/dev/null 2>&1; then
        return
    fi
    if ! command -v yay >/dev/null 2>&1; then
        echo "Error: sd-cli (stable-diffusion.cpp) is missing and yay is not" \
            "installed; install stable-diffusion.cpp with Vulkan, then re-run" >&2
        exit 1
    fi
    log "Installing stable-diffusion.cpp (Vulkan)"
    # The prebuilt package first; the -git one builds from source.
    yay -S --needed --noconfirm stable-diffusion.cpp-vulkan-bin \
        || yay -S --needed --noconfirm stable-diffusion.cpp-vulkan-git
    if ! command -v sd-cli >/dev/null 2>&1; then
        echo "Error: stable-diffusion.cpp installed but sd-cli is not on PATH" >&2
        exit 1
    fi
}

check_vulkan() {
    # No `grep -q`: it exits on the first match, and under pipefail the
    # resulting SIGPIPE in sd-cli would read as "no GPU".
    if ! sd-cli --list-devices 2>/dev/null | grep '^Vulkan' >/dev/null; then
        echo "Warning: sd-cli sees no Vulkan GPU; generation will be very slow." >&2
    fi
}

install_sd_cpp
check_vulkan
