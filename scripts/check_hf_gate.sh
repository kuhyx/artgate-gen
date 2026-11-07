#!/bin/bash
# Report whether this HF token may download a gated repo; open its page if not.
# Usage: check_hf_gate.sh <org/repo> [--verify-only]   (exit 0 accepted, 1 not)
set -euo pipefail

readonly TOKEN_FILE="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"

main() {
    local repo="${1:?usage: $0 <org/repo> [--verify-only]}" verify_only=0 code
    [[ "${2:-}" == "--verify-only" ]] && verify_only=1
    [[ -r "$TOKEN_FILE" ]] || { echo "no HF token at $TOKEN_FILE" >&2; exit 2; }
    code="$(curl -s -o /dev/null -I -w '%{http_code}' \
        -H "Authorization: Bearer $(<"$TOKEN_FILE")" \
        "https://huggingface.co/$repo/resolve/main/model_index.json")"
    if [[ "$code" =~ ^[23][0-9][0-9]$ ]]; then
        echo "$repo: accepted"
        return 0
    fi
    echo "$repo: NOT accepted ($code)"
    if [[ $verify_only -eq 0 ]]; then
        xdg-open "https://huggingface.co/$repo" >/dev/null 2>&1 || true
    fi
    return 1
}

main "$@"
