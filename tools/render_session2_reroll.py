"""Re-render the 4 backdrop strips session 2 found failing SEAM_ENERGY on v.

Deliberately NOT a re-roll of random seeds against the original prompts --
that was ruled out because the failure is prompt-correlated (all 3 desert-dusk
seeds failed, 0 of 3 desert-storm), so the fix has to change what is asked
for, not which noise seeds it. Owns its own prompt table rather than
importing ``render_session1.BACKDROPS``: session 1's script is that
milestone's provenance record and reproduces the 9 strips actually accepted
there, so it must not carry session-2-only prompt edits.

Measured cause of the failure: the vertical seam ratio compares a wrap-edge
luma jump (``seam_v``, sky-to-ground brightness gap) against the mean
interior row-to-row jump (``inner_v``, how much local texture exists). Desert
dusk's smooth gradient has low ``inner_v`` (8.1-12.2) carrying a large
``seam_v`` (46-80); passing storm strips reach ``inner_v`` 5.9-9.3 from
cloud/rain/rock detail. The lever is texture, not brightness -- flattening
the sky-to-ground gap would also push ``NO_VERTICAL_STRUCTURE``'s row-luma
range toward its own floor of 20.0.

Same seeds and target as session 1, only the prompt and finisher call repeat.
"""

from __future__ import annotations

from pathlib import Path

from artgate_gen import pipeline
from artgate_gen.console import emit
from artgate_gen.targets import TARGETS

ROOT = Path(__file__).resolve().parent.parent
RAW_BACKDROP = ROOT / "art" / "raw-backdrop"
FINISHED = ROOT / "art" / "backdrops-256x64"

# Session-2-only prompt variants, texture-widened from render_session1's
# BACKDROPS entries of the same key. Two passes so far:
#   pass 1 (clouds + ridgelines on dusk only): inner_v 8.1-12.2 -> 8.4-13.4,
#     closed the ratio only on desert-dusk_s2 (5.92 -> 4.04, still short).
#   pass 2 (wider texture on both families): closed desert-dusk_s2 (2.29) and
#     desert-night_s2 (3.52), but desert-dusk_s0/s1 stayed short (5.94, 6.06)
#     -- both need inner_v roughly 1.5-1.9x their current value, beyond
#     anything measured across all 9 original strips or either retry pass.
# Pass 3 targets dusk_s0/s1 specifically with texture across the FULL strip
# height (foreground scrub/rock, midground dust devils, not just sky/ridge),
# on the theory that pass 1-2's texture was concentrated near the top third
# and never touched the lower rows that anchor seam_v's row-0-to-row-63
# comparison.
BACKDROPS_RETRY = {
    "desert-dusk": (
        "desert highway at dusk, orange sky, scattered wispy clouds, saguaro "
        "cacti, distant layered mountain ridgelines with visible rock texture, "
        "dust devils on the horizon, uneven cracked earth and scattered rocks "
        "across the full foreground, tire tracks and gravel texture on road "
        "shoulder, scrub brush silhouettes"
    ),
}

# (prompt key, seed) pairs still failing SEAM_ENERGY's vertical axis after two
# retry passes. desert-dusk_s2 and desert-night_s2 passed on pass 2 and are
# not retried again.
RETRY = (
    ("desert-dusk", 0),
    ("desert-dusk", 1),
)


def main() -> int:
    """Re-render the 4 failing stems and overwrite their raw and finished art.

    Returns:
        0 on success.
    """
    pipe = pipeline.load_pipeline()
    target = TARGETS[256, 64]
    for name, seed in RETRY:
        prompt = BACKDROPS_RETRY[name]
        stem = f"{name}_s{seed}"
        image = pipeline.generate(pipe, prompt, seed, target)
        image.save(RAW_BACKDROP / f"{stem}.png")
        target.finisher(image, target.canvas, target.max_colors).save(
            FINISHED / f"{stem}.png"
        )
        emit(f"reroll {stem} {image.size} -> {target.canvas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
