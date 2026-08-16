"""Generate game art with SDXL plus a pixel-art LoRA, then gate it.

The package is split by dependency weight rather than by feature: everything
that needs torch lives behind :mod:`artgate_gen.pipeline`, and the rest --
finishing, the gate, repair, the CLI -- runs on numpy and Pillow alone. That
seam is what lets the base install stay small and the test suite run without a
GPU or a model download.
"""
