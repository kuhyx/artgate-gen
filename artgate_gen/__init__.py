"""Generate game art with SDXL plus a pixel-art LoRA, then gate it.

The package is split by dependency weight rather than by feature: everything
that renders lives behind :mod:`artgate_gen.pipeline` (which drives the
stable-diffusion.cpp ``sd-cli`` binary), and the rest --
finishing, the gate, repair, the CLI -- runs on numpy and Pillow alone. That
seam is what lets the base install stay small and the test suite run without a
GPU or a model download.
"""
