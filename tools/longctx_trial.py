"""Re-render the SDXL portraits and backdrops on one backbone, recording every render.

Reads roadside-assistance's ``art/MANIFEST.json``, renders each portrait and the
chosen backdrops with the same seed, style and size, plus one long-prompt twin,
and writes every outcome (failures included) back under ``longctx-<backend>``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

import torch

from artgate_gen import backends, prompt_budget
from artgate_gen.console import emit
from artgate_gen.targets import TARGETS

if TYPE_CHECKING:
    from types import ModuleType

    from artgate_gen._types import Pipeline

LONG_DETAIL = (
    "small silver stud earring in the left ear, a thin pale scar across the right "
    "eyebrow, faint freckles across the nose and cheeks, headphone cable tucked "
    "into the jacket collar, jacket collar turned up with a worn brass zipper, a "
    "tiny round enamel pin of a red car on the lapel, soft warm rim light from the "
    "left, cool teal shadow on the right side of the face, muted dusty orange "
    "background, clean dark outline around the hair, two-tone cel shading on the "
    "skin, bright single-pixel catchlight in each eye, calm steady breathing, "
    "slightly parted lips, chin level with the camera, ears half hidden by hair"
)
COPIED = ("seed", "style", "negative", "render", "canvas", "max_colors")
MB = 2**20


@dataclass(frozen=True)
class Job:
    """One render: a source SDXL entry, as is or with ``LONG_DETAIL`` appended."""

    source: dict[str, Any]
    long: bool

    @property
    def name(self) -> str:
        """Entry and file name; the long twin gets a ``_long`` suffix."""
        return f"{self.source['name']}_long" if self.long else str(self.source["name"])

    @property
    def prompt(self) -> str:
        """The subject prompt for this arm."""
        base = str(self.source["prompt"])
        return f"{base}, {LONG_DETAIL}" if self.long else base


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="<art>/*-trial dir")
    parser.add_argument("--backend", choices=sorted(backends.MODULES), default="sd35")
    parser.add_argument("--backdrops", nargs="+", default=["coast_s0", "desert_s0"])
    parser.add_argument("--long-subject", default="neutral_s11")
    return parser


def _jobs(
    parser: argparse.ArgumentParser,
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[Job]:
    """Pick every portrait, the named backdrops, and the long twin's source."""
    portraits = [e for e in entries if e["category"] == "portraits"]
    backdrops = {e["name"]: e for e in entries if e["category"] == "backdrops"}
    missing = [n for n in args.backdrops if n not in backdrops]
    long_source = next((e for e in portraits if e["name"] == args.long_subject), None)
    if missing or long_source is None:
        parser.error(f"not in the manifest: {missing or args.long_subject}")
    sources = portraits + [backdrops[n] for n in args.backdrops]
    return [Job(e, long=False) for e in sources] + [Job(long_source, long=True)]


def _render(
    module: ModuleType, pipe: Pipeline, job: Job, folder: Path, art: Path
) -> dict[str, Any]:
    """Render ``job``, save raw and finished PNGs, and return its manifest entry."""
    src = job.source
    target = replace(
        TARGETS[tuple(src["canvas"])],
        style=src["style"],
        negative=src["negative"],
        render=tuple(src["render"]),
    )
    prompt = module.full_prompt(job.prompt, target)
    path = folder / f"{job.name}.png"
    entry: dict[str, Any] = {k: src[k] for k in COPIED} | {
        "name": job.name,
        "category": f"longctx-{folder.name}",
        "file": path.relative_to(art).as_posix(),
        "prompt": prompt,
        "source": f"{src['category']}/{src['name']}",
        "arm": "long" if job.long else "short",
        "model": module.MODEL,
        "lora": module.LORA,
        "lora_weights": module.LORA_WEIGHTS,
        "lora_scale": getattr(module, "LORA_SCALE", 1.0),
        "steps": module.STEPS,
        "guidance": module.GUIDANCE,
        "t5_max_len": getattr(module, "T5_MAX_LEN", None),
        "offload": getattr(module, "OFFLOAD", False),
        "clip_tokens": prompt_budget.token_count(pipe, prompt),
    }
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        image = module.generate(pipe, job.prompt, src["seed"], target)
        entry |= {
            "status": "ok",
            "seconds": round(time.perf_counter() - start, 2),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / MB),
        }
        (folder / "raw").mkdir(parents=True, exist_ok=True)
        image.save(folder / "raw" / path.name)
        colours = src["max_colors"] or target.max_colors
        target.finisher(image, target.canvas, colours).save(path)
    except (RuntimeError, ValueError) as exc:
        return entry | {"status": "error", "error": str(exc)}
    return entry


def _store(manifest: Path, category: str, fresh: list[dict[str, Any]]) -> None:
    """Replace ``category``'s entries with ``fresh`` at the end; others keep order."""
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    merged = [e for e in entries if e["category"] != category] + fresh
    manifest.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the trial; exit 1 if any render failed."""
    parser = _parser()
    args = parser.parse_args(argv)
    art = args.manifest.parent.resolve()
    out = args.out.resolve()
    if not out.is_relative_to(art) or not out.name.endswith("-trial"):
        parser.error(f"--out must be a *-trial folder inside {art}")
    jobs = _jobs(parser, json.loads(args.manifest.read_text(encoding="utf-8")), args)
    module = backends.load(args.backend)
    pipe: Pipeline = module.load_pipeline()
    folder = out / args.backend
    fresh: list[dict[str, Any]] = []
    for job in jobs:
        fresh.append(_render(module, pipe, job, folder, art))
        _store(args.manifest, f"longctx-{args.backend}", fresh)
    ok = [e for e in fresh if e["status"] == "ok"]
    mean = fmean([e["seconds"] for e in ok] or [0.0])
    peak = max((e["peak_vram_mb"] for e in ok), default=0)
    emit(
        f"{args.backend}: {len(ok)}/{len(fresh)} ok, {mean:.1f} s/image, peak {peak} MB"
    )
    return 0 if len(ok) == len(fresh) else 1


if __name__ == "__main__":
    raise SystemExit(main())
