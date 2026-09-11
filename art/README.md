# Generated art

Everything here was produced by `../generate.py` (SDXL base 1.0 +
`nerijs/pixel-art-xl`) and checked by the artgate gates. It is committed as the
reference for what this pipeline actually produces — including its failures.

| Directory | What it is |
|---|---|
| `icons-64px/` | 31 finished icons at 64px, the recommended size. All pass `pixel_icon_64.toml`. |
| `icons-32px/` | 18 at 32px, passing `pixel_icon.toml`. Kept to show what the 32x downscale costs. |
| `raw-1024px/` | The unfinished SDXL renders behind 18 of the icons. Often better than any downscale of them. |
| `contact-sheets/` | Comparison sheets from the bake-off and the size/background experiments. |

## Honest quality notes

The gates check canvas, palette, binary alpha, margin and contrast. **They do
not judge whether the art looks good**, and everything here passed them.

By eye, roughly 9 of the 12 in the newest set (`crown` … `arrow`) read
correctly. Known failures, kept rather than hidden:

- **arrow** — renders as *two* crossed arrows. Wrong across 6 seeds and 2
  phrasings; the negative prompt's "multiple objects" does not catch it.
- **bow** — ambiguous without an arrow; also unrecovered across 6 seeds.
- **rope** — reads as a ring or mirror rather than a coil. The best-looking
  retry was rejected by the gate for `MARGIN_TOO_SMALL`, so the original stands.

`apple` was recovered by retrying: the first render was a whole tree.

`cat` (the one non-item subject) reads correctly as a sitting black cat, but
only at the silhouette level: the prompt's glowing eyes and sticking-out tongue
are both lost at 64px. A face-level detail is below what this pipeline resolves.

> **Refined 2026-08-16.** That last sentence was too general, and it cost a
> milestone's worth of doubt about whether this pipeline could render a face at
> all. It can. What does not survive is **detail below roughly 1/8 of the
> canvas** — the cat's eyes are small features on a whole-body subject, whereas
> a portrait spends the entire canvas on the face.

## Portraits and backdrops (2026-08-16)

Added with the non-square render path. Everything below comes from human review
of real output, not from the gates — which, as stated above, do not judge
whether art looks good.

**Portraits need 256px.** 64px passes every gate and reads as "a face", but
manual review found it unusable, and 128 too. Generate at 1024, downscale to
256, and leave it **unquantised** — a 32-colour cap eats facial detail.

Three prompt findings, each from rejected output:

- **Three-quarter framing beats front-facing.** `"front facing"` produced flat,
  interchangeable faces; `"three-quarter view"` was markedly better on every
  expression.
- **Open-mouth smiles read as creepy.** At pixel resolution teeth become an
  undifferentiated pale block; 3 of 9 portraits were rejected for this alone.
  Use `"faint closed-mouth smile"`. The toothy version stays genuinely useful
  for deliberately uncanny work — it just has to be chosen, not stumbled into.
- **A fixed seed holds character identity** across changed expression clauses,
  verified over three seeds x three expressions. Pick one seed per character
  and vary only the expression clause.

**Do not use `ICON_STYLE` for a portrait**: "game item icon, centered, plain
black background" describes an object on a void, not a face.

**The backdrop wrap overlap is 512px, not 256.** At 256 the worst seam across
nine real renders measured 6.01 against a 4.0 threshold — and the failing strip
looked fine by eye, which is precisely why the metric exists. At 512 the same
nine measured 2.27 worst. Measure with `tools/seam_probe.py`; never substitute
a different metric.

**Record prompt + seed for everything, including experiments.** A PNG whose
recipe is lost cannot be varied, fixed or regenerated. In this session a raw
render was overwritten and the exact image behind a reported number became
unrecoverable. `~/src/roadside-assistance/tools/artgen.py` is a manifest
implementation whose recipes are verified to reproduce images bit-for-bit.

## SEAM_ENERGY vertical axis (2026-08-18)

Session 2 added a vertical check to SEAM_ENERGY, same 4.0 threshold as the
existing horizontal one. It retroactively failed 4 of the 9 backdrop strips
above: a smooth sky-to-ground gradient scores a large luma jump between the
top and bottom row (`seam_v`) while carrying very little interior row-to-row
texture (`inner_v`), and the ratio blows past threshold even though nothing
about the strip reads as a seam by eye. Storm strips (clouds, rain, rock
detail) passed 3/3 without changes; desert-dusk failed 3/3, desert-night 1/3.

Fix that worked: **add interior texture to the prompt** (scattered clouds,
layered ridgelines, foreground rock/scrub detail), which raises `inner_v`
without flattening the brightness gap that `NO_VERTICAL_STRUCTURE` depends
on. Confirmed NOT working: pixel-level post-processing (a vertical taper
analogous to the horizontal `wrap_blend`) — the row0-vs-row63 comparison is
too indirect for a local blend to move predictably; two prototypes moved
some strips the right way and others the wrong way with no stable pattern,
and were discarded rather than shipped.

Three texture-prompt passes fixed desert-dusk_s0, desert-dusk_s2 and
desert-night_s2. **desert-dusk_s1 did not close** (v stayed at 5.6-6.1
across all three attempts, and its `inner_v` even moved the wrong direction
on the third, most heavily-texture-worded pass) and was left on its
session-1 art rather than kept as an unaccepted, still-failing re-render.
Reported as an open finding, not solved: this specific seed/prompt
combination may have a real ceiling below what interior texture alone can
raise `inner_v` to, at this resolution and pipeline.

Also measured: rendering is byte-identical within one process (same
prompt+seed twice, max pixel diff 0), but the same prompt+seed measured
~0.5 different in `v` across two separate process invocations — a real,
if modest, cross-process variance source. Strips passing with less than
~0.5 of margin (desert-night_s2 at v=3.52) should be treated as marginal,
not settled.

## Contact sheets worth reading

- `SIZE_COMPARE.png` — the same subjects at 32/48/64px. This is why 64px is the
  default; look at the shield's radial pattern and the coin's emblem.
- `BG_REMOVAL_COMPARE.png` — four-corner flood vs the border-ring flood now
  used. Opaque coverage is identical to 0.1%; the difference is edge pixels.
- `ROWC_AGX_FIX.png` — Blender's AgX view transform vs Standard, from the same
  bake-off. Not this pipeline, but the same lesson about defaults.
- `HUMAN_GATE.png` — the original four-route bake-off sheet. Row C (3D) is
  stale: it predates the AgX fix.

## Licence

These outputs are generated by SDXL base 1.0 (CreativeML Open RAIL++-M) with
the `nerijs/pixel-art-xl` LoRA. No third-party art is redistributed here — the
CC0/CC-BY-SA corpus used to *calibrate* the colour cap is not included in this
repo, only the statistics derived from it.
