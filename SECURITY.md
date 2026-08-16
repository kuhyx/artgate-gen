# Security Policy

## Supported versions

The `main` branch is the supported version. Fixes land there and go out in the
next tagged release.

## Reporting a vulnerability

Report privately through GitHub's [security advisory
form](https://github.com/kuhyx/artgate-gen/security/advisories/new) rather than
a public issue. Expect an acknowledgement within a week.

## Threat model

Worth stating plainly, because this is a tool that writes files and shells out:

- **The MCP server reads and writes caller-supplied paths.** `generate_icons`
  creates `out_dir` and writes PNGs into it; `gate_icon` and `describe_icon`
  read any path they are given. The server does not sandbox these paths, so it
  has whatever filesystem access its own process has. Run it as a user whose
  reach you are comfortable with, particularly when the caller is an
  autonomous agent.
- **Model weights are downloaded from Hugging Face** on first use, from the
  three repositories named in `artgate_gen/pipeline.py`. They are executed as
  model weights, not as code, but they are third-party artifacts.
- **Prompts are passed to the model verbatim.** Nothing here filters what is
  generated; the deterministic gate measures canvas, palette, alpha, scale,
  contrast and margin, and makes no judgement about content.
- **The gate never adjudicates with a language model.** Verdicts come from
  measurement and are reproducible.

## What is not a vulnerability

The gate passing an ugly icon. It measures conformance to a written spec, not
quality — that judgement stays with a human.
