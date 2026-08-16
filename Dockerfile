# Base install only, deliberately: no `[generate]` extra.
#
# This image exists so an automated inspector can start the server and read its
# tool list. Pulling the CUDA stack in would make the build multi-gigabyte and
# slow, and a build that fails or times out costs both the quality score and
# search visibility. `gate_icon` and `describe_icon` need no model, and
# `generate_icons` reports the missing dependency rather than crashing the
# server at import.
FROM python:3.12-slim

WORKDIR /app

# Copy metadata first so the dependency layer is cached independently of source
# edits.
COPY pyproject.toml README.md LICENSE ./
COPY artgate_gen ./artgate_gen

RUN pip install --no-cache-dir .

# Fail the build here rather than at inspection time if the style TOMLs did not
# make it into the install.
RUN python -c "\
from pathlib import Path; \
import artgate_gen; \
specs = sorted((Path(artgate_gen.__file__).parent / 'styles').glob('*.toml')); \
assert specs, 'style specs missing from the install'; \
print(f'{len(specs)} style specs present')"

# stdio transport: the client speaks to the server over stdin/stdout.
ENTRYPOINT ["artgate-gen-mcp"]
