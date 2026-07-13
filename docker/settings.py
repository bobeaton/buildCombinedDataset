"""Docker webservice configuration -- all overridable via `docker run -e VAR=value`.

Unlike the CLI scripts (src/infer.py, src/infer_file.py), which hardcode host paths as
constants, this server reads them from the environment since the container sees its own
mounted paths, not the host's. See buildDocker.ps1 for how these map to `-v` mounts.
"""

import os

PORT = int(os.environ.get("PORT", "8000"))

# Set to a non-empty value to require an `Authorization: <API_KEY>` header on every request.
API_KEY = os.environ.get("API_KEY", "")

# Which transcription-variant model to use when a request doesn't specify one.
DEFAULT_VARIANT = os.environ.get("DEFAULT_VARIANT", "b")

# Required: the buildCombinedDataset project (src/, checkpoints/, data/, model_init*/,
# tokenizer*/), mounted read-only from the host. Everything under /api/v1/tts/synthesize*
# needs this.
PROJECT_PATH = os.environ.get("PROJECT_PATH", "/app/project")

# Optional: raw reference wav recordings (FCBH/wavs on the host), only needed for the
# `compare` option of /api/v1/tts/synthesize/. Left unset -> that option returns a clear
# error instead of failing to start the whole server.
WAV_DIR = os.environ.get("WAV_DIR") or None

# Optional: folder containing characterMapping.json, only needed by
# /api/v1/tts/synthesize-file/. Left unset -> that endpoint returns a clear error.
CHARACTER_MAPPING_DIR = os.environ.get("CHARACTER_MAPPING_DIR") or None
CHARACTER_MAPPING_FILENAME = os.environ.get("CHARACTER_MAPPING_FILENAME", "characterMapping.json")
