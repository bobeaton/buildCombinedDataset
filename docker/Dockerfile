# Build context must be the buildCombinedDataset project root (not this docker/ folder),
# since requirements.txt lives there -- see buildDocker.ps1, which handles this.
FROM python:3.11-slim

# build-essential (gcc/cc) is required at *runtime*, not just build time: recent torch
# dispatches some ops (e.g. bmm inside SpeechT5's attention) through a Triton-JIT-compiled
# kernel, which needs a C compiler available in the container to build itself on first use.
# Confirmed via: `RuntimeError: Failed to find C compiler` from triton/runtime/build.py.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV CC=gcc

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/server.py docker/settings.py docker/openapi.yaml ./
COPY docker/templates ./templates

# Everything the model needs (src/, checkpoints/, data/, model_init*/, tokenizer*/) is
# volume-mounted at runtime rather than baked into the image -- see buildDocker.ps1.
# This keeps the image itself small and lets code/model updates on the host take effect
# on container restart without a rebuild.
# Optional additional mounts: WAV_DIR (raw reference recordings, for synthesize compare
# mode) and CHARACTER_MAPPING_DIR (characterMapping.json, for synthesize-file).

EXPOSE 8000
CMD ["python", "server.py"]
