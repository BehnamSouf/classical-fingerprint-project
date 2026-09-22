# Dockerfile
FROM python:3.12-slim

# System dependencies:
# - build-essential: some scientific packages (scipy) may need to compile
# - libgl1: required by scikit-image / matplotlib backends in headless envs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency metadata first so Docker can cache this layer --
# rebuilding only re-installs dependencies when pyproject.toml changes,
# not on every source code edit.
COPY pyproject.toml .

# Install the project in editable mode with its dependencies.
# A placeholder src/ is needed for `pip install -e .` to succeed before
# the real source is copied in the next step.
RUN mkdir -p src/fingerprint_dataset && touch src/fingerprint_dataset/__init__.py
RUN pip install --no-cache-dir -e .

# Now copy the actual source code.
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/

# Data and output are expected to be mounted as volumes at runtime, not
# baked into the image (see docker run examples below).
RUN mkdir -p data output

# Default: run the test suite. Override with `docker run ... <command>`
# to run an evaluation script or an interactive shell instead.
CMD ["python", "scripts/evaluate_minutiae_matching.py"]