# syntax=docker/dockerfile:1

########################
# Stage 1: build
########################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

WORKDIR /curator

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build dependencies + Python 3.12
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        git \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" | gpg --dearmor -o /etc/apt/keyrings/deadsnakes.gpg 2>/dev/null \
       || curl -fsSL "https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/dists/jammy/Release.gpg" -o /etc/apt/keyrings/deadsnakes.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade packaging tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel hatchling editables

# Install prebuilt llama-cpp-python CUDA 12.4 wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary \
        "llama-cpp-python>=0.3.8" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Copy metadata and source code
COPY pyproject.toml README.md* LICENSE* ./
COPY src /curator/src/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
        .

# Clean up bytecode and tests without corrupting ELF binaries (NO strip on .so files)
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true


########################
# Stage 2: runtime
########################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install base runtime packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        libgomp1 \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" | gpg --dearmor -o /etc/apt/keyrings/deadsnakes.gpg 2>/dev/null \
       || curl -fsSL "https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/dists/jammy/Release.gpg" -o /etc/apt/keyrings/deadsnakes.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
    && apt-get purge -y curl gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3

# Copy virtual environment intact
COPY --from=builder /opt/venv /opt/venv

WORKDIR /curator

RUN useradd --create-home --shell /bin/bash curator \
    && chown -R curator:curator /curator /opt/venv

USER curator

ENTRYPOINT ["/opt/venv/bin/python3", "-m", "mirna_curator.main"]