

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
# FROM debian:stable-slim
WORKDIR /curator
RUN set -x && \
    apt-get update \
    && apt-get install -y git \
    wget \
    zip \
    unzip \
    libbz2-dev \
    libncurses-dev \
    libssl-dev \
    libffi-dev \
    libreadline6-dev \
    libsqlite3-dev \
    liblzma-dev \
    libxml2-dev \
    xz-utils \
    libxmlsec1-dev \
    python3-venv \
    python3-pip

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="/curator/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.11 
RUN  pyenv global 3.11
    
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

COPY pyproject.toml .

COPY src /curator/src/

RUN pip install .