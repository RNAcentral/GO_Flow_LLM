

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS cuda_build_stage_1
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

RUN git clone https://github.com/abetlen/llama-cpp-python.git --branch v0.3.2 . \
&& git submodule update --init --recursive

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="/curator/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.11 
RUN  pyenv global 3.11

FROM cuda_build_stage_1 AS cuda_build
ARG CMAKE_ARGS="-DGGML_CUDA=on"
#  -DCMAKE_CUDA_ARCHITECTURES=all -DGGML_CUDA_FORCE_MMQ=on"
RUN pip install build wheel
RUN python -m build --wheel 

FROM debian:stable-slim AS final
COPY --from=cuda_build llama_cpp_python-0.3.2-cp311-cp311-linux_x86_64.whl .

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

RUN pip install llama_cpp_python-0.3.2-cp311-cp311-linux_x86_64.whl

COPY pyproject.toml .

COPY src /curator/src/

RUN pip install .