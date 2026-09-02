ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG ACE_STEP_COMMIT=ca1e85fe9430179831e6bc6be790c332190a3866

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_CACHE=/runpod-volume/huggingface/hub \
    ACESTEP_CHECKPOINTS_DIR=/runpod-volume/ace-step/checkpoints \
    ACESTEP_PROJECT_ROOT=/opt/ace-step \
    ACESTEP_DISABLE_TQDM=true \
    TOKENIZERS_PARALLELISM=false \
    TMPDIR=/runpod-volume/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
      build-essential \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libffi-dev \
      libsndfile1 \
      libsndfile1-dev \
      libssl-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3.11 \
      python3.11-dev \
      python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /opt/ace-step \
    && cd /opt/ace-step \
    && git checkout "${ACE_STEP_COMMIT}" \
    && uv sync --frozen --no-dev --python python3.11 \
    && uv pip install --python /opt/ace-step/.venv/bin/python runpod==1.7.13

WORKDIR /opt/ace-step
COPY handler.py /app/handler.py
RUN /opt/ace-step/.venv/bin/python -m py_compile /app/handler.py

CMD ["/opt/ace-step/.venv/bin/python", "-u", "/app/handler.py"]
