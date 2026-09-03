FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG SEED_VC_COMMIT=51383efd921027683c89e5348211d93ff12ac2a8
ARG DEMUCS_COMMIT=2883f3db65617d6d178c6ed10d869dc14e44e59b
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_RETRIES=20 \
    HF_HOME=/runpod-volume/huggingface \
    TMPDIR=/runpod-volume/tmp \
    SEED_VC_ROOT=/opt/seed-vc
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ffmpeg build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --timeout 1200 --retries 20 --upgrade pip setuptools wheel
RUN git clone https://github.com/Plachtaa/seed-vc.git /opt/seed-vc \
    && cd /opt/seed-vc \
    && git checkout "$SEED_VC_COMMIT" \
    && python3 -m pip install --timeout 1200 --retries 20 -r requirements.txt \
    && python3 -m pip install --timeout 1200 --retries 20 runpod==1.7.13
RUN git clone https://github.com/adefossez/demucs.git /opt/demucs \
    && cd /opt/demucs \
    && git checkout "$DEMUCS_COMMIT" \
    && python3 -m pip install --timeout 1200 --retries 20 .
COPY handler.py /app/handler.py
RUN python3 -m py_compile /app/handler.py
WORKDIR /opt/seed-vc
CMD ["python3","-u","/app/handler.py"]
