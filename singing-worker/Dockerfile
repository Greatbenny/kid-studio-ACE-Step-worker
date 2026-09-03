FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG SEED_VC_COMMIT=51383efd921027683c89e5348211d93ff12ac2a8
ARG DEMUCS_COMMIT=2883f3db65617d6d178c6ed10d869dc14e44e59b
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/runpod-volume/huggingface TMPDIR=/runpod-volume/tmp SEED_VC_ROOT=/opt/seed-vc
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ffmpeg build-essential libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/Plachtaa/seed-vc.git /opt/seed-vc && cd /opt/seed-vc && git checkout "$SEED_VC_COMMIT" && pip install -r requirements.txt && pip install runpod==1.7.13
RUN git clone https://github.com/adefossez/demucs.git /opt/demucs && cd /opt/demucs && git checkout "$DEMUCS_COMMIT" && pip install .
COPY handler.py /app/handler.py
RUN python3 -m py_compile /app/handler.py
WORKDIR /opt/seed-vc
CMD ["python3","-u","/app/handler.py"]
