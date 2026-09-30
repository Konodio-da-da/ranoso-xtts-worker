FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV COQUI_TOS_AGREED=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    soundfile \
    coqui-tts \
    huggingface_hub

# Download model checkpoint files directly without executing CUDA code
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='coqui/XTTS-v2', local_dir='/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
