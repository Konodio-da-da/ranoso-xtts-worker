FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
# Auto-accept the Coqui CPML license terms
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

# Pre-download weights with TOS auto-agreed
RUN python3 -c 'from TTS.api import TTS; TTS("tts_models/multilingual/multi-dataset/xtts_v2")'

COPY rp_handler.py /app/rp_handler.py

CMD ["python", "-u", "/app/rp_handler.py"]
