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
    "transformers<4.46.0" \
    "numpy<2.0.0" \
    runpod \
    requests \
    soundfile \
    coqui-tts \
    huggingface_hub

# Pre-bake model checkpoint so worker never spends runtime pulling weights
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='coqui/XTTS-v2', local_dir='/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2')"

# Verify torch and TTS import cleanly during the GitHub Actions build
RUN python3 -c "import torch; import torchaudio; from TTS.api import TTS; print('PyTorch CUDA Ready:', torch.cuda.is_available())"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
