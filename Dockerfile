FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model weights so cold-starts are faster
RUN python3 -c 'from TTS.api import TTS; TTS("tts_models/multilingual/multi-dataset/xtts_v2")'

COPY handler.py .
CMD ["python3", "-u", "handler.py"]
