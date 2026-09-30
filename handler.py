import os
import io
import time
import base64
import torch
import torchaudio
import runpod
from TTS.api import TTS

# Pre-load model weights into GPU VRAM during container initialization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing voice cloning model on {DEVICE}...")

# Using Coqui XTTS-v2 for zero-shot speaker latent conditioning
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
print("Model loaded successfully into VRAM.")

def handler(job):
    job_input = job.get("input", {})
    
    text = job_input.get("text", "").strip()
    speaker_wav_url = job_input.get("speaker_wav_url")
    language = job_input.get("language", "en")
    
    if not text:
        return {"error": "Missing 'text' in input payload"}
    if not speaker_wav_url:
        return {"error": "Missing 'speaker_wav_url' reference stem"}

    start_time = time.time()

    try:
        # Download reference audio stem if a URL is provided
        temp_speaker_path = f"/tmp/ref_{job['id']}.wav"
        if speaker_wav_url.startswith("http://") or speaker_wav_url.startswith("https://"):
            import requests
            r = requests.get(speaker_wav_url, timeout=30)
            with open(temp_speaker_path, "wb") as f:
                f.write(r.content)
            reference_audio = temp_speaker_path
        else:
            reference_audio = speaker_wav_url

        # Synthesize audio with cloned voice
        output_wav_path = f"/tmp/out_{job['id']}.wav"
        
        tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            language=language,
            file_path=output_wav_path,
            split_sentences=True
        )

        # Calculate exact generated audio duration
        info = torchaudio.info(output_wav_path)
        duration_seconds = round(info.num_frames / info.sample_rate, 2)
        duration_minutes = round(duration_seconds / 60, 2)

        # Read output and encode to base64
        with open(output_wav_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Cleanup temp disk files
        if os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)
        if os.path.exists(output_wav_path):
            os.remove(output_wav_path)

        compute_latency = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "audio_base64": audio_base64,
            "format": "wav",
            "sample_rate": info.sample_rate,
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "compute_latency_seconds": compute_latency
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

runpod.serverless.start({"handler": handler})
