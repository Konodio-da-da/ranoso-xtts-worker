import os
import io
import time
import base64
import requests
import torch
import torchaudio
import runpod

# Auto-accept Coqui XTTS CPML License non-interactively
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing voice cloning model on {DEVICE}...")

# Initialize XTTS-v2 with automatic terms acceptance
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(DEVICE)
print("Model loaded successfully into VRAM.")

def handler(job):
    job_input = job.get("input", {})
    
    text = job_input.get("text", "").strip()
    speaker_wav_url = job_input.get("speaker_wav_url")
    language = job_input.get("language", "en")
    
    if not text:
        return {"status": "error", "message": "Missing 'text' in input payload"}
    if not speaker_wav_url:
        return {"status": "error", "message": "Missing 'speaker_wav_url' reference stem"}

    start_time = time.time()
    job_id = job.get("id", str(int(time.time())))
    temp_speaker_path = f"/tmp/ref_{job_id}.wav"
    output_wav_path = f"/tmp/out_{job_id}.wav"

    try:
        # Download reference audio stem if a URL is provided
        if speaker_wav_url.startswith("http://") or speaker_wav_url.startswith("https://"):
            r = requests.get(speaker_wav_url, timeout=45)
            r.raise_for_status()
            with open(temp_speaker_path, "wb") as f:
                f.write(r.content)
            reference_audio = temp_speaker_path
        else:
            reference_audio = speaker_wav_url

        # Synthesize audio with cloned voice
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
    finally:
        # Guarantee disk cleanup
        if os.path.exists(temp_speaker_path):
            try:
                os.remove(temp_speaker_path)
            except OSError:
                pass
        if os.path.exists(output_wav_path):
            try:
                os.remove(output_wav_path)
            except OSError:
                pass

runpod.serverless.start({"handler": handler})
