import os
import io
import time
import base64
import requests
import subprocess
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

def convert_to_clean_wav(input_path, output_path):
    """
    Transcodes any incoming audio format (AAC, M4A, OGG, MP3, WebM)
    into 22.05kHz 16-bit mono PCM WAV for Coqui speaker conditioning.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-acodec", "pcm_s16le",
        "-ar", "22050",
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg transcode error: {result.stderr.decode('utf-8')}")

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
    
    raw_download_path = f"/tmp/raw_ref_{job_id}"
    normalized_wav_path = f"/tmp/norm_ref_{job_id}.wav"
    output_wav_path = f"/tmp/out_{job_id}.wav"

    try:
        # Download reference audio stem if a URL is provided
        if speaker_wav_url.startswith("http://") or speaker_wav_url.startswith("https://"):
            r = requests.get(speaker_wav_url, timeout=45)
            r.raise_for_status()
            with open(raw_download_path, "wb") as f:
                f.write(r.content)
            source_file = raw_download_path
        else:
            source_file = speaker_wav_url

        # Transcode any incoming audio format to clean PCM WAV
        convert_to_clean_wav(source_file, normalized_wav_path)

        # Synthesize audio with cloned voice
        tts.tts_to_file(
            text=text,
            speaker_wav=normalized_wav_path,
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
        # Guarantee disk cleanup for all temporary files
        for p in [raw_download_path, normalized_wav_path, output_wav_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

runpod.serverless.start({"handler": handler})
