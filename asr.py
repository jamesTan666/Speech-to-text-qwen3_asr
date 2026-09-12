"""Core speech-to-text logic: Qwen3-ASR-0.6B via the Hugging Face Inference API.

UI-free on purpose so it can be imported by app.py, a CLI, or any other caller.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"  # provider="auto" picks a serving provider; no ":preferred" suffix
# (huggingface_hub >=1.0 validates model as a plain repo id and rejects "<repo>:preferred")
MAX_FILE_MB = 25
SUPPORTED_EXTS = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".webm"}

load_dotenv()

_client = None


class TranscriptionError(Exception):
    """Raised with a human-readable message instead of leaking a traceback."""


def get_client():
    """Build the InferenceClient once, lazily, so importing this module never fails."""
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise TranscriptionError(
                "HF_TOKEN not set - copy .env.example to .env and put your "
                "Hugging Face token in it, then restart the app."
            )
        _client = InferenceClient(provider="auto", api_key=token)
    return _client


def _extract_text(output) -> str:
    """Pull the transcript out of whatever shape the provider returned."""
    text = getattr(output, "text", None)
    if text is None and isinstance(output, dict):
        text = output.get("text")
    if text is None:
        text = str(output)
    return text.strip()


def _friendly(err: Exception) -> str:
    """Turn an API failure into something worth showing a user."""
    raw = str(err)
    status = getattr(getattr(err, "response", None), "status_code", None)
    if status is None:
        for code in (401, 403, 404, 413, 429, 503):
            if str(code) in raw:
                status = code
                break

    if isinstance(err, StopIteration) or status == 404:
        hint = (
            "No Inference Provider is currently serving Qwen/Qwen3-ASR-0.6B. "
            "Check the model page's 'Inference Providers' section."
        )
    elif status in (401, 403):
        hint = (
            "Hugging Face rejected the token. Make sure it is valid and has the "
            "'Make calls to Inference Providers' permission enabled."
        )
    elif status == 413:
        hint = f"The provider rejected the file as too large (limit is around {MAX_FILE_MB} MB)."
    elif status == 429:
        hint = "Rate limited by Hugging Face. Wait a moment and try again."
    elif status == 503:
        hint = "The model is loading or the provider is busy. Try again in a few seconds."
    elif "timed out" in raw.lower() or "timeout" in raw.lower():
        hint = "The request timed out. Try again, or use a shorter clip."
    else:
        hint = "Transcription failed."

    return f"{hint}\n\nDetails: {type(err).__name__}: {raw}"


def transcribe(audio_path) -> str:
    """Transcribe an audio file and return the text.

    Raises TranscriptionError with a readable message on any failure.
    """
    if not audio_path:
        raise TranscriptionError("No audio provided - record a clip or upload a file first.")

    path = Path(audio_path)
    if not path.is_file():
        raise TranscriptionError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise TranscriptionError(
            f"Unsupported audio format '{ext or path.name}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        raise TranscriptionError("That audio file is empty (0 bytes).")
    if size_mb > MAX_FILE_MB:
        raise TranscriptionError(
            f"File is {size_mb:.1f} MB, over the {MAX_FILE_MB} MB limit. Use a shorter clip."
        )

    client = get_client()
    try:
        output = client.automatic_speech_recognition(str(path), model=MODEL_ID)
    except TranscriptionError:
        raise
    except Exception as err:  # noqa: BLE001 - every failure becomes a readable message
        raise TranscriptionError(_friendly(err)) from err

    text = _extract_text(output)
    if not text:
        raise TranscriptionError("The model returned an empty transcript - was there any speech?")
    return text
