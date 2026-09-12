# Speech to Text (Qwen3-ASR-0.6B)

A small Gradio web app that turns speech into text. Record from your microphone or drop
in an audio file; transcription runs on [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
through Hugging Face Inference Providers, so nothing is downloaded or run locally.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env    # then edit .env and paste your HF token
```

The token needs the **"Make calls to Inference Providers"** permission
(https://huggingface.co/settings/tokens). `.env` is gitignored - never commit it.

## Run

```powershell
.\.venv\Scripts\python.exe app.py
```

Opens http://127.0.0.1:7861. Click the microphone to record (the browser will ask for
permission the first time) or drag an audio file onto the upload area, then press
**Transcribe**. The transcript has a copy button and a `.txt` download.

## Files

| File | Purpose |
| --- | --- |
| `asr.py` | Core transcription. UI-free, importable from anything. `transcribe(path) -> str`. |
| `app.py` | Gradio UI and the button handler. |
| `.env` | Your `HF_TOKEN` (gitignored). |

Use it from a script or another module:

```python
from asr import transcribe, TranscriptionError

try:
    print(transcribe("sample1.flac"))
except TranscriptionError as err:
    print(err)
```

## Limits

- Formats: `.flac .wav .mp3 .m4a .ogg .webm`, up to 25 MB (`MAX_FILE_MB` in `asr.py`).
- Audio is sent as-is - no local resampling or chunking, so no ffmpeg needed. Very long
  recordings may hit provider limits; split them yourself if so.
- Language is auto-detected. The HF ASR task endpoint exposes no portable language
  parameter for this model, so there is deliberately no language selector.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `HF_TOKEN not set` | Create `.env` from `.env.example` and restart. |
| `Hugging Face rejected the token` | Token is invalid or missing the Inference Providers permission. |
| `No Inference Provider is currently serving...` | Provider availability changed; check the model page's "Inference Providers" section. |
| `Rate limited` / `model is loading` | Wait a few seconds and retry. |
| Port 7861 already in use | Change `server_port` at the bottom of `app.py`. |

Note: `model=` must be the plain repo id `Qwen/Qwen3-ASR-0.6B`. The `:preferred` suffix
from the model page's snippet is rejected by `huggingface_hub >= 1.0`, which validates the
value as a repo id - `provider="auto"` already handles provider selection.

## Security

If a token has ever been pasted into a chat, an issue, or a shared terminal, rotate it at
https://huggingface.co/settings/tokens.
