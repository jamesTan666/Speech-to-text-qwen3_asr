"""Gradio web UI for speech-to-text with Qwen3-ASR-0.6B."""

import tempfile
import time
from pathlib import Path

import gradio as gr

from asr import MAX_FILE_MB, SUPPORTED_EXTS, TranscriptionError, transcribe

OUT_DIR = Path(tempfile.gettempdir()) / "qwen3_asr_transcripts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run(audio_path):
    """Transcribe and return (status_markdown, transcript, download_file)."""
    if not audio_path:
        return "Record a clip or upload a file, then press **Transcribe**.", "", None

    started = time.time()
    try:
        text = transcribe(audio_path)
    except TranscriptionError as err:
        return f"**Error**\n\n{err}", "", None

    out_file = OUT_DIR / f"transcript_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    out_file.write_text(text, encoding="utf-8")

    elapsed = time.time() - started
    words = len(text.split())
    status = f"Done in {elapsed:.1f}s - {words} word{'s' if words != 1 else ''}."
    return status, text, gr.update(value=str(out_file), visible=True)


with gr.Blocks(title="Qwen3-ASR Speech to Text") as demo:
    gr.Markdown(
        "# Qwen3-ASR Speech to Text\n"
        "Record from your microphone or drop in an audio file. "
        "Transcription runs on `Qwen/Qwen3-ASR-0.6B` via Hugging Face Inference Providers.\n\n"
        f"Supported: {', '.join(sorted(SUPPORTED_EXTS))} - up to {MAX_FILE_MB} MB. "
        "Language is auto-detected."
    )

    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Audio",
            )
            transcribe_btn = gr.Button("Transcribe", variant="primary")
            clear_btn = gr.ClearButton(value="Clear")

        with gr.Column():
            status = gr.Markdown("Record a clip or upload a file, then press **Transcribe**.")
            transcript = gr.Textbox(
                label="Transcript",
                lines=12,
                buttons=["copy"],  # gradio>=6 replaced show_copy_button with buttons=[...]
                placeholder="The transcript will appear here...",
            )
            download = gr.File(label="Download .txt", visible=False)

    transcribe_btn.click(
        fn=run,
        inputs=[audio_in],
        outputs=[status, transcript, download],
        api_name="transcribe",
    )
    clear_btn.add([audio_in, transcript, download])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, inbrowser=True)
