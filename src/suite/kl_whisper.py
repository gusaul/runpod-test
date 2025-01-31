from faster_whisper import WhisperModel

from kl_log import logger

def TranscribeAudioToSrt(audio_path: str, srt_output: str, model_size: str = "large-v2"):
    """
    Faster-Whisper-based transcription with word-level timestamps. Writes out an SRT file.
    """
    logger.info(f"Transcribing audio to SRT with Faster-Whisper ({model_size})...")
    model = WhisperModel(model_size, compute_type="float16")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)

    with open(srt_output, "w", encoding="utf-8") as fp:
        for idx, seg in enumerate(segments, start=1):
            start_time = seg.start
            end_time = seg.end
            text = seg.text.strip()

            start_srt = (f"{int(start_time // 3600):02d}:"
                         f"{int(start_time % 3600 // 60):02d}:"
                         f"{int(start_time % 60):02d},"
                         f"{int((start_time % 1)*1000):03d}")
            end_srt = (f"{int(end_time // 3600):02d}:"
                       f"{int(end_time % 3600 // 60):02d}:"
                       f"{int(end_time % 60):02d},"
                       f"{int((end_time % 1)*1000):03d}")

            fp.write(f"{idx}\n{start_srt} --> {end_srt}\n{text}\n\n")

    logger.info(f"Subtitle file saved: {srt_output}")