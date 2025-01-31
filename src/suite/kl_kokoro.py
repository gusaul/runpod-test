import os
import uuid
from typing import List
from kokoro import KPipeline
import soundfile as sf
from pydub import AudioSegment

from kl_log import logger

def GenerateTTSAudio(
    text: str,
    lang_code: str,
    voice: str,
    output_dir: str
) -> str:
    """
    Generates TTS audio from the given text. Kokoro might produce multiple .wav segments
    which we then combine into one.
    Returns path to the combined file.
    """
    pipeline = KPipeline(lang_code=lang_code)

    segment_files = []
    logger.info("Generating TTS with Kokoro pipeline...")
    generator = pipeline(text, voice=voice, speed=1)
    for i, (graphemes, phonemes, audio) in enumerate(generator):
        seg_path = os.path.join(output_dir, f"tts_segment_{i}.wav")
        sf.write(seg_path, audio, 24000)
        segment_files.append(seg_path)

    # Combine them into one
    combined_path = os.path.join(output_dir, "combined_audio.wav")
    combine_audio_files(segment_files, combined_path)
    logger.info(f"TTS completed. Combined audio saved: {combined_path}")
    return combined_path


def combine_audio_files(files: List[str], output_file: str) -> None:
    """
    Uses pydub to combine multiple .wav files into one.
    """
    combined = AudioSegment.silent(duration=0)
    for wav in files:
        seg = AudioSegment.from_wav(wav)
        combined += seg
    combined.export(output_file, format="wav")
