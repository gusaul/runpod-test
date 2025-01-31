from faster_whisper import WhisperModel

def transcribe_audio_to_srt(audio_path, srt_output="output.srt", model_size="large-v2"):
    """
    Transcribes an audio file and generates an SRT subtitle file using Faster-Whisper.

    Args:
        audio_path (str): Path to the audio file.
        srt_output (str): Path to save the generated subtitle file.
        model_size (str): Whisper model size ("large-v2" for best accuracy).
    """
    
    # Load the Faster-Whisper model
    model = WhisperModel(model_size, compute_type="float16")  # Use float16 for speed
    
    # Transcribe the audio
    segments, _ = model.transcribe(audio_path, word_timestamps=True)

    # Open SRT file for writing
    with open(srt_output, "w", encoding="utf-8") as srt_file:
        for idx, segment in enumerate(segments, start=1):
            start_time = segment.start
            end_time = segment.end
            text = segment.text.strip()

            # Convert timestamps to SRT format
            start_srt = f"{int(start_time // 3600):02}:{int(start_time % 3600 // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
            end_srt = f"{int(end_time // 3600):02}:{int(end_time % 3600 // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"

            # Write to file
            srt_file.write(f"{idx}\n{start_srt} --> {end_srt}\n{text}\n\n")

    print(f"Subtitle file saved as {srt_output}")

# Example usage
transcribe_audio_to_srt("combined_audio.wav", "subtitles.srt")