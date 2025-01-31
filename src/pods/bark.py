from bark import generate_audio, SAMPLE_RATE
import soundfile as sf
import torch


@torch.inference_mode()
def text_to_speech_bark(text, output_file="output.wav", language="en"):
    print(f"Generating speech for: {text}")

    # Generate audio from text
    audio_array = generate_audio(text, language=language)

    # Save audio to a file
    sf.write(output_file, audio_array, SAMPLE_RATE)
    print(f"Audio saved to {output_file}")


if __name__ == "__main__":
    # Creative and expressive text input
    text = (
        "Welcome to the magical world of imagination! [cheerful] "
        "Today, we embark on a journey like no other. [excited] "
        "Along the way, we'll encounter fantastical creatures, breathtaking landscapes, and... [pause] "
        "unforgettable adventures! [laughs] "
        "Are you ready? Let the adventure begin! [cheerful]"
    )
    output_path = "./test_run/expressive_bark_output.wav"

    text_to_speech_bark(text, output_path)