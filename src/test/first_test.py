import os
from TTS.api import TTS
from diffusers import StableDiffusionPipeline
import torch
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips

# === Configuration === #
script_text = """
Welcome to our AI-powered video tutorial. 
In this video, we'll explore how AI can revolutionize your content creation journey.
"""
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Prompts for visual generation
prompts = [
    "A futuristic city skyline at sunset",
    "A robot working on a computer in a modern office",
    "An AI assistant explaining a concept on a digital screen"
]

# === Step 1: Text-to-Speech (TTS) === #
def generate_audio(script, output_path):
    print("Generating audio...")
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=True)
    tts.tts_to_file(text=script, file_path=output_path)

audio_path = os.path.join(output_dir, "output_audio.wav")
generate_audio(script_text, audio_path)

# === Step 2: Text-to-Image (Stable Diffusion) === #
def generate_images(prompts, output_dir):
    print("Generating images...")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe.to("cuda")
    image_paths = []
    for i, prompt in enumerate(prompts):
        image = pipe(prompt).images[0]
        image_path = os.path.join(output_dir, f"output_image_{i}.png")
        image.save(image_path)
        image_paths.append(image_path)
    return image_paths

image_paths = generate_images(prompts, output_dir)

# === Step 3: Assemble Video === #
def create_video(image_paths, audio_path, output_path):
    print("Creating video...")
    audio = AudioFileClip(audio_path)
    clips = []
    for image_path in image_paths:
        clip = ImageClip(image_path, duration=5)  # 5 seconds per image
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")
    video = video.set_audio(audio)
    video.write_videofile(output_path, fps=24)

video_path = os.path.join(output_dir, "final_video.mp4")
create_video(image_paths, audio_path, video_path)

print(f"Video created successfully: {video_path}")