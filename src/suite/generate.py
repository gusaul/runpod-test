import os
import uuid
import json
import argparse
from typing import Dict, Any

import torch

from kl_schema import INPUT_SCHEMA, ValidateUserInput
from kl_log import logger
from kl_sdxl import SDXL_MODELS
from kl_kokoro import GenerateTTSAudio
from kl_whisper import TranscribeAudioToSrt
from kl_video_built import CreateVideoWithSubtitles

# ------------------------------------------------------------------------------------
#                 Orchestrating the Entire Pipeline
# ------------------------------------------------------------------------------------
def run_entire_pipeline(validated_input: Dict[str, Any]):
    """
    1) TTS -> single WAV
    2) Generate images for each prompt (base + refiner)
    3) Transcribe -> SRT
    4) Create final video with random Ken Burns
    """

    # Unpack
    story_script = validated_input["story_script"]
    lang_code = validated_input["lang_code"]
    voice = validated_input["voice"]
    image_prompts = validated_input["image_prompts"]
    width, height = validated_input["video_resolution"]
    img_width, img_height = validated_input["image_resolution"]
    job_id = validated_input["job_id"] or str(uuid.uuid4())
    output_video_name = validated_input["output_video_name"]
    video_fps = validated_input["video_fps"]

    # SDXL
    scheduler_name = validated_input["scheduler"]
    steps_base = validated_input["num_inference_steps_base"]
    steps_refiner = validated_input["num_inference_steps_refiner"]
    guidance_scale = validated_input["guidance_scale"]
    high_noise_frac = validated_input["high_noise_frac"]
    strength = validated_input["strength"]
    negative_prompt = validated_input["sdxl_negative_prompt"]
    seed = validated_input["seed"]

    os.makedirs(job_id, exist_ok=True)
    output_dir = os.path.join(".", job_id)

    # 1. TTS
    audio_path = GenerateTTSAudio(
        text=story_script,
        lang_code=lang_code,
        voice=voice,
        output_dir=output_dir
    )

    torch.cuda.empty_cache()

    # 2. Generate images
    generated_images = []
    for i, prompt in enumerate(image_prompts):
        img = SDXL_MODELS.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            scheduler_name=scheduler_name,
            num_inference_steps_base=steps_base,
            num_inference_steps_refiner=steps_refiner,
            guidance_scale=guidance_scale,
            high_noise_frac=high_noise_frac,
            strength=strength,
            seed=seed,
            width=img_width,
            height=img_height
        )
        save_path = os.path.join(job_id, f"generated_{i}.png")
        img.save(save_path)
        generated_images.append(save_path)
        torch.cuda.empty_cache()

    # 3. Subtitles
    srt_path = os.path.join(job_id, "subtitles.srt")
    TranscribeAudioToSrt(audio_path, srt_path, model_size="large-v2")
    torch.cuda.empty_cache()

    # 4. Create video
    CreateVideoWithSubtitles(
        image_paths=generated_images,
        audio_path=audio_path,
        srt_path=srt_path,
        output_path=os.path.join(job_id, output_video_name),
        video_fps=video_fps,
        width=width,
        height=height
    )
    torch.cuda.empty_cache()
    logger.info("Pipeline complete!")


# ------------------------------------------------------------------------------------
#                                 MAIN
# ------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI Video Creation Pipeline")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to a JSON file containing pipeline parameters."
    )
    args = parser.parse_args()

    # Load JSON from file
    with open(args.input_json, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    # Validate
    validated_input = ValidateUserInput(user_data, INPUT_SCHEMA)

    # Run pipeline
    run_entire_pipeline(validated_input)


if __name__ == "__main__":
    main()