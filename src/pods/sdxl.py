import os
import concurrent.futures
import logging

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import load_image

from rp_schemas import INPUT_SCHEMA

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

torch.cuda.empty_cache()

# ----------------------------- Validation Function ----------------------------- #
def validate(input_data, schema):
    validated_input = {}
    errors = []

    for key, rules in schema.items():
        if key in input_data:
            value = input_data[key]

            # Check type
            if not isinstance(value, rules['type']):
                errors.append(f"Invalid type for '{key}'. Expected {rules['type'].__name__}.")
                continue

            # Check constraints
            if 'constraints' in rules and not rules['constraints'](value):
                errors.append(f"Value for '{key}' does not satisfy constraints.")
                continue

            validated_input[key] = value
        else:
            # Use default if provided
            if 'default' in rules:
                validated_input[key] = rules['default']
            elif rules.get('required', False):
                errors.append(f"Missing required input: '{key}'.")

    return {'validated_input': validated_input, 'errors': errors} if errors else {'validated_input': validated_input}

# ------------------------------- Model Handler ------------------------------ #
class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        base_pipe = base_pipe.to("cuda", silence_dtype_warnings=True)
        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe

    def load_refiner(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        refiner_pipe = refiner_pipe.to("cuda", silence_dtype_warnings=True)
        refiner_pipe.enable_xformers_memory_efficient_attention()
        return refiner_pipe

    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_base = executor.submit(self.load_base)
            future_refiner = executor.submit(self.load_refiner)

            self.base = future_base.result()
            self.refiner = future_refiner.result()


MODELS = ModelHandler()

# ---------------------------------- Helper ---------------------------------- #
def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

def _save_images_locally(images, job_id):
    output_dir = f"./{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []

    for index, image in enumerate(images):
        image_path = os.path.join(output_dir, f"{index}.png")
        image.save(image_path)
        image_paths.append(image_path)

    return image_paths

@torch.inference_mode()
def generate_image(job_input):
    '''
    Generate an image from text using your Model
    '''
    validated = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated:
        return {"error": validated['errors']}
    job_input = validated['validated_input']

    generator = torch.Generator("cuda").manual_seed(job_input['seed'] or int.from_bytes(os.urandom(2), "big"))

    MODELS.base.scheduler = make_scheduler(
        job_input['scheduler'], MODELS.base.scheduler.config)

    if job_input['image_url']:
        init_image = load_image(job_input['image_url']).convert("RGB")
        output = MODELS.refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=init_image,
            generator=generator
        ).images
    else:
        latent_image = MODELS.base(
            prompt=job_input['prompt'],
            negative_prompt=job_input['negative_prompt'],
            height=job_input['height'],
            width=job_input['width'],
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            num_images_per_prompt=job_input['num_images'],
            generator=generator
        ).images

        output = MODELS.refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=latent_image,
            num_images_per_prompt=job_input['num_images'],
            generator=generator
        ).images

    return _save_images_locally(output, job_input['job_id'])

if __name__ == "__main__":
    test_input = {
        "prompt": "A highly detailed, photorealistic cinematic rendering of a hippopotamus wearing a tuxedo and playing a grand piano on a grand concert hall stage. Intricate details, realistic textures, volumetric lighting, 8k resolution, hyper-realistic, cinematic atmosphere, dramatic spotlight, golden highlights, ultra sharp focus",
        "negative_prompt": "low resolution, blurry, distorted, cartoonish, text, watermark, logo, worst quality, poorly drawn, unrealistic proportions",
        "height": 1024,
        "width": 1024,
        "scheduler": "DPMSolverMultistep",
        "num_inference_steps": 40,
        "refiner_inference_steps": 60,
        "guidance_scale": 9.0,
        "strength": 0.3,
        "num_images": 1,
        "high_noise_frac": 0.7,
        "job_id": "test_run"
    }

    result = generate_image(test_input)
    print(result)