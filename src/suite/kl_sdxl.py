import os
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

from kl_log import logger

# ------------------------------------------------------------------------------------
#                       3. SDXL (Base + Refiner) Model Handler
# ------------------------------------------------------------------------------------
def make_scheduler(name: str, config):
    """
    Instantiate a scheduler from config. You can add more schedulers if needed.
    """
    schedulers = {
        "PNDM": PNDMScheduler.from_config,
        "KLMS": LMSDiscreteScheduler.from_config,
        "DDIM": DDIMScheduler.from_config,
        "K_EULER": EulerDiscreteScheduler.from_config,
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config,
    }
    if name not in schedulers:
        raise ValueError(f"Scheduler '{name}' is not supported.")
    return schedulers[name](config)


class SDXLModelHandler:
    """
    Loads the SDXL base and refiner pipelines. We do text->image with base,
    then refine that output with the refiner.
    """
    def __init__(self):
        self.base = None
        self.refiner = None
        self._load_pipelines()

    def _load_pipelines(self):
        logger.info("Loading SDXL pipelines (base + refiner)...")

        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )

        # Load base
        self.base = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False
        )
        self.base.to("cuda", silence_dtype_warnings=True)
        self.base.enable_xformers_memory_efficient_attention()

        # Load refiner
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False
        )
        self.refiner.to("cuda", silence_dtype_warnings=True)
        self.refiner.enable_xformers_memory_efficient_attention()

        logger.info("SDXL pipelines loaded successfully.")

    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        scheduler_name: str,
        num_inference_steps_base: int,
        num_inference_steps_refiner: int,
        guidance_scale: float,
        high_noise_frac: float,
        strength: float,
        seed: int,
        width: int,
        height: int
    ):
        """
        2-step generation:
          1) text -> image with base
          2) refine with refiner
        """
        # Seed handling
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)

        # Update base scheduler
        self.base.scheduler = make_scheduler(scheduler_name, self.base.scheduler.config)

        # Step 1: Base T2I
        base_output = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps_base,
            guidance_scale=guidance_scale,
            denoising_end=high_noise_frac,
            generator=generator
        )
        base_image = base_output.images[0]

        # Step 2: Refine
        refiner_output = self.refiner(
            prompt=prompt,
            image=base_image,
            num_inference_steps=num_inference_steps_refiner,
            strength=strength,
            generator=generator
        )
        refined_image = refiner_output.images[0]

        return refined_image


# Instantiate a global model handler (you could also lazy-init).
SDXL_MODELS = SDXLModelHandler()