import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image(
    "hippo.png"
)
prompt = "An expert hippo pianist playing piano in front of thousands audience with a funny expression yet professional gesture."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_frames=161,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)

'''
ERROR:
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 218.00 MiB. GPU 0 has a total capacty of 15.60 GiB of which 158.81 MiB is free. Process 2110062 has 15.44 GiB memory in use. Of the allocated memory 14.60 GiB is allocated by PyTorch, and 733.93 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
'''