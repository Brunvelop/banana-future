import torch
from torch import autocast
from diffusers import DiffusionPipeline
import base64
from io import BytesIO
import os

def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    model = DiffusionPipeline.from_pretrained(
        "nitrosocke/Future-Diffusion",
        torch_dtype=torch.float16
    ).to('cuda')

def _generate_latent(height, width, seed=None, device="cuda"):
    generator = torch.Generator(device=device)

    # Get a new random seed, store it and use it as the generator state
    if not seed:
        seed = generator.seed()
    generator = generator.manual_seed(seed)
    
    image_latent = torch.randn(
        (1, model.unet.in_channels, height // 8, width // 8),
        generator = generator,
        device = device
    )
    return image_latent.type(torch.float16)
    

def inference(model_inputs:dict):
    global model

    latent = _generate_latent(64*6, 64*6)
    with autocast("cuda"):
            images = model(
        prompt = "future style "+ model_inputs.get('prompt', None) +" cinematic lights, trending on artstation, avengers endgame, emotional",
        height=64*6,
        width=64*6,
        num_inference_steps = 20,
        guidance_scale = 7.5,
        negative_prompt="duplicate heads bad anatomy extra legs text",
        num_images_per_prompt = 1,
        return_dict=False,
        latents = latent
    )
    image = images[0][0]
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {'image_base64': image_base64}
