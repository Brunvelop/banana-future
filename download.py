import os
import torch
from diffusers import DiffusionPipeline

def download_model():

    model = DiffusionPipeline.from_pretrained(
        "nitrosocke/Future-Diffusion",
        torch_dtype=torch.float16
    ) 
    
if __name__ == "__main__":
    download_model()
