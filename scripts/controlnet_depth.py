import torch 
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation, AutoImageProcessor
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from src.data import load_img_if_path
import pathlib as pl
from src.utils import show_img
from torch import Tensor

def get_dino_depth(input_img: Tensor, image_processor, model, resize: bool = True) -> Tensor:
    if resize:
        input_img = input_img * 255
    
    inputs = image_processor(images=input_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    outputs = predicted_depth

    return outputs



if __name__ == "__main__":
    image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-small-nyu")
    model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-small-nyu")
    example_img_path = 
    depth_img = get_dino_depth(example_img, image_processor, model)


