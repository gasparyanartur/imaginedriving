from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline

from src.utils import get_device
from src.data import read_PIL_img, save_PIL_img


@dataclass
class ModelId:
    sd_v2_1 = "stabilityai/stable-diffusion-2-1"
    sd_v1_5 = "runwayml/stable-diffusion-v1-5"
    sdxl_ref_v1_0 = "stabilityai/stable-diffusion-xl-refiner-1.0"


sd_models = {
    ModelId.sd_v1_5,
    ModelId.sd_v2_1
}

sdxl_models = {
    ModelId.sdxl_ref_v1_0
}


def init_sd_pipe(model_id: str, device: torch.device):
    return StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to(device)


def init_sdxl_pipe(model_id: str, device: torch.device):
    return StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to(device)


class ImageToImageDiffusionModel:
    def __init__(self, model_id: str = ModelId.sdxl_ref_v1_0, device_name: str = None) -> None:
        self.device = get_device() if device_name is None else torch.device(device_name)
        
        if model_id in sd_models:
            self.pipe = init_sd_pipe(model_id, self.device)

        elif model_id in sdxl_models:
            self.pipe = init_sdxl_pipe(model_id, self.device)

        else:
            raise NotImplementedError

    def forward(self, image: Image, prompt: str = "", strength: float=0.1, guidance_scale: float=0.0, num_steps: int = 50):
        img = self.pipe(
            image=image,
            prompt=prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps
        )
        return img.images[0]


def diffuse_images_to_dir(model: ImageToImageDiffusionModel, img_paths: Iterable[Path], dst_dir: Path):
    dst_dir.mkdir(exist_ok=True, parents=True)
    for src_img_path in img_paths:
        src_img = read_PIL_img(src_img_path)
        dst_img = model.forward(src_img)

        save_PIL_img(dst_dir / src_img_path.name, dst_img)
