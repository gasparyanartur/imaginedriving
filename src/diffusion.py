from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms.v2 as tvtf2
from torch import Tensor

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline

from src.utils import get_device
from src.data import read_image, save_image


@dataclass
class ModelId:
    sd_v2_1 = "stabilityai/stable-diffusion-2-1"
    sd_v1_5 = "runwayml/stable-diffusion-v1-5"
    sdxl_ref_v1_0 = "stabilityai/stable-diffusion-xl-refiner-1.0"


sd_models = {ModelId.sd_v1_5, ModelId.sd_v2_1}

sdxl_models = {ModelId.sdxl_ref_v1_0}


class ImageToImageDiffusionModel:
    def __init__(
        self,
        model_id: str = ModelId.sdxl_ref_v1_0,
        device_name: str = None,
        low_mem_mode: bool = False,
        post_processing=None,
    ) -> None:
        device = get_device() if device_name is None else torch.device(device_name)

        if model_id in sd_models:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, variant="fp16"
            )

        elif model_id in sdxl_models:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, variant="fp16"
            )

        else:
            raise NotImplementedError

        if low_mem_mode:
            pipe.enable_sequential_cpu_offload()

        else:
            pipe = pipe.to(device)

        if post_processing is None:
            post_processing = tvtf2.Compose([tvtf2.PILToTensor()])

        self.pipe = pipe
        self.device = device
        self.post_processing = post_processing

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


    def forward(
        self,
        image: Tensor,
        prompt: str = "",
        strength: float = 0.1,
        guidance_scale: float = 0.0,
        num_steps: int = 50,
        generator: torch.Generator | Iterable[torch.Generator] = None,
        seeds: Iterable[int] = None,
    ) -> Tensor:
        if generator is None:
            if seeds is not None:
                generator = [torch.Generator(self.device).manual_seed(s) for s in seeds]

        batch_size = (len(generator) if generator else 1)
        if batch_size > 1:
            if isinstance(prompt, str):
                prompt = [prompt] 
            
            if len(prompt) == 1:
                prompt = prompt * batch_size

            if len(image.shape) < 4:
                image = image.expand(batch_size, *image.shape)

        print(len(prompt), len(generator), len(image))
        img = self.pipe(
            image=image,
            prompt=prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
            batch_size=batch_size     
        )
        img = self.post_processing(img)
        return torch.stack(img.images)


def diffuse_images_to_dir(
    model: ImageToImageDiffusionModel, img_paths: Iterable[Path], dst_dir: Path
):
    dst_dir.mkdir(exist_ok=True, parents=True)
    for src_img_path in img_paths:
        src_img = read_image(src_img_path)
        dst_img = model.forward(src_img)

        save_image(dst_dir / src_img_path.name, dst_img)
