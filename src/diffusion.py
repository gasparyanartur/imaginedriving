from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms.v2 as tvtf2
from torch import Tensor

from diffusers import StableDiffusionXLImg2ImgPipeline

from src.utils import get_device
from src.data import read_image, save_image


@dataclass
class ModelId:
    sdxl_base = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"


class BaseImg2ImgModel(ABC):
    def __init__(
        self, device_name: str = None, post_processing=None, prompt=""
    ) -> None:
        device = get_device() if device_name is None else torch.device(device_name)

        if post_processing is None:
            post_processing = tvtf2.Compose([tvtf2.PILToTensor()])

        self.device = device
        self.post_processing = post_processing
        self.prompt = prompt

    def forward(
        self,
        image: Tensor,
    ):
        img = torch.stack(self._forward(image=image))
        img = self.post_processing(img)
        return img

    @abstractmethod
    def _forward(self, image: Tensor):
        raise NotImplementedError


class SDXLFull(BaseImg2ImgModel):
    def __init__(
        self,
        n_steps: int = 50,
        refiner_threshold: float = 0.7,
        strength_base: float = 0.2,
        strength_refiner: float = 0.2,
        base_model_id: str = ModelId.sdxl_base,
        refiner_model_id=ModelId.sdxl_refiner,
        device_name: str = None,
        post_processing=None,
        prompt="",
        **model_kwargs,
    ) -> None:
        super().__init__(device_name, post_processing, prompt)

        self.base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(self.device)

        self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_model_id,
            text_encoder_2=self.base_pipe.text_encoder_2,
            vae=self.base_pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)

        self.n_steps = n_steps
        self.refiner_threshold = refiner_threshold
        self.strength_base = strength_base
        self.strength_refiner = strength_refiner
        self.model_kwargs = model_kwargs

    def _forward(self, image: Tensor):
        image = self.base_pipe(
            prompt=self.prompt,
            image=image,
            num_inference_steps=self.n_steps,
            denoising_end=self.refiner_threshold,
            strength=self.strength_base,
            output_type="latent",
        ).images

        image = self.refiner_pipe(
            prompt=self.prompt,
            image=image,
            num_inference_steps=self.n_steps,
            denoising_start=self.refiner_threshold,
            strength = self.strength_refiner
        ).images

        return image


def diffuse_images_to_dir(
    model: BaseImg2ImgModel, img_paths: Iterable[Path], dst_dir: Path
):
    dst_dir.mkdir(exist_ok=True, parents=True)
    for src_img_path in img_paths:
        src_img = read_image(src_img_path)
        dst_img = model.forward(src_img)

        save_image(dst_dir / src_img_path.name, dst_img)
