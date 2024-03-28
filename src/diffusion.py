from typing import Any
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import logging

import torch
from torch import Tensor

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    retrieve_latents,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL

from src.utils import (
    get_device,
    validate_same_len,
    batch_if_not_iterable,
)
from src.data import (
    save_image,
    DynamicDataset,
    suffixes
)
from src.configuration import save_yaml


default_prompt = "dashcam recording, urban driving scene, video, autonomous driving, detailed cars, traffic scene, pandaset, kitti, high resolution, realistic, detailed, camera video, dslr, ultra quality, sharp focus, crystal clear, 8K UHD, 10 Hz capture frequency 1/2.7 CMOS sensor, 1920x1080"
default_negative_prompt = "face, human features, unrealistic, artifacts, blurry, noisy image, NeRF, oil-painting, art, drawing, poor geometry, oversaturated, undersaturated, distorted, bad image, bad photo"


@dataclass
class ModelId:
    sdxl_base = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"
    sdxl_turbo = "stabilityai/sdxl-turbo"


class DiffusionModel(ABC):
    load_model = None

    @abstractmethod
    def diffuse_sample(self, sample: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def diffuse_to_dir(
        self, src_dataset: DynamicDataset, dst_dir: Path, **kwargs
    ) -> None:
        dst_dir.mkdir(exist_ok=True, parents=True)

        for sample in src_dataset:
            diff_img = self.diffuse_sample(sample, **kwargs)["rgb"]
            dst_path = dst_dir / sample["dataset"] / sample["scene"] / sample["sample"]
            dst_path = dst_path.with_suffix(suffixes["rgb", src_dataset.name])
            dst_path.parent.mkdir(exist_ok=True, parents=True)
            save_image(dst_path, diff_img)

    @abstractproperty
    def vae(self):
        raise NotImplementedError

    @abstractproperty
    def image_processor(self):
        raise NotImplementedError


class SDXLFull(DiffusionModel):
    def __init__(
        self,
        base_model_id: str = ModelId.sdxl_base,
        refiner_model_id: str | None = ModelId.sdxl_refiner,
        device: torch.device = get_device(),
    ) -> None:
        super().__init__()

        self.use_refiner = refiner_model_id is not None

        self.base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(device)

        self.refiner_pipe = (
            StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_model_id,
                text_encoder_2=self.base_pipe.text_encoder_2,
                vae=self.base_pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(device)
            if self.use_refiner
            else None
        )

    def diffuse_sample(
        self,
        sample: dict[str, Any],
        base_strength: float = 0.2,
        refiner_strength: float = 0.2,
        base_denoising_start: int = None,
        base_denoising_end: int = None,
        refiner_denoising_start: int = None,
        refiner_denoising_end: int = None,
        original_size: tuple[int, int] = (1024, 1024),
        target_size: tuple[int, int] = (1024, 1024),
        prompt: str = default_prompt,
        negative_prompt: str = default_negative_prompt,
        base_num_steps: int = 50,
        refiner_num_steps: int = 50,
        base_gen: torch.Generator | Iterable[torch.Generator] = None,
        refiner_gen: torch.Generator | Iterable[torch.Generator] = None,
        base_kwargs: dict[str, any] = None,
        refiner_kwargs: dict[str, any] = None,
    ):
        image = sample["rgb"]

        image = batch_if_not_iterable(image)
        base_gen = batch_if_not_iterable(base_gen)
        refiner_gen = batch_if_not_iterable(refiner_gen)
        validate_same_len(image, base_gen, refiner_gen)

        base_kwargs = base_kwargs or {}
        image = self.base_pipe(
            image=image,
            generator=base_gen,
            output_type="latent" if self.use_refiner else "pt",
            strength=base_strength,
            denoising_start=base_denoising_start,
            denoising_end=base_denoising_end,
            num_inference_steps=base_num_steps,
            original_size=original_size,
            target_size=target_size,
            prompt=prompt,
            negative_prompt=negative_prompt,
            **base_kwargs,
        ).images

        if self.use_refiner:
            refiner_kwargs = refiner_kwargs or {}
            image = self.refiner_pipe(
                image=image,
                generator=refiner_gen,
                output_type="pt",
                strength=refiner_strength,
                denoising_start=refiner_denoising_start,
                denoising_end=refiner_denoising_end,
                num_inference_steps=refiner_num_steps,
                original_size=original_size,
                target_size=target_size,
                prompt=prompt,
                negative_prompt=negative_prompt,
                **refiner_kwargs,
            ).images

        return {"rgb": image}

    @property
    def vae(self) -> AutoencoderKL:
        return self.base_pipe.vae

    @property
    def image_processor(self) -> VaeImageProcessor:
        return self.base_pipe.image_processor


def encode_img(
    img_processor: VaeImageProcessor, vae: AutoencoderKL, img: Tensor, seed: int = 0
) -> Tensor:
    img = img_processor.preprocess(img)

    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        original_vae_dtype = vae.dtype
        upcast_vae(vae)  # Ensure float32 to avoid overflow
        img = img.float()

    latents = vae.encode(img.to("cuda"))

    if needs_upcasting:
        vae.to(original_vae_dtype)

    latents = retrieve_latents(latents, generator=torch.manual_seed(seed))
    latents = latents * vae.config.scaling_factor

    return latents


def decode_img(
    img_processor: VaeImageProcessor, vae: AutoencoderKL, latents: Tensor
) -> Tensor:
    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        original_vae_dtype = vae.dtype
        upcast_vae(vae)
        latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)

    img = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    if needs_upcasting:
        vae.to(original_vae_dtype)

    img = img_processor.postprocess(img, output_type="pt")
    return img


def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    vae.post_quant_conv.to(dtype)
    vae.decoder.conv_in.to(dtype)
    vae.decoder.mid_block.to(dtype)

    return vae


def load_img2img_model(**kwargs: dict[str, Any]) -> DiffusionModel:
    logging.info(f"Loading diffusion model...")

    match kwargs.get("model_name"):
        case "sdxlbase" | None:
            base_model_id = kwargs.get("base_model_id") or ModelId.sdxl_base
            refiner_model_id = None

            model = SDXLFull(base_model_id, refiner_model_id)

        case _:
            raise NotImplementedError

    logging.info(f"Finished loading diffusion model")
    return model


DiffusionModel.load_model = load_img2img_model


def diffusion_from_config_to_dir(
    src_dataset: DynamicDataset,
    dst_dir: Path,
    model_config: dict[str, Any],
    model: DiffusionModel = None,
):
    if model_config is not None:
        model_config_params = model_config["model_config_params"]
        model_forward_params = model_config["model_forward_params"]
    else:
        model_config_params = {}
        model_forward_params = {}

    if model is None:
        model = load_img2img_model(**model_config_params)

    dst_dir.mkdir(exist_ok=True, parents=True)
    model.diffuse_to_dir(src_dataset, dst_dir, **model_forward_params)
    save_yaml(model_config, dst_dir / "config.yml")

    logging.info(f"Finished diffusion.")
