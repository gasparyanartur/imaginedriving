from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import PIL
import torch
import torchvision.transforms.v2 as tvtf2
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
    combine_kwargs
)
from src.data import read_image, save_image


def pipeline_output_to_tensor(imgs: Iterable[PIL.Image.Image]) -> Tensor:
    return torch.stack([tvtf2.functional.pil_to_tensor(img) for img in imgs])


@dataclass
class ModelId:
    sdxl_base = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"


class ImgToImgModel(ABC):
    @abstractmethod
    def img_to_img(self, img: Tensor, *args, **kwargs) -> dict[str, any]:
        raise NotImplementedError


class SDXLBase(ImgToImgModel):
    def __init__(
        self, model_id: str = ModelId.sdxl_base, device: torch.device = get_device()
    ) -> None:
        super().__init__()

        self.base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(device)


class SDXLBase(ImgToImgModel):
    def __init__(
        self,
        n_steps: int = 50,
        strength: float = 0.2,
        prompt: str = "",
        model_id=ModelId.sdxl_base,
        device: torch.device = get_device(),
    ) -> None:
        super().__init__()

        #self.pipe: StableDiffusionXLImg2ImgPipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, )

class SDXLFull(ImgToImgModel):
    def __init__(
        self,
        n_steps: int = 50,
        mixture_threshold: float = 0.7,
        base_strength: float = 0.2,
        base_prompt: str = "",
        base_model_id: str = ModelId.sdxl_base,
        refiner_strength: float = 0.2,
        refiner_prompt: str = None,
        refiner_model_id=ModelId.sdxl_refiner,
        device: torch.device = get_device(),
    ) -> None:
        super().__init__()

        self.base_pipe: StableDiffusionXLImg2ImgPipeline = (
            StableDiffusionXLImg2ImgPipeline.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(device)
        )

        self.refiner_pipe: StableDiffusionXLImg2ImgPipeline = (
            StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_model_id,
                text_encoder_2=self.base_pipe.text_encoder_2,
                vae=self.base_pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(device)
        )

        self.base_kwargs = {
            "num_inference_steps": n_steps,
            "prompt": base_prompt,
            "strength": base_strength,
            "denoising_end": mixture_threshold,
        }

        self.refiner_kwargs = {
            "num_inference_steps": n_steps,
            "prompt": refiner_prompt or base_prompt,
            "strength": refiner_strength,
            "denoising_start": mixture_threshold,
        }

    def img_to_img(
        self,
        img: Tensor,
        base_gen: torch.Generator | Iterable[torch.Generator] = None,
        refiner_gen: torch.Generator | Iterable[torch.Generator] = None,
        base_extra_kwargs: dict[str, any] = None,
        refiner_extra_kwargs: dict[str, any] = None,
    ):
        base_kwargs = combine_kwargs(self.base_kwargs, base_extra_kwargs)
        refiner_kwargs = combine_kwargs(self.refiner_kwargs, refiner_extra_kwargs)

        img = batch_if_not_iterable(img)
        base_gen = batch_if_not_iterable(base_gen)
        refiner_gen = batch_if_not_iterable(refiner_gen)
        validate_same_len(img, base_gen, refiner_gen)

        img = self.base_pipe(image=img, output_type="latent", **base_kwargs).images
        img = self.refiner_pipe(image=img, **refiner_kwargs).images
        img = pipeline_output_to_tensor(img)

        return {"imgs": img}


def diffuse_images_to_dir(
    model: ImgToImgModel, img_paths: Iterable[Path], dst_dir: Path
):
    dst_dir.mkdir(exist_ok=True, parents=True)
    for src_img_path in img_paths:
        src_img = read_image(src_img_path)
        dst_img = model.img_to_img(src_img)
        save_image(dst_dir / src_img_path.name, dst_img)


def encode_img(
    img_processor: VaeImageProcessor, vae: AutoencoderKL, img: Tensor, seed: int = 0
) -> Tensor:
    upcast_vae(vae)  # Ensure float32 to avoid overflow
    img = img_processor.preprocess(img)
    latents = vae.encode(img.to("cuda"))
    latents = retrieve_latents(latents, generator=torch.manual_seed(seed))
    latents = latents * vae.config.scaling_factor
    latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)

    return latents


def decode_img(
    img_processor: VaeImageProcessor, vae: AutoencoderKL, latents: Tensor
) -> Tensor:
    img = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    img = img_processor.postprocess(img, output_type="pt")
    return img


def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    vae.post_quant_conv.to(dtype)
    vae.decoder.conv_in.to(dtype)
    vae.decoder.mid_block.to(dtype)

    return vae
