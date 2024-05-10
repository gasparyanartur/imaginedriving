from typing import Any
from functools import lru_cache
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import logging

import torch
from torch import Tensor

from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline
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
from src.data import save_yaml


default_prompt = "dashcam recording, urban driving scene, video, autonomous driving, detailed cars, traffic scene, pandaset, kitti, high resolution, realistic, detailed, camera video, dslr, ultra quality, sharp focus, crystal clear, 8K UHD, 10 Hz capture frequency 1/2.7 CMOS sensor, 1920x1080"
default_negative_prompt = "face, human features, unrealistic, artifacts, blurry, noisy image, NeRF, oil-painting, art, drawing, poor geometry, oversaturated, undersaturated, distorted, bad image, bad photo"


@dataclass
class ModelId:
    sd_v1_5 = "runwayml/stable-diffusion-v1-5"
    sd_v2_1 = "stabilityai/stable-diffusion-2-1"
    sdxl_base_v1_0 = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner_v1_0 = "stabilityai/stable-diffusion-xl-refiner-1.0"
    sdxl_turbo_v1_0 = "stabilityai/sdxl-turbo"


sdxl_models = {ModelId.sdxl_base_v1_0, ModelId.sdxl_refiner_v1_0, ModelId.sdxl_turbo_v1_0}
sd_models = {ModelId.sd_v1_5}

def is_sdxl_model(model_id: str) -> bool:
    return model_id in sdxl_models

def is_sdxl_vae(model_id: str) -> bool:
    return model_id == "madebyollin/sdxl-vae-fp16-fix" or is_sdxl_model(model_id)


def prep_model(pipe, device=get_device(), low_mem_mode: bool = False, compile: bool = True):
    if compile:
        try:
            pipe.unet = torch.compile(pipe.unet, fullgraph=True)
        except AttributeError:
            logging.warn(f"No unet found in Pipe. Skipping compiling")
    
    
    if low_mem_mode: 
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)


    return pipe


class DiffusionModel(ABC):
    load_model = None

    @abstractmethod
    def diffuse_sample(self, sample: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def diffuse_to_dir(
        self, src_dataset: DynamicDataset, dst_dir: Path, id_range: tuple[int, int, int] = None, **kwargs
    ) -> None:
        assert "meta" in src_dataset.data_getters

        dst_dir.mkdir(exist_ok=True, parents=True)

        id_range = id_range or (0, 0, 0)
        for sample in src_dataset.iter_range(*id_range, verbose=True):
            meta = sample["meta"]
            dst_path = dst_dir / meta["dataset"] / meta["scene"] / meta["sample"]
            dst_path = dst_path.with_suffix(suffixes["rgb", src_dataset.name])
            dst_path.parent.mkdir(exist_ok=True, parents=True)

            diff_img = self.diffuse_sample(sample, **kwargs)["rgb"]
            save_image(dst_path, diff_img)

    @property
    @abstractmethod
    def vae(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def image_processor(self):
        raise NotImplementedError


class SDPipe(DiffusionModel):
    def __init__(
        self,
        configs: dict[str, Any] = None,
        device: torch.device = get_device(),
    ) -> None:
        super().__init__()
        if configs is None:
            configs = {}

        base_model_id = configs.get("base_model_id", ModelId.sd_v1_5)
        refiner_model_id = configs.get("refiner_model_id", None)
        low_mem_mode = configs.get("low_mem_mode", False)
        compile_model = configs.get("compile_model", False)

        self.use_refiner = refiner_model_id is not None

        if self.use_refiner:
            raise NotImplementedError

        
        self.base_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.base_pipe = prep_model(self.base_pipe, low_mem_mode=low_mem_mode, device=device, compile=compile_model)
        

        if self.use_refiner:
            self.refiner_pipe = (
                StableDiffusionImg2ImgPipeline.from_pretrained(
                    refiner_model_id,
                    text_encoder_2=self.base_pipe.text_encoder_2,
                    vae=self.base_pipe.vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                )
            )
            self.refiner_pipe = prep_model(self.refiner_pipe, low_mem_mode=low_mem_mode, device=device, compile=compile_model)

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


class SDXLPipe(DiffusionModel):
    def __init__(
        self,
        configs: dict[str, Any] = None,
        device: torch.device = get_device(),
    ) -> None:
        super().__init__()
        # TODO: add model compilation
        # TODO: Combine with regular SD pipe
        raise NotImplementedError

        if configs is None:
            configs = {
                "base_model_id": ModelId.sdxl_base_v1_0,
                "refiner_model_id": None
            }

        base_model_id = configs.get("base_model_id")
        refiner_model_id = configs.get("refiner_model_id")
        low_mem_mode = configs.get("low_mem_mode")
        self.use_refiner = refiner_model_id is not None

        self.base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        if low_mem_mode: 
            self.base_pipe.enable_model_cpu_offload()
        else:
            self.base_pipe = self.base_pipe.to(device)

        if self.use_refiner:
            self.refiner_pipe = (
                StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_model_id,
                    text_encoder_2=self.base_pipe.text_encoder_2,
                    vae=self.base_pipe.vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                )
            )

            if low_mem_mode:
                self.refiner_pipe.enable_model_cpu_offload()
            else:
                self.refiner_pipe = self.refiner_pipe.to(device)

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


model_name_to_constructor = {
    "sdxl_base": SDXLPipe,
    "sdxl_full": SDXLPipe,
    "sd_base": SDPipe,
    "sd_full": SDPipe,
    None: SDXLPipe,
}


def encode_img(
    img_processor: VaeImageProcessor, vae: AutoencoderKL, img: Tensor, device, seed: int = None, sample_latent: bool = False
) -> Tensor:
    img = img_processor.preprocess(img)

    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        original_vae_dtype = vae.dtype
        upcast_vae(vae)  # Ensure float32 to avoid overflow
        img = img.float()

    latents = vae.encode(img.to(device))
    if sample_latent:
        latents = latents.latent_dist.sample()

    else:
        if needs_upcasting:
            vae.to(original_vae_dtype)

        latents = retrieve_latents(
            latents,
            generator=(torch.manual_seed(seed) if seed is not None else None)
        )

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

def get_noised_img(img, timestep, pipe, noise_scheduler, seed=None):
    vae = pipe.vae
    img_processor = pipe.image_processor
        
    with torch.no_grad():
        model_input = encode_img(img_processor, vae, img, sample_latent=True, device=vae.device, seed=seed)  
        noise = torch.randn_like(model_input, device=vae.device)
        timestep = noise_scheduler.timesteps[timestep]
        timesteps = torch.tensor([timestep], device=vae.device)
        noisy_model_input = noise_scheduler.add_noise(
            model_input, noise, timesteps
        )
        img_pred = decode_img(img_processor, vae, noisy_model_input)

    return img_pred

def load_img2img_model(model_config_params: dict[str, Any], device=get_device()) -> DiffusionModel:
    logging.info(f"Loading diffusion model...")

    model_name = model_config_params.get("model_name")
    constructor = model_name_to_constructor.get(model_name)
    if not constructor:
        raise NotImplementedError

    model = constructor(configs=model_config_params, device=device)
    logging.info(f"Finished loading diffusion model")
    return model


DiffusionModel.load_model = load_img2img_model


def diffusion_from_config_to_dir(
    src_dataset: DynamicDataset,
    dst_dir: Path,
    model_config: dict[str, Any],
    model: DiffusionModel = None,
    id_range: tuple[int, int, int] = None,
    device=get_device()
):
    if model_config is not None:
        model_config_params = model_config["model_config_params"]
        model_forward_params = model_config["model_forward_params"]
    else:
        model_config_params = {}
        model_forward_params = {}

    if model is None:
        model = load_img2img_model(model_config_params=model_config_params, device=device)

    dst_dir.mkdir(exist_ok=True, parents=True)
    model.diffuse_to_dir(src_dataset, dst_dir, id_range=id_range, **model_forward_params)
    save_yaml(dst_dir / "config.yml", model_config)

    logging.info(f"Finished diffusion.")


@lru_cache(maxsize=4)
def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tokens = text_inputs.input_ids
    return tokens


def encode_tokens(text_encoder, tokens, using_sdxl):
    if using_sdxl:
        raise NotImplementedError

    prompt_embeds = text_encoder(tokens)

    return {
        "embeds": prompt_embeds.last_hidden_state
    }
