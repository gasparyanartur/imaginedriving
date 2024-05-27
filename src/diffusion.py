from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Dict, Iterable, Any
from functools import lru_cache
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import logging
import re

import torch
from torch import Tensor

from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    retrieve_latents,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL
from diffusers.models.controlnet import ControlNetModel, ControlNetOutput

from src.utils import (
    get_device,
    validate_same_len,
    batch_if_not_iterable,
)
from src.data import save_image, DynamicDataset, suffixes
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


sdxl_models = {
    ModelId.sdxl_base_v1_0,
    ModelId.sdxl_refiner_v1_0,
    ModelId.sdxl_turbo_v1_0,
}
sd_models = {ModelId.sd_v1_5}


def is_sdxl_model(model_id: str) -> bool:
    return model_id in sdxl_models


def is_sdxl_vae(model_id: str) -> bool:
    return model_id == "madebyollin/sdxl-vae-fp16-fix" or is_sdxl_model(model_id)


def prep_model(
    pipe, device=get_device(), low_mem_mode: bool = False, compile: bool = True
):
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
        self,
        src_dataset: DynamicDataset,
        dst_dir: Path,
        id_range: tuple[int, int, int] = None,
        **kwargs,
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
        **kwargs,
    ) -> None:
        super().__init__()
        if configs is None:
            configs = {}

        configs.update(kwargs)

        if "base_model_id" not in configs and "model_id" in configs:
            configs["base_model_id"] = configs["model_id"]

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
        self.base_pipe = prep_model(
            self.base_pipe,
            low_mem_mode=low_mem_mode,
            device=device,
            compile=compile_model,
        )

        if self.use_refiner:
            self.refiner_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                refiner_model_id,
                text_encoder_2=self.base_pipe.text_encoder_2,
                vae=self.base_pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.refiner_pipe = prep_model(
                self.refiner_pipe,
                low_mem_mode=low_mem_mode,
                device=device,
                compile=compile_model,
            )
        self.tokenizer = self.base_pipe.tokenizer
        self.text_encoder = self.base_pipe.text_encoder
        self.device = device

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
        batch_size = len(image)

        image = batch_if_not_iterable(image)
        base_gen = batch_if_not_iterable(base_gen)
        refiner_gen = batch_if_not_iterable(refiner_gen)
        validate_same_len(image, base_gen, refiner_gen)

        if base_gen:
            base_gen = base_gen * batch_size

        if refiner_gen:
            refiner_gen = refiner_gen * batch_size

        base_kwargs = base_kwargs or {}
        if prompt is not None:
            with torch.no_grad():
                tokens = tokenize_prompt(self.tokenizer, prompt).to(self.device)
                prompt_embeds = encode_tokens(
                    self.text_encoder, tokens, using_sdxl=False
                )["embeds"]
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)

        if negative_prompt is not None:
            with torch.no_grad():
                negative_tokens = tokenize_prompt(self.tokenizer, negative_prompt).to(
                    self.device
                )
                negative_prompt_embeds = encode_tokens(
                    self.text_encoder, negative_tokens, using_sdxl=False
                )["embeds"]

            negative_prompt_embeds = negative_prompt_embeds.expand(batch_size, -1, -1)

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
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
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
                "refiner_model_id": None,
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
            self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_model_id,
                text_encoder_2=self.base_pipe.text_encoder_2,
                vae=self.base_pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
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
    img_processor: VaeImageProcessor,
    vae: AutoencoderKL,
    img: Tensor,
    device,
    seed: int = None,
    sample_latent: bool = False,
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
            latents, generator=(torch.manual_seed(seed) if seed is not None else None)
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
        model_input = encode_img(
            img_processor, vae, img, sample_latent=True, device=vae.device, seed=seed
        )
        noise = torch.randn_like(model_input, device=vae.device)
        timestep = noise_scheduler.timesteps[timestep]
        timesteps = torch.tensor([timestep], device=vae.device)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
        img_pred = decode_img(img_processor, vae, noisy_model_input)

    return img_pred


def load_img2img_model(
    model_config_params: dict[str, Any], device=get_device()
) -> DiffusionModel:
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
    device=get_device(),
):
    if model_config is not None:
        model_config_params = model_config["model_config_params"]
        model_forward_params = model_config["model_forward_params"]
    else:
        model_config_params = {}
        model_forward_params = {}

    if model is None:
        model = load_img2img_model(
            model_config_params=model_config_params, device=device
        )

    dst_dir.mkdir(exist_ok=True, parents=True)
    model.diffuse_to_dir(
        src_dataset, dst_dir, id_range=id_range, **model_forward_params
    )
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

    return {"embeds": prompt_embeds.last_hidden_state}


def get_diffusion_cls(
    model_id: str,
) -> StableDiffusionImg2ImgPipeline | StableDiffusionXLImg2ImgPipeline:
    if is_sdxl_model(model_id):
        return StableDiffusionXLImg2ImgPipeline
    else:
        return StableDiffusionImg2ImgPipeline


def get_random_timesteps(noise_strength, total_num_timesteps, device, batch_size):
    # Sample a random timestep for each image
    timesteps = torch.randint(
        int((1 - noise_strength) * total_num_timesteps),
        total_num_timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    return timesteps


def draw_from_bins(start, end, n_draws, device, include_last: bool = False):
    values = torch.zeros(n_draws + int(include_last), dtype=torch.long, device=device)
    buckets = torch.round(torch.linspace(start, end, n_draws + 1)).int()

    for i in range(n_draws):
        values[i] = torch.randint(buckets[i], buckets[i + 1], (1,))

    if include_last:
        values[-1] = end

    return values


def get_ordered_timesteps(
    noise_strength,
    total_num_timesteps,
    device,
    num_timesteps=None,
    sample_from_bins: bool = True,
):
    if num_timesteps is None:
        num_timesteps = total_num_timesteps

    start_step = int((1 - noise_strength) * total_num_timesteps)
    end_step = total_num_timesteps - 1

    # Make sure the last one is total_num_timesteps-1
    if sample_from_bins:
        timesteps = draw_from_bins(
            start_step, end_step, num_timesteps - 1, include_last=True, device=device
        )
    else:
        timesteps = torch.round(
            torch.linspace(start_step, end_step, num_timesteps, device=device)
        ).to(torch.long)
    return timesteps







class PeftCompatibleControlNet(ControlNetModel):
    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            controlnet_cond: torch.FloatTensor,
            conditioning_scale: float = 1.0,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guess_mode: bool = False,
            return_dict: bool = True,
        ) -> Union[ControlNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:
            """
            The [`ControlNetModel`] forward method.

            Args:
                sample (`torch.FloatTensor`):
                    The noisy input tensor.
                timestep (`Union[torch.Tensor, float, int]`):
                    The number of timesteps to denoise an input.
                encoder_hidden_states (`torch.Tensor`):
                    The encoder hidden states.
                controlnet_cond (`torch.FloatTensor`):
                    The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
                conditioning_scale (`float`, defaults to `1.0`):
                    The scale factor for ControlNet outputs.
                class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                    Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
                timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                    Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                    timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                    embeddings.
                attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                    An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                    is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                    negative values to the attention scores corresponding to "discard" tokens.
                added_cond_kwargs (`dict`):
                    Additional conditions for the Stable Diffusion XL UNet.
                cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                    A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
                guess_mode (`bool`, defaults to `False`):
                    In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                    you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
                return_dict (`bool`, defaults to `True`):
                    Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

            Returns:
                [`~models.controlnet.ControlNetOutput`] **or** `tuple`:
                    If `return_dict` is `True`, a [`~models.controlnet.ControlNetOutput`] is returned, otherwise a tuple is
                    returned where the first element is the sample tensor.
            """
            # check channel order
            channel_order = self.config.controlnet_conditioning_channel_order

            if channel_order == "rgb":
                # in rgb order by default
                ...
            elif channel_order == "bgr":
                controlnet_cond = torch.flip(controlnet_cond, dims=[1])
            else:
                raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # 1. time
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])

            t_emb = self.time_proj(timesteps)

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=sample.dtype)

            emb = self.time_embedding(t_emb, timestep_cond)
            aug_emb = None

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")

                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)

                class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                emb = emb + class_emb

            if self.config.addition_embed_type is not None:
                if self.config.addition_embed_type == "text":
                    aug_emb = self.add_embedding(encoder_hidden_states)

                elif self.config.addition_embed_type == "text_time":
                    if "text_embeds" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                        )
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    if "time_ids" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                        )
                    time_ids = added_cond_kwargs.get("time_ids")
                    time_embeds = self.add_time_proj(time_ids.flatten())
                    time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                    add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                    add_embeds = add_embeds.to(emb.dtype)
                    aug_emb = self.add_embedding(add_embeds)

            emb = emb + aug_emb if aug_emb is not None else emb

            # 2. pre-process
            sample = self.conv_in(sample)

            controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
            sample = sample + controlnet_cond

            # 3. down
            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                down_block_res_samples += res_samples

            # 4. mid
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample = self.mid_block(sample, emb)

            # 5. Control net blocks

            controlnet_down_block_res_samples = ()

            #controlnet_down_blocks = next((b for a, b in self.controlnet_down_blocks.named_children() if a == "modules_to_save"))["default"]
            controlnet_down_blocks = next((b for a, b in self.controlnet_down_blocks.named_children() if a == "original_module"))
            for down_block_res_sample, controlnet_block in zip(down_block_res_samples, controlnet_down_blocks):
                down_block_res_sample = controlnet_block(down_block_res_sample)
                controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = controlnet_down_block_res_samples

            mid_block_res_sample = self.controlnet_mid_block(sample)

            # 6. scaling
            if guess_mode and not self.config.global_pool_conditions:
                scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
                scales = scales * conditioning_scale
                down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
                mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
            else:
                down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
                mid_block_res_sample = mid_block_res_sample * conditioning_scale

            if self.config.global_pool_conditions:
                down_block_res_samples = [
                    torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
                ]
                mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

            if not return_dict:
                return (down_block_res_samples, mid_block_res_sample)

            return ControlNetOutput(
                down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
            )



def get_matching(model, patterns: Iterable[Union[re.Pattern, str]] = (".*",)):
    for i, pattern in enumerate(patterns):
        if isinstance(pattern, str):
            patterns[i] = re.compile(pattern)

    li = []
    for name, mod in model.named_modules():
        for pattern in patterns:
            if pattern.match(name):
                li.append((name, mod))
    return li



def parse_target_ranks(target_ranks, prefix=r""):
    parsed_targets = {}

    for name, item in target_ranks.items():
        if not item:
            continue

        match name:
            case "":
                continue

            case "downblocks":
                assert isinstance(item, dict)
                parsed_targets.update(
                    parse_target_ranks(item, rf"{prefix}.*down_blocks")
                )

            case "midblocks":
                assert isinstance(item, dict)
                parsed_targets.update(
                    parse_target_ranks(item, rf"{prefix}.*mid_blocks")
                )

            case "upblocks":
                assert isinstance(item, dict)
                parsed_targets.update(
                    parse_target_ranks(item, rf"{prefix}.*up_blocks")
                )

            case "attn":
                assert isinstance(item, int)
                parsed_targets[f"{prefix}.*attn.*to_[kvq]"] = item
                parsed_targets[ rf"{prefix}.*attn.*to_out\.0"] = item


            case "resnet":
                assert isinstance(item, int)
                parsed_targets[rf"{prefix}.*resnets.*conv\d*"] = item
                parsed_targets[rf"{prefix}.*resnets.*time_emb_proj"] = item

            case "ff":
                assert isinstance(item, int)
                parsed_targets[rf"{prefix}.*ff\.net\.0\.proj"] = item
                parsed_targets[rf"{prefix}.*ff\.net\.2"] = item

            case "proj":
                assert isinstance(item, int)
                parsed_targets[rf"{prefix}.*attentions.*proj_in"] = item
                parsed_targets[rf"{prefix}.*attentions.*proj_out"] = item

            case "_":
                raise NotImplementedError(f"Unrecognized target: {name}")

    return parsed_targets