from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Dict, Iterable, Any, List
from functools import lru_cache
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import logging
import re

import torch
from torch import nn, Tensor

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    retrieve_latents,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL
from diffusers.models.controlnet import ControlNetModel, ControlNetOutput, UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL


import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transform
from torchmetrics.image import PeakSignalNoiseRatio

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from src.control_lora import ControlLoRAModel
from src.utils import (
    get_device,
    validate_same_len,
    batch_if_not_iterable,
)
from src.data import save_image, DynamicDataset, suffixes
from src.data import save_yaml


default_prompt = ""
default_negative_prompt = ""

LOWER_DTYPES = {"fp16", "bf16"}
DTYPE_CONVERSION = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _make_metric(name, device, **kwargs):
    match name:
        case "psnr":
            metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

        case "mse":
            metric = nn.MSELoss().to(device)

        case _:
            raise NotImplementedError

    return metric

@dataclass
class DiffusionModelId:
    sd_v1_5 = "runwayml/stable-diffusion-v1-5"
    sd_v2_1 = "stabilityai/stable-diffusion-2-1"
    sdxl_base_v1_0 = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner_v1_0 = "stabilityai/stable-diffusion-xl-refiner-1.0"
    sdxl_turbo_v1_0 = "stabilityai/sdxl-turbo"

@dataclass
class DiffusionModelType:
    sd: str = "sd"
    cn: str = "cn"
    mock: str = "mock"

def prep_hf_pipe(
    pipe: Union[StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline],
    device: torch.device = get_device(),
    low_mem_mode: bool = False,
    compile: bool = True,
    num_inference_steps: int = 50,
) -> Union[StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline]:
    if compile:
        try:
            pipe.unet = torch.compile(pipe.unet, fullgraph=True)
        except AttributeError:
            logging.warn(f"No unet found in Pipe. Skipping compiling")

    if low_mem_mode:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=True)
    pipe.noise_scheduler.set_timesteps(num_inference_steps)

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    return pipe


@dataclass
class DiffusionModelConfig:
    model_type: str = DiffusionModelType.sd
    model_id: str = DiffusionModelId.sd_v2_1

    low_mem_mode: bool = False
    """If applicable, prioritize options which lower GPU memory requirements at the expense of performance."""

    compile_model: bool = False
    """If applicable, compile Diffusion pipeline using available torch backend."""

    lora_weights: Optional[str] = None
    """Path to lora weights for the base diffusion model. Loads if applicable."""

    noise_strength: Optional[float] = 0.2
    """How much noise to apply during inference. 1.0 means complete gaussian."""

    num_inference_steps: Optional[int] = 50
    """Across how many timesteps the diffusion denoising occurs. Higher number gives better diffusion at expense of performance."""

    enable_progress_bar: bool = False
    """Create a progress bar for the denoising timesteps during inference."""

    metrics: Tuple[str, ...] = ("psnr", "mse")

    losses: Tuple[str, ...] = ("mse",)


@dataclass
class ControlNetConfig:
    conditioning_signals: Tuple[str, ...] = ()


class DiffusionModel(ABC):
    config: DiffusionModelConfig

    @abstractmethod
    def get_diffusion_output(
        self, sample: Dict[str, Any], *args, **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_diffusion_metrics(
        self, batch_pred: Dict[str, Any], batch_gt: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_diffusion_losses(
        self,
        batch_pred: Dict[str, Any],
        batch_gt: Dict[str, Any],
        metrics_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: DiffusionModelConfig, **kwargs) -> "DiffusionModel":
        model_type_to_constructor = {
            DiffusionModelType.sd: StableDiffusionModel,
            DiffusionModelType.cn: ControlNetDiffusionModel,
            DiffusionModelType.mock: MockDiffusionModel,
        }
        model = model_type_to_constructor[config.model_type]

        if config.compile_model and config.lora_weights:
            logging.warning(
                "Compiling the model currently leads to a bug when a LoRA is loaded, proceed with caution"
            )

        return model(config=config, **kwargs)


class MockDiffusionModel(DiffusionModel):
    def __init__(
        self, config: DiffusionModelConfig, device=get_device(), *args, **kwargs
    ) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.config = config

        self.diffusion_metrics = {
            metric_name: _make_metric(metric_name, device)
            for metric_name in config.metrics
        }
        self.diffusion_losses = {
            loss_name: _make_metric(loss_name, device) for loss_name in config.losses
        }

    def get_diffusion_output(
        self,
        sample: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        image = sample["rgb"]

        if len(image.shape) == 3:
            image = image[None, ...]

        return {"rgb": image}

    def get_diffusion_metrics(
        self, batch_pred: Dict[str, Any], batch_gt: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Currently only handles RGB case, assumes all metrics take in an RGB image.
        rgb_pred = batch_pred["rgb"]
        rgb_gt = batch_gt["rgb"]

        return {
            metric_name: metric(rgb_pred, rgb_gt)
            for metric_name, metric in self.diffusion_metrics.items()
        }

    def get_diffusion_losses(
        self,
        batch_pred: Dict[str, Any],
        batch_gt: Dict[str, Any],
        metrics_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Currently only handles RGB case, assumes all metrics take in an RGB image.
        rgb_pred = batch_pred["rgb"]
        rgb_gt = batch_gt["rgb"]

        loss_dict = {}
        for loss_name, loss in self.diffusion_losses.items():
            if loss_name in metrics_dict:
                loss_dict[loss_name] = metrics_dict[loss_name]
                continue

            loss_dict[loss_name] = loss(rgb_pred, rgb_gt)

        return loss_dict



class StableDiffusionModel(DiffusionModel):
    def __init__(
        self,
        config: DiffusionModelConfig,
        device: torch.device = get_device(),
        dtype: torch.dtype = torch.float16,
        use_safetensors: bool = True,
        variant: Optional[str] = None,
        unet: UNet2DConditionModel = None,
        vae: AutoencoderKL = None,
        tokenizer: CLIPTokenizer = None,
        scheduler: KarrasDiffusionSchedulers = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config
        self.device = device

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            variant=variant,
            use_safetensors=use_safetensors,
            unet=unet,
            vae=vae,
            tokenizer=tokenizer,
            scheduler=scheduler
        )

        self.pipe = prep_hf_pipe(
            self.pipe,
            low_mem_mode=config.low_mem_mode,
            device=device,
            compile=config.compile_model,
            num_inference_steps=config.num_inference_steps
        )

        if config.lora_weights:
            self.pipe.load_lora_weights(config.lora_weights)

        if verbose:
            logging.info(f"Ignoring unrecognized kwargs: {kwargs.keys()}")

        self.diffusion_metrics = {
            metric_name: _make_metric(metric_name, device)
            for metric_name in config.metrics
        }
        self.diffusion_losses = {
            loss_name: _make_metric(loss_name, device) for loss_name in config.losses
        }

    def get_diffusion_output(
        self,
        sample: Dict[str, Any],
        **kwargs,
    ):
        """Denoise image with diffusion model.

        Interesting kwargs:
        - image
        - generator
        - output_type
        - strength
        - num_inference_steps
        - prompt_embeds
        - negative_prompt_embeds

        - denoising_start (sdxl)
        - denoising_end (sdxl)
        - original_size (sdxl)
        - target_size (sdxl)
        """
        image = sample["rgb"]
        image = batch_if_not_iterable(image)

        batch_size = len(image)

        if image.size(1) == 3:
            channel_first = True
        elif image.size(3) == 3:
            channel_first = False
        else:
            raise ValueError(f"Image needs to be BCHW or BHWC, received {image.shape}")

        if not channel_first:
            image = image.permute(0, 3, 1, 2)  # Diffusion model is channel first

        kwargs = kwargs or {}
        kwargs["image"] = image
        kwargs["output_type"] = kwargs.get("output_type", "pt")
        kwargs["strength"] = kwargs.get("strength", self.config.noise_strength)
        kwargs["num_inference_steps"] = kwargs.get(
            "num_inference_steps", self.config.num_inference_steps
        )

        if "generator" in kwargs:
            kwargs["generator"] = batch_if_not_iterable(kwargs["generator"])


        if (
            "generator" in kwargs
            and len(kwargs["generator"]) <= 1
            and batch_size > 1
        ):
            raise ValueError(f"Number of generators must match number of images")

        # Convert any existing prompts to prompt embeddings, utilizing memoization.
        # Ensure there is at least one prompt embedding passed to the pipeline.
        prompt_embed_keys = []
        for prefix, suffix in it.product(["", "negative_"], ["", "_two"]):
            prompt_key = f"{prefix}prompt{suffix}"
            prompt_embed_key = f"{prefix}prompt_embeds{suffix}"

            if prompt_key in kwargs:
                prompt_embed_keys.append(prompt_embed_key)
                prompt = kwargs.pop(prompt_key)
                if prompt_embed_key not in kwargs:
                    with torch.no_grad():
                        kwargs[prompt_embed_key] = embed_prompt(
                            self.pipe.tokenizer, self.pipe.text_encoder, prompt
                        )

            if prompt_embed_key in kwargs:
                prompt_embed_keys.append(prompt_embed_key)

        # If no promp embed keys were passed, create one from an empty prompt
        if not prompt_embed_keys:
            prompt_embed_keys.append("prompt_embeds")
            with torch.no_grad():
                kwargs["prompt_embeds"] = embed_prompt(
                    self.pipe.tokenizer, self.pipe.text_encoder, ""
                )

        # Ensure batch size of prompts matches batch size of images
        for prompt_embed_key in prompt_embed_keys:
            kwargs[prompt_key] = batch_if_not_iterable(kwargs[prompt_key], single_dim=2)

            embed_size = kwargs[prompt_embed_key].shape
            if embed_size[0] == 1 and batch_size > 1:
                kwargs[prompt_embed_key] = kwargs[prompt_embed_key].expand(
                    batch_size * embed_size[0], embed_size[1], embed_size[2]
                )

        image = self.pipe(
            **kwargs,
        ).images

        if isinstance(image, list):
            image = torch.stack(image)

        if not channel_first:
            image = image.permute(0, 2, 3, 1)

        return {"rgb": image}

    def get_diffusion_metrics(
        self, batch_pred: Dict[str, Any], batch_gt: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Currently only handles RGB case, assumes all metrics take in an RGB image.
        rgb_pred = batch_pred["rgb"]
        rgb_gt = batch_gt["rgb"]

        return {
            metric_name: metric(rgb_pred, rgb_gt)
            for metric_name, metric in self.diffusion_metrics.items()
        }

    def get_diffusion_losses(
        self,
        batch_pred: Dict[str, Any],
        batch_gt: Dict[str, Any],
        metrics_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Currently only handles RGB case, assumes all metrics take in an RGB image.
        rgb_pred = batch_pred["rgb"]
        rgb_gt = batch_gt["rgb"]

        loss_dict = {}
        for loss_name, loss in self.diffusion_losses.items():
            if loss_name in metrics_dict:
                loss_dict[loss_name] = metrics_dict[loss_name]
                continue

            loss_dict[loss_name] = loss(rgb_pred, rgb_gt)

        return loss_dict


def encode_img(
    img_processor: VaeImageProcessor,
    vae: AutoencoderKL,
    img: Tensor,
    device,
    seed: int = None,
    sample_mode: str = "sample",
) -> Tensor:
    img = img_processor.preprocess(img)

    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        original_vae_dtype = vae.dtype
        upcast_vae(vae)  # Ensure float32 to avoid overflow
        img = img.float()

    latents = vae.encode(img.to(device))
    latents = retrieve_latents(
        latents,
        generator=(torch.manual_seed(seed) if seed is not None else None),
        sample_mode=sample_mode,
    )
    latents = latents * vae.config.scaling_factor

    if needs_upcasting:
        vae.to(original_vae_dtype)

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


@lru_cache(maxsize=4)
def tokenize_prompt(
    tokenizer: CLIPTokenizer, prompt: Union[str, Iterable[str]]
) -> torch.Tensor:
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tokens = text_inputs.input_ids
    return tokens


def encode_tokens(
    text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
    tokens: torch.Tensor,
) -> torch.Tensor:
    prompt_embeds = text_encoder(tokens.to(text_encoder.device))
    return prompt_embeds.last_hidden_state


@lru_cache(maxsize=4)
def _embed_hashable_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
    prompt: Union[str, Tuple[str, ...]],
) -> torch.Tensor:
    tokens = tokenize_prompt(tokenizer, prompt)
    embeddings = encode_tokens(text_encoder, tokens)
    return embeddings


def embed_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
    prompt: Union[str, Tuple[str, ...]],
    use_cache: bool = True
) -> torch.Tensor:
    if not isinstance(
        prompt, str
    ):  # Convert list to tuple to make it hashable for memoization
        prompt = tuple(prompt)

    if use_cache:
        embeddings = _embed_hashable_prompt(tokenizer, text_encoder, prompt)
    else:
        tokens = tokenize_prompt(tokenizer, prompt)
        embeddings = encode_tokens(text_encoder, tokens)
    
    return embeddings

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
                parsed_targets.update(parse_target_ranks(item, rf"{prefix}.*up_blocks"))

            case "attn":
                assert isinstance(item, int)
                parsed_targets[f"{prefix}.*attn.*to_[kvq]"] = item
                parsed_targets[rf"{prefix}.*attn.*to_out\.0"] = item

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



