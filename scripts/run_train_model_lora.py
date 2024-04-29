from typing import Any, Iterable
from pathlib import Path
from argparse import ArgumentParser
import logging
import os
from dataclasses import dataclass, asdict, field
import math
import itertools as it
import tqdm
import shutil

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import transformers
import torchvision.transforms.v2 as transforms
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_snr,
)
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from huggingface_hub import create_repo, upload_folder
from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.data import setup_project, DynamicDataset
from src.diffusion import is_sdxl_model, is_sdxl_vae


check_min_version("0.27.0")
logger = get_logger(__name__, log_level="INFO")



def import_text_encoder_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    match model_class:
        case "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel

        case "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection

        case _:
            raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = ArgumentParser(
        "train_model", description="Finetune a given model on a dataset"
    )
    parser.add_argument("config_path", type=Path)

    args = parser.parse_args()
    return args


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=False,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def compute_vae_encodings(batch, vae):
    images = batch.pop("rgb")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return {"model_input": model_input.cpu()}


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def compute_time_ids(original_size, crops_coords_top_left, target_size, device, dtype):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], device=device, dtype=dtype)
    return add_time_ids


def preprocess_rgb(
    rgb, target_size: tuple[int, int], resizer, flipper, cropper, center_crop: bool
):
    # preprocess rgb:
    # out: rgb, original_size, crop_top_left, target_size

    th, tw = target_size
    h0, w0 = rgb.shape[-2:]

    if resizer:
        rgb = resizer(rgb)

    if flipper:
        rgb = flipper(rgb)

    if cropper:
        if center_crop:
            cy = (h0 - th) // 2
            cx = (w0 - tw) // 2
            rgb = cropper(rgb)

        else:
            cy, cx, h, w = cropper.get_params(rgb, target_size)
            rgb = transforms.functional.crop(rgb, cy, cx, h, w)
    else:
        cx, cy = 0, 0
        h, w = h0, w0

    original_size = h0, w0
    crop_top_left = cy, cx

    return {
        "rgb": rgb,
        "original_size": original_size,
        "crop_top_left": crop_top_left,
        "target_size": target_size,
    }


def preprocess_prompt(prompt: list[str], tokenizers):
    assert 1 <= len(tokenizers) <= 2

    if isinstance(prompt, str):
        prompt = [prompt]

    input_ids_one = tokenize_prompt(tokenizers[0], prompt)[0]
    input_ids_two = (
        tokenize_prompt(tokenizers[1], prompt)[0] if len(tokenizers) > 1 else None
    )

    return {"input_ids_one": input_ids_one, "input_ids_two": input_ids_two}


def preprocess_sample(
    batch, resizer, flipper, cropper, tokenizers, target_size, center_crop: bool = False
):
    # Ignore negative prompt :)
    rgb = preprocess_rgb(
        batch["rgb"], target_size, resizer, flipper, cropper, center_crop
    )
    prompt = preprocess_prompt(batch["prompt"]["positive_prompt"], tokenizers)

    sample = {
        "original_size": rgb["original_size"],
        "crop_top_left": rgb["crop_top_left"],
        "target_size": rgb["target_size"],
        "rgb": rgb["rgb"],
        "input_ids_one": prompt["input_ids_one"],
        "meta": batch["meta"],
        "positive_prompt": batch["prompt"]["positive_prompt"],
    }
    if prompt.get("input_ids_two"):
        sample["input_ids_two"] = prompt["input_ids_two"]

    return sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[list[str], Any]:
    original_size = [sample["original_size"] for sample in batch]
    crop_top_left = [sample["crop_top_left"] for sample in batch]
    target_size = [sample["target_size"] for sample in batch]
    rgb = torch.stack([sample["rgb"] for sample in batch]).to(memory_format=torch.contiguous_format, dtype=torch.float32)
    meta = [sample["meta"] for sample in batch]
    input_ids_one = torch.stack([sample["input_ids_one"] for sample in batch])
    positive_prompt = [sample["positive_prompt"] for sample in batch]

    input_ids_two = torch.stack([sample["input_ids_two"] for sample in batch]) if "input_ids_two" in batch[0] else None

    batch = {
        "original_size": original_size,
        "crop_top_left": crop_top_left,
        "target_size": target_size,
        "rgb": rgb,
        "input_ids_one": input_ids_one,
        "meta": meta,
        "positive_prompt": positive_prompt
    }

    if input_ids_two:
        batch["input_ids_two"] = input_ids_two

    return batch

@dataclass(init=True)
class TrainLoraConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    #model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Path to pretrained VAE model with better numerical stability.
    # More details: https://github.com/huggingface/diffusers/pull/4038.
    #vae_id: str = "madebyollin/sdxl-vae-fp16-fix"
    vae_id: str = None

    mixed_precision: str = None
    revision: str = None
    variant: str = None
    prediction_type: str = None

    train_text_encoder: bool = False
    gradient_checkpointing: bool = False
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = None
    resume_from_checkpoint: bool = False
    n_epochs: int = 100
    num_train_timesteps: int = None
    max_train_samples: int = None
    val_freq: int = 10
    val_strength: float = 0.8

    keep_vae_full: bool = True

    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
    noise_offset: float = 0

    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    allow_tf32: bool = True
    scale_lr: bool = False
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 16
    dataloader_num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    snr_gamma: float = None
    max_grad_norm: float = 1.0
    lora_rank: int = 4
    compile_model: bool = False

    # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_scheduler_kwargs: dict[str, any] = field(default_factory=dict)

    image_width: int = 1920
    image_height: int = 1080

    center_crop: bool = False
    flip_prob: float = 0.5

    crop_height: float = 512
    crop_width: float = 512
    resize_factor: float = 1.0

    seed: int = None


def main(args):
    # ========================
    # ===   Setup script   ===
    # ========================

    config_path = args.config_path
    config = setup_project(config_path)


    output_dir = Path(config["output_dir"])
    logging_dir = Path(output_dir, "logs")

    report_to: str = config.get("report_to", "wandb")
    entity: str = config.get("entity", "arturruiqi")
    group: str = config.get("group", "finetune-lora")
    project: str = config.get("project", "master-thesis")
    push_to_hub: bool = config.get("push_to_hub", False)  # Not Implemented
    hub_token: str | None = config.get("hub_token", None)

    model_config = TrainLoraConfig(**config["model"])
    using_sdxl = is_sdxl_model(model_config.model_id)
    
    mixed_precision = model_config.mixed_precision

    if (model_config.vae_id is not None) and (
        (using_sdxl and not is_sdxl_vae(model_config.vae_id))
        or (not using_sdxl and is_sdxl_vae(model_config.vae_id))
    ):
        raise ValueError(
            f"Mismatch between model_id and vae_id. Both models need to either be SDXL or SD, but received {model_config.model_id} and {model_config.vae_id}"
        )

    if torch.backends.mps.is_available() and mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )
    accelerator_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=model_config.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=[report_to],
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_kwargs],
    )
    logging.info(f"Number of cuda detected devices: {torch.cuda.device_count()}, Using device: {accelerator.device}, distributed: {accelerator.distributed_type}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if model_config.seed is not None:
        set_seed(model_config.seed)

    if accelerator.is_main_process:
        output_dir.mkdir(exist_ok=True, parents=True)
        logging_dir.mkdir(exist_ok=True)

        if push_to_hub:
            raise NotImplementedError

    if report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

        if hub_token is not None:
            raise ValueError(
                "You cannot use both report_to=wandb and hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

        import wandb
        if accelerator.is_local_main_process:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", project),
                entity=os.environ.get("WANDB_ENTITY", entity),
                dir=os.environ.get("WANDB_DIR", str(logging_dir)),
                group=os.environ.get("WANDB_GROUP", group),
                reinit=True,
                config=asdict(model_config)
            )

    # =======================
    # ===   Load models   ===
    # =======================
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction="none").to(accelerator.device)

    noise_scheduler = DDPMScheduler.from_pretrained(
        model_config.model_id, subfolder="scheduler"
    )

    tokenizer_one = AutoTokenizer.from_pretrained(
        model_config.model_id,
        subfolder="tokenizer",
        revision=model_config.revision,
        use_fast=False,
    )
    text_encoder_cls_one = import_text_encoder_class_from_model_name_or_path(
        model_config.model_id, model_config.revision, subfolder="text_encoder"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        model_config.model_id,
        subfolder="text_encoder",
        revision=model_config.revision,
        variant=model_config.variant,
    )

    if using_sdxl:
        tokenizer_two = AutoTokenizer.from_pretrained(
            model_config.model_id,
            subfolder="tokenizer_2",
            revision=model_config.revision,
            use_fast=False,
        )
        text_encoder_cls_two = import_text_encoder_class_from_model_name_or_path(
            model_config.model_id, model_config.revision, subfolder="text_encoder_2"
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            model_config.model_id,
            subfolder="text_encoder_2",
            revision=model_config.revision,
            variant=model_config.variant,
        )
    else:
        tokenizer_two = None
        text_encoder_cls_two = None
        text_encoder_two = None

    if (
        model_config.vae_id is None
    ):  # Need VAE fix for stable diffusion XL, see https://github.com/huggingface/diffusers/pull/4038
        vae = AutoencoderKL.from_pretrained(
            model_config.model_id,
            subfolder="vae",
            revision=model_config.revision,
            variant=model_config.variant,
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            model_config.vae_id,
            subfolder=None,
            revision=model_config.revision,
            variant=model_config.variant,
        )

    unet = UNet2DConditionModel.from_pretrained(
        model_config.model_id,
        subfolder="unet",
        revision=model_config.revision,
        variant=model_config.variant,
    )

    # ===================
    # === Config Lora ===
    # ===================

    if model_config.compile_model:
        torch_backend = "cudagraphs"
        vae = torch.compile(vae, backend=torch_backend)
        unet = torch.compile(unet, backend=torch_backend)
        text_encoder_one = torch.compile(text_encoder_one, backend=torch_backend)
        if text_encoder_two:
            text_encoder_two = torch.compile(text_encoder_two, backend=torch_backend)


    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    if text_encoder_two:
        text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)

    if (
        (using_sdxl and model_config.vae_id is None) or model_config.keep_vae_full
    ):  # The VAE in SDXL is in float32 to avoid NaN losses.
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    if text_encoder_two:
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    unet_lora_config = LoraConfig(
        r=model_config.lora_rank,
        lora_alpha=model_config.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    if model_config.train_text_encoder:
        text_lora_config = LoraConfig(
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

        if text_encoder_two:
            text_encoder_two.add_adapter(text_lora_config)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if model_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    # ============================
    # === Prepare optimization ===
    # ============================

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder attn layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                unwrapped_model = unwrap_model(model)
                state_model_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)
                )

                if isinstance(unwrapped_model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = state_model_dict

                elif isinstance(unwrapped_model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = state_model_dict

                elif text_encoder_two and isinstance(
                    unwrapped_model, type(unwrap_model(text_encoder_two))
                ):
                    text_encoder_two_lora_layers_to_save = state_model_dict

                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            if using_sdxl:
                StableDiffusionXLImg2ImgPipeline.save_lora_weights(
                    save_directory=output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                    text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
                )

            else:
                StableDiffusionImg2ImgPipeline.save_lora_weights(
                    save_directory=output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while model := models.pop():
            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model

            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model

            elif text_encoder_two and isinstance(
                model, type(unwrap_model(text_encoder_two))
            ):
                text_encoder_two_ = model

            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {
            f'{k.replace("unet.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            unet_, unet_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if model_config.train_text_encoder:
            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_
            )

            if using_sdxl:
                _set_state_dict_into_text_encoder(
                    lora_state_dict,
                    prefix="text_encoder_2.",
                    text_encoder=text_encoder_two_,
                )

        # Make sure the trainable params are in float32.
        if mixed_precision == "fp16":
            models = [unet_]
            if model_config.train_text_encoder:
                models.append(text_encoder_one_)

                if text_encoder_two:
                    models.append(text_encoder_two_)

            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if model_config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if model_config.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

            if text_encoder_two:
                text_encoder_two.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # Note: this applies to the A100
    if model_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    learning_rate = model_config.learning_rate
    if model_config.scale_lr:
        learning_rate = (
            learning_rate
            * model_config.gradient_accumulation_steps
            * model_config.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if mixed_precision == "fp16":
        models = [unet]
        if model_config.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

    models_to_optimize = [unet]
    if model_config.train_text_encoder:
        models_to_optimize.append(text_encoder_one)
        if text_encoder_two:
            models_to_optimize.append(text_encoder_two)

    params_to_optimize = []
    for model in models_to_optimize:
        for param in model.parameters():
            if param.requires_grad:
                params_to_optimize.append(param)

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(model_config.adam_beta1, model_config.adam_beta2),
        weight_decay=model_config.adam_weight_decay,
        eps=model_config.adam_epsilon,
    )

    # ======================
    # ===   Setup data   ===
    # ======================

    dataset_config = config["datasets"]
    train_dataset_config = dataset_config["train_data"]
    val_dataset_config = dataset_config["val_data"]

    train_dataset = DynamicDataset.from_config(train_dataset_config)
    val_dataset = DynamicDataset.from_config(val_dataset_config)


    # Preprocessing

    def nearest_multiple(x: float | int, m: int) -> int:
        return int(int(x / m) * m)

    crop_height = model_config.crop_height or model_config.image_height 
    crop_width = model_config.crop_width or model_config.image_width 
    resize_factor = model_config.resize_factor or 1

    crop_size = (
        nearest_multiple(crop_height * resize_factor, 8),
        nearest_multiple(crop_width * resize_factor, 8)
    )
    downsample_size = (
        nearest_multiple(model_config.image_height * resize_factor, 8),
        nearest_multiple(model_config.image_width * resize_factor, 8)
    ) 

    if resize_factor != 1:
        train_resizer = transforms.Resize(downsample_size, interpolation=transforms.InterpolationMode.BILINEAR)
    else:
        train_resizer = None

    if model_config.flip_prob > 0:
        train_flipper = transforms.RandomHorizontalFlip(p=model_config.flip_prob)
    else:
        train_flipper = None

    if (crop_height != model_config.image_height) or (crop_width != model_config.image_width):
        train_cropper = (
            transforms.CenterCrop(crop_size)
            if model_config.center_crop
            else transforms.RandomCrop(crop_size)
        )
    else:
        train_cropper = None

    if resize_factor != 1:
        val_resizer = transforms.Resize(downsample_size, interpolation=transforms.InterpolationMode.BILINEAR)
    else:
        val_resizer = None

    val_flipper = None
    if (crop_height != model_config.image_height) or (crop_width != model_config.image_width):
        val_cropper = transforms.CenterCrop(crop_size)
    else:
        val_cropper = None


    tokenizers = [tokenizer_one]
    if tokenizer_two:
        tokenizers.append(tokenizer_two)

    with accelerator.main_process_first():
        train_dataset.preprocess_func = lambda batch: preprocess_sample(batch, train_resizer, train_flipper, train_cropper, tokenizers, crop_size, model_config.center_crop)
        val_dataset.preprocess_func = lambda batch: preprocess_sample(batch, val_resizer, val_flipper, val_cropper, tokenizers, crop_size, True)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=model_config.train_batch_size,
        num_workers=model_config.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=model_config.pin_memory
    )

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=model_config.train_batch_size,
        num_workers=model_config.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=model_config.pin_memory
    )

    # ==========================
    # ===   Setup training   ===
    # ==========================

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / model_config.gradient_accumulation_steps
    )
    max_train_steps = model_config.n_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        model_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=model_config.lr_warmup_steps
        * model_config.gradient_accumulation_steps,
        num_training_steps=max_train_steps * model_config.gradient_accumulation_steps,
        **model_config.lr_scheduler_kwargs
    )

    # Prepare everything with our `accelerator`.
    if model_config.train_text_encoder:
        if text_encoder_two:
            (
                unet,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = accelerator.prepare(
                unet,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )

        else:
            unet, text_encoder_one, optimizer, train_dataloader, lr_scheduler = (
                accelerator.prepare(
                    unet, text_encoder_one, optimizer, train_dataloader, lr_scheduler
                )
            )

    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / model_config.gradient_accumulation_steps
    )
    max_train_steps = model_config.n_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    n_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Override number of timesteps if set
    if model_config.num_train_timesteps:
        num_train_timesteps = model_config.num_train_timesteps 
    else:
        num_train_timesteps = noise_scheduler.config.num_train_timesteps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(group, config=asdict(model_config))


    # Some configurations require autocast to be disabled.
    enable_autocast = True
    if torch.backends.mps.is_available() or (
        accelerator.mixed_precision == "fp16" or accelerator.mixed_precision == "bf16"
    ):
        enable_autocast = False

    # ===================
    # === Train model ===
    # ===================
    total_batch_size = (
        model_config.train_batch_size
        * accelerator.num_processes
        * model_config.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {n_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {model_config.train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {model_config.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    if model_config.resume_from_checkpoint:
        if model_config.resume_from_checkpoint != "latest":
            path = os.path.basename(model_config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{model_config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(Path(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0


    progress_bar = tqdm.tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    for epoch in range(first_epoch, n_epochs):
        unet.train()
        if model_config.train_text_encoder:
            text_encoder_one.train()
            if text_encoder_two:
                text_encoder_two.train()

        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                model_input = vae.encode(batch["rgb"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(model_input)
                if model_config.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += model_config.noise_offset * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1),
                        device=model_input.device,
                    )

                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    num_train_timesteps,
                    (bsz,),
                    device=model_input.device,
                    dtype=torch.long,
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(
                    model_input, noise, timesteps
                )

                text_encoders = [text_encoder_one]
                text_input_ids_list = [batch["input_ids_one"]]
                if text_encoder_two:
                    text_encoders.append(text_encoder_two)
                    text_input_ids_list.append(batch["input_ids_two"])

    

                unet_added_conditions = {}
                if using_sdxl:
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=text_encoders, text_input_ids_list=text_input_ids_list
                    )
                    add_time_ids = torch.cat(
                        [
                            compute_time_ids(s, c, ts, accelerator.device, weight_dtype)
                            for s, c, ts in zip(
                                batch["original_size"],
                                batch["crop_top_left"],
                                batch["target_size"],
                            )
                        ]
                    )
                    unet_added_conditions["time_ids"] = add_time_ids
                    unet_added_conditions["text_embeds"] = pooled_prompt_embeds

                else:
                    prompt_embeds = text_encoders[0](text_input_ids_list[0])[0]

                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if model_config.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=model_config.prediction_type
                    )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                if model_config.snr_gamma is None:
                    loss = nn.functional.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, model_config.snr_gamma * torch.ones_like(timesteps)],
                        dim=1,
                    ).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = nn.functional.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                if config.get("debug_loss", False) and "meta" in batch:
                    for meta in batch["meta"]:
                        accelerator.log({"loss_for_" + meta.path: loss}, step=global_step)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(model_config.train_batch_size)).mean()
                train_loss += avg_loss.item() / model_config.gradient_accumulation_steps                    

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, model_config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % model_config.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if model_config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= model_config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - model_config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = Path(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if accelerator.is_main_process and ((global_step // model_config.train_batch_size // len(train_dataloader)) % model_config.val_freq) == 0:
            # Validation

            logger.info(
                f"Running validation..."
            )
            # create pipeline
            if using_sdxl:
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_config.model_id,
                    vae=vae.to(dtype=weight_dtype),
                    text_encoder=unwrap_model(text_encoder_one).to(dtype=weight_dtype),
                    text_encoder_2=unwrap_model(text_encoder_two).to(dtype=weight_dtype),
                    unet=unwrap_model(unet).to(dtype=weight_dtype),
                    revision=model_config.revision,
                    variant=model_config.variant,
                    torch_dtype=weight_dtype
                )

            else:
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_config.model_id,
                    vae=vae,
                    text_encoder=unwrap_model(text_encoder_one),
                    unet=unwrap_model(unet),
                    revision=model_config.revision,
                    variant=model_config.variant,
                    torch_dtype=weight_dtype
                )

            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            for step, batch in enumerate(val_dataloader):
                rgb = batch["rgb"]

                val_bs = len(rgb)
                random_seeds = np.arange(val_bs * step, val_bs * (step+1))  
                generator = [torch.Generator(device=accelerator.device).manual_seed(int(seed)) for seed in random_seeds]


                output_imgs = pipeline(image=rgb, prompt=batch["positive_prompt"], generator=generator, output_type="pt", strength=model_config.val_strength).images

                # Benchmark
                ssim_values = ssim_metric(
                    output_imgs.to(dtype=torch.float32, device=accelerator.device),
                    batch["rgb"].to(dtype=torch.float32, device=accelerator.device)
                )

                if len(ssim_values.shape) == 0:
                    ssim_values = [ssim_values.item()]

                ssim_value_dict = {str(i): float(v) for i, v in enumerate(ssim_values)}
                log_values = {"val_ssim": ssim_value_dict}
                accelerator.log(log_values, step=global_step)

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in output_imgs])
                        tracker.writer.add_images("val_images", np_images, epoch, dataformats="NHWC")

                    if tracker.name == "wandb":
                        tracker.log_images({
                            "ground_truth": [
                                wandb.Image(img, caption=f"{meta['dataset']} - {meta['scene']} - {meta['sample']}")
                                for (img, meta) in (zip(batch["rgb"], batch["meta"]))
                            ],
                            "val_images": [
                                wandb.Image(img, caption=f"{meta['dataset']} - {meta['scene']} - {meta['sample']}")
                                for (img, meta) in (zip(output_imgs, batch["meta"]))
                            ]
                        }, step=global_step)

            del pipeline
            torch.cuda.empty_cache()

     # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if model_config.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            if text_encoder_two:
                text_encoder_two = unwrap_model(text_encoder_two)

            text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_one))
            if text_encoder_two:
                text_encoder_2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_two))

        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        if using_sdxl:
            StableDiffusionXLImg2ImgPipeline.save_lora_weights(
                save_directory=output_dir,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_lora_layers,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers,
            )

        else:
            StableDiffusionImg2ImgPipeline.save_lora_weights(
                save_directory=output_dir,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_lora_layers
            )

        del unet
        del text_encoder_one
        del text_encoder_two
        del text_encoder_lora_layers
        del text_encoder_2_lora_layers

        torch.cuda.empty_cache()

        # Final inference
        # Make sure vae.dtype is consistent with the unet.dtype
        if mixed_precision == "fp16":
            vae.to(weight_dtype)

        if using_sdxl:
            pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_config.model_id,
                vae=vae,
                revision=model_config.revision,
                variant=model_config.variant,
                torch_dtype=weight_dtype,
            )

        else:
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_config.model_id,
                vae=vae,
                revision=model_config.revision,
                variant=model_config.variant,
                torch_dtype=weight_dtype
            )

        pipeline = pipeline.to(accelerator.device)
        pipeline.load_lora_weights(output_dir)

        for step, batch in enumerate(val_dataloader):
            val_bs = len(batch["rgb"])
            random_seeds = np.arange(val_bs * step, val_bs * (step+1))  
            generator = [torch.Generator(device=accelerator.device).manual_seed(int(seed)) for seed in random_seeds]

            with torch.autocast(
                accelerator.device.type,
                enabled=enable_autocast,
            ):
                output_imgs = pipeline(image=batch["rgb"], prompt=batch["positive_prompt"], generator=generator, output_type="pt", strength=model_config.val_strength).images

            # Benchmark
            ssim_values = ssim_metric(
                output_imgs.to(dtype=torch.float32, device=accelerator.device),
                batch["rgb"].to(dtype=torch.float32, device=accelerator.device))


            if len(ssim_values.shape) == 0:
                ssim_values = [ssim_values.item()]

            ssim_value_dict = {str(i): float(v) for i, v in enumerate(ssim_values)}
            log_values = {"test_ssim": ssim_value_dict}
            accelerator.log(log_values, step=global_step)

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in output_imgs])
                    tracker.writer.add_images("test_images", np_images, epoch, dataformats="NHWC")

                if tracker.name == "wandb":
                    tracker.log_images({
                        "ground_truth": [
                            wandb.Image(img, caption=f"{meta['dataset']} - {meta['scene']} - {meta['sample']}")
                            for (img, meta) in (zip(batch["rgb"], batch["meta"]))
                        ],
                        "test_images": [
                            wandb.Image(img, caption=f"{meta['dataset']} - {meta['scene']} - {meta['sample']}")
                            for (img, meta) in (zip(output_imgs, batch["meta"]))
                        ]
                    }, step=global_step)


        del pipeline
        torch.cuda.empty_cache()

    accelerator.end_training()



if __name__ == "__main__":
    args = parse_args()
    main(args)
