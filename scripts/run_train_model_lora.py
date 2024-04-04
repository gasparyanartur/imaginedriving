from typing import Any, Iterable
from pathlib import Path
from argparse import ArgumentParser
import logging
import os
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
import torchvision.transforms.v2 as transforms
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
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
import wandb

from src.data import setup_project, DynamicDataset


check_min_version("0.23.0.dev0")
logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = ArgumentParser(
        "train_model", description="Finetune a given model on a dataset"
    )
    parser.add_argument("config_path", type=Path)

    args = parser.parse_args()
    return args


def embed_prompt(prompt: str, tokenizers, text_encoders):
    prompt_embeds_list = []
    captions = [prompt]

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

    return {
        "prompt_embeds": prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
    }


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
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
    batch, resizer, flipper, cropper, target_size, center_crop: bool = False
):
    all_rgbs = []
    original_sizes = []
    crop_top_lefts = []

    th, tw = target_size

    for rgb in batch["rgb"]:
        h0, w0 = rgb.shape[-2:]

        rgb = resizer(rgb)
        rgb = flipper(rgb)

        if center_crop:
            cy = (h0 - th) // 2
            cx = (w0 - tw) // 2
            rgb = cropper(rgb)

        else:
            cy, cx, h, w = cropper.get_params(rgb, target_size)
            rgb = transforms.functional.crop(rgb, cy, cx, h, w)

        original_sizes.append((h0, w0))
        crop_top_lefts.append((cy, cx))
        all_rgbs.append(rgb)

    return {
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
        "pixel_values": all_rgbs,
    }


@dataclass(init=True)
class TrainLoraConfig:
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Path to pretrained VAE model with better numerical stability.
    # More details: https://github.com/huggingface/diffusers/pull/4038.
    vae_id: str = "madebyollin/sdxl-vae-fp16-fix"

    report_to: str = "wandb"
    mixed_precision: str = None
    revision: str = None
    variant: str = None

    lora_rank: int = 4
    train_text_encoder: bool = False
    gradient_checkpointing: bool = False

    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    allow_tf32: bool = True
    scale_lr: bool = False
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 16
    dataloader_num_workers: int = 0
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08

    prompt: str = "dashcam video, self-driving car, urban driving scene"
    negative_prompt: str = ""

    image_width: int = 1024
    image_height: int = 1024

    center_crop: bool = False
    flip_prob: float = 0.5

    hub_token: str = None


def main(args):
    # ========================
    # ===   Setup script   ===
    # ========================

    config_path = args.config_path
    config = setup_project(config_path)

    output_dir = Path(config["output_dir"])
    logging_dir = output_dir / "logs"

    model_config = TrainLoraConfig(**config["model"])

    # Sanity checks
    if model_config.report_to == "wandb" and model_config.hub_token is not None:
        raise ValueError(
            "You cannot use both report_to=wandb and hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if model_config.report_to == "wandb":
        assert is_wandb_available()

    if torch.backends.mps.is_available() and model_config.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Setup accelerator

    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )
    accelerator_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=model_config.mixed_precision,
        log_with=model_config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_kwargs],
    )

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if (seed := model_config.get("seed")) is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        output_dir.mkdir(exist_ok=True, parents=True)
        logging_dir.mkdir(exist_ok=True)

        if model_config["push_to_hub"]:
            # repo_id = create_repo(
            #    repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            # ).repo_id
            raise NotImplementedError

    # =======================
    # ===   Load models   ===
    # =======================

    tokenizer_one = AutoTokenizer.from_pretrained(
        model_config.model_id,
        subfolder="tokenizer",
        revision=model_config.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        model_config.model_id,
        subfolder="tokenizer_2",
        revision=model_config.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        model_config.model_id, model_config.revision, subfolder="text_encoder"
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        model_config.model_id, model_config.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_config.model_id, subfolder="scheduler"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        model_config.model_id,
        subfolder="text_encoder",
        revision=model_config.revision,
        variant=model_config.variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        model_config.model_id,
        subfolder="text_encoder_2",
        revision=model_config.revision,
        variant=model_config.variant,
    )

    vae = AutoencoderKL.from_pretrained(
        model_config.vae_id
        or model_config.model_id,  # Custom VAE is preferred due to bug
        subfolder="vae" if model_config.vae_id is None else None,
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

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if model_config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif model_config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    unet.to(accelerator.device, dtype=weight_dtype)

    if model_config.vae_id is None:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_config = LoraConfig(
        r=model_config.lora_rank,
        lora_alpha=model_config.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if model_config.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_config = LoraConfig(
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

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
                if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(
                    unwrap_model(model), type(unwrap_model(text_encoder_one))
                ):
                    text_encoder_one_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )
                elif isinstance(
                    unwrap_model(model), type(unwrap_model(text_encoder_two))
                ):
                    text_encoder_two_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
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

            _set_state_dict_into_text_encoder(
                lora_state_dict,
                prefix="text_encoder_2.",
                text_encoder=text_encoder_two_,
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if model_config.mixed_precision == "fp16":
            models = [unet_]
            if model_config.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if model_config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if model_config.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # Note: this applies to the A100
    if model_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if model_config.scale_lr:
        learning_rate = (
            learning_rate
            * model_config.gradient_accumulation_steps
            * model_config.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if model_config.mixed_precision == "fp16":
        models = [unet]
        if model_config.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_text_encoder:
        params_to_optimize = (
            params_to_optimize
            + list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
            + list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
        )

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

    prompt = config["prompt"]
    prompt_embed = embed_prompt(
        prompt, (tokenizer_one, tokenizer_two), (text_encoder_one, text_encoder_two)
    )

    dataset_config = config["datasets"]
    train_dataset_config = dataset_config["train_data"]
    val_dataset_config = dataset_config["val_data"]

    train_dataset = DynamicDataset.from_config(train_dataset_config)
    val_dataset = DynamicDataset.from_config(val_dataset_config)

    # Preprocessing
    target_size = (model_config.img_height, model_config.img_width)
    resolution = min(target_size)

    train_resizer = transforms.Resize(
        resolution, interpolation=transforms.InterpolationMode.BILINEAR
    )
    train_flipper = transforms.RandomHorizontalFlip(p=model_config.flip_prob)
    train_cropper = (
        transforms.CenterCrop(target_size)
        if model_config.center_crop
        else transforms.RandomCrop(target_size)
    )

    with accelerator.main_process_first():
        train_dataset.shuffle_index()
        train_dataset.data_getters["rgb"].set_extra_transforms(
            train_resizer, train_flipper, train_cropper
        )

    def collate_fn(batch: Iterable[dict[str, Any]]) -> dict[str, Iterable[Any]]:
        # TODO

        result = {
            "pixel_values": None,
            "prompt_embeds_one": None,
            "prompt_embeds_two": None,
            "original_sizes": None,
            "crop_top_lefts": None
        }
        return None
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=model_config.train_batch_size,
        num_workers=model_config.dataloader_num_workers
    )

    # =======================
    # ===   Train model   ===
    # =======================

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    # TODO
    ...


if __name__ == "__main__":
    args = parse_args()
    main(args)
