from typing import Any, Iterable
from pathlib import Path
from argparse import ArgumentParser
import logging
import os
from dataclasses import dataclass
import math
import itertools as it
import tqdm

import numpy as np
import torch
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

from src.data import setup_project, DynamicDataset
from src.diffusion import is_sdxl_model, is_sdxl_vae


check_min_version("0.28.0.dev0")
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
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
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


def preprocess_rgbs(
    rgbs, target_size: tuple[int, int], resizer, flipper, cropper, center_crop: bool
):
    rgbs_pp = []
    original_sizes = []
    target_sizes = []
    crop_top_lefts = []

    for rgb in rgbs:
        th, tw = target_size
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

        original_size = h0, w0
        crop_top_left = cy, cx

        original_sizes.append(original_size)
        crop_top_lefts.append(crop_top_left)
        rgbs_pp.append(rgb)
        target_sizes.append(target_size)

    return {"rgb": rgbs_pp, "original_size": original_sizes, "crop_top_left": crop_top_lefts, "target_size": target_sizes}


def preprocess_prompts(prompt: list[str], tokenizers):
    assert 1 <= len(tokenizers) <= 2

    if isinstance(prompt, str):
        prompt = [prompt]

    input_ids_one = tokenize_prompt(tokenizers[0], prompt)
    input_ids_two = tokenize_prompt(tokenizers[1], prompt) if len(tokenizers) > 1 else None

    return {
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two
    }


def preprocess_batch(
    batch, resizer, flipper, cropper, tokenizers, target_size, center_crop: bool = False
):
    # Ignore negative prompt :)

    rgbs = preprocess_rgbs(batch["rgb"], target_size, resizer, flipper, cropper, center_crop)
    prompts = preprocess_prompts(batch["prompt"]["positive_prompt"], tokenizers)

    return {
        "original_size": rgbs["original_size"],
        "crop_top_left": rgbs["crop_top_left"],
        "target_size": rgbs["target_size"],
        "rgb": rgbs["rgb"],
        "input_ids_one": prompts["input_ids_one"],
        "input_ids_two": prompts["input_ids_two"]
    }


@dataclass(init=True)
class TrainLoraConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Path to pretrained VAE model with better numerical stability.
    # More details: https://githugb.com/huggingface/diffusers/pull/4038.
    vae_id: str = "madebyollin/sdxl-vae-fp16-fix"

    report_to: str = "wandb"
    mixed_precision: str = None
    revision: str = None
    variant: str = None

    train_text_encoder: bool = False
    gradient_checkpointing: bool = False
    resume_from_checkpoint: bool = False
    n_epochs: int = 100

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
    
    # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
    lr_scheduler = "constant"
    lr_warmup_steps = 500

    lora_rank: int = 4
    prompt: str = "dashcam video, self-driving car, urban driving scene"
    negative_prompt: str = ""

    image_width: int = 1024
    image_height: int = 1024

    center_crop: bool = False
    flip_prob: float = 0.5

    push_to_hub: bool = False       # Not Implemented
    hub_token: str = None
    seed: int = None


def main(args):
    # ========================
    # ===   Setup script   ===
    # ========================

    config_path = args.config_path
    config = setup_project(config_path)

    output_dir = Path(config["output_dir"])
    logging_dir = Path(output_dir, "logs")

    model_config = TrainLoraConfig(**config["model"])
    using_sdxl = is_sdxl_model(model_config.model_id)

    if (model_config.vae_id is not None) and (
        (using_sdxl and not is_sdxl_vae(model_config.vae_id)) or 
        (not using_sdxl and is_sdxl_vae(model_config.vae_id))
    ):
        raise ValueError(f"Mismatch between model_id and vae_id. Both models need to either be SDXL or SD, but received {model_config.model_id} and {model_config.vae_id}")

    if model_config.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
     
        if model_config.hub_token is not None:
            raise ValueError(
                "You cannot use both report_to=wandb and hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

        import wandb


    if torch.backends.mps.is_available() and model_config.mixed_precision == "bf16":
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
        mixed_precision=model_config.mixed_precision,
        log_with=model_config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_kwargs],
    )

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

        if model_config.push_to_hub:
            raise NotImplementedError

    # =======================
    # ===   Load models   ===
    # =======================
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


    if model_config.vae_id is None:     # Need VAE fix for stable diffusion XL, see https://github.com/huggingface/diffusers/pull/4038
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

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    if text_encoder_two:
        text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if model_config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif model_config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)

    if using_sdxl and model_config.vae_id is None:      # The VAE in SDXL is in float32 to avoid NaN losses.
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
                state_model_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))

                if isinstance(unwrapped_model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = state_model_dict

                elif isinstance(
                    unwrapped_model, type(unwrap_model(text_encoder_one))
                ):
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
                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                    text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
                )

            else:
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=output_dir, 
                    unet_lora_layers=unet_lora_layers_to_save, text_encoder_lora_layers=text_encoder_one_lora_layers_to_save
                )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while (model := models.pop()):
            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model

            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model

            elif text_encoder_two and isinstance(model, type(unwrap_model(text_encoder_two))):
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
        if model_config.mixed_precision == "fp16":
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


    models_to_optimize = [unet]
    if model_config.train_text_encoder:
        models_to_optimize.append(text_encoder_one)
        if text_encoder_two:
            models_to_optimize.append(text_encoder_two)

    params_to_optimize = it.chain(filter(lambda p: p.requires_grad, model.parameters()) for model in models_to_optimize)
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
        train_dataset.data_getters["rgb"].set_extra_transforms(
            train_resizer, train_flipper, train_cropper
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=model_config.train_batch_size,
        num_workers=model_config.dataloader_num_workers
    )

    # ==========================
    # ===   Setup training   ===
    # ==========================

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / model_config.gradient_accumulation_steps)
    max_train_steps = model_config.n_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        model_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=model_config.lr_warmup_steps * model_config.gradient_accumulation_steps,
        num_training_steps=max_train_steps * model_config.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if model_config.train_text_encoder:
        if text_encoder_two:        
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
            )

        else:
            unet, text_encoder_one, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder_one, optimizer, train_dataloader, lr_scheduler
            )

    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = model_config.n_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    n_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Some configurations require autocast to be disabled.
    enable_autocast = True
    if torch.backends.mps.is_available() or (
        accelerator.mixed_precision == "fp16" or accelerator.mixed_precision == "bf16"
    ):
        enable_autocast = False

    # ===================
    # === Train model ===
    # ===================
    total_batch_size = model_config.train_batch_size * accelerator.num_processes * model_config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {model_config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {model_config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {model_config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    if model_config.resume_from_checkpoint:
        # TODO: Implement resuming
        raise NotImplementedError

    else:
        initial_global_step = 0

    
    progress_bar = tqdm.tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, model_config.num_train_epochs):



if __name__ == "__main__":
    args = parse_args()
    main(args)
