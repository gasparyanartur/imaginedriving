# Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

from typing import Any, Iterable
from pathlib import Path
from argparse import ArgumentParser
import logging
import os
from dataclasses import dataclass, asdict, field
import math
import itertools as it
import torch.utils
import torch.utils.data
import tqdm
import shutil
import functools

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
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
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
from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.data import setup_project, DynamicDataset
from src.diffusion import is_sdxl_model, is_sdxl_vae, get_noised_img
from src.diffusion import tokenize_prompt
from src.diffusion import encode_tokens
from src.utils import get_env


check_min_version("0.27.0")
logger = get_logger(__name__, log_level="INFO")


@dataclass(init=True)
class LoraTrainState:
    project_name: str = "ImagineDriving"
    project_dir: str = None
    cache_dir: str = None
    datasets: dict[str, Any] = field(default_factory=dict)

    model_id: str = "stabilityai/stable-diffusion-2-1"
    # model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Path to pretrained VAE model with better numerical stability.
    # More details: https://github.com/huggingface/diffusers/pull/4038.
    # vae_id: str = "madebyollin/sdxl-vae-fp16-fix"
    vae_id: str = None

    weights_dtype: str = "fp32"
    vae_dtype: str = "fp32"

    revision: str = None
    variant: str = None
    prediction_type: str = None

    gradient_checkpointing: bool = False
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = None
    resume_from_checkpoint: bool = False
    n_epochs: int = 100
    max_train_samples: int = None
    val_freq: int = 10

    train_noise_strength: float = 0.5
    val_noise_num_steps: int = 30
    val_noise_strength: float = 0.25

    keep_vae_full: bool = True

    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
    noise_offset: float = 0

    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    allow_tf32: bool = True
    torch_backend: str = "cudagraphs"

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

    max_train_steps: int = None  # Gets set later
    num_update_steps_per_epoch: int = None  # Gets set later

    global_step: int = 0
    epoch = 0

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

    trainable_models: list[str] = ("unet",)

    loggers: list[str] = ("wandb",)
    wandb_project: str = "ImagineDriving"
    wandb_entity: str = "arturruiqi"
    wandb_group: str = "finetune-lora"

    output_dir: str = None
    logging_dir: str = None

    push_to_hub: bool = False  # Not Implemented
    hub_token: str = None

    seed: int = None


_lower_dtypes = {"fp16", "bf16"}
_dtype_conversion = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def find_checkpoint_paths(cp_dir: str, cp_prefix: str = "checkpoint", cp_delim="-"):
    cp_paths = os.listdir(cp_dir)
    cp_paths = [d for d in cp_paths if d.startswith(cp_prefix)]
    cp_paths = sorted(cp_paths, key=lambda x: int(x.split(cp_delim)[1]))
    return cp_paths


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


def preprocess_rgb(rgb, preprocessors, target_size: tuple[int, int], center_crop: bool):
    # preprocess rgb:
    # out: rgb, original_size, crop_top_left, target_size

    th, tw = target_size
    h0, w0 = rgb.shape[-2:]

    if resizer := preprocessors.get("resizer"):
        rgb = resizer(rgb)

    if flipper := preprocessors.get("flipper"):
        rgb = flipper(rgb)

    if cropper := preprocessors.get("cropper"):
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


def preprocess_prompt(prompt: list[str], models):
    assert "tokenizer" in models

    if isinstance(prompt, str):
        prompt = [prompt]

    input_ids = tokenize_prompt(models["tokenizer"], prompt)[0]
    return {"input_ids": input_ids}


def preprocess_sample(batch, preprocessors):
    rgb = preprocessors["rgb"](batch["rgb"])
    prompt = preprocessors["prompt"](batch["prompt"])

    sample = {
        "rgb": rgb["rgb"],
        "original_size": rgb["original_size"],
        "crop_top_left": rgb["crop_top_left"],
        "target_size": rgb["target_size"],
        "input_ids": prompt["input_ids"],
        "meta": batch["meta"],
        "positive_prompt": batch["prompt"]["positive_prompt"],
    }

    return sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[list[str], Any]:
    original_size = [sample["original_size"] for sample in batch]
    crop_top_left = [sample["crop_top_left"] for sample in batch]
    target_size = [sample["target_size"] for sample in batch]
    rgb = torch.stack([sample["rgb"] for sample in batch]).to(
        memory_format=torch.contiguous_format, dtype=torch.float32
    )
    meta = [sample["meta"] for sample in batch]
    input_ids_one = torch.stack([sample["input_ids_one"] for sample in batch])
    positive_prompt = [sample["positive_prompt"] for sample in batch]

    input_ids_two = (
        torch.stack([sample["input_ids_two"] for sample in batch])
        if "input_ids_two" in batch[0]
        else None
    )

    batch = {
        "original_size": original_size,
        "crop_top_left": crop_top_left,
        "target_size": target_size,
        "rgb": rgb,
        "input_ids_one": input_ids_one,
        "meta": meta,
        "positive_prompt": positive_prompt,
    }

    if input_ids_two:
        batch["input_ids_two"] = input_ids_two

    return batch


def save_model_hook(loaded_models, weights, accelerator, models, lora_train_state):
    if not accelerator.is_main_process:
        return

    # there are only two options here. Either are just the unet attn processor layers
    # or there are the unet and text encoder attn layers
    layers_to_save = {}

    while loaded_model := loaded_models.pop():
        # Map the list of loaded_models given by accelerator to keys given in lora_train_state.
        # NOTE: This mapping is done by type, so two objects of the same type will be treated as the same object.
        # TODO: Find a better way of mapping this.

        unwrapped_model = unwrap_model(accelerator, loaded_model)
        state_model_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(loaded_model)
        )
        for model_name, model in models.items():
            if isinstance(unwrapped_model, unwrap_model(accelerator, model)):
                layers_to_save[model_name] = state_model_dict
                break

        # make sure to pop weight so that corresponding model is not saved again
        if weights:
            weights.pop()

    # TODO: Extend for more models
    StableDiffusionImg2ImgPipeline.save_lora_weights(
        save_directory=lora_train_state.output_dir,
        unet_lora_layers=layers_to_save.get("unet"),
        text_encoder_lora_layers=layers_to_save.get("text_encoder"),
    )


def load_model_hook(
    loaded_models,
    input_dir,
    accelerator: Accelerator,
    models,
    lora_train_state: LoraTrainState,
):
    loaded_models_dict = {}

    while loaded_model := loaded_models.pop():
        # Map the list of loaded_models given by accelerator to keys given in lora_train_state.
        # NOTE: This mapping is done by type, so two objects of the same type will be treated as the same object.
        # TODO: Find a better way of mapping this.

        for model_name, model in models.items():
            if isinstance(loaded_model, unwrap_model(accelerator, model)):
                loaded_models_dict[model_name] = loaded_model
                break

        else:
            raise ValueError(f"unexpected save model: {loaded_model.__class__}")

    lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)

    if "unet" in models:
        unet_state_dict = {
            f'{k.replace("unet.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            loaded_models_dict["unet"], unet_state_dict, adapter_name="default"
        )

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

    if "text_encoder" in models:
        _set_state_dict_into_text_encoder(
            lora_state_dict,
            prefix="text_encoder.",
            text_encoder=loaded_models_dict["text_encoder"],
        )

    # Make sure the trainable params are in float32.
    if lora_train_state.weights_dtype in _lower_dtypes:
        models_to_cast = [
            loaded_model
            for model_name, loaded_model in loaded_models_dict.items()
            if model_name in lora_train_state.trainable_models
        ]
        cast_training_params(models_to_cast, dtype=torch.float32)


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def nearest_multiple(x: float | int, m: int) -> int:
    return int(int(x / m) * m)


def prepare_preprocessors(models, lora_train_state: LoraTrainState):
    crop_height = lora_train_state.crop_height or lora_train_state.image_height
    crop_width = lora_train_state.crop_width or lora_train_state.image_width
    resize_factor = lora_train_state.resize_factor or 1

    crop_size = (
        nearest_multiple(crop_height * resize_factor, 8),
        nearest_multiple(crop_width * resize_factor, 8),
    )
    downsample_size = (
        nearest_multiple(lora_train_state.image_height * resize_factor, 8),
        nearest_multiple(lora_train_state.image_width * resize_factor, 8),
    )

    preprocessors = {"train": {}, "val": {}}

    _train_rgb_preprocess_steps = {}
    _val_rgb_preprocess_steps = {}

    if resize_factor != 1:
        _train_rgb_preprocess_steps["resizer"] = transforms.Resize(
            downsample_size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        _val_rgb_preprocess_steps["resizer"] = transforms.Resize(
            downsample_size, interpolation=transforms.InterpolationMode.BILINEAR
        )

    if lora_train_state.flip_prob > 0:
        _train_rgb_preprocess_steps["flipper"] = transforms.RandomHorizontalFlip(
            p=lora_train_state.flip_prob
        )

    if (crop_height != lora_train_state.image_height) or (
        crop_width != lora_train_state.image_width
    ):
        _train_rgb_preprocess_steps["cropper"] = (
            transforms.CenterCrop(crop_size)
            if lora_train_state.center_crop
            else transforms.RandomCrop(crop_size)
        )
        _val_rgb_preprocess_steps["cropper"] = transforms.CenterCrop(crop_size)

    preprocessors["train"]["rgb"] = functools.partial(
        preprocess_rgb,
        preprocessor=_train_rgb_preprocess_steps,
        target_size=crop_size,
        center_crop=lora_train_state.center_crop,
    )

    preprocessors["val"]["rgb"] = functools.partial(
        preprocess_rgb,
        preprocessor=_val_rgb_preprocess_steps,
        target_size=crop_size,
        center_crop=lora_train_state.center_crop,
    )

    preprocessors["train"]["prompt"] = functools.partial(
        preprocess_prompt, models=models
    )

    preprocessors["val"]["prompt"] = functools.partial(preprocess_prompt, models=models)

    return preprocessors


def get_diffusion_cls(
    model_id: str,
) -> StableDiffusionImg2ImgPipeline | StableDiffusionXLImg2ImgPipeline:
    if is_sdxl_model(model_id):
        return StableDiffusionXLImg2ImgPipeline
    else:
        return StableDiffusionImg2ImgPipeline


def save_lora_weights(
    accelerator: Accelerator, models, train_state: LoraTrainState
) -> None:
    lora_state_dicts = {}

    if "unet" in train_state.trainable_models:
        lora_state_dicts["unet_lora_layers"] = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrap_model(accelerator, models["unet"]))
        )

    if "text_encoder" in train_state.trainable_models:
        lora_state_dicts["text_encoder_lora_layers"] = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrap_model(accelerator, models["text_encoder"]))
        )

    get_diffusion_cls(train_state.model_id).save_lora_weights(
        save_directory=train_state, **lora_state_dicts
    )


def get_diffusion_noise(
    size: Iterable[int], device: torch.device, train_state: LoraTrainState
) -> Tensor:
    noise = torch.randn(size)
    if train_state.noise_offset:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += train_state.noise_offset * torch.randn(
            (size[0], size[1], 1, 1),
            device=device,
        )
    return noise


def get_diffusion_loss(
    models, train_state: LoraTrainState, model_input, model_pred, noise, timesteps
):
    noise_scheduler = models["noise_scheduler"]

    match noise_scheduler.config.prediction_type:
        case "epsilon":
            target = noise
        case "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        case _:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

    # Get the target for loss depending on the prediction type
    if train_state.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=train_state.prediction_type)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )

    if train_state.snr_gamma is None:
        loss = nn.functional.mse_loss(
            model_pred.float(), target.float(), reduction="mean"
        )
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack(
            [snr, train_state.snr_gamma * torch.ones_like(timesteps)],
            dim=1,
        ).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = nn.functional.mse_loss(
            model_pred.float(), target.float(), reduction="none"
        )
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

    return loss


def get_timesteps(noise_strength, total_num_timesteps, device, batch_size):
    # Sample a random timestep for each image
    timesteps = torch.randint(
        int((1 - noise_strength) * total_num_timesteps),
        total_num_timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    return timesteps


def save_checkpoint(accelerator: Accelerator, lora_train_state: LoraTrainState) -> None:
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if lora_train_state.checkpoints_total_limit is not None:
        cp_paths = find_checkpoint_paths(lora_train_state.output_dir)

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(cp_paths) >= lora_train_state.checkpoints_total_limit:
            num_to_remove = len(cp_paths) - lora_train_state.checkpoints_total_limit + 1
            cps_to_remove = cp_paths[0:num_to_remove]

            logger.info(
                f"{len(cp_paths)} checkpoints already exist, removing {len(cps_to_remove)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(cps_to_remove)}")

            for cp_path in cps_to_remove:
                cp_path = os.path.join(lora_train_state.output_dir, cp_path)
                shutil.rmtree(cp_path)

    cp_path = os.path.join(
        lora_train_state.output_dir, f"checkpoint-{lora_train_state.global_step}"
    )
    accelerator.save_state(cp_path)
    logger.info(f"Saved state to {cp_path}")


def resume_from_checkpoint(accelerator, train_state: LoraTrainState):
    if train_state.resume_from_checkpoint:

        if train_state.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            cp_paths = find_checkpoint_paths(train_state.output_dir)

            if len(cp_paths) == 0:
                accelerator.print(
                    f"Checkpoint '{train_state.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                return

            path = cp_paths[-1]

        else:
            path = os.path.basename(train_state.resume_from_checkpoint)

        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(Path(args.output_dir, path))

        train_state.global_step = int(path.split("-")[1])
        train_state.epoch = (
            train_state.global_step // train_state.num_update_steps_per_epoch
        )

    else:
        train_state.initial_global_step = 0


def prepare_models(lora_train_state: LoraTrainState, device):
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    if is_sdxl_vae(lora_train_state.vae_id) or lora_train_state.keep_vae_full:
        # The VAE in SDXL is in float32 to avoid NaN losses.
        lora_train_state.vae_dtype = "fp32"
    else:
        lora_train_state.vae_dtype = lora_train_state.weights_dtype

    models = {}
    models["noise_scheduler"] = DDPMScheduler.from_pretrained(
        lora_train_state.model_id, subfolder="scheduler"
    )
    models["tokenizer"] = AutoTokenizer.from_pretrained(
        lora_train_state.model_id,
        subfolder="tokenizer",
        revision=lora_train_state.revision,
        use_fast=False,
    )
    text_encoder_cls = import_text_encoder_class_from_model_name_or_path(
        lora_train_state.model_id, lora_train_state.revision, subfolder="text_encoder"
    )
    models["text_encoder"] = text_encoder_cls.from_pretrained(
        lora_train_state.model_id,
        subfolder="text_encoder",
        revision=lora_train_state.revision,
        variant=lora_train_state.variant,
    )
    models["vae"] = (
        AutoencoderKL.from_pretrained(lora_train_state.vae_id, subfolder=None)
        if (
            lora_train_state.vae_id
            # Need VAE fix for stable diffusion XL, see https://github.com/huggingface/diffusers/pull/4038
        )
        else AutoencoderKL.from_pretrained(
            lora_train_state.model_id,
            subfolder="vae",
            revision=lora_train_state.revision,
            variant=lora_train_state.variant,
        )
    )
    models["unet"] = UNet2DConditionModel.from_pretrained(
        lora_train_state.model_id,
        subfolder="unet",
        revision=lora_train_state.revision,
        variant=lora_train_state.variant,
    )
    models["image_processor"] = VaeImageProcessor()

    # ===================
    # === Config Lora ===
    # ===================

    if lora_train_state.compile_model:
        models["vae"] = torch.compile(
            models["vae"], backend=lora_train_state.torch_backend
        )
        models["unet"] = torch.compile(
            models["unet"], backend=lora_train_state.torch_backend
        )
        models["text_encoder"] = torch.compile(
            models["text_encoder"], backend=lora_train_state.torch_backend
        )

    # We only train the additional adapter LoRA layers
    models["vae"].requires_grad_(False)
    models["unet"].requires_grad_(False)
    models["text_encoder"].requires_grad_(False)

    models["vae"].to(device, dtype=lora_train_state.vae_dtype)
    models["unet"].to(device, dtype=lora_train_state.weights_dtype)
    models["text_encoder"].to(device, dtype=lora_train_state.weights_dtype)

    if "unet" in lora_train_state.trainable_models:
        models["unet"].add_adapter(
            LoraConfig(
                r=lora_train_state.lora_rank,
                lora_alpha=lora_train_state.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
        )

    if "text_encoder" in lora_train_state.trainable_models:
        models["text_encoder"].add_adapter(
            LoraConfig(
                r=lora_train_state.lora_rank,
                lora_alpha=lora_train_state.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
        )

    for model_name in lora_train_state.trainable_models:
        if model_name in models:
            if lora_train_state.gradient_checkpointing:
                models[model_name].enable_gradient_checkpointing()

    # Make sure the trainable params are in float32.
    cast_training_params(
        [models[model_name] for model_name in lora_train_state.trainable_models],
        dtype=torch.float32,
    )

    return models


def validate_model(
    accelerator,
    train_state: LoraTrainState,
    dataloader: torch.utils.data.DataLoader,
    models,
    metrics: dict[str, Any],
    run_prefix: str,
) -> None:
    logging.info(f"Running: {run_prefix}")

    using_wandb = any(tracker.name == "wandb" for tracker in accelerator.trackers)
    if using_wandb:
        import wandb

    weights_dtype = _dtype_conversion[train_state.weights_dtype]
    vae_dtype = _dtype_conversion[train_state.vae_dtype]
    noise_scheduler = models["noise_scheduler"]

    # create pipeline
    pipeline_args = {}

    if "vae" in models:
        pipeline_args["vae"] = models["vae"].to(dtype=vae_dtype)

    if "text_encoder" in models:
        pipeline_args["text_encoder"] = unwrap_model(
            accelerator, models["text_encoder"]
        )

    if "unet" in models:
        pipeline_args["unet"] = unwrap_model(accelerator, models["unet"])

    # TODO: Support SDXL
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        train_state.model_id,
        torch_dtype=weights_dtype,
        revision=train_state.revision,
        variant=train_state.variant,
        **pipeline_args,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    noise_scheduler.set_timesteps(train_state.val_noise_num_steps)

    for step, batch in enumerate(dataloader):
        input_imgs = batch["rgb"].to(dtype=weights_dtype, device=accelerator.device)
        batch_size = len(input_imgs)
        random_seeds = np.arange(batch_size * step, batch_size * (step + 1))

        generator = [
            torch.Generator(device=accelerator.device).manual_seed(int(seed))
            for seed in random_seeds
        ]

        output_imgs = pipeline(
            image=input_imgs,
            prompt=batch["positive_prompt"],
            generator=generator,
            output_type="pt",
            strength=train_state.val_noise_strength,
        ).images.to(dtype=weights_dtype, device=accelerator.device)

        val_start_timestep = int(
            (1 - train_state.val_noise_strength) * len(noise_scheduler.timesteps)
        )
        noised_imgs = get_noised_img(
            input_imgs,
            timestep=val_start_timestep,
            pipe=pipeline,
            noise_scheduler=noise_scheduler,
            device=accelerator.device,
        )

        # Benchmark
        for metric_name, metric in metrics.items():
            values = metric(output_imgs, input_imgs)
            if len(values.shape) == 0:
                values = [values.item()]

            value_dict = {str(i): float(v) for i, v in enumerate(values)}
            accelerator.log(
                {f"{run_prefix}_{metric_name}": value_dict},
                step=train_state.global_step,
            )

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_images(
                    f"{run_prefix}_images",
                    np.stack([np.asarray(img) for img in output_imgs]),
                    train_state.epoch,
                    dataformats="NHWC",
                )

            if tracker.name == "wandb" and using_wandb:
                tracker.log_images(
                    {
                        "ground_truth": [
                            wandb.Image(
                                img,
                                caption=f"{meta['dataset']} - {meta['scene']} - {meta['sample']}",
                            )
                            for (img, meta) in (zip(batch["rgb"], batch["meta"]))
                        ],
                        f"{run_prefix}_images": [
                            wandb.Image(
                                img,
                                caption=f"{meta['dataset']} - {meta['scene']} - {meta['sample']}",
                            )
                            for (img, meta) in (zip(output_imgs, batch["meta"]))
                        ],
                        "noised_images": [
                            wandb.Image(
                                img,
                                caption=f"{meta['dataset']} - {meta['scene']} - {meta['sample']}",
                            )
                            for (img, meta) in (zip(noised_imgs, batch["meta"]))
                        ],
                    },
                    step=train_state.global_step,
                )


def train_epoch(
    config: dict[str, Any],
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_state: LoraTrainState,
    dataloader: torch.utils.data.DataLoader,
    models,
    params_to_optimize,
    progress_bar: tqdm.tqdm,
):
    weights_dtype = _dtype_conversion[train_state.weights_dtype]
    vae_dtype = _dtype_conversion[train_state.vae_dtype]
    using_sdxl = is_sdxl_model(train_state.model_id)
    noise_scheduler = models["noise_scheduler"]

    models["unet"].train()
    if "text_encoder" in train_state.trainable_models:
        models["text_encoder"].train()

    train_loss = 0.0
    for batch in dataloader:
        with accelerator.accumulate(models["unet"]):
            model_input = (
                models["vae"]
                .encode(
                    models["image_processor"].preprocess(batch["rgb"]).to(vae_dtype)
                )
                .latent_dist.sample()
                * models["vae"].config.scaling_factor
            ).to(weights_dtype)

            noise = get_diffusion_noise(
                model_input.size, model_input.device, train_state
            )
            timesteps = get_timesteps(
                train_state.train_noise_strength,
                noise_scheduler.config.num_train_timesteps,
                model_input.device,
                model_input.size(0),
            )

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            unet_added_conditions = {}
            token_embed_outputs = encode_tokens(
                models["text_encoder"], batch["input_ids"], using_sdxl
            )

            if using_sdxl:
                add_time_ids = torch.cat(
                    [
                        compute_time_ids(s, c, ts, accelerator.device, weights_dtype)
                        for s, c, ts in zip(
                            batch["original_size"],
                            batch["crop_top_left"],
                            batch["target_size"],
                        )
                    ]
                )
                unet_added_conditions["time_ids"] = add_time_ids
                unet_added_conditions["pooled_embeds"] = token_embed_outputs[
                    "pooled_embeds"
                ]

            model_pred = models["unet"](
                noisy_model_input,
                timesteps,
                token_embed_outputs["embeds"],
                added_cond_kwargs=unet_added_conditions,
                return_dict=False,
            )[0]

            loss = get_diffusion_loss(
                models, train_state, model_input, model_pred, noise, timesteps
            )

            if config.get("debug_loss", False) and "meta" in batch:
                for meta in batch["meta"]:
                    accelerator.log(
                        {"loss_for_" + meta.path: loss}, step=train_state.global_step
                    )

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(
                loss.repeat(train_state.train_batch_size)
            ).mean()
            train_loss += avg_loss.item() / train_state.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    params_to_optimize, train_state.max_grad_norm
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            train_state.global_step += 1
            accelerator.log({"train_loss": train_loss}, step=train_state.global_step)
            train_loss = 0.0

            if accelerator.is_main_process and (
                train_state.global_step % train_state.checkpointing_steps == 0
            ):
                save_checkpoint(accelerator, train_state)

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)


def main(args):
    # ========================
    # ===   Setup script   ===
    # ========================

    config = setup_project(args.config_path)

    train_state = LoraTrainState(**config["model"])

    using_sdxl = is_sdxl_model(train_state.model_id)
    if (train_state.vae_id is not None) and (
        (using_sdxl and not is_sdxl_vae(train_state.vae_id))
        or (not using_sdxl and is_sdxl_vae(train_state.vae_id))
    ):
        raise ValueError(
            f"Mismatch between model_id and vae_id. Both models need to either be SDXL or SD, but received {train_state.model_id} and {train_state.vae_id}"
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=train_state.gradient_accumulation_steps,
        mixed_precision=train_state.weights_dtype,
        log_with=train_state.loggers,
        project_config=ProjectConfiguration(
            project_dir=train_state.output_dir,
            logging_dir=train_state.logging_dir,
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    logging.info(
        f"Number of cuda detected devices: {torch.cuda.device_count()}, Using device: {accelerator.device}, distributed: {accelerator.distributed_type}"
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

    if train_state.seed is not None:
        set_seed(train_state.seed)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # Note: this applies to the A100
    if train_state.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if torch.backends.mps.is_available() and train_state.weights_dtype == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if accelerator.is_main_process:
        Path(train_state.output_dir).mkdir(exist_ok=True, parents=True)
        Path(train_state.logging_dir).mkdir(exist_ok=True)

        if train_state.push_to_hub:
            raise NotImplementedError

    if train_state.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

        if train_state.hub_token is not None:
            raise ValueError(
                "You cannot use both report_to=wandb and hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

        import wandb

        if accelerator.is_local_main_process:
            wandb.init(
                project=train_state.wandb_project or get_env("WANDB_PROJECT"),
                entity=train_state.wandb_entity or get_env("WANDB_ENTITY"),
                dir=train_state.logging_dir or get_env("WANDB_DIR"),
                group=train_state.wandb_group or get_env("WANDB_GROUP"),
                reinit=True,
                config=asdict(train_state),
            )

    # =======================
    # ===   Load models   ===
    # =======================

    metrics = {
        "ssim": StructuralSimilarityIndexMeasure(
            data_range=(0.0, 1.0), reduction="none"
        ).to(accelerator.device)
    }

    models = prepare_models(train_state, accelerator.device)

    # ============================
    # === Prepare optimization ===
    # ============================

    accelerator.register_save_state_pre_hook(
        functools.partial(
            save_model_hook,
            accelerator=accelerator,
            models=models,
            lora_train_state=train_state,
        )
    )
    accelerator.register_load_state_pre_hook(
        functools.partial(
            load_model_hook,
            accelerator=accelerator,
            models=models,
            lora_train_state=train_state,
        )
    )

    # ======================
    # ===   Setup data   ===
    # ======================

    preprocessors = prepare_preprocessors(models, train_state)
    train_dataset = DynamicDataset.from_config(
        train_state.datasets["train_data"],
        preprocess_func=functools.partial(
            preprocess_sample, preprocessors=preprocessors["train"]
        ),
    )
    val_dataset = DynamicDataset.from_config(
        train_state.datasets["val_data"],
        preprocess_func=functools.partial(
            preprocess_sample, preprocessors=preprocessors["val"]
        ),
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_state.train_batch_size,
        num_workers=train_state.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=train_state.pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=train_state.train_batch_size,
        num_workers=train_state.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=train_state.pin_memory,
    )

    # ==========================
    # ===   Setup training   ===
    # ==========================

    trainable_models = [
        models[model_name] for model_name in train_state.trainable_models
    ]
    params_to_optimize = list(
        filter(
            lambda p: p.requires_grad,
            it.chain(*(model.parameters() for model in trainable_models)),
        )
    )

    if train_state.scale_lr:
        train_state.learning_rate = (
            train_state.learning_rate
            * train_state.gradient_accumulation_steps
            * train_state.train_batch_size
            * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        lr=train_state.learning_rate,
        betas=(train_state.adam_beta1, train_state.adam_beta2),
        weight_decay=train_state.adam_weight_decay,
        eps=train_state.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    train_state.num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_state.gradient_accumulation_steps
    )
    train_state.max_train_steps = (
        train_state.n_epochs * train_state.num_update_steps_per_epoch
    )

    lr_scheduler = get_scheduler(
        train_state.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_state.lr_warmup_steps
        * train_state.gradient_accumulation_steps,
        num_training_steps=train_state.max_train_steps
        * train_state.gradient_accumulation_steps,
        **train_state.lr_scheduler_kwargs,
    )

    optimizer, train_dataloader, lr_scheduler, *trainable_models = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler, *trainable_models
    )
    for model, model_name in zip(trainable_models, train_state.trainable_models):
        models[model_name] = model

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    train_state.num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_state.gradient_accumulation_steps
    )
    train_state.max_train_steps = (
        train_state.n_epochs * train_state.num_update_steps_per_epoch
    )

    # Afterwards we recalculate our number of training epochs
    train_state.n_epochs = math.ceil(
        train_state.max_train_steps / train_state.num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(train_state.wandb_group, config=asdict(train_state))

    # ===================
    # === Train model ===
    # ===================
    total_batch_size = (
        train_state.train_batch_size
        * accelerator.num_processes
        * train_state.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {train_state.n_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {train_state.train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {train_state.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {train_state.max_train_steps}")

    if train_state.resume_from_checkpoint:
        resume_from_checkpoint(accelerator, train_state)

    progress_bar = tqdm.tqdm(
        range(0, train_state.max_train_steps),
        initial=train_state.global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    while train_state.epoch < train_state.n_epochs:
        train_epoch(
            config,
            accelerator,
            optimizer,
            lr_scheduler,
            train_state,
            train_dataloader,
            models,
            params_to_optimize,
            progress_bar,
        )
        torch.cuda.empty_cache()

        if train_state.epoch % train_state.val_freq == 0:
            validate_model(
                accelerator,
                train_state,
                val_dataloader,
                models,
                metrics,
                "val",
            )
        torch.cuda.empty_cache()

        if (
            train_state.global_step >= train_state.max_train_steps
            or train_state.epoch > train_state.n_epochs
        ):
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:

        # Final inference
        validate_model(
            accelerator,
            train_state,
            val_dataloader,
            models,
            metrics,
            "test",
        )

        # Save the lora layers
        save_lora_weights(accelerator, train_state)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
