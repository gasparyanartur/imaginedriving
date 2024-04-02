from typing import Any
from pathlib import Path
from argparse import ArgumentParser
import logging
import os

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForTextEncoding
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
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub import create_repo, upload_folder
import wandb

from src.data import setup_project


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


def main(args):
    # ========================
    # ===   Setup script   ===
    # ========================

    config_path = args.config_path
    config = setup_project(config_path)

    output_dir = Path(config["output_dir"])
    logging_dir = output_dir / "logs"

    model_config: dict[str, Any] = config["model"]
    report_to = model_config.get("report_to")
    mixed_precision = model_config.get("mixed_precision")
    log_with = model_config.get("log_with")
    revision = model_config.get("revision")
    variant = model_config.get("variant")

    # Sanity checks
    if report_to == "wandb" and model_config.get("hub_token") is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if report_to == "wandb":
        assert is_wandb_available()

    if torch.backends.mps.is_available() and mixed_precision == "bf16":
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
        mixed_precision=mixed_precision,
        log_with=log_with,
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
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
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

    model_id = model_config["pretrained_model_name_or_path"]

    tokenizer_one = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", revision=revision, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer_2", revision=revision, use_fast=False
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        model_id, revision, subfolder="text_encoder"
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        model_id, revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        model_id, subfolder="text_encoder", revision=revision, variant=variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        model_id, subfolder="text_encoder_2", revision=revision, variant=variant
    )

    # ======================
    # ===   Setup data   ===
    # ======================

    # TODO
    prompt = config["prompt"]
    prompt_embed = embed_prompt(prompt)

    # =======================
    # ===   Train model   ===
    # =======================

    # TODO
    ...


if __name__ == "__main__":
    args = parse_args()
    main(args)
