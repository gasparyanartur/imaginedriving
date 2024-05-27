import argparse
import torchvision.transforms.v2 as tvtf

import torch
from diffusers import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionImg2ImgPipeline
from diffusers.pipelines import StableDiffusionControlNetImg2ImgPipeline
from peft import get_peft_model, LoraConfig, inject_adapter_in_model

from src.diffusion import PeftCompatibleControlNet, parse_target_ranks
from src.data import read_image, norm_img_crop_pipeline
from src.utils import get_device, show_img

from src.control_lora import ControlLoRAModel


model_id = "stabilityai/stable-diffusion-2-1"
unet_target_ranks = {
    "downblocks": {"attn": 4, "resnet": 4, "ff": 8, "proj": 8},
    "midblocks": {"attn": 8, "resnet": 8, "ff": 16, "proj": 16},
    "upblocks": {"attn": 8, "resnet": 8, "ff": 16, "proj": 16},
}

controlnet_target_ranks = {
    "downblocks": {"attn": 8, "resnet": 8, "ff": 16, "proj": 16},
    "midblocks": {"attn": 8, "resnet": 8, "ff": 16, "proj": 16},
}


def main_peft(args):
    device = get_device()

    unet_ranks = parse_target_ranks(unet_target_ranks)
    controlnet_ranks = parse_target_ranks(controlnet_target_ranks)

    peft_unet_conf = LoraConfig(
        r=8,
        init_lora_weights="gaussian",
        target_modules="|".join(unet_ranks.keys()),
        rank_pattern=unet_ranks
    )

    peft_controlnet_conf = LoraConfig(
        r=8,
        init_lora_weights="gaussian",
        target_modules="|".join(controlnet_ranks.keys()),
        rank_pattern=controlnet_ranks,
        modules_to_save=["controlnet_down_blocks", "controlnet_mid_block", "controlnet_cond_embedding", ]
    )

    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float32)
    controlnet = PeftCompatibleControlNet.from_unet(unet)

    unet_injected = inject_adapter_in_model(peft_unet_conf, unet)
    controlnet_injected = inject_adapter_in_model(peft_controlnet_conf, controlnet)

    pipe_base = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, unet=unet_injected, torch_dtype=torch.float32)
    pipe_base.to(device)

    im1 = read_image("reference/pandaset-01/renders/0m/01.jpg", tf_pipeline=norm_img_crop_pipeline).to(device)[None, ...]
    im2 = read_image("reference/pandaset-01/renders/2m/01.jpg", tf_pipeline=norm_img_crop_pipeline).to(device)[None, ...]

    with torch.no_grad():
        base_out = pipe_base("", im1)

    del pipe_base

    pipe_cn = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_id, unet=unet_injected, controlnet=controlnet_injected, torch_dtype=torch.float32)
    pipe_cn.to(device)

    with torch.no_grad():
        cn_out = pipe_cn("", image=im1, control_image=im2)

    del pipe_cn

    show_img((base_out, cn_out))

def main_controllora(args):
    device = get_device()

    unet_ranks = parse_target_ranks(unet_target_ranks)
    controlnet_ranks = parse_target_ranks(controlnet_target_ranks)

    peft_unet_conf = LoraConfig(
        r=8,
        init_lora_weights="gaussian",
        target_modules="|".join(unet_ranks.keys()),
        rank_pattern=unet_ranks
    )

    peft_controlnet_conf = LoraConfig(
        r=8,
        init_lora_weights="gaussian",
        target_modules="|".join(controlnet_ranks.keys()),
        rank_pattern=controlnet_ranks,
        modules_to_save=["controlnet_down_blocks", "controlnet_mid_block", "controlnet_cond_embedding", ]
    )

    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float32)
    controlnet = ControlLoRAModel.from_unet(unet, lora_linear_rank=4, lora_conv2d_rank=4, use_dora=True)

    unet = inject_adapter_in_model(peft_unet_conf, unet)
    #controlnet_injected = inject_adapter_in_model(peft_controlnet_conf, controlnet)

    pipe_base = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float32)
    pipe_base.to(device)

    im1 = read_image("reference/pandaset-01/renders/0m/01.jpg", tf_pipeline=norm_img_crop_pipeline).to(device)[None, ...]
    im2 = read_image("reference/pandaset-01/renders/2m/01.jpg", tf_pipeline=norm_img_crop_pipeline).to(device)[None, ...]

    with torch.no_grad():
        base_out = pipe_base("", im1)

    del pipe_base

    pipe_cn = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_id, unet=unet, controlnet=controlnet, torch_dtype=torch.float32)
    pipe_cn.to(device)

    with torch.no_grad():
        cn_out = pipe_cn("", image=im1, control_image=im2)

    del pipe_cn

    show_img((base_out, cn_out))


def main(args):
    match args.controlnet_type:
        case "peft":
            main_peft(args)
        case "controllora":
            main_controllora(args)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--controlnet_type", "-ct", default="peft", choices=["peft", "controllora"])
    args = parser.parse_args()

    main(args)