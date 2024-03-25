import argparse
from pathlib import Path
import torch

from src.diffusion import load_img2img_model, encode_img, decode_img
from src.configuration import setup_project
from src.data import read_image, save_image


if __name__ == "__main__":
    setup_project()

    parser = argparse.ArgumentParser("encode-decode", description="Demonstration of encoder-decoder functionaltiy of specific model")
    parser.add_argument("img_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    img_path = args.img_path
    output_path = args.output_path

    model = load_img2img_model()
    vae = model.vae
    img_processor = model.image_processor

    img_start = read_image(img_path=args.img_path)

    with torch.no_grad():
        img = encode_img(img_processor, vae, img_start)
        img_out = decode_img(img_processor, vae, img)

    save_image(output_path, img_out)