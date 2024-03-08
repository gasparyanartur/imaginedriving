import argparse
from pathlib import Path

from src.diffusion import ImageToImageDiffusionModel, ModelId, encode_img, decode_img
from src.configuration import setup_project
from src.data import load_img_paths_from_dir, read_image
from src.utils import show_img


if __name__ == "__main__":
    setup_project()

    parser = argparse.ArgumentParser("encode-decode", description="Demonstration of encoder-decoder functionaltiy of specific model")
    parser.add_argument("img_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model_id", type=str, default=ModelId.sdxl_base_v1_0)
    parser.add_argument("--low_mem", action="store_false")
    args = parser.parse_args()

    diff_model = ImageToImageDiffusionModel(model_id=args.model_id)
    vae = diff_model.pipe.vae
    img_processor = diff_model.pipe.image_processor

    img_start = read_image(img_path=args.img_path)
    img = encode_img(img_processor, vae, img_start)
    img_out = decode_img(img_processor, vae, img)

    show_img((img_start, img_out), args.output_path)