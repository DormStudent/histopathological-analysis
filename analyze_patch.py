import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config


def analyze_patch(
    image_path: str | Path, output_path: str | Path | None = None
) -> np.ndarray:
    """Extract and optionally save a UNI embedding for one histology patch."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"image file not found: {image_path}")

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".npy":
            raise ValueError("output path must end in .npy")

    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        embedding = model(image_tensor).squeeze(0).cpu().numpy()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embedding)

    return embedding


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a UNI embedding from a histology image patch."
    )
    parser.add_argument("image", type=Path, help="Path to a histology patch image")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the saved .npy embedding (default: <image>_uni_embedding.npy)",
    )
    args = parser.parse_args()

    output_path = args.output or args.image.with_name(
        f"{args.image.stem}_uni_embedding.npy"
    )
    embedding = analyze_patch(args.image, output_path)
    print(f"Saved UNI embedding with shape {embedding.shape} to {output_path}")


if __name__ == "__main__":
    main()
