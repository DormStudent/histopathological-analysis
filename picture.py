import os
import numpy as np
from openslide import OpenSlide


def getting_image(wsi_path, output, patch_size=256):
    if not os.path.exists(wsi_path):
        print(f'File not found: {wsi_path}')
        return

    if patch_size <= 0:
        raise ValueError('patch_size must be greater than zero')

    slide = OpenSlide(wsi_path)
    try:
        width, height = slide.dimensions
        os.makedirs(output, exist_ok=True)
        patch_idx = 0

        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                if x + patch_size <= width and y + patch_size <= height:
                    patch = slide.read_region(
                        (x, y), 0, (patch_size, patch_size)
                    )
                    patch_rgb = patch.convert('RGB')
                    np_patch = np.array(patch_rgb)

                    if np_patch.mean() > 215:
                        continue

                    patch_filename = os.path.join(
                        output, f'patch_x{x}_y{y}_{patch_idx}.png'
                    )
                    patch_rgb.save(patch_filename)
                    patch_idx += 1
    finally:
        slide.close()
