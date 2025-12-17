#!/usr/bin/env python3
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import argparse

# Handle very large images
Image.MAX_IMAGE_PIXELS = None

def _compute_split_bounds(width, height, split_ratio):
    """
    Compute exact integer split boundaries so the grid fully covers the image.
    Returns:
        xs, ys where each is a list of boundaries (len = split_ratio + 1).
    """
    xs = np.linspace(0, width, split_ratio + 1, dtype=int)
    ys = np.linspace(0, height, split_ratio + 1, dtype=int)
    return xs, ys

def split_jpg(jpg_path, split_ratio, tiles_dir):
    """
    Split a single JPG into split_ratio x split_ratio tiles and save as JPGs.
    Returns base filename (without extension) and (width, height).
    """
    base = os.path.splitext(os.path.basename(jpg_path))[0]
    os.makedirs(tiles_dir, exist_ok=True)

    with Image.open(jpg_path) as img:
        img = img.convert("RGB")  # ensure 3-channel
        width, height = img.width, img.height

        xs, ys = _compute_split_bounds(width, height, split_ratio)

        for i in tqdm(range(split_ratio), desc="Splitting rows"):
            for j in tqdm(range(split_ratio), desc="Splitting cols", leave=False):
                x0, x1 = xs[j], xs[j+1]
                y0, y1 = ys[i], ys[i+1]
                tile = img.crop((x0, y0, x1, y1))
                out_name = f"{base}_{i}_{j}.jpg"
                tile.save(os.path.join(tiles_dir, out_name), "JPEG", quality=95)

    return base, (width, height)

def save_downscaled(jpg_path, original_size, downscaled_dir, base, factor):
    """
    Save a downscaled version (1/factor) of the original JPG.
    """
    os.makedirs(downscaled_dir, exist_ok=True)
    w, h = original_size
    dw, dh = max(1, w // factor), max(1, h // factor)

    with Image.open(jpg_path) as img:
        img = img.convert("RGB")
        down = img.resize((dw, dh), Image.LANCZOS)

    out_path = os.path.join(downscaled_dir, f"{base}_downscaled_1to{factor}.jpg")
    down.save(out_path, "JPEG", quality=90)
    print(f"Downscaled image saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Split a JPG into tiles and save a downscaled full image."
    )
    parser.add_argument("--jpg", help="Path to input .jpg file", default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023_mavic_p4/p4/test_data_1/mavic3_pro_10_2_2023_transparent_mosaic_group1_stitched_full.jpg")
    parser.add_argument("--tiles_dir", help="Folder to save split tiles", default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023_mavic_p4/p4/test_data_1/mavic3_pro_10_2_2023_transparent_mosaic_group1_stitched_full/split/")
    parser.add_argument("--merged_dir", help="Folder to save stitched full-size JPG", default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023_mavic_p4/p4/test_data_1/mavic3_pro_10_2_2023_transparent_mosaic_group1_stitched_full/")
    parser.add_argument("--downscaled_dir", help="Folder to save downscaled JPG", default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023_mavic_p4/p4/test_data_1/mavic3_pro_10_2_2023_transparent_mosaic_group1_stitched_full/")
    parser.add_argument("--split_ratio", type=int, default=3, help="Splits per axis (default: 3)")
    parser.add_argument("--downscale_factor", type=int, default=10, help="Downscale ratio (default: 10)")
    args = parser.parse_args()

    # Ensure output folders exist
    for d in [args.tiles_dir, args.downscaled_dir]:
        os.makedirs(d, exist_ok=True)

    base, orig_size = split_jpg(args.jpg, args.split_ratio, args.tiles_dir)
    save_downscaled(args.jpg, orig_size, args.downscaled_dir, base, args.downscale_factor)

if __name__ == "__main__":
    main()
