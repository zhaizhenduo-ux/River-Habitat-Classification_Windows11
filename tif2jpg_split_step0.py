from PIL import Image
import numpy as np
import rasterio
import os
from tqdm import tqdm
import psutil
import argparse

Image.MAX_IMAGE_PIXELS = None

def _compute_split_bounds(width, height, split_ratio):
    xs = np.linspace(0, width, split_ratio + 1, dtype=int)
    ys = np.linspace(0, height, split_ratio + 1, dtype=int)
    return xs, ys

def raster_rgb(tif_path, split_ratio, band_num, tiles_dir, base):
    with rasterio.open(tif_path) as src:
        band = src.read(band_num)
        height, width = band.shape
    image_data = np.expand_dims(band, axis=-1).astype(band.dtype)

    xs, ys = _compute_split_bounds(width, height, split_ratio)

    for i in tqdm(range(split_ratio), desc=f"Band {band_num}: rows"):
        for j in tqdm(range(split_ratio), desc=f"Band {band_num}: cols", leave=False):
            y0, y1 = ys[i], ys[i+1]
            x0, x1 = xs[j], xs[j+1]
            split_arr = image_data[y0:y1, x0:x1, 0]
            split_image = Image.fromarray(split_arr).convert("L")
            out_name = f'{base}_{i}_{j}_{band_num-1}.png'
            split_image.save(os.path.join(tiles_dir, out_name))

def merge_channels_to_jpg(red_path, green_path, blue_path, output_path):
    red = Image.open(red_path).convert("L")
    green = Image.open(green_path).convert("L")
    blue = Image.open(blue_path).convert("L")
    rgb = Image.merge("RGB", (red, green, blue))
    rgb.save(output_path, "JPEG", quality=95)
    for p in (red_path, green_path, blue_path):
        try: os.remove(p)
        except OSError: pass

def stitch_jpg_tiles_to_full(tif_path, split_ratio, tiles_dir, merged_dir, base):
    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
    print("h and w", width, height)
    os.makedirs(merged_dir, exist_ok=True)
    out_path = os.path.join(merged_dir, f"{base}_stitched_full.jpg")

    xs, ys = _compute_split_bounds(width, height, split_ratio)
    full_img = Image.new("RGB", (width, height))

    for i in tqdm(range(split_ratio), desc="Stitching rows"):
        for j in tqdm(range(split_ratio), desc="Stitching cols", leave=False):
            tile_path = os.path.join(tiles_dir, f"{base}_{i}_{j}.jpg")
            tile = Image.open(tile_path).convert("RGB")
            x0, x1 = xs[j], xs[j+1]
            y0, y1 = ys[i], ys[i+1]
            region_w, region_h = x1 - x0, y1 - y0
            if tile.size != (region_w, region_h):
                tile = tile.resize((region_w, region_h), Image.BILINEAR)
            full_img.paste(tile, (x0, y0))

    full_img.save(out_path, "JPEG", quality=95)
    print(f"Stitched full image saved to: {out_path}")
    return out_path, (width, height)

def save_downscaled(full_img_path, original_size, downscaled_dir, base, factor):
    os.makedirs(downscaled_dir, exist_ok=True)
    w, h = original_size
    dw, dh = max(1, w // factor), max(1, h // factor)
    img = Image.open(full_img_path)
    down = img.resize((dw, dh), Image.LANCZOS)
    out_path = os.path.join(downscaled_dir, f"{base}_stitched_full_downscaled_1to{factor}.jpg")
    down.save(out_path, "JPEG", quality=90)
    print(f"Downscaled stitched image saved to: {out_path}")

def tif2jpg_split(tif_path, split_ratio, tiles_dir, merged_dir, downscaled_dir, downscale_factor):
    base = os.path.splitext(os.path.basename(tif_path))[0]
    os.makedirs(tiles_dir, exist_ok=True)

    for band_idx in range(3):
        raster_rgb(tif_path, split_ratio, band_idx + 1, tiles_dir, base)

    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
    xs, ys = _compute_split_bounds(width, height, split_ratio)

    for i in tqdm(range(split_ratio), desc="Merging bands to JPG (rows)"):
        for j in tqdm(range(split_ratio), desc="Merging bands to JPG (cols)", leave=False):
            red = os.path.join(tiles_dir, f'{base}_{i}_{j}_0.png')
            green = os.path.join(tiles_dir, f'{base}_{i}_{j}_1.png')
            blue = os.path.join(tiles_dir, f'{base}_{i}_{j}_2.png')
            out = os.path.join(tiles_dir, f'{base}_{i}_{j}.jpg')
            merge_channels_to_jpg(red, green, blue, out)

    full_img_path, orig_size = stitch_jpg_tiles_to_full(tif_path, split_ratio, tiles_dir, merged_dir, base)
    save_downscaled(full_img_path, orig_size, downscaled_dir, base, downscale_factor)

def main():
    parser = argparse.ArgumentParser(description="Split TIF into tiles, merge to full JPG, and save a downscaled version.")
    parser.add_argument("--tif", help="Path to input .tif file", default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023/p4/11_03/p4-11-3-reprocess_transparent_mosaic_group1.tif")
    parser.add_argument("--tiles_dir", help="Folder to save split tiles", default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023/p4/11_03/split_tmp/")
    parser.add_argument("--merged_dir", help="Folder to save stitched full-size JPG", default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023/p4/11_03")
    parser.add_argument("--downscaled_dir", help="Folder to save downscaled JPG", default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023/p4/11_03")
    parser.add_argument("--split_ratio", type=int, default=4, help="Splits per axis (default: 3)")
    parser.add_argument("--downscale_factor", type=int, default=6, help="Downscale ratio (default: 10)")
    args = parser.parse_args()

    for d in [args.tiles_dir, args.merged_dir, args.downscaled_dir]:
        os.makedirs(d, exist_ok=True)

    tif2jpg_split(args.tif, args.split_ratio, args.tiles_dir, args.merged_dir, args.downscaled_dir, args.downscale_factor)

if __name__ == "__main__":
    main()
