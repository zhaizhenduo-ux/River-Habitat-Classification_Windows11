#!/usr/bin/env python3
import os
import re
import argparse
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # handle very large images safely

DIR_INDEX_RE = re.compile(r"^(?P<prefix>.*)_(?P<i>\d+)_(?P<j>\d+)$")

def _find_tile_dirs(root_dir, split_ratio_w, split_ratio_h, tile_name):
    """
    Find subfolders named like '..._<i>_<j>' and ensure each contains tile_name.
    Uses split_ratio_h (rows, i) and split_ratio_w (cols, j).
    Returns a dict mapping (i, j) -> full path to the tile image file.
    """
    mapping = {}
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        m = DIR_INDEX_RE.match(entry.name)
        if not m:
            continue
        i = int(m.group("i"))  # row index
        j = int(m.group("j"))  # col index
        if not (0 <= i < split_ratio_h and 0 <= j < split_ratio_w):
            continue
        tile_path = os.path.join(entry.path, tile_name)
        if os.path.isfile(tile_path):
            mapping[(i, j)] = tile_path

    # Basic validation: ensure full grid is present
    missing = []
    for i in range(split_ratio_h):
        for j in range(split_ratio_w):
            if (i, j) not in mapping:
                missing.append((i, j))
    if missing:
        msg = "Missing tiles for positions: " + ", ".join([f"({i},{j})" for i, j in missing])
        raise FileNotFoundError(msg)

    return mapping

def _pil_format_from_ext(ext):
    ext = ext.lower()
    if ext in [".jpg", ".jpeg"]:
        return "JPEG"
    if ext == ".png":
        return "PNG"
    # Fallback to PNG if unknown (lossless & widely supported)
    return "PNG"

def _choose_common_mode(sample_mode, target_ext):
    """
    Pick a safe common mode for pasting. For PNG prefer RGBA if alpha present,
    otherwise RGB. For JPEG use RGB. If tiles are grayscale 'L', keep 'L'.
    """
    if sample_mode == "L":
        return "L"
    if target_ext.lower() in [".jpg", ".jpeg"]:
        return "RGB"
    # PNG: prefer RGBA if alpha might exist, else RGB
    return "RGBA" if "A" in sample_mode else "RGB"

def merge_tiles(root_dir, split_ratio_w, split_ratio_h, tile_name, output_dir):
    """
    Merge tiles (found in subfolders) into a single image.
    Returns the merged image path and its (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)

    mapping = _find_tile_dirs(root_dir, split_ratio_w, split_ratio_h, tile_name)

    # Determine extension and PIL save format from the tile name
    _, ext = os.path.splitext(tile_name)
    pil_fmt = _pil_format_from_ext(ext)

    # Load one tile to determine mode preference
    sample = Image.open(next(iter(mapping.values())))
    common_mode = _choose_common_mode(sample.mode, ext)

    # Compute total width and height by summing tile sizes per row/col
    widths_by_col = [0] * split_ratio_w
    heights_by_row = [0] * split_ratio_h

    # Measure widths using the first row, heights using the first column
    for j in range(split_ratio_w):
        with Image.open(mapping[(0, j)]) as im:
            widths_by_col[j] = im.width
    for i in range(split_ratio_h):
        with Image.open(mapping[(i, 0)]) as im:
            heights_by_row[i] = im.height

    total_width = sum(widths_by_col)
    total_height = sum(heights_by_row)

    # Build paste coordinates (x offsets per col, y offsets per row)
    x_offsets = [0]
    for j in range(1, split_ratio_w):
        x_offsets.append(x_offsets[-1] + widths_by_col[j-1])
    y_offsets = [0]
    for i in range(1, split_ratio_h):
        y_offsets.append(y_offsets[-1] + heights_by_row[i-1])

    # Create the full canvas
    full = Image.new(common_mode, (total_width, total_height))

    # Paste all tiles
    for i in tqdm(range(split_ratio_h), desc="Merging rows"):
        for j in tqdm(range(split_ratio_w), desc="Merging cols", leave=False):
            tile_path = mapping[(i, j)]
            with Image.open(tile_path) as im:
                im = im.convert(common_mode)
                full.paste(im, (x_offsets[j], y_offsets[i]))

    # Construct output filename using root folder basename
    base = os.path.basename(os.path.normpath(root_dir))
    merged_name = f"{base}_merged{ext}"
    merged_path = os.path.join(output_dir, merged_name)

    # Save with sensible defaults
    if pil_fmt == "JPEG":
        full.save(merged_path, pil_fmt, quality=95, optimize=True)
    else:
        # PNG or fallback
        full.save(merged_path, pil_fmt, optimize=True)

    print(f"Merged image saved to: {merged_path}")
    return merged_path, (total_width, total_height), pil_fmt, ext

def save_downscaled(src_path, original_size, output_dir, base, factor, pil_fmt, ext):
    """
    Save a downscaled (1/factor) version of src_path, preserving suffix.
    """
    os.makedirs(output_dir, exist_ok=True)
    w, h = original_size
    dw = max(1, w // factor)
    dh = max(1, h // factor)
    with Image.open(src_path) as im:
        # Choose target mode compatible with format
        target_mode = "RGB" if pil_fmt == "JPEG" else ("RGBA" if "A" in im.mode else "RGB")
        if im.mode != target_mode and target_mode in ["RGB", "RGBA", "L"]:
            im = im.convert(target_mode)
        down = im.resize((dw, dh), Image.LANCZOS)

    out_name = f"{base}_merged_downscaled_1to{factor}{ext}"
    out_path = os.path.join(output_dir, out_name)

    if pil_fmt == "JPEG":
        down.save(out_path, pil_fmt, quality=90, optimize=True)
    else:
        down.save(out_path, pil_fmt, optimize=True)

    print(f"Downscaled image saved to: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(
        description="Merge tiles from subfolders named like '..._<i>_<j>' and save merged + downscaled images."
    )
    parser.add_argument("--root_dir", default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023_mavic_p4/p4/test_data_1/mavic3_pro_10_2_2023_transparent_mosaic_group1/split/",
                        help="Path to the folder that contains subfolders named like '..._<i>_<j>'")
    # Backward-compatible: if h/w not provided, use split_ratio for both
    parser.add_argument("--split_ratio", type=int, default=3,
                        help="Splits per axis if --split_ratio_h/--split_ratio_w not provided (default: 3)")
    parser.add_argument("--split_ratio_h", type=int,
                        help="Number of splits along height (rows, i). Overrides --split_ratio when set.")
    parser.add_argument("--split_ratio_w", type=int,
                        help="Number of splits along width (cols, j). Overrides --split_ratio when set.")
    parser.add_argument("--tile_name", default="processed_mask.png",
                        help='Filename of the image inside each subfolder (e.g., "prediction.jpg" or "prediction.png")')
    parser.add_argument("--output_dir", default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023_mavic_p4/p4/test_data_1/mavic3_pro_10_2_2023_transparent_mosaic_group1/step1_result/",
                        help="Folder to save the merged and downscaled images")
    parser.add_argument("--downscale_factor", type=int, default=10,
                        help="Downscale ratio (1/factor), default=10")
    args = parser.parse_args()

    # Resolve rectangular grid
    split_h = args.split_ratio_h if args.split_ratio_h is not None else args.split_ratio
    split_w = args.split_ratio_w if args.split_ratio_w is not None else args.split_ratio
    if split_h <= 0 or split_w <= 0:
        raise ValueError("split ratios must be positive integers")

    os.makedirs(args.output_dir, exist_ok=True)

    merged_path, size_wh, pil_fmt, ext = merge_tiles(
        args.root_dir, split_w, split_h, args.tile_name, args.output_dir
    )

    base = os.path.basename(os.path.normpath(args.root_dir))
    save_downscaled(
        merged_path, size_wh, args.output_dir, base, args.downscale_factor, pil_fmt, ext
    )

if __name__ == "__main__":
    main()
