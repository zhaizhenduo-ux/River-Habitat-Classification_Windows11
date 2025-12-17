#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import label, distance_transform_edt

Image.MAX_IMAGE_PIXELS = None

def batch_process_images(folder_path: str, mask_filename: str):
    """
    Walk through subfolders of `folder_path`. In each subfolder, look for a mask file named
    `mask_filename` (e.g., 'processed_mask.png') and an original RGB image named '<subdir>.jpg'
    in the parent folder. If both exist, process and save outputs in the subfolder.
    """
    # List all subdirectories in the given folder path
    for subdir in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, subdir)
        if not os.path.isdir(sub_path):
            continue

        mask_path = os.path.join(sub_path, mask_filename)
        original_image_path = os.path.join(folder_path, f'{subdir}.jpg')

        if os.path.exists(mask_path) and os.path.exists(original_image_path):
            output_mask_path = os.path.join(sub_path, 'processed_mask_domain_adjust.png')
            output_color_image_path = os.path.join(sub_path, 'processed_mask_domain_adjust_color.jpg')
            process_images(mask_path, original_image_path, output_color_image_path, output_mask_path)
        else:
            # Optional: uncomment to see which items were skipped
            # print(f"Skipping '{subdir}': missing {mask_path} or {original_image_path}")
            pass

def process_images(mask_path, color_image_path, output_color_image_path, output_mask_path):
    # Open images
    mask = Image.open(mask_path).convert("L")      # single-channel (grayscale)
    color_image = Image.open(color_image_path).convert("RGB")

    # Convert to numpy
    mask_array = np.array(mask)
    color_image_array = np.array(color_image)

    # Find largest connected component of class 4
    labeled_array, num_features = label(mask_array == 4)
    if num_features > 0:
        sizes = np.bincount(labeled_array.ravel())
        # sizes[0] is background; guard in case all zeros
        if sizes.size > 1:
            largest_component = np.argmax(sizes[1:]) + 1  # +1 to skip background
            largest_area_mask = (labeled_array == largest_component).astype(int)
        else:
            largest_area_mask = np.zeros_like(mask_array, dtype=int)
    else:
        largest_area_mask = np.zeros_like(mask_array, dtype=int)

    # Distance transform (to boundary of largest class-4 component)
    distance = distance_transform_edt(1 - largest_area_mask)

    # Pixels of interest: classes 2 or 5 (fixed the duplicate 2)
    target_pixels = (mask_array == 2) | (mask_array == 5)
    target_indices = np.where(target_pixels)

    # Relabel to class 6 if within threshold distance
    for i, j in zip(*target_indices):
        if distance[i, j] < 400:
            mask_array[i, j] = 6

    # Overlay colors half-transparent
    modified_color_image = color_image_array.copy()
    colors = {
        0: (0, 255, 0),       # Green
        1: (255, 105, 180),   # Pink
        2: (255, 165, 0),     # Orange
        3: (255, 0, 0),       # Red
        4: (255, 255, 0),     # Yellow
        5: (0, 0, 255),       # Blue
        6: (128, 128, 128),   # Gray
    }
    for value, color in colors.items():
        mask_area = (mask_array == value)
        if np.any(mask_area):
            modified_color_image[mask_area] = (
                0.5 * modified_color_image[mask_area] + 0.5 * np.array(color)
            ).astype(np.uint8)

    # Save results
    Image.fromarray(modified_color_image).save(output_color_image_path)
    Image.fromarray(mask_array.astype(np.uint8)).save(output_mask_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process masks and overlay domain adjustment."
    )
    parser.add_argument(
        "--folder",
        default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/data_2023_mavic_p4/p4/test_data_1/mavic3_pro_10_2_2023_transparent_mosaic_group1/split",
        help="Path to the parent folder containing subfolders and the '<subdir>.jpg' originals."
    )
    parser.add_argument(
        "--mask_name",
        default="processed_mask.png",
        help="Mask filename inside each subfolder (default: processed_mask.png)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    batch_process_images(args.folder, args.mask_name)
