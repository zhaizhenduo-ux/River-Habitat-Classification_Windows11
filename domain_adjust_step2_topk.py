#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import label, distance_transform_edt

Image.MAX_IMAGE_PIXELS = None

def batch_process_images(folder_path: str, mask_filename: str, top_k: int):
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
            process_images(mask_path, original_image_path, output_color_image_path, output_mask_path, top_k)
        else:
            # Optional: uncomment to see which items were skipped
            # print(f"Skipping '{subdir}': missing {mask_path} or {original_image_path}")
            pass

def process_images(mask_path, color_image_path, output_color_image_path, output_mask_path, top_k):
    # Open images
    mask = Image.open(mask_path).convert("L")      # single-channel (grayscale)
    color_image = Image.open(color_image_path).convert("RGB")

    # Convert to numpy
    mask_array = np.array(mask)
    color_image_array = np.array(color_image)

    # Find top-K largest connected components of class 4 (each must be ≥10% of total mask area)
    labeled_array, num_features = label(mask_array == 4)

    total_area = mask_array.size
    min_size = int(0.001 * total_area)  # 10% threshold

    largest_k_mask = np.zeros_like(mask_array, dtype=np.uint8)

    if num_features > 0:
        sizes = np.bincount(labeled_array.ravel())
        # sizes[0] is background; components are sizes[1:]
        if sizes.size > 1:
            component_sizes = sizes[1:]
            # sort component indices by size (desc)
            order_desc = np.argsort(component_sizes)[::-1]
            # keep only components meeting the ≥10% threshold
            valid_labels = [idx + 1 for idx in order_desc if component_sizes[idx] >= min_size]
            # pick up to K of those valid components
            k = int(min(max(1, top_k), len(valid_labels)))
            for lbl in valid_labels[:k]:
                largest_k_mask |= (labeled_array == lbl).astype(np.uint8)

    # If no valid components, skip distance-based relabeling entirely
    if np.any(largest_k_mask):
        # Distance transform (to boundary of union of selected class-4 components)
        distance = distance_transform_edt(1 - largest_k_mask)

        # Pixels of interest: classes 2 or 5
        target_pixels = (mask_array == 2) | (mask_array == 5)
        target_indices = np.where(target_pixels)

        # Relabel to class 6 if within threshold distance to ANY of the selected components
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
        default="/media/zzd/hrhsr/backup/Documents/img_seg/River/data/third_test/inference_images_Unet",
        help="Path to the parent folder containing subfolders and the '<subdir>.jpg' originals."
    )
    parser.add_argument(
        "--mask_name",
        default="processed_mask.png",
        help="Mask filename inside each subfolder (default: processed_mask.png)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Number of largest class-4 components to consider (default: 3)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    batch_process_images(args.folder, args.mask_name, args.top_k)
