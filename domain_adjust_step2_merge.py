#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import label, distance_transform_edt

Image.MAX_IMAGE_PIXELS = None

# -------- 命名解析：..._i_j --------
IDX_RE = re.compile(r".*_(?P<i>\d+)_(?P<j>\d+)$")
def _parse_ij_from_subdir(name):
    m = IDX_RE.match(name)
    if not m:
        return None
    return int(m.group("i")), int(m.group("j"))

# -------- 颜色叠加保存 --------
def _save_color_overlay(color_image_array, mask_array, output_color_image_path):
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
                0.5 * modified_color_image[mask_area] + 0.5 * np.array(color, dtype=np.uint8)
            ).astype(np.uint8)
    Image.fromarray(modified_color_image).save(output_color_image_path)

# -------- 读取所有 tile 掩膜（允许尺寸不同）--------
def _load_all_tile_masks(folder_path, mask_filename):
    """
    返回:
      mapping: dict[(i,j)] = (subdir_path, mask_array, h, w)
      max_i, max_j: 最大下标
    """
    mapping = {}
    max_i = max_j = -1

    for subdir in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, subdir)
        if not os.path.isdir(sub_path):
            continue
        ij = _parse_ij_from_subdir(subdir)
        if ij is None:
            continue

        mask_path = os.path.join(sub_path, mask_filename)
        if not os.path.exists(mask_path):
            continue

        mask_img = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask_img, dtype=np.uint8)
        h, w = mask_arr.shape

        i, j = ij
        mapping[(i, j)] = (sub_path, mask_arr, h, w)
        max_i = max(max_i, i)
        max_j = max(max_j, j)

    if not mapping:
        raise FileNotFoundError(f"No subfolder with a valid mask '{mask_filename}' found under: {folder_path}")

    return mapping, max_i, max_j

# -------- 计算网格几何（每行高、每列宽、累计偏移）--------
def _compute_grid_geometry(mapping, split_ratio):
    """
    针对可能不一致的 tile 尺寸，按行取最大高度，按列取最大宽度，计算累计偏移。
    返回:
      heights_by_row, widths_by_col, y_offsets, x_offsets, total_H, total_W
    """
    heights_by_row = [0] * split_ratio
    widths_by_col  = [0] * split_ratio

    for (i, j), (_, _, h, w) in mapping.items():
        if 0 <= i < split_ratio:
            heights_by_row[i] = max(heights_by_row[i], h)
        if 0 <= j < split_ratio:
            widths_by_col[j]  = max(widths_by_col[j],  w)

    y_offsets = [0] * split_ratio
    x_offsets = [0] * split_ratio
    for i in range(1, split_ratio):
        y_offsets[i] = y_offsets[i-1] + heights_by_row[i-1]
    for j in range(1, split_ratio):
        x_offsets[j] = x_offsets[j-1] + widths_by_col[j-1]

    total_H = sum(heights_by_row)
    total_W = sum(widths_by_col)
    return heights_by_row, widths_by_col, y_offsets, x_offsets, total_H, total_W

# -------- 合并大掩膜（并返回几何用于裁剪）--------
def _merge_masks(mapping, split_ratio, void_val=255):
    (heights_by_row, widths_by_col,
     y_offsets, x_offsets, total_H, total_W) = _compute_grid_geometry(mapping, split_ratio)

    merged = np.full((total_H, total_W), void_val, dtype=np.uint8)

    for (i, j), (_, tile_arr, h, w) in mapping.items():
        if not (0 <= i < split_ratio and 0 <= j < split_ratio):
            continue
        y0, x0 = y_offsets[i], x_offsets[j]
        y1, x1 = y0 + h, x0 + w
        merged[y0:y1, x0:x1] = tile_arr

    return merged, heights_by_row, widths_by_col, y_offsets, x_offsets

# -------- 在合并图上找 class=4 的 top-K 并返回联合mask --------
def _find_topk_union(mask_arr, cls_val, top_k):
    labeled_array, num_features = label(mask_arr == cls_val)
    if num_features <= 0:
        return np.zeros_like(mask_arr, dtype=np.uint8)

    sizes = np.bincount(labeled_array.ravel())
    if sizes.size <= 1:
        return np.zeros_like(mask_arr, dtype=np.uint8)

    component_sizes = sizes[1:]  # 跳过背景
    k = int(min(max(1, top_k), component_sizes.size))
    top_k_idx = np.argpartition(component_sizes, -k)[-k:]

    union = np.zeros_like(mask_arr, dtype=np.uint8)
    for idx in top_k_idx:
        lbl = idx + 1
        union |= (labeled_array == lbl).astype(np.uint8)
    return union

# -------- 主流程（合并→全局距离→逐tile重标注）--------
def batch_process_images(folder_path: str, mask_filename: str, top_k: int, split_ratio: int):
    """
    1) 读取所有子块掩膜并合并为大掩膜（容忍边缘尺寸±1）
    2) 在合并掩膜上寻找 class=4 的前K大连通域
    3) 计算全局距离变换
    4) 对每个子块：裁剪对应窗口距离图，将距离<400的 class 2/5 改为 6
    5) 保存回各自子文件夹：processed_mask_domain_adjust.png / _color.jpg
    """
    # 1) gather tiles
    mapping, _, _ = _load_all_tile_masks(folder_path, mask_filename)

    # 2) merge
    merged_mask, heights_by_row, widths_by_col, y_offsets, x_offsets = _merge_masks(
        mapping, split_ratio, void_val=255
    )

    # 3) top-k union on merged
    union_topk = _find_topk_union(merged_mask, cls_val=4, top_k=top_k)

    # 4) global distance transform（到top-K联合区域边界的距离）
    distance = distance_transform_edt(1 - union_topk.astype(np.uint8))

    # 5) per-tile relabel using cropped distance
    for (i, j), (sub_path, mask_arr, h, w) in mapping.items():
        y0, x0 = y_offsets[i], x_offsets[j]
        y1, x1 = y0 + h, x0 + w

        dist_tile = distance[y0:y1, x0:x1]
        tile_mask = mask_arr.copy()

        target_pixels = (tile_mask == 2) | (tile_mask == 5)
        ti, tj = np.where(target_pixels)
        if ti.size > 0:
            within = dist_tile[ti, tj] < 400
            tile_mask[ti[within], tj[within]] = 6

        # 保存掩膜
        output_mask_path = os.path.join(sub_path, 'processed_mask_domain_adjust.png')
        Image.fromarray(tile_mask.astype(np.uint8)).save(output_mask_path)

        # 保存颜色叠加（若存在对应 jpg 原图）
        original_image_path = os.path.join(folder_path, os.path.basename(sub_path) + ".jpg")
        if os.path.exists(original_image_path):
            color_image = Image.open(original_image_path).convert("RGB")
            color_image_array = np.array(color_image)
            output_color_image_path = os.path.join(sub_path, 'processed_mask_domain_adjust_color.jpg')
            _save_color_overlay(color_image_array, tile_mask, output_color_image_path)
        # 若找不到原图，静默跳过颜色叠加

# -------- 保留原函数签名（为兼容旧调用，不在新流程中使用）--------
def process_images(mask_path, color_image_path, output_color_image_path, output_mask_path, top_k):
    mask = Image.open(mask_path).convert("L")
    color_image = Image.open(color_image_path).convert("RGB")
    mask_array = np.array(mask)
    color_image_array = np.array(color_image)

    labeled_array, num_features = label(mask_array == 4)
    if num_features > 0:
        sizes = np.bincount(labeled_array.ravel())
        if sizes.size > 1:
            component_sizes = sizes[1:]
            k = int(min(max(1, top_k), component_sizes.size))
            top_k_idx = np.argpartition(component_sizes, -k)[-k:]
            largest_k_mask = np.zeros_like(mask_array, dtype=int)
            for idx in top_k_idx:
                lbl = idx + 1
                largest_k_mask |= (labeled_array == lbl).astype(int)
        else:
            largest_k_mask = np.zeros_like(mask_array, dtype=int)
    else:
        largest_k_mask = np.zeros_like(mask_array, dtype=int)

    distance = distance_transform_edt(1 - largest_k_mask)
    target_pixels = (mask_array == 2) | (mask_array == 5)
    target_indices = np.where(target_pixels)
    for i, j in zip(*target_indices):
        if distance[i, j] < 400:
            mask_array[i, j] = 6

    _save_color_overlay(color_image_array, mask_array, output_color_image_path)
    Image.fromarray(mask_array.astype(np.uint8)).save(output_mask_path)

# -------- 参数与入口 --------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process masks with global merged-mask adjustment (top-K class-4 union + distance relabel)."
    )
    parser.add_argument(
        "--folder",
        default="/media/zzd/hrhsr/backup/Documents/img_seg/River/river_classification_pipeline/data/test_data/10-9-1sthalf_transparent_mosaic_group1/split",
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
        default=4,
        help="Number of largest class-4 components to consider on the merged mask (default: 3)."
    )
    parser.add_argument(
        "--split_ratio",
        type=int,
        default=6,
        help="Grid split ratio (e.g., 6 means a 6x6 grid)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    batch_process_images(args.folder, args.mask_name, args.top_k, args.split_ratio)
