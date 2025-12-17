import numpy as np
import cv2
from PIL import Image
import pandas as pd
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
from scipy import stats
import random
import gc
import torchvision.models as models
import torchvision.transforms as transforms
import json
import glob
import os
import argparse
from skimage import segmentation
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

def convert_png_to_jpg(input_path, output_path):
    colors = {
        0: (0, 255, 0),       # Green
        1: (255, 105, 180),   # Pink
        2: (255, 165, 0),     # Orange
        3: (255, 0, 0),       # Red
        4: (255, 255, 0),     # Yellow
        5: (0, 0, 255),       # Blue
        6: (128, 128, 128),   # Gray
    }

    img_png = Image.open(input_path)
    data = np.array(img_png)

    rgb_array = np.zeros(data.shape + (3,), dtype=np.uint8)
    for pixel_value, color in colors.items():
        mask = data == pixel_value
        rgb_array[mask] = color

    img_jpg = Image.fromarray(rgb_array)
    img_jpg.save(output_path)


def apply_semi_transparent_mask(image_data, color_mask_data, output_path, alpha=0.5):
    if image_data.shape != color_mask_data.shape:
        raise ValueError("Image data and color mask data must have the same shape.")

    alpha = np.clip(alpha, 0, 1)
    blended_image = (1 - alpha) * image_data + alpha * color_mask_data
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, blended_image)
    return blended_image


def find_mode(arr1, arr2):
    assert arr1.shape == arr2.shape, "The input arrays must have the same shape."
    output = np.zeros(arr1.shape, dtype=int)
    true_indices = np.where(arr2)
    if stats.mode(arr1[true_indices]).mode.shape[0] == 0:
        return output
    mode = stats.mode(arr1[true_indices])[0][0]
    output[true_indices] = mode
    return mode

def find_mode_1(arr1, arr2):
    assert arr1.shape == arr2.shape, "The input arrays must have the same shape."
    output = np.zeros(arr1.shape, dtype=int)

    # True indices where arr2 == True
    true_indices = np.where(arr2)
    num_true_pixels = len(true_indices[0])

    if num_true_pixels == 0:
        print("No True pixels in arr2.")
        return output

    # Compute mode
    mode_result = stats.mode(arr1[true_indices], keepdims=True)
    if mode_result.mode.shape[0] == 0:
        print("No mode found.")
        return output

    mode_value = mode_result.mode[0]
    mode_count = mode_result.count[0]

    # Print stats
    print(f"Region size (True pixels in arr2): {num_true_pixels}")
    print(f"Mode value: {mode_value}, Count: {mode_count}")

    # Fill output with mode value where arr2 is True
    output[true_indices] = mode_value
    return mode_value

def segment_anything_auto(image, output_img, output_folder_path):
    resize_ratio = 8
    image_resize = cv2.resize(image, (image.shape[1] // resize_ratio, image.shape[0] // resize_ratio))
    output_img_resize = cv2.resize(output_img.astype("uint8"), (image.shape[1] // resize_ratio, image.shape[0] // resize_ratio))

    result_padding = np.zeros(output_img_resize.shape, dtype=np.uint8)
    result_padding_color = np.zeros((*output_img_resize.shape, 3), dtype=np.uint8)

    sam_result_mask = np.zeros(output_img_resize.shape, dtype=np.uint32)
    tmp_sam_mask = np.full(output_img_resize.shape, False, dtype=bool)

    sam_checkpoint = "./saved_model_spec/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.5,
        crop_n_layers=0,
        crop_n_points_downscale_factor=0,
        min_mask_region_area=100,
    )

    masks = mask_generator.generate(image_resize)
    tmp_mask = np.full(output_img_resize.shape, True, dtype=bool)
    # print("!!!!!!!!!!!!!!!!!!!!!!!")
    # print(len(masks))
    # print("!!!!!!!!!!!!!!!!!!!!!!!")
    # if len(masks) > 500:
    #     mask_generator = SamAutomaticMaskGenerator(
    #         model=sam,
    #         points_per_side=32,
    #         pred_iou_thresh=0.6,
    #         stability_score_thresh=0.6,
    #         crop_n_layers=1,
    #         crop_n_points_downscale_factor=1,
    #         min_mask_region_area=100,
    #     )

    #     masks = mask_generator.generate(image_resize)
    #     tmp_mask = np.full(output_img_resize.shape, True, dtype=bool)
    #     print("######################")
    #     print(len(masks))
    #     print("######################")
    for k in range(len(masks)):
        if find_mode(output_img_resize, masks[k]['segmentation']) != 255:
            sam_result_mask = np.where(masks[k]['segmentation'], k + 1, sam_result_mask)

    tmp_counter = 1
    for k in range(len(masks)):
        tmp_sam_mask = np.where(sam_result_mask == k + 1, True, False)
        if not tmp_sam_mask.any():
            continue
        tmp_mask = np.where(tmp_sam_mask, False, tmp_mask)
        tmp = find_mode(output_img_resize, tmp_sam_mask)
        result_padding = np.where(tmp_sam_mask, tmp, result_padding)
        random_color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        # if "1_4_3" in output_folder_path and tmp == 4:
        #     print(k)
        #     random_color = [10*tmp_counter, 10*tmp_counter, 10*tmp_counter]
        #     tmp_counter = tmp_counter + 1
        #     if k == 67:
        #         random_color = [200, 0, 0]
        result_padding_color = np.where(tmp_sam_mask[..., None], random_color, result_padding_color)
        # tmp_mask = np.where(masks[k]['segmentation'], False, tmp_mask)
        # tmp = find_mode(output_img_resize, masks[k]['segmentation'])
        # result_padding = np.where(masks[k]['segmentation'], tmp, result_padding)
        # random_color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        # if "1_4_3" in output_folder_path and tmp == 4:
        #     print(k)
        #     random_color = [10*tmp_counter, 10*tmp_counter, 10*tmp_counter]
        #     tmp_counter = tmp_counter + 1
        #     if k == 67:
        #         random_color = [200, 0, 0]

        # result_padding_color = np.where(masks[k]['segmentation'][..., None], random_color, result_padding_color)

    tmp = find_mode(output_img_resize, tmp_mask)
    result_padding = np.where(tmp_mask, output_img_resize, result_padding)

    # Build safe file paths
    predict_step3_path = os.path.join(output_folder_path, "predict_step3.png")
    sam_seg_result_path = os.path.join(output_folder_path, "SAM_seg_result.jpg")
    predict_color_path = os.path.join(output_folder_path, "predict_color.jpg")
    predict_color_mix_path = os.path.join(output_folder_path, "predict_color_mix.jpg")
    # resize_mask_path = os.path.join(output_folder_path, "resize_mask.png")

    # cv2.imwrite(resize_mask_path, output_img_resize*20)
    cv2.imwrite(predict_step3_path, cv2.resize(result_padding.astype("uint8"), (image.shape[1], image.shape[0])))
    cv2.imwrite(sam_seg_result_path, result_padding_color)
    convert_png_to_jpg(predict_step3_path, predict_color_path)
    apply_semi_transparent_mask(image[:, :, [2, 1, 0]], np.array(Image.open(predict_color_path))[:, :, [2, 1, 0]], predict_color_mix_path, alpha=0.5)

    try:
        os.remove(predict_color_path)
    except OSError:
        pass

    return cv2.resize(result_padding.astype("uint8"), (image.shape[1], image.shape[0]))


def process_images(image_list, output_folder_path, gt_file_name):
    for image_path in image_list:
        # if "1_4_3" not in image_path:
        #     continue
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        sub_output_folder_path = os.path.join(output_folder_path, image_name)

        if not os.path.exists(sub_output_folder_path):
            print(f"No classification result for {image_name}!")
            continue

        img_org = np.array(Image.open(image_path))
        processed_output_img = np.array(Image.open(os.path.join(sub_output_folder_path, gt_file_name)))
        segment_anything_auto(img_org, processed_output_img, sub_output_folder_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--model_selection", type=str, default="", help="Model selection")
    parser.add_argument('--image_root', type=str,
                        default='/media/zzd/hrhsr/backup/Documents/img_seg/River/data/leaf_on_off_train_infer/inference/leaf_off/2024_2025_tmp/',
                        help='The root dir where images are stored')
    parser.add_argument('--image_path', type=str, default='',
                        help='Path to a single image (optional)')
    parser.add_argument("--gt_file_name", type=str, default="output_mask.png")
    parser.add_argument("--output_folder_path", type=str,
                        default='/media/zzd/hrhsr/backup/Documents/img_seg/River/data/leaf_on_off_train_infer/inference/leaf_off/2024_2025_tmp/',
                        help="Output folder path")
    args = parser.parse_args()

    root_dir = args.image_root
    output_folder_path = args.output_folder_path
    gt_file_name = args.gt_file_name

    if args.image_path:
        image_list = [args.image_path]
    else:
        image_list = sorted(glob.glob(os.path.join(root_dir, '*.JPG'))) + \
                     sorted(glob.glob(os.path.join(root_dir, '*.jpg')))

    print(f"Found {len(image_list)} images.")
    process_images(image_list, output_folder_path, gt_file_name)


if __name__ == "__main__":
    main()
