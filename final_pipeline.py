import os
import argparse
import subprocess
import time
from PIL import Image
import math

Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(description="Final Pipeline")
parser.add_argument("--image_path", type=str, default="/media/zzd/hrhsr/backup/Documents/img_seg/River/river_classification_pipeline/data/test_data/P4_10_2_2023_transparent_mosaic_group1.tif", help="The path of the image")
parser.add_argument('--output_folder_path', type = str, default= '/media/zzd/hrhsr/backup/Documents/img_seg/River/river_classification_pipeline/data/test_data/P4_10_2_2023_transparent_mosaic_group1/', help = 'output_folder_path')
# parser.add_argument("--split_ratio", type=int, default=3, help="Splits per axis (default: 3)")
# parser.add_argument("--downscale_factor", type=int, default=10, help="Downscale ratio (default: 10)")
       
args = parser.parse_args()

image_path = args.image_path
output_folder_path = args.output_folder_path
# split_ratio = args.split_ratio
# downscale_factor = args.downscale_factor

ext = os.path.splitext(image_path)[1].lower()  # normalize to lowercase

if ext in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}:
    with Image.open(image_path) as img:
        width = img.width
        height = img.height
    print(f"Image width: {width}")
    split_ratio = math.ceil(width/5000)
    split_ratio_h = math.ceil(height/5000)
    split_ratio_w = math.ceil(width/5000)
    downscale_factor = split_ratio * 1
    print(split_ratio_h, split_ratio_w, downscale_factor)

else:
    print("Unsupported file type")

if ext in {".jpg", ".jpeg", ".png"}:
    command0 = "python jpg_split_step0.py --jpg " + str(image_path) + " --tiles_dir " + str(os.path.join(output_folder_path, "split")) + " --merged_dir " + str(output_folder_path) + " --downscaled_dir " + str(output_folder_path) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor)
elif ext in {".tif", ".tiff"}:
    command0 = "python tif2jpg_split_step0_downscale_h_w_ratio.py --tif " + str(image_path) + " --tiles_dir " + str(os.path.join(output_folder_path, "split")) + " --merged_dir " + str(output_folder_path) + " --downscaled_dir " + str(output_folder_path) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor)+ " --split_ratio_h " + str(split_ratio_h)+ " --split_ratio_w " + str(split_ratio_w)
else:
    print("Input image not valid")
command1 = "python inference_ortho_list_step1.py --model fcn32s_vgg16_voc --input-dir " +  str(os.path.join(output_folder_path, "split")) + " --save-folder ./models_balance/"
command2 = "python domain_adjust_step2_topk.py --folder " +  str(os.path.join(output_folder_path, "split")) + " --mask_name processed_mask.png" #+ " --split_ratio " + str(split_ratio)
command3 = "python SAM_inference_step3.py --image_root " + str(os.path.join(output_folder_path, "split")) + " --output_folder_path " + str(os.path.join(output_folder_path, "split")) + " --gt_file_name processed_mask_domain_adjust.png"
command4 = "python debris_detection_step4.py --p " +  str(os.path.join(output_folder_path, "split")) + " -m "+ "./models/model_first -t 0.5"

start_time0 = time.time()
os.system(command0)
print(f'Image split time consumption: {time.time() - start_time0} sec')
start_time1 = time.time()
os.system(command1)
print(f'Habitat segmentation time consumption: {time.time() - start_time1} sec')
# os.system("python split_merge_scale_downscale.py --tile_name color_mask.jpg" + " --root_dir " + str(os.path.join(output_folder_path, "split")) + " --output_dir " + str(os.path.join(output_folder_path, "step1_result")) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor))
os.system("python split_merge_scale_downscale_h_w_ratio.py --tile_name color_mask.jpg" + " --root_dir " + str(os.path.join(output_folder_path, "split")) + " --output_dir " + str(os.path.join(output_folder_path, "step1_result")) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor)+ " --split_ratio_h " + str(split_ratio_h)+ " --split_ratio_w " + str(split_ratio_w))
start_time2 = time.time()
os.system(command2)
print(f'Domain adjust time consumption: {time.time() - start_time2} sec')
# os.system("python split_merge_scale_downscale.py --tile_name processed_mask_domain_adjust_color.jpg" + " --root_dir " + str(os.path.join(output_folder_path, "split")) + " --output_dir " + str(os.path.join(output_folder_path, "step2_result")) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor))
os.system("python split_merge_scale_downscale_h_w_ratio.py --tile_name processed_mask_domain_adjust_color.jpg" + " --root_dir " + str(os.path.join(output_folder_path, "split")) + " --output_dir " + str(os.path.join(output_folder_path, "step2_result")) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor)+ " --split_ratio_h " + str(split_ratio_h)+ " --split_ratio_w " + str(split_ratio_w))
start_time3 = time.time()
os.system(command3)
print(f'SAM applied time consumption: {time.time() - start_time3} sec')

# os.system("python split_merge_scale.py --tile_name predict_color_mix.jpg" + " --root_dir " + str(os.path.join(output_folder_path, "split")) + " --output_dir " + str(os.path.join(output_folder_path, "step3_result")) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor))
os.system("python split_merge_scale_downscale_h_w_ratio.py --tile_name predict_color_mix.jpg" + " --root_dir " + str(os.path.join(output_folder_path, "split")) + " --output_dir " + str(os.path.join(output_folder_path, "step3_result")) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor)+ " --split_ratio_h " + str(split_ratio_h)+ " --split_ratio_w " + str(split_ratio_w))
# os.system("python split_merge_scale.py --tile_name predict_step3.png" + " --root_dir " + str(os.path.join(output_folder_path, "split")) + " --output_dir " + str(os.path.join(output_folder_path, "step3_result")) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor))
os.system("python split_merge_scale_h_w_ratio.py --tile_name predict_step3.png" + " --root_dir " + str(os.path.join(output_folder_path, "split")) + " --output_dir " + str(os.path.join(output_folder_path, "step3_result")) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor)+ " --split_ratio_h " + str(split_ratio_h)+ " --split_ratio_w " + str(split_ratio_w))

start_time4 = time.time()
os.system(command4)
print(f'Debris detection time consumption: {time.time() - start_time4} sec')
os.system("python split_merge_scale_downscale_h_w_ratio.py --tile_name debris_detection.jpg" + " --root_dir " + str(os.path.join(output_folder_path, "split")) + " --output_dir " + str(os.path.join(output_folder_path, "step4_result")) + " --split_ratio " + str(split_ratio) + " --downscale_factor " + str(downscale_factor)+ " --split_ratio_h " + str(split_ratio_h)+ " --split_ratio_w " + str(split_ratio_w))


