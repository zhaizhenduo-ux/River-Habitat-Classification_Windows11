import numpy as np
import os
import sys
import argparse
import torch
import cv2
from PIL import Image
from torchvision import transforms


Image.MAX_IMAGE_PIXELS = None

# Function to process individual images
def process_images(mask_path, color_image_path, output_color_image_path, output_mask_path):
    # Open images
    mask = Image.open(mask_path).convert("L")  # Convert mask to single channel (grayscale)
    color_image = Image.open(color_image_path).convert("RGB")  # Ensure color image is RGB

    # Convert images to numpy arrays
    mask_array = np.array(mask)
    color_image_array = np.array(color_image)
    
    # Create a copy of the color image to modify
    modified_color_image = color_image_array.copy()
    
    # Define colors
    colors = {
        0: (0, 255, 0),       # Green
        1: (255, 105, 180),   # Pink
        2: (255, 165, 0),     # Orange
        3: (255, 0, 0),       # Red
        4: (255, 255, 0),     # Yellow
        5: (0, 0, 255),       # Blue
        6: (128, 128, 128),   # Gray
    }

    # Create a white image for black areas
    white_area = np.all(color_image_array == [0, 0, 0], axis=-1)

    # Turn black regions in color image to black
    modified_color_image[white_area] = [0, 0, 0]

    # Also turn corresponding mask pixels to white
    mask_array[white_area] = 255

    # Apply half-transparent color overlay based on mask value
    for value, color in colors.items():
        mask_area = mask_array == value
        modified_color_image[mask_area] = (0.5 * modified_color_image[mask_area] + 0.5 * np.array(color)).astype(int)

    # Convert modified arrays back to images
    modified_color_image = Image.fromarray(modified_color_image)
    modified_mask_image = Image.fromarray(mask_array)

    # Save the modified images
    modified_color_image.save(output_color_image_path)
    modified_mask_image.save(output_mask_path)

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete
from core.models import get_model
Image.MAX_IMAGE_PIXELS = None
# Parser for command line arguments
parser = argparse.ArgumentParser(
    description='Predict segmentation result from images in a folder')
parser.add_argument('--model', type=str, default='fcn32s_vgg16_voc',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='pascal_aug', 
                    choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys'],
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-dir', type=str, required=True,
                    help='path to the folder containing input images')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the prediction results')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # Prepare the model
    model = get_model(config.model, local_rank=config.local_rank, pretrained=True, root=config.save_folder).to(device)
    print('Finished loading model!')
    model.eval()

    # Process each image in the input directory
    input_dir = config.input_dir
    for img_filename in os.listdir(input_dir):
        if img_filename.endswith('.jpg'):
            img_path = os.path.join(input_dir, img_filename)
            
            # Prepare subfolder for the image output
            image_name = os.path.splitext(img_filename)[0]
            output_subdir = os.path.join(input_dir, image_name)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # Open and transform the image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)

            # Split the image into 1024x1024 patches
            patch_size = 1024
            height, width, _ = image.shape
            stride = patch_size  # No overlap

            # Create an empty array to hold the full prediction
            full_pred = np.zeros((height, width), dtype=np.int32)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            with torch.no_grad():
                for y in range(0, height, stride):
                    for x in range(0, width, stride):
                        patch = image[y:y+patch_size, x:x+patch_size]
                        patch_height, patch_width, _ = patch.shape

                        # If the patch size is less than 1024x1024, pad it
                        if patch_height < patch_size or patch_width < patch_size:
                            padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                            padded_patch[:patch_height, :patch_width] = patch
                            patch = padded_patch

                        # Transform the patch
                        patch = Image.fromarray(patch)
                        patch = transform(patch).unsqueeze(0).to(device)

                        # Predict the patch
                        output = model(patch)
                        pred = torch.argmax(output[0], 1).squeeze(0).cpu().numpy()

                        # Crop the padding if necessary
                        pred = pred[:patch_height, :patch_width]

                        # Place the prediction in the full_pred array
                        full_pred[y:y+patch_height, x:x+patch_width] = pred

            # Convert the prediction to a color image
            outname = os.path.splitext(os.path.split(img_path)[-1])[0] + '_full.png'
            cv2.imwrite(outname, full_pred*1)
            #mask = get_color_pallete(full_pred, config.dataset)
            mask_path = os.path.join(output_subdir, 'processed_mask.png')
            #mask.save(os.path.join(config.outdir, outname))

            # Process the images and save color mask
            color_image_path = img_path
            output_color_image_path = os.path.join(output_subdir, 'color_mask.jpg')
            process_images(outname, color_image_path, output_color_image_path, mask_path)
            try: os.remove(outname)
            except OSError: pass

if __name__ == '__main__':
    demo(args)
