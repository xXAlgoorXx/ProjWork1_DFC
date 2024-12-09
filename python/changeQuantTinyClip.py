# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner
import onnx
import sys
import os
import numpy as np
import torch
import clip
import open_clip
from pathlib import Path
from onnxsim import simplify
from PIL import Image
from tqdm import tqdm
import folderManagment.pathsToFolders as ptf  # Controlls all paths #

# preprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, PILToTensor
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

model_name = "TinyClip-19M-16Bit"
datafolder = ptf.dataBaseFolder
input_folder = ptf.Dataset5Patch
calibFolder = datafolder / "calibData"


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def transform(n_px):
    """
    n_px: input resolution of the network
    """
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


chosen_hw_arch = "hailo8l"

har_path = ptf.HarPath
hef_path = ptf.Hefpath


x_y_pixel = 224
preprocess = transform(x_y_pixel)
images_list = [img_name for img_name in os.listdir(
    input_folder) if os.path.splitext(img_name)[1] == ".jpg"]
images_list = images_list[0:2048]
calib_dataset = np.zeros((len(images_list),  x_y_pixel, x_y_pixel, 3))

# for idx, img_name in tqdm(enumerate(sorted(images_list))):
#     img = Image.open(os.path.join(input_folder, img_name))
#     # img = PILToTensor(img)
#     img_preproc = preprocess(img)
#     img_transposed = np.transpose(img_preproc.numpy(), (1, 2, 0))
#     # input_data = (img_transposed * 255).astype(np.uint8)  # Assuming image is already normalized
#     calib_dataset[idx, :, :, :] = img_transposed
calib_dataset = np.load(calibFolder / f"calib_set_{model_name}.npy")
np.save(calibFolder / f"calib_set_{model_name}.npy", calib_dataset)

# Second, we will load our parsed HAR from the Parsing Tutorial

hailo_model_har_path = f"models/Harfiles/TinyCLIP-ResNet-19M_hailo_model.har"
assert os.path.isfile(
    hailo_model_har_path), "Please provide valid path for HAR file"
print(f"Model from {hailo_model_har_path}")
runner = ClientRunner(har=str(hailo_model_har_path), hw_arch=chosen_hw_arch)

# Batch size is 8 by default
alls_lines = [#"model_optimization_config(compression_params, auto_16bit_weights_ratio=0)\n",
              "model_optimization_flavor(optimization_level=2)\n",
              "model_optimization_config(calibration, batch_size=16, calibset_size=2048)\n",]  # From tutorial

# Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # From Lia
# alls = "normalization1 = normalization([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])\n"

# Load the model script to ClientRunner so it will be considered on optimization
runner.load_model_script("".join(alls_lines))

# Call Optimize to perform the optimization process
runner.optimize(calib_dataset)

# Save the result state to a Quantized HAR file
quantized_model_har_path = str(har_path / f"{model_name}_quantized_model.har")
runner.save_har(quantized_model_har_path)
print(f"saved model at {quantized_model_har_path}")
