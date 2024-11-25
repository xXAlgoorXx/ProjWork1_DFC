# General imports used throughout the tutorial
# file operations
import json
import os
import numpy as np
import tensorflow as tf
from IPython.display import SVG
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.eager.context import eager_mode
from pathlib import Path

# import the hailo sdk client relevant classes
from hailo_sdk_client import ClientRunner, InferenceContext

# preprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, PILToTensor
import torch

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC




harPath = Path("hailoDFC/Harfiles")
datafolder = Path("../Data")
input_folder = datafolder / '02_data/hexagon_images/candolle_5patches_224px'
calibFolder = datafolder / "calibData"

model_name ="RN50_simple"

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
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# First, we will prepare the calibration set. Resize the images to the correct size and crop them.
def preproc(image, output_height=224, output_width=224, resize_side=256):
    """imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px"""
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h * scale), int(w * scale)])
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)

        return tf.squeeze(cropped_image)

x_y_pixel = 288
preprocess = transform(x_y_pixel)
images_list = [img_name for img_name in os.listdir(input_folder) if os.path.splitext(img_name)[1] == ".jpg"]
images_list = images_list[0:1024]
calib_dataset = np.zeros((len(images_list), x_y_pixel, x_y_pixel, 3))
for idx, img_name in enumerate(sorted(images_list)):
    img = Image.open(os.path.join(input_folder, img_name))
    # img = PILToTensor(img)
    img_preproc = preprocess(img)
    img_transposed = np.transpose(img_preproc.numpy(),(1,2,0))
    calib_dataset[idx, :, :, :] = img_transposed

np.save(calibFolder / f"calib_set_{model_name}.npy", calib_dataset)

# Second, we will load our parsed HAR from the Parsing Tutorial

hailo_model_har_name = f"{model_name}_hailo_model.har"
hailo_model_har_path = harPath / hailo_model_har_name
assert os.path.isfile(hailo_model_har_path), "Please provide valid path for HAR file"
runner = ClientRunner(har=str(hailo_model_har_path),hw_arch="hailo8l")
# By default it uses the hw_arch that is saved on the HAR. For overriding, use the hw_arch flag.

# Now we will create a model script, that tells the compiler to add a normalization on the beginning
# of the model (that is why we didn't normalize the calibration set;
# Otherwise we would have to normalize it before using it)

# Batch size is 8 by default
alls = "normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n" # From tutorial

# Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # From Lia
# alls = "normalization1 = normalization([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])\n"

# Load the model script to ClientRunner so it will be considered on optimization
runner.load_model_script(alls)

# Call Optimize to perform the optimization process
runner.optimize(calib_dataset)

# Save the result state to a Quantized HAR file
quantized_model_har_path = f"{model_name}_quantized_model.har"
runner.save_har(quantized_model_har_path)
print(f"saved model at {quantized_model_har_path}")