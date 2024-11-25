# import the ClientRunner class from the hailo_sdk_client package

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, PILToTensor
import onnxruntime as ort
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
from splitCompilation_utils import CompilerHelper


# Own modules
print(Path.cwd())
sys.path.append(str(Path.cwd()))
import pathsToFolders as ptf  # Controlls all paths #
chosen_hw_arch = "hailo8l"

har_path = ptf.HarPath
hef_path = ptf.Hefpath
input_folder = ptf.Dataset5Patch
calibFolder = ptf.dataBaseFolder / "calibData"


upperModel_path = "hailoDFC/models/splitModels/modified_upper_clip_simpel.onnx"
lowerModel_path = "hailoDFC/models/splitModels/modified_lower_clip_simpel.onnx"
model_name_upper = "upperClip"
model_name_lower = "lowerClip"

godHelper = CompilerHelper(chosen_hw_arch,har_path,input_folder,calibFolder,288,upperModel_path,lowerModel_path,hef_path)
godHelper.compileOnnxtoHar()
godHelper.optimizeHar()
godHelper.compilerHEF()


# try:
#     from torchvision.transforms import InterpolationMode
#     BICUBIC = InterpolationMode.BICUBIC
# except ImportError:
#     BICUBIC = Image.BICUBIC


# def _convert_image_to_rgb(image):
#     return image.convert("RGB")


# def transform(self, n_px):
#     """
#     n_px: input resolution of the network
#     """
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073),
#                   (0.26862954, 0.26130258, 0.27577711)),  # for channels (R,G,B)
#     ])

# dummy_input = torch.randn(
#     1, 3, 288, 288, device="cpu").numpy()

# upperModel = ort.InferenceSession(upperModel_path)
# output_upper = upperModel.run(None, {'onnx::Cast_0': dummy_input})
# output_upper = np.reshape(output_upper, (1, 1, 40, 64))

