import torch
import clip
from PIL import Image
import onnx
from onnx import helper

vision_arch = "RN50x4"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(clip.available_models())
model, preprocess = clip.load("RN50", device=device)
model2, preprocess2 = clip.load("RN50x4", device=device)

# Preprocess image
image = preprocess(Image.open("pics/constructionsite.png")).unsqueeze(0).to(device)

# ==================== #
# Export Model as ONNX #
# ==================== #

visualEmbedding = model.visual
visualEmbedding.eval()

torch.onnx.export(visualEmbedding,  # Model being run
         image,  # Model input
         f"models/{vision_arch}.onnx",  # Output model location
         export_params=True,  # Store trained parameters
         opset_version=17,  # ONNX opset version
         verbose=False,
         do_constant_folding=True,  # Optimize constant folding
         input_names=['modelInput'],  # Input name
         output_names=['modelOutput']  # Output name
         )

print(f"Model saved as {vision_arch}.onnx")

