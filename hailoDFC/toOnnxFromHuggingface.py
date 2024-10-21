from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel

import optimum.exporters.onnx

model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text=["a photo of a cat", "a photo of a dog"]
inputs = processor(text, images=image, return_tensors="pt", padding=True)

vision_arch = "tinyVit"
image = inputs.data["pixel_values"]
visualEmbedding = model.vision_model
visualEmbedding.eval()

torch.onnx.export(visualEmbedding,  # Model being run
         image,  # Model input
         f"models/{vision_arch}.onnx",  # Output model location
         input_names=['modelInput'],  # Input name
         output_names=['modelOutput']  # Output name
         )

print(f"Model saved as {vision_arch}.onnx")