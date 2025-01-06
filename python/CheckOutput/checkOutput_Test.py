from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, PILToTensor
import onnxruntime as ort
import clip
import torch
from PIL import Image
import numpy as np
import json
import open_clip
from onnx import numpy_helper
import onnx
from hailo_sdk_client import ClientRunner, InferenceContext
import os
from tqdm import tqdm
import sys
from pathlib import Path

import matplotlib.pyplot as plt
# Own modules
import folderManagment.pathsToFolders as ptf  # Controlls all paths
# preprocessing

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


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


class RestOfGraphOnnx:
    """
    GemmLayer which got cut off
    """

    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)

    def __call__(self, input):
        # input = np.array(list(input.values())[0]).squeeze()
        if input.ndim == 1:
            input = input[np.newaxis, :]
        result = self.session.run(
            None, {"/attnpool/Reshape_7_output_0": input})
        result = np.array(result).squeeze(0)
        return result
    
    
def loadJson(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        data = json.load(file)

    return data

def checkOutput(onnx_path, quantized_model_har_path, postProcessing_onnx, inputSize,  testImage,model, text, text_json):
    x_y_pixel = inputSize
    chosen_hw_arch = "hailo8l"
    preprocess = transform(x_y_pixel)
    postProcess = RestOfGraphOnnx(postProcessing_onnx)
    textEmb2 = np.array(loadJson(text_json)["2"]["embeddings"])
    image = preprocess(Image.open(testImage)).unsqueeze(0)
    session = ort.InferenceSession(onnx_path)

    # # Run inference
    # imageEmb_onnx = session.run(None, {"input": image.numpy()})
    # transpose_ouput = np.array(imageEmb_onnx)#.reshape(1, -1)
    # postp_onnx = postProcess(transpose_ouput)

    print("Prepare Calib Dataset")
    calib_data = np.zeros((1,  x_y_pixel, x_y_pixel, 3))
    img_preproc = image.squeeze()
    
    # change dim arangment for Hailo
    img_transposed = np.transpose(img_preproc.numpy(), (1, 2, 0))
    calib_data[0, :, :, :] = img_transposed

    img_transposed = [np.transpose(image.numpy().squeeze(), (1, 2, 0))]
    
    # Quantized
    runner = ClientRunner(har=quantized_model_har_path,
                          hw_arch=chosen_hw_arch)
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        output_quantized_har = runner.infer(ctx, calib_data)
        output_quantized_har = output_quantized_har[0][0]
        output_quantized_har = postProcess(output_quantized_har)
    
    print("\n=== Text Emb check ===")
    text_features = model.encode_text(text).detach().numpy()
    print(textEmb2-text_features)
     
    fig, ax = plt.subplots(1,2)
    
    print("\n=== TinyCLIP PC ===")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    ax[0].plot(text_probs[0])
    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
    
    
    print("\n=== TinyCLIP HAR Quantized ===")
    with torch.no_grad():
        image_features = torch.tensor(
            output_quantized_har, dtype=torch.float32)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    ax[1].plot(text_probs[0])
    
    plt.ylabel("Probs")
    plt.show()
    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

if __name__ == "__main__":
    names2 = ["architectural", "office", "residential", "school", "manufacturing",
              "cellar", "laboratory", "construction site", "mining", "tunnel"]
    model_name = "TinyCLIP-ResNet-19M-Text-19M"
    tokenizer = open_clip.get_tokenizer(model_name)
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=f"tinyClipModels/{model_name}-LAION400M.pt"
    )
    onnx_path = "models/baseAndSimple/TinyCLIP-ResNet-19M_simplified.onnx"
    textEmb_path = "models/textEmbeddings/textEmbeddings_TinyCLIP-ResNet-19M-Text-19M.json"
    
    checkOutput(onnx_path = onnx_path,
                quantized_model_har_path = "models/Harfiles/TinyCLIP-ResNet-19M_quantized_model.har",
                postProcessing_onnx = "models/RestOfGraphONNX/RestOf_TinyCLIP-ResNet-19M_simplified.onnx",
                inputSize = 224,
                testImage = "temp/panorama_00007_0003_3.jpg",
                model = model, 
                text = tokenizer(names2),
                text_json = textEmb_path)
    
