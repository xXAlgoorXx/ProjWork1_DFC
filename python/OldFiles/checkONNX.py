
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

sys.path.append(str(Path.cwd() / "python"))

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


def printNodeNams(onnxObj_path):
    # Load the ONNX model
    onnxObj = onnx.load(onnxObj_path)
    graph = onnxObj.graph
    name = graph.name
    # List all nodes in the graph to find your target layer
    for node in graph.node:
        print(f"Node name: {node.name}, Node type: {node.op_type}")

    # Extract parameters (weights/biases) from the initializer
    parameters = {}
    for initializer in graph.initializer:
        param_name = initializer.name
        param_data = numpy_helper.to_array(
            initializer)  # Convert to NumPy array
        parameters[param_name] = param_data

    # Example: Get weights and bias for a specific layer
    layer_name = "/attnpool/Gemm"
    weights = parameters.get(f"attnpool.c_proj.weight", None)
    bias = parameters.get(f"attnpool.c_proj.bias", None)

    if weights is not None:
        print(f"Weights for {layer_name}: {weights}")
        print(f"Shape: {weights.shape}")
    else:
        print(f"No weights found for {layer_name}")

    if bias is not None:
        print(f"Bias for {layer_name}: {bias}")
        print(f"Shape: {bias.shape}")
    else:
        print(f"No bias found for {layer_name}")

    return bias, weights, name


def get_GemmLayer(onnxPath):
    name = onnxPath.split("/")[-1]
    if "simplified" in name:
        name = name.split('_')[0]
    name = name.split('.')[0]

    bias, weights, _ = printNodeNams(onnxPath)
    gemm_layer = {
        "bias": bias.tolist(),
        "weights": weights.tolist(),
    }
    return gemm_layer


def loadJson(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        data = json.load(file)

    return data


class RestOfGraph:
    """
    GemmLayer which got cut off
    """

    def __init__(self, weightJson_path):
        self.json = loadJson(weightJson_path)
        self.bias = np.array(self.json["bias"])
        self.weight = np.array(self.json["weights"])

    def __call__(self, input):
        result = np.dot(input, self.weight.T) + self.bias
        return result

    def checkWeights(self, otherjson):
        bias2 = otherjson["bias"]
        weights2 = otherjson["weights"]
        print("CHECK")
        print(self.bias - bias2)
        print(self.weight - weights2)

class RestOfGraphOnnx:
    """
    GemmLayer which got cut off
    """
    def __init__(self,onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        
    def __call__(self,input):
        # input = np.array(list(input.values())[0]).squeeze()
        if input.ndim == 1:
            input = input[np.newaxis,:]
        result = self.session.run(None, {"/attnpool/Reshape_7_output_0": input})
        result = np.array(result).squeeze(0)
        return result


def findBestIndex(img_path, onnx_path, inputSize, postprocess_json, text):
    img_path = img_path
    x_y_pixel = inputSize
    preprocess = transform(x_y_pixel)

    image = preprocess(Image.open(testImage)).unsqueeze(0)
    session = ort.InferenceSession(onnx_path)
    postProcess = RestOfGraph(postprocess_json)

    # Run inference
    imageEmb_onnx = session.run(None, {"input": image.numpy()})
    transpose_ouput = np.array(imageEmb_onnx).reshape(50, -1)
    postp_onnx_50 = postProcess(transpose_ouput)

    with torch.no_grad():
        maxProb = 0
        maxIndex = 0
        for i, postp_onnx in enumerate(postp_onnx_50):

            image_features = torch.tensor(postp_onnx, dtype=torch.float32)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @
                          text_features.T).softmax(dim=-1)
            if text_probs[1] > maxProb:
                maxProb = text_probs[1]
                maxIndex = i

    return maxIndex, maxProb


def compareTinyCLip(img_path, onnx_path, onnxMod_path, har_path, cHar_path, qHar_path, inputSize, postprocess_json, model, text):
    img_path = img_path
    model_har_path = har_path
    compiled_model_har_path = cHar_path
    quantized_model_har_path = qHar_path
    x_y_pixel = inputSize
    chosen_hw_arch = "hailo8l"
    preprocess = transform(x_y_pixel)
    model.eval()
    index = 0

    image = preprocess(Image.open(img_path)).unsqueeze(0)

    session = ort.InferenceSession(onnxMod_path)
    session2 = ort.InferenceSession(onnx_path)
    postProcess = RestOfGraph(postprocess_json)

    imageEmb_onnx2 = session2.run(None, {"input": image.numpy()})

    # Run inference
    imageEmb_onnx = session.run(None, {"input": image.numpy()})
    transpose_ouput = np.array(imageEmb_onnx)[0]
    transpose_ouput = np.transpose(transpose_ouput, (0, 2, 1, 3))
    transpose_ouput = transpose_ouput.reshape(1, 50, 22*64)
    postp_onnx_50 = postProcess(transpose_ouput)
    postp_onnx = postp_onnx_50[:, index]

    print("Prepare Calib Dataset")
    calib_data = np.zeros((1,  x_y_pixel, x_y_pixel, 3))
    img_preproc = image.squeeze()
    # change dim arangment for Hailo
    img_transposed = np.transpose(img_preproc.numpy(), (1, 2, 0))
    calib_data[0, :, :, :] = img_transposed

    img_transposed = [np.transpose(image.numpy().squeeze(), (1, 2, 0))]

    # Nativ
    runner = ClientRunner(har=model_har_path,
                          hw_arch=chosen_hw_arch)
    with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
        output_har = runner.infer(ctx, calib_data)
        output_har = output_har[0][0]
        output_har = postProcess(output_har)
        output_har = output_har[index]

    # Compiled
    runner = ClientRunner(har=compiled_model_har_path,
                          hw_arch=chosen_hw_arch)
    with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
        output_compiled_har = runner.infer(ctx, calib_data)
        output_compiled_har = output_compiled_har[0][0]
        output_compiled_har = postProcess(output_compiled_har)[index]

    # Quantized
    runner = ClientRunner(har=quantized_model_har_path,
                          hw_arch=chosen_hw_arch)
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        output_quantized_har = runner.infer(ctx, calib_data)
        output_quantized_har = output_quantized_har[0][0]
        output_quantized_har = postProcess(output_quantized_har)[index]

    print("\n=== TinyCLIP PC ===")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

    print("\n=== TinyCLIP ONNX Modified ===")
    with torch.no_grad():
        image_features = torch.tensor(postp_onnx, dtype=torch.float32)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

    print("\n=== TinyCLIP ONNX ===")
    with torch.no_grad():
        image_features = torch.tensor(
            np.array(imageEmb_onnx2), dtype=torch.float32)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

    print("\n=== TinyCLIP HAR Nativ ===")
    with torch.no_grad():
        image_features = torch.tensor(output_har, dtype=torch.float32)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs.tolist())  # prints: [[1., 0., 0.]]

    print("\n=== TinyCLIP HAR Compiled ===")
    with torch.no_grad():
        image_features = torch.tensor(output_compiled_har, dtype=torch.float32)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs.tolist())  # prints: [[1., 0., 0.]]

    print("\n=== TinyCLIP HAR Quantized ===")
    with torch.no_grad():
        image_features = torch.tensor(
            output_quantized_har, dtype=torch.float32)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs.tolist())  # prints: [[1., 0., 0.]]


def compareCLip(img_path, onnx_path, har_path, cHar_path, qHar_path, inputSize, postprocess_json, model, text):
    img_path = img_path
    model_har_path = har_path
    compiled_model_har_path = cHar_path
    quantized_model_har_path = qHar_path
    x_y_pixel = inputSize
    chosen_hw_arch = "hailo8l"
    preprocess = transform(x_y_pixel)
    model.eval()

    image = preprocess(Image.open(testImage)).unsqueeze(0)

    session = ort.InferenceSession(onnx_path)
    postProcess = RestOfGraph(postprocess_json)

    # Run inference
    imageEmb_onnx = session.run(None, {"input": image.numpy()})
    transpose_ouput = np.array(imageEmb_onnx).reshape(1, -1)
    postp_onnx = postProcess(transpose_ouput)

    print("Prepare Calib Dataset")
    calib_data = np.zeros((1,  x_y_pixel, x_y_pixel, 3))
    img_preproc = image.squeeze()
    # change dim arangment for Hailo
    img_transposed = np.transpose(img_preproc.numpy(), (1, 2, 0))
    calib_data[0, :, :, :] = img_transposed

    img_transposed = [np.transpose(image.numpy().squeeze(), (1, 2, 0))]

    # Nativ
    runner = ClientRunner(har=model_har_path,
                          hw_arch=chosen_hw_arch)
    with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
        output_har = runner.infer(ctx, calib_data)
        output_har = output_har[0][0]
        output_har = postProcess(output_har)

    # Compiled
    runner = ClientRunner(har=compiled_model_har_path,
                          hw_arch=chosen_hw_arch)
    with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
        output_compiled_har = runner.infer(ctx, calib_data)
        output_compiled_har = output_compiled_har[0][0]
        output_compiled_har = postProcess(output_compiled_har)

    # Quantized
    runner = ClientRunner(har=quantized_model_har_path,
                          hw_arch=chosen_hw_arch)
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        output_quantized_har = runner.infer(ctx, calib_data)
        output_quantized_har = output_quantized_har[0][0]
        output_quantized_har = postProcess(output_quantized_har)

    print("\n=== CLIP PC ===")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

    print("\n=== CLIP ONNX ===")
    with torch.no_grad():
        image_features = torch.tensor(postp_onnx, dtype=torch.float32)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

    print("\n=== CLIP HAR Nativ ===")
    with torch.no_grad():
        image_features = torch.tensor(output_har, dtype=torch.float32)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs.tolist())  # prints: [[1., 0., 0.]]

    print("\n=== CLIP HAR Compiled ===")
    with torch.no_grad():
        image_features = torch.tensor(output_compiled_har, dtype=torch.float32)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs.tolist())  # prints: [[1., 0., 0.]]

    print("\n=== CLIP HAR Quantized ===")
    with torch.no_grad():
        image_features = torch.tensor(
            output_quantized_har, dtype=torch.float32)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs.tolist())  # prints: [[1., 0., 0.]]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = ["a Human", "a cat", "a dog", "a white cat", "a small dog"]
    classes = ["construction site", "town", "city",
              "country side", "alley", "parking lot", "forest"]
    # TinyClip
    testImage = "models/temp/images.jpeg"
    testImage = "temp/panorama_00002_0014_2.jpg"
    modelMod_path = "models/modified/modified_TinyCLIP-ResNet-19M_simplified.onnx"
    model_path = "models/baseAndSimple/TinyCLIP-ResNet-19M_simplified.onnx"
    model_har_path = "models/Harfiles/TinyCLIP-ResNet-19M_hailo_model.har"
    compiled_model_har_path = "models/Harfiles/TinyCLIP-ResNet-19M_compiled_model.har"
    quantized_model_har_path = "models/Harfiles/TinyCLIP-ResNet-19M_quantized_model.har"
    postprocess_json = "models/RestOfGraph/gemmLayer_TinyCLIP-ResNet-19M.json"

    model_name = "TinyCLIP-ResNet-19M-Text-19M"
    tokenizer = open_clip.get_tokenizer(model_name)
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=f"tinyClipModels/{model_name}-LAION400M.pt"
    )

    text = tokenizer(classes)
    print("TinyClip")
    compareTinyCLip(
        img_path=testImage,
        onnxMod_path=modelMod_path,
        onnx_path=model_path,
        har_path=model_har_path,
        cHar_path=compiled_model_har_path,
        qHar_path=quantized_model_har_path,
        model=model,
        inputSize=224,
        postprocess_json=postprocess_json,
        text=text,)

    # print(findBestIndex(img_path = testImage,
    #                     onnx_path = model_path,
    #                     inputSize = 224,
    #                     postprocess_json = postprocess_json,
    #                     text = text))
    # CLIP
    testImage = "models/temp/images.jpeg"
    model_path = "models/modified/modified_CLIP_RN50x4_simplified.onnx"
    model_har_path = "models/Harfiles/CLIP_RN50x4_hailo_model.har"
    compiled_model_har_path = "models/Harfiles/CLIP_RN50x4_compiled_model.har"
    quantized_model_har_path = "models/Harfiles/CLIP_RN50x4_quantized_model.har"
    postprocess_json = "models/RestOfGraph/gemmLayer_CLIP_RN50x4.json"
    model, preprocess = clip.load('RN50x4', device)
    text = clip.tokenize(classes)

    print("\nCLIP RN50x4")
    compareCLip(img_path=testImage,
                onnx_path=model_path,
                har_path=model_har_path,
                cHar_path=compiled_model_har_path,
                qHar_path=quantized_model_har_path,
                model=model,
                inputSize=288,
                postprocess_json=postprocess_json,
                text=text,)
