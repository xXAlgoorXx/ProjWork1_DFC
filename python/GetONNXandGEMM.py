import torch
import clip
from torch.autograd import Variable
import open_clip
import onnx
import onnxsim
from onnx import numpy_helper
import os
import re
import json
import folderManagment.pathsToFolders as ptf


def simplify_onnx_model(onnx_path):
    """
    Simplify the ONNX model graph.
    """
    model = onnx.load(onnx_path)
    print(f"Simplifying ONNX model: {onnx_path}")
    simplified_model, check = onnxsim.simplify(model)
    if not check:
        raise ValueError(f"Simplification failed for {onnx_path}")

    # Save the simplified model
    simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
    onnx.save(simplified_model, simplified_path)
    print(f"Simplified model saved at: {simplified_path}")


def get_modelnames():
    """
    Retrieve available CLIP and Tiny CLIP ResNet model names.
    """
    clip_models = []
    tiny_clip_models = []
    for clipmodel in clip.available_models():
        if "RN" in clipmodel:
            clip_models.append(clipmodel)

    print("Open CLIP Models:", open_clip.list_models())
    for clipmodel in open_clip.list_models():
        if "ResNet" in clipmodel and "Tiny" in clipmodel:
            tiny_clip_models.append(clipmodel)
    return clip_models, tiny_clip_models


def load_onnx_clip(clip_models):
    """Export CLIP models to ONNX format."""
    device = torch.device("cpu")
    os.makedirs("models/baseAndSimple", exist_ok=True)
    Clip_onnx_paths = []

    for model_name in clip_models:
        model, preprocess = clip.load(model_name, device=device)
        vit = model.visual
        input_size = vit.input_resolution
        print(f"Processing model: {model_name}, Input size: {input_size}")

        # Create dummy input
        dummy_input = torch.randn(
            1, 3, input_size, input_size, requires_grad=True, device=device)

        # Export the model to ONNX format
        output_path = f"models/baseAndSimple/CLIP_{model_name.replace('/', '_')}.onnx"
        vit.eval()  # Set model to evaluation mode
        torch.onnx.export(
            vit,                    # Model being run
            # Model input (or a tuple for multiple inputs)
            dummy_input,
            output_path,            # Where to save the model
            opset_version=15,       # ONNX version to export the model to
            do_constant_folding=True,  # Whether to execute constant folding for optimization
            input_names=["input"],  # Name of the input node
            output_names=["output"],  # Name of the output node
        )
        print(f"Model {model_name} exported to {output_path}")
        simplify_onnx_model(output_path)
        Clip_onnx_paths.append(output_path)
    return Clip_onnx_paths


def load_onnx_tinyCLIP(tiny_clip_models):
    """Export Tiny CLIP models to ONNX format."""
    device = torch.device("cpu")
    os.makedirs("models/baseAndSimple", exist_ok=True)
    TinyClip_onnx_paths = []

    for model_name in tiny_clip_models:
        # Create the Tiny CLIP model and its transforms
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=f"tinyClipModels/{model_name}-LAION400M.pt"
        )
        vit = model.visual
        input_size = model.visual.image_size

        # Extract the first occurrence of '19M' or similar pattern
        match = re.search(r'\d+M', model_name)
        params = match.group(0) if match else "UnknownParams"

        # Tiny CLIP uses transforms to prepare inputs, here we simulate the shape
        dummy_input = torch.randn(
            1, 3, input_size, input_size, requires_grad=True, device=device)

        # Create a specific name for Tiny CLIP ONNX model
        output_path = f"models/baseAndSimple/TinyCLIP-ResNet-{params}.onnx"

        vit.eval()  # Set model to evaluation mode
        torch.onnx.export(
            vit,                    # Model being run
            # Model input (or a tuple for multiple inputs)
            dummy_input,
            output_path,            # Where to save the model
            opset_version=15,       # ONNX version to export the model to
            do_constant_folding=True,  # Whether to execute constant folding for optimization
            input_names=["input"],  # Name of the input node
            output_names=["output"],  # Name of the output node
        )
        print(f"Tiny CLIP Model {model_name} exported to {output_path}")
        simplify_onnx_model(output_path)
        TinyClip_onnx_paths.append(output_path)
    return TinyClip_onnx_paths


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


def get_GemmLayer(onnxPaths_list):
    for onnxPath in onnxPaths_list:
        name = onnxPath.split("/")[-1]
        if "simplified" in name:
            name = name.split('_')[0]
        name = name.split('.')[0]

        bias, weights, _ = printNodeNams(onnxPath)
        gemm_layer = {
            "bias": bias.tolist(),
            "weights": weights.tolist(),
        }
        gemmpath = str(ptf.gemmpath / f"gemmLayer_{name}.json")
        with open(gemmpath, 'w') as f:
            json.dump(gemm_layer, f)


if __name__ == "__main__":
    print("Retrieving models...")
    clip_models, tiny_clip_models = get_modelnames()
    print("Exporting CLIP models to ONNX...")
    Clip_onnx_paths = load_onnx_clip(clip_models)
    print("Exporting Tiny CLIP models to ONNX...")
    TinyClip_onnx_paths = load_onnx_tinyCLIP(tiny_clip_models)

    print("Get Gemm layer attributes CLIP")
    get_GemmLayer(Clip_onnx_paths)
    print("Get Gemm layer attributes TinyCLIP")
    get_GemmLayer(TinyClip_onnx_paths)
    print("Done!")
