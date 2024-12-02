import onnxruntime as ort
import onnx
from onnx import numpy_helper
import torch
import clip
import numpy as np
from torch.autograd import Variable
import json

def printNodeNams(onnxObj):
    # Load the ONNX model
    graph = onnxObj.graph
    name = graph.name
    # List all nodes in the graph to find your target layer
    for node in graph.node:
        print(f"Node name: {node.name}, Node type: {node.op_type}")

    # Extract parameters (weights/biases) from the initializer
    parameters = {}
    for initializer in graph.initializer:
        param_name = initializer.name
        param_data = numpy_helper.to_array(initializer)  # Convert to NumPy array
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
        
    return bias,weights,name

# m = onnx.load("hailoDFC/models/baseAndSimple/RN50x4_simple.onnx")
# weights = m.graph.initializer
# for weight in weights:
#     w = numpy_helper.to_array(weight)
#     print(w.shape)
    
# w = numpy_helper.to_array(weights[0])
# bias = numpy_helper.to_array(weights[1])

# print(f"Weights: {w.shape}\nBias: {bias.shape}")

# weightMatrix = np.array(w)
# bias = np.array(bias)


if __name__ == "__main__":
    onnxObj = onnx.load("hailoDFC/models/baseAndSimple/RN50x4_simple.onnx")
    bias,weights,name = printNodeNams(onnxObj)
    gemm_layer = {
    "bias":bias.tolist(),
    "weights":weights.tolist(),
    }
    with open('gemm_layer.json', 'w') as f:
        json.dump(gemm_layer, f)
