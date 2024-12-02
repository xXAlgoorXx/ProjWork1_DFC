import onnxruntime as ort
import clip
import torch
from PIL import Image
import numpy as np
import json

def loadJson(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        data = json.load(file)

    return data

class RestOfGraph:
    """
    GemmLayer which got cut off
    """
    
    def __init__(self,weightJson_path):
        self.json = loadJson(weightJson_path)
        self.bias = np.array(self.json["bias"])
        self.weight = np.array(self.json["weights"])
        
    def __call__(self,input):
        input = np.array(list(input.values())[0]).squeeze()
        result = np.dot(self.weight,input) + self.bias
        return result



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50x4", device=device)
    # Load the model and create InferenceSession
    img_path  = "hailoDFC/testImages/panorama_00009_0012_0.jpg"
    model_path = "/home/lukasschoepf/Documents/ProjWork1_DFC/hailoDFC/models/baseAndSimple/RN50x4_simple.onnx"
    image = Image.open(img_path)
    image = preprocess(image).unsqueeze(0).to(device)
    imageEmb_clip = model.encode_image(image).detach().numpy().squeeze()
        
    session = ort.InferenceSession(model_path)
    # "Load and preprocess the input image inputTensor"
    # Run inference
    imageEmb_onnx = session.run(None, {"onnx::Cast_0": image.numpy()})
    imageEmb_onnx = np.array(imageEmb_onnx)
    print(imageEmb_clip-imageEmb_onnx)