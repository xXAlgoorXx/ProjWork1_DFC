import torch
import clip
from torch.autograd import Variable

vision_arch = "RN50"
device = "cpu"
model, preprocess = clip.load("RN50", device=device)

vit = model.visual
inputSize = model.visual.input_resolution
print(inputSize)
dummy_input = Variable(torch.randn(3, inputSize, inputSize,requires_grad=True, device="cpu"))
dummy_input = torch.randn(1,3, inputSize, inputSize,requires_grad=True, device="cpu")

# ==================== #
# Export Model as ONNX #
# ==================== #
vit.eval()
torch.onnx.export(vit,          # model being run 
         dummy_input,                 # model input (or a tuple for multiple inputs) 
         "CLIP_RN50.onnx",       # where to save the model  
         opset_version=15,      # the ONNX version to export the model to
         do_constant_folding=True,# whether to execute constant folding for optimization
        input_names=["input"],
        output_names=["output"],
        )