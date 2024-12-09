import torch
import clip
from PIL import Image
import onnx
import onnxsim
import os
import numpy as np
from hailo_sdk_client import ClientRunner
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
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

model_name = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device=device)

inputRes = model.visual.input_resolution
image = preprocess(Image.open("models/temp/images.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

vit = model.visual
text_enc = model.transformer
input_size = vit.input_resolution
print(f"Processing model: {model_name}, Input size: {input_size}")

# Create dummy input
dummy_input = torch.randn(
    1, 3, input_size, input_size, requires_grad=True, device=device)

# Export the model to ONNX format
modelPath = f"models/temp/CLIP_{model_name.replace('/', '_')}_Vision_e.onnx"
vit.eval()  # Set model to evaluation mode
torch.onnx.export(
    vit,                    # Model being run
    # Model input (or a tuple for multiple inputs)
    dummy_input,
    modelPath,            # Where to save the model
    opset_version=15,       # ONNX version to export the model to
    do_constant_folding=True,  # Whether to execute constant folding for optimization
    input_names=["input"],  # Name of the input node
    output_names=["output"],  # Name of the output node
    training=torch.onnx.TrainingMode.EVAL
    )
model = onnx.load(modelPath)
simplified_model, check = onnxsim.simplify(model)
simplified_path = modelPath.replace(".onnx", "_simplified.onnx")
onnx.save(simplified_model, simplified_path)

chosen_hw_arch = "hailo8l"
input_folder = "../Data/02_data/hexagon_images/candolle_5patch"
har_path = "models/temp/"
hef_path = "models/temp/"

runner = ClientRunner(hw_arch=chosen_hw_arch)
onnx_model = onnx.load(modelPath)
input_shape = [[d.dim_value for d in _input.type.tensor_type.shape.dim]
                for _input in onnx_model.graph.input][0]

model_path_simple = modelPath

if "TinyCLIP" in modelPath:
    model_name = modelPath.split("/")[-1]
    if "modified" in model_name:
        model_name = model_name.split('_')[1]
    if "simplified" in model_name:
        model_name = model_name.split('_')[0]
        
else:
    model_name = modelPath.split("/")[-1]
    if "modified" in model_name:
        model_name = model_name.split('_')[1] + "_" + model_name.split('_')[2]   
        model_name = model_name.split('_')[1]
    else:
        model_name = model_name.split('_')[0] + "_" + model_name.split('_')[1] 
                
    
print(f"Take model from {model_path_simple}")
hn, npz = runner.translate_onnx_model(
    model_path_simple,
    model_name,
    start_node_names=["input"],
    disable_shape_inference=True,
    net_input_shapes=input_shape
)

hailo_model_har_name = str(har_path / f"{model_name}_hailo_model.har")
runner.save_har(hailo_model_har_name)

x_y_pixel = input_shape[3]
preprocess = transform(x_y_pixel)
images_list = [img_name for img_name in os.listdir(
    input_folder) if os.path.splitext(img_name)[1] == ".jpg"]
images_list = images_list[0:1024]
calib_dataset = np.zeros((len(images_list),  x_y_pixel, x_y_pixel, 3))

for idx, img_name in enumerate(sorted(images_list)):
    img = Image.open(os.path.join(input_folder, img_name))
    img_preproc = preprocess(img)
    img_transposed = np.transpose(
    img_preproc.numpy(), (1, 2, 0))  # change dim arangment for Hailo
    calib_dataset[idx, :, :, :] = img_transposed

# Second, we will load our parsed HAR from the Parsing Tutorial

hailo_model_har_name = f"{model_name}_hailo_model.har"
hailo_model_har_path = har_path / hailo_model_har_name
assert os.path.isfile(
    hailo_model_har_path), "Please provide valid path for HAR file"
print(f"Model from {hailo_model_har_path}")
runner = ClientRunner(
    har=str(hailo_model_har_path), hw_arch=chosen_hw_arch)

# Call Optimize to perform the optimization process
runner.optimize(calib_dataset)

# Save the result state to a Quantized HAR file
quantized_model_har_path = str(
    har_path / f"{model_name}_quantized_model.har")
runner.save_har(quantized_model_har_path)
print(f"saved model at {quantized_model_har_path}")
quantized_model_har_path = str(
    har_path / f"{model_name}_quantized_model.har")
print(f"Model used:{model_name}_quantized_model.har")
runner = ClientRunner(har=quantized_model_har_path,
                        hw_arch=chosen_hw_arch)
# By default it uses the hw_arch that is saved on the HAR. It is not recommended to change the hw_arch after Optimization.

hef = runner.compile()

file_name = str(hef_path / f"{model_name}.hef")
with open(file_name, "wb") as f:
    f.write(hef)

har_path = har_path + f"/{model_name}_compiled_model.har"
runner.save_har(har_path)
