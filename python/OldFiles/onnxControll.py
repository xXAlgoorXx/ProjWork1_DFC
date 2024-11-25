# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner
import onnx
print(onnx.__version__)
chosen_hw_arch = "hailo8l"

onnx_model_name = "clip_simpel"
onnx_path = f"hailoDFC/models/{onnx_model_name}.onnx"
onnx_path = f"hailoDFC/HailoDevModel/clip_resnet_50x4.onnx"
onnx_path = f"{onnx_model_name}.onnx"
# onnx_path = f"clip_resnet_50x4_simple.onnx"
# runner = ClientRunner(hw_arch=chosen_hw_arch)


# onnx_model_name = "clip_resnet_50x4_simple"
# onnx_path = "clip_resnet_50x4_simple.onnx"
runner = ClientRunner(hw_arch=chosen_hw_arch)

hn, npz = runner.translate_onnx_model(
    str(onnx_path),
    onnx_model_name
)

hailo_model_har_name = f"hailoDFC/Harfiles/{onnx_model_name}_hailo_model.har"
runner.save_har(hailo_model_har_name)


# onnx_model_name = "clip_resnet_50x4_simple"
# onnx_path = f"hailoDFC/models/{onnx_model_name}.onnx"
# onnx_path = f"hailoDFC/HailoDevModel/clip_resnet_50x4_simple.onnx"
# onnx_path = f"clip_resnet_50x4_simple.onnx"
# runner = ClientRunner(hw_arch=chosen_hw_arch)

# hn, npz = runner.translate_onnx_model(
#     onnx_path,
#     onnx_model_name,
# )
