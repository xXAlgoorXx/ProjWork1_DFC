# General imports used throughout the tutorial
import tensorflow as tf
from IPython.display import SVG

# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner
import onnx
print(onnx.__version__)
chosen_hw_arch = "hailo8"

onnx_model_name = "RN50x4"
onnx_path = f"models/{onnx_model_name}.onnx"

runner = ClientRunner(hw_arch=chosen_hw_arch)

hn, npz = runner.translate_onnx_model(
    onnx_path,
    onnx_model_name,
    # start_node_names=["modelInput"],
    # end_node_names=["modelOutput"],
    # disable_rt_metadata_extraction=True,
    # disable_shape_inference=True,
    # net_input_shapes={"modelInput": [1, 3, 224, 224]}
)

hailo_model_har_name = f"Harfiles/{onnx_model_name}_hailo_model.har"
runner.save_har(hailo_model_har_name)
