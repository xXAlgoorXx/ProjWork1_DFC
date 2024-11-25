from hailo_sdk_client import ClientRunner
from pathlib import Path
import sys

# Own modules
sys.path.append("/home/lukasschoepf/Documents/ProjWork1_DFC")

import pathsToFolders as ptf #Controlls all paths


model_name = "RN50_simple"
quantized_model_har_path = str(ptf.QuantizedPath / f"{model_name}_quantized_model.har")

runner = ClientRunner(har=quantized_model_har_path)
# By default it uses the hw_arch that is saved on the HAR. It is not recommended to change the hw_arch after Optimization.

hef = runner.compile()

file_name = f"{model_name}.hef"
with open(file_name, "wb") as f:
    f.write(hef)

har_path =ptf.HarPath/ f"{model_name}_compiled_model.har"
runner.save_har(har_path)
