# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner,InferenceContext
import onnx
import os
import numpy as np
from onnxsim import simplify
from PIL import Image
from google.protobuf.json_format import MessageToDict
from os import walk
import time
from tqdm import tqdm
import random
# Own modules
import folderManagment.pathsToFolders as ptf  # Controlls all paths
from random import sample
# preprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, PILToTensor
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

layer_to_convert = {"RN50":57,"RN50x4":88,"RN50x16":130,"RN50x64":109,"TinyCLIP-ResNet-19M":56,"TinyCLIP-ResNet-30M":56,}
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

def createLayerScripts(x):
            scriptList = []
            for x in range(1,x):
                scriptList.append(f'quantization_param(conv{x}, precision_mode=a16_w16)\n')
                
            return scriptList

def getxImagesPerClass(dataFolder,x):
    """
    Images in output dict is randomly shuffled
    """
    # define second degree classes 
    in_arch = [7,10,18,27,29,32,36,1,28,6,33,40,30,31,24]#[7,18,31]#
    out_constr = [8,16,22]#[16]#
    in_constr = [9,13,39,12] #[12]#
    out_urb = [2,20,38,26,15,42,44,4,23]#[15,2,23]#
    out_forest = [17]
    
    in_arch_list = []
    out_constr_list = []
    in_constr_list = []
    out_urb_list = []
    out_forest_list = []
    
    files = os.listdir(dataFolder)
    for file in files:
        classNumber = int(file.split("_")[1])
        if classNumber in in_arch:
            in_arch_list.append(file)
            continue
        if classNumber in out_constr:
            out_constr_list.append(file)
            continue
        if classNumber in in_constr:
            in_constr_list.append(file)
            continue
        if classNumber in out_urb:
            out_urb_list.append(file)
            continue
        if classNumber in out_forest:
            out_forest_list.append(file)
            continue
        print(f"{file} no match")    
    
    random.shuffle(in_arch_list)
    random.shuffle(out_constr_list)
    random.shuffle(in_constr_list)
    random.shuffle(out_urb_list)
    random.shuffle(out_forest_list)
    
    in_arch_list = in_arch_list[:x]
    out_constr_list = out_constr_list[:x]
    in_constr_list = in_constr_list[:x]
    out_urb_list = out_urb_list[:x]
    out_forest_list = out_forest_list[:x]
    
    imagedict = {"in_arch":in_arch_list,
                 "out_constr":out_constr_list,
                 "in_constr":in_constr_list,
                 "out_urb":out_urb_list,
                 "out_forest":out_forest_list
                 }
    
    return imagedict

def getCalbirationData(dataFolder,x):
    """
    Get calibration data
    """
    imagedict = getxImagesPerClass(dataFolder,x)
    calibData = []
    for value in imagedict.values():
        calibData.append(value)
    calibData = [item for imageList in calibData for item in imageList]#flatten list
    return calibData

def getONNXList(path):
    filenames = next(walk(path), (None, None, []))[2]
    filenames = [path + "/" + filename for filename in filenames]
    return filenames


def compileHARamdHEF(onnxList):
    start_time = time.time()
    datafolder = ptf.dataBaseFolder
    input_folder = ptf.Dataset5Patch
    chosen_hw_arch = "hailo8l"

    for modelPath in onnxList:
        har_path = ptf.har16Bit
        hef_path = ptf.har16Bit
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
                model_name = model_name.split(
                    '_')[1] + "_" + model_name.split('_')[2]
                model_name = model_name.split('_')[1]
            else:
                model_name = model_name.split(
                    '_')[0] + "_" + model_name.split('_')[1]

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
        images_list = sample(images_list,1024)
        # images_list = images_list[0:2048]
        # images_list = getCalbirationData(input_folder,13)
        calib_dataset = np.zeros((len(images_list),  x_y_pixel, x_y_pixel, 3))
        if os.path.exists(f"Data/calibData{model_name}.npy"):
            calib_dataset = np.load(f"Data/calibData{model_name}.npy")
        else:
            print("Prepare Calib Dataset")
            for idx, img_name in tqdm(enumerate(sorted(images_list)),desc = "Calib Dataset"):
                img = Image.open(os.path.join(input_folder, img_name))
                img_preproc = preprocess(img)
                img_transposed = np.transpose(
                    img_preproc.numpy(), (1, 2, 0))  # change dim arangment for Hailo
                calib_dataset[idx, :, :, :] = img_transposed
            np.save(f"Data/calibData{model_name}.npy",calib_dataset)
        # Second, we will load our parsed HAR from the Parsing Tutorial

        hailo_model_har_name = f"{model_name}_hailo_model.har"
        hailo_model_har_path = har_path / hailo_model_har_name
        assert os.path.isfile(
            hailo_model_har_path), "Please provide valid path for HAR file"
        print(f"Model from {hailo_model_har_path}")
        runner = ClientRunner(
            har=str(hailo_model_har_path), hw_arch=chosen_hw_arch)

        # Call Optimize to perform the optimization process
        runner.optimize_full_precision(calib_dataset)
        
        
            
        convScript = createLayerScripts(layer_to_convert[model_name])

        # Batch size is 8 by default
        alls_lines = ['quantization_param(input_layer1, precision_mode=a16_w16)\n',
                    'quantization_param({ew_add*}, precision_mode=a16_w16)\n',
                    'quantization_param({avgpool*}, precision_mode=a16_w16)\n',
                    'quantization_param({matmul*}, precision_mode=a16_w16)\n',
                    #'quantization_param({conv*}, precision_mode=a16_w16)\n',
                    #'quantization_param({format_conversion*}, precision_mode=a16_w16)\n',
                    "model_optimization_flavor(optimization_level=2)\n",
                    "model_optimization_config(calibration, batch_size=16, calibset_size=1024)\n",
                    ]  # From tutorial

        alls_lines.extend(convScript)

        runner.load_model_script("".join(alls_lines))

        runner.optimize(calib_dataset)
            
        # Save the result state to a Quantized HAR file
        quantized_model_har_path = str(
            har_path / f"{model_name}_quantized_model_16Bit.har")
        runner.save_har(quantized_model_har_path)
        print(f"saved model at {quantized_model_har_path}")
        quantized_model_har_path = str(
            har_path / f"{model_name}_quantized_model_16Bit.har")
        print(f"Model used:{model_name}_quantized_model_16Bit.har")
        runner = ClientRunner(har=quantized_model_har_path,
                              hw_arch=chosen_hw_arch)
        # By default it uses the hw_arch that is saved on the HAR. It is not recommended to change the hw_arch after Optimization.

        hef = runner.compile()

        file_name = str(hef_path / f"{model_name}.hef")
        with open(file_name, "wb") as f:
            f.write(hef)

        har_path = ptf.HarPath / f"{model_name}_compiled_model.har"
        runner.save_har(har_path)
    print("Total Compilation Time:")
    print(f"{time.time() - start_time:.3} seconds")


if __name__ == "__main__":
    onnxFiles_path = getONNXList("models/modified")
    TinyFiles_path = [path for path in onnxFiles_path if "Tiny" in path]
    [onnxFiles_path.remove(path) for path in TinyFiles_path]
    
    print(f"Compile ONNX models: {onnxFiles_path}")
    compileHARamdHEF(onnxFiles_path)
