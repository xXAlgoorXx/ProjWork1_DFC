from hailo_sdk_client import ClientRunner
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, PILToTensor
import os
from PIL import Image
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class CompilerHelper:

    def __init__(self, chosen_hw_arch, har_path, input_folder, calibData_folder, inputResolution, upperOnnx_path, lowerOnnx_path, hef_path) -> None:
        self.lowerOnnx_path = lowerOnnx_path
        self.upperOnnx_path = upperOnnx_path
        self.chosen_hw_arch = chosen_hw_arch
        self.har_path = har_path
        self.input_folder = input_folder
        self.calibData_folder = calibData_folder
        self.inputResolution = inputResolution
        self.lowerOnnx_path = lowerOnnx_path
        self.upperOnnx_path = upperOnnx_path
        self.hef_path = hef_path

    def compileOnnxtoHar(self):
        print("Compiling onnx to har")
        self.harLower_path = self._compile(self.lowerOnnx_path, "lower")
        self.harUpper_path = self._compile(self.upperOnnx_path, "upper")

    def optimizeHar(self):
        calib_dataset_upper, calib_dataset_lower = self.prepareCalibData()
        assert os.path.isfile(
            self.harLower_path), "Please provide valid path for HAR file"
        assert os.path.isfile(
            self.harUpper_path), "Please provide valid path for HAR file"
        self.upperRunner = ClientRunner(
            har=str(self.harUpper_path), hw_arch=self.chosen_hw_arch)
        self.lowerRunner = ClientRunner(
            har=str(self.harLower_path), hw_arch=self.chosen_hw_arch)

        # Call Optimize to perform the optimization process
        self.upperRunner.optimize(calib_dataset_upper)
        self.lowerRunner.optimize(calib_dataset_lower)

        # Save the result state to a Quantized HAR file
        model_name = "upper"
        quantized_upper_har_path = str(
            self.har_path / f"{model_name}_quantized_model.har")
        model_name = "lower"
        quantized_lower_har_path = str(
            self.har_path / f"{model_name}_quantized_model.har")

        self.upperRunner.save_har(quantized_upper_har_path)
        self.lowerRunner.save_har(quantized_lower_har_path)
        print(
            f"saved model at {quantized_upper_har_path} and {quantized_lower_har_path}")

    def compilerHEF(self):
        hef_upper = self.upperRunner.compile()
        hef_lower = self.lowerRunner.compile()

        self._writeHef(hef_upper, self.hef_path, "upper")
        self._writeHef(hef_lower, self.hef_path, "lower")

    def _compile(self, model_path, model_name):
        runner = ClientRunner(hw_arch=self.chosen_hw_arch)

        print(f"Take model from {model_path}")
        hn, npz = runner.translate_onnx_model(
            model_path,
            model_name
        )

        hailo_model_har_path = self.har_path / f"{model_name}_hailo_model.har"
        runner.save_har(str(hailo_model_har_path))
        return hailo_model_har_path

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def _transform(self, n_px):
        """
        n_px: input resolution of the network
        """
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),  # for channels (R,G,B)
        ])

    def _writeHef(self, hef, hef_path, model_name):
        file_name = str(hef_path / f"{model_name}.hef")
        with open(file_name, "wb") as f:
            f.write(hef)

    def prepareCalibData(self):
        print("Prepare calibration dataset")
        preprocess = self._transform(self.inputResolution)
        images_list = [img_name for img_name in os.listdir(
            self.input_folder) if os.path.splitext(img_name)[1] == ".jpg"]
        images_list = images_list[0:1024]
        calib_dataset_upper = np.zeros(
            (len(images_list), self.inputResolution, self.inputResolution, 3))
        calib_dataset_lower = np.zeros(
            (len(images_list), 40, 64, 1))
        upperModel = ort.InferenceSession(self.upperOnnx_path)

        for idx, img_name in tqdm(enumerate(sorted(images_list)), desc="Calib data", leave=True):
            img = Image.open(os.path.join(self.input_folder, img_name))
            img_preproc = preprocess(img).numpy()
            # img_transposed = np.transpose(img_preproc.numpy(), (1, 2, 0))
            # input_data = (img_transposed * 255).astype(np.uint8)

            # Get inputnode from graph
            img_forModel = img_preproc[np.newaxis, :]
            output_upper = upperModel.run(None, {'onnx::Cast_0': img_forModel})
            output_upper = np.array(output_upper).squeeze(
                axis=0).squeeze(axis=0)
            output_upper = np.reshape(output_upper,  (40, 64, 1))
            calib_dataset_lower[idx, :, :, :] = output_upper
            img_transposed = np.transpose(img_preproc, (1, 2, 0))
            calib_dataset_upper[idx, :, :, :] = img_transposed

        model_name = "upper"
        np.save(self.calibData_folder /
                f"calib_set_{model_name}.npy", calib_dataset_upper)

        model_name = "lower"
        np.save(self.calibData_folder /
                f"calib_set_{model_name}.npy", calib_dataset_lower)

        return calib_dataset_upper, calib_dataset_lower
