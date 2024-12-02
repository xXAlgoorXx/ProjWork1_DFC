import os
from pathlib import Path

# Data folders
dataBaseFolder = Path("../Data")
HexagonDataFolder = dataBaseFolder / "02_data/hexagon_images"
Dataset5Patch = HexagonDataFolder / "candolle_5patch"
Dataset5Patch224px = HexagonDataFolder / "candolle_5patches_224px"
DatasetPanorama = HexagonDataFolder / "candolle_panorama"
evaluationFolder = dataBaseFolder / "Evaluation"
evaluationFolder2 = dataBaseFolder / "Evaluation2"
evaluationFolder25Patch = evaluationFolder2 / "5Patch"
evaluationFolder5Patch = evaluationFolder / "5Patch"
evaluationFolderPanorama = evaluationFolder / "Panorama"
onnxFolder = Path("models/baseAndSimple")
tinyClipModels = Path("tinyClipModels")
HarPath = Path("models/Harfiles")
QuantizedPath = Path("models/QuantizedModels")
Hefpath = Path("models/Heffiles")

folders  = [dataBaseFolder,HexagonDataFolder,evaluationFolder,evaluationFolder5Patch,evaluationFolderPanorama,tinyClipModels,HarPath,QuantizedPath,onnxFolder,Hefpath]

for folder in folders:
    if not folder.is_dir():
        raise Exception(f"Folder {folder} does not exist")