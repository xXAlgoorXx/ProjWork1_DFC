import os
from pathlib import Path

"""
Manages the location of folders
"""

# Data folders
dataBaseFolder = Path("../Data")
modelsBasefolder = Path("models")
HexagonDataFolder = dataBaseFolder / "02_data/hexagon_images"
Dataset5Patch = HexagonDataFolder / "candolle_5patch"
Dataset5Patch224px = HexagonDataFolder / "candolle_5patches_224px"
DatasetPanorama = HexagonDataFolder / "candolle_panorama"
evalBaseFolder = Path("Data")
evaluationFolder = evalBaseFolder / "Evaluation"
evaluationFolder2 = evalBaseFolder / "Evaluation2"
evaluationFolder5Patchcombined = evaluationFolder / "5Patchcombined"
evaluationFolder5Patch = evaluationFolder / "5Patch"
evaluationFolderPanorama = evaluationFolder / "Panorama"
onnxFolder = modelsBasefolder / "baseAndSimple"
compiledonnx_path = modelsBasefolder / "compiledOnnx"
modifiedonnx_path = modelsBasefolder / "modified"
EvaluationQuantized_path = dataBaseFolder / "Evaluation_Quantized"
har16Bit = modelsBasefolder / "Har16Bit"
tinyClipModels = Path("tinyClipModels")
HarPath = Path("models/Harfiles")
Hefpath = Path("models/Heffiles")
gemmpath= Path("models/RestOfGraph")
textEmbpath = Path("models/textEmbeddings")

folders  = [dataBaseFolder,HexagonDataFolder,evaluationFolder,evaluationFolder5Patch,evaluationFolderPanorama,tinyClipModels,HarPath,onnxFolder,Hefpath,gemmpath]

for folder in folders:
    if not folder.is_dir():
        raise Exception(f"Folder {folder} does not exist")