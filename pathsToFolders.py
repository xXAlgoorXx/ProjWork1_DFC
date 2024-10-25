import os
from pathlib import Path

dataBaseFolder = Path("../Data")
HexagonDataFolder = dataBaseFolder / "02_data/hexagon_images"
evaluationFolder = dataBaseFolder / "Evaluation"
tinyClipModels = Path("tinyClipModels")
Dataset5Patch224px = HexagonDataFolder / "candolle_5patches_224px"

folders  = [dataBaseFolder,HexagonDataFolder,evaluationFolder,tinyClipModels]

for fodler in folders:
    if not fodler.is_dir():
        raise Exception(f"Folder {fodler} does not exist")