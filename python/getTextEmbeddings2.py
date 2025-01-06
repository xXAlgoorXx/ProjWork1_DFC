import folderManagment.pathsToFolders as ptf  # Controlls all paths
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from pathlib import Path
import clip
import open_clip
from fvcore.nn import FlopCountAnalysis

import torch
import torchvision
import torchvision.transforms as T
from torcheval.metrics import Throughput

# Own modules
sys.path.append("/home/lukasschoepf/Documents/ProjWork1_DFC")


def saveAsJson(dict, path, name):
    # Serializing json
    json_object = json.dumps(dict, indent=4)

    if isinstance(path, Path):
        path = str(path)

    # Writing to json
    with open(f"{path}/{name}.json", "w") as outfile:
        outfile.write(json_object)


def getTextembeddings(model, text, tokens):
    # embeddedTextDict ={}
    # for disc,token in zip(text,tokens):
    #     text_features = model.encode_text(token)
    #     embeddedTextDict[disc] = text_features
    # return embeddedTextDict
    embeddedTextDict = {}
    model.eval()
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    embeddedTextDict["Discription"] = text
    embeddedTextDict["embeddings"] = text_features.tolist()
    return embeddedTextDict


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use5Scentens = True
    evalFodler = ptf.evaluationFolderPanorama
    datafolder = ptf.DatasetPanorama
    # From Lia
    # define text prompts
    names2 = ["architectural", "office", "residential", "school", "manufacturing",
              "cellar", "laboratory", "construction site", "mining", "tunnel"]
    names3 = ["construction site", "town", "city",
              "country side", "alley", "parking lot", "forest"]

    # 5 sentences to use as text prompt
    prompts = [
        "a photo of a {}.",
        "a picture of a {}.",
        "an image of a {}.",
        "a {} scene.",
        "a picture showing a scene of a {}."
    ]
    sent2 = [prompt.format(label) for label in names2 for prompt in prompts]
    sent3 = [prompt.format(label) for label in names3 for prompt in prompts]

    print("Clip Models:", clip.available_models())
    # Evaluate every Resnet model
    resnetModels = []
    for clipmodel in clip.available_models():
        if "RN" in clipmodel:
            resnetModels.append(clipmodel)

    print("Open Clip Models:", open_clip.list_models())
    for clipmodel in open_clip.list_models():
        if "ResNet" in clipmodel and "Tiny" in clipmodel:
            resnetModels.append(clipmodel)

    for modelname in tqdm(resnetModels, position=0, desc="Models"):
        for mode in [True, False]:
            use5Scentens = mode
            try:
                if os.path.exists(ptf.tinyClipModels / f"modelname"):
                    print(f"Model {modelname} available")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    modelname, device=device, pretrained=str(ptf.tinyClipModels / f"{modelname}-LAION400M.pt"))

                openClipTokenizer = open_clip.get_tokenizer(modelname)
                text1 = openClipTokenizer(["indoor", "outdoor"]).to(device)
                if use5Scentens:
                    text2 = openClipTokenizer(sent2).to(device)
                    text3 = openClipTokenizer(sent3).to(device)

                else:
                    text2 = openClipTokenizer(names2).to(device)
                    text3 = openClipTokenizer(names3).to(device)
            except:
                model, preprocess = clip.load(modelname, device=device)

                text1 = clip.tokenize(["indoor", "outdoor"]).to(device)
                if use5Scentens:
                    text2 = clip.tokenize(sent2).to(device)
                    text3 = clip.tokenize(sent3).to(device)
                else:
                    text2 = clip.tokenize(names2).to(device)
                    text3 = clip.tokenize(names3).to(device)

            model.eval()

            dictlvl1 = getTextembeddings(model, ["indoor", "outdoor"], text1)
            dictlvl2 = getTextembeddings(model, names2, text2)
            dictlvl3 = getTextembeddings(model, names3, text3)

            textembeddings = {"1": dictlvl1, "2": dictlvl2,
                              "3": dictlvl3, "Use5Scentens": use5Scentens}
            if use5Scentens == True:
                saveAsJson(textembeddings, ptf.textEmbpath,
                           f"textEmbeddings_{modelname}_5S")
            else:
                saveAsJson(textembeddings, ptf.textEmbpath,
                           f"textEmbeddings_{modelname}")


if __name__ == "__main__":
    main()
