from Evaluation_utils import get_max_class_with_threshold, get_pred, get_throughput, get_trueClass, find_majority_element, printAndSaveHeatmap, get_modelnames
import folderManagment.pathsToFolders as ptf  # Controlls all paths
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torcheval.metrics import Throughput
import torchvision.transforms as T
import torchvision
import torch
from fvcore.nn import FlopCountAnalysis
import open_clip
import clip
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Own modules
sys.path.append("/home/lukasschoepf/Documents/ProjWork1_DFC")


def main(evalFodler, datafolder, use5Scentens=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    resnetModels = get_modelnames()
    accuracy_models = []
    for modelname in tqdm(resnetModels, position=0, desc="Models"):

        # Path to csv
        if use5Scentens:
            csv_path_predictions = evalFodler / \
                f'pred_{modelname}_5patches_5scentens.csv'
        else:
            csv_path_predictions = evalFodler / \
                f'pred_{modelname}_5patches.csv'

        # check if csv already exists
        if os.path.exists(csv_path_predictions):
            df_5patch = get_trueClass(pd.read_csv(csv_path_predictions))
        else:
            try:
                if os.path.exists(ptf.tinyClipModels / f"modelname"):
                    print(f"Model {modelname} available")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    modelname, device=device, pretrained=str(ptf.tinyClipModels / f"{modelname}-LAION400M.pt"))

                # tokenize text prompts
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
                # tokenize text prompts
                text1 = clip.tokenize(["indoor", "outdoor"]).to(device)
                if use5Scentens:
                    text2 = clip.tokenize(sent2).to(device)
                    text3 = clip.tokenize(sent3).to(device)
                else:
                    text2 = clip.tokenize(names2).to(device)
                    text3 = clip.tokenize(names3).to(device)
            df_pred = get_pred(datafolder, text1, text2, text3,
                               preprocess, model, use5Scentens)
            df_pred.to_csv(csv_path_predictions, index=False)
            df_5patch = get_trueClass(pd.read_csv(csv_path_predictions))
        df = df_5patch.copy()
        df['y_predIO'] = df.apply(
            get_max_class_with_threshold, axis=1, threshold=0.8)

        # set the outdoor classes to 0 when the image was classified as indoor
        # set the indoor classes to 0 when the image was classified as outdoor
        df.loc[df['y_predIO'] == 'In', [
            'Out_Constr', 'Out_Urban', 'Forest']] = 0
        df.loc[df['y_predIO'] == 'Out', ['In_Arch', 'In_Constr']] = 0

        # create the new column y_predIO
        columns = ['In_Arch', 'In_Constr', 'Out_Constr', 'Out_Urban', 'Forest']
        df['y_pred'] = df[columns].idxmax(axis=1)

        # evaluate performance of model
        y_test = df['ClassTrue']
        y_pred = df['y_pred']

        # majority counts
        y_test_s = []
        majority_pred = []

        if "5patch" in str(datafolder):
            # iterate through the input array in chunks of 5
            for i in range(0, len(y_test), 5):

                patches = y_test[i:i+5]
                majority_element = find_majority_element(patches)
                y_test_s.append(majority_element)

                patches = y_pred[i:i+5]
                majority_element = find_majority_element(patches)
                majority_pred.append(majority_element)

            # conpute indoor/outdoor classification accuracy score
            replacements = {
                "In_Arch": "In",
                "In_Constr": "In",
                "Forest": "Out",
                "Out_Constr": "Out",
                "Out_Urban": "Out"
            }

            IO_pred = [replacements.get(item, item) for item in majority_pred]
            IO_true = [replacements.get(item, item) for item in y_test_s]

            accuracy = accuracy_score(IO_true, IO_pred)
        else:
            accuracy = accuracy_score(y_test, y_pred)
        accuracy_models.append(accuracy)
        print(f'Accuracy: {accuracy:.3f}')

        printAndSaveHeatmap(df, modelname, evalFodler, use5Scentens)

    # Parameter Evaluation
    df_perf_acc = pd.DataFrame(
        columns=["Modelname", "Params Vis", "Params Text"])

    csvName = "modelPerformance"
    if "5patch" in str(datafolder):
        csvName += "_5patches"
    if use5Scentens:
        csvName += "_5scentens"
    csvName += ".csv"

    csv_path_perforemance = evalFodler / csvName
    for i, modelname in enumerate(tqdm(resnetModels, position=0, desc="Params")):
        try:
            if os.path.exists(ptf.tinyClipModels / modelname):
                print(f"Model {modelname} available")
            model, _, preprocess = open_clip.create_model_and_transforms(
                modelname, device=device, pretrained=str(ptf.tinyClipModels / f"{modelname}-LAION400M.pt"))
        except:
            model, preprocess = clip.load(modelname, device=device)

        # Get prameter count
        modelsParamCountVis = (sum(p.numel()
                               for p in model.visual.parameters()))
        modelsParamCounttext = (sum(p.numel()
                                for p in model.transformer.parameters()))

        # Append new row to the DataFrame
        df_perf_acc = df_perf_acc._append({
            "Modelname": modelname,
            "Params Vis": modelsParamCountVis,
            "Params Text": modelsParamCounttext
        }, ignore_index=True)
    accuracy_models = ['%.3f' % elem for elem in accuracy_models]
    df_perf_acc["Accuracy"] = accuracy_models

    # Throughput evaluation
    throughput_model = []
    for i, modelname in enumerate(tqdm(resnetModels, position=0, desc="Params")):
        try:
            if os.path.exists(ptf.tinyClipModels / f"modelname"):
                print(f"Model {modelname} available")
            model, _, preprocess = open_clip.create_model_and_transforms(
                modelname, device=device, pretrained=str(ptf.tinyClipModels / f"{modelname}-LAION400M.pt"))

            # tokenize text prompts
            openClipTokenizer = open_clip.get_tokenizer(modelname)
            text1 = openClipTokenizer(["indoor", "outdoor"]).to(device)
            text2 = openClipTokenizer(names2).to(device)
            text3 = openClipTokenizer(names3).to(device)

        except:
            model, preprocess = clip.load(modelname, device=device)

            # tokenize text prompts
            text1 = clip.tokenize(["indoor", "outdoor"]).to(device)
            text2 = clip.tokenize(names2).to(device)
            text3 = clip.tokenize(names3).to(device)
        throughput = get_throughput(
            datafolder, text1, text2, text3, preprocess, model)
        throughput_model.append(throughput)
        print(throughput_model)
    throughput_model = ['%.2f' % elem for elem in throughput_model]
    df_perf_acc["Throughput (it/s)"] = throughput_model
    df_perf_acc.to_csv(csv_path_perforemance, index=False)


if __name__ == "__main__":
    use5Scentens = False
    evalFodler = ptf.evaluationFolder5Patch
    datafolder = ptf.Dataset5Patch
    main(evalFodler, datafolder, use5Scentens)
