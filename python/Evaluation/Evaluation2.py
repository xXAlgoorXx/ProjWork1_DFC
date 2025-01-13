
from Evaluation_utils import get_max_class_with_threshold, get_pred2, get_throughput2, get_trueClass, find_majority_element, printAndSaveHeatmap, get_modelnames, get_throughput_image2,printAndSaveClassReport,printClassificationReport
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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

cwd = os.getcwd()
newPath = cwd + "/python"
print(newPath)
sys.path.append(newPath)
import folderManagment.pathsToFolders as ptf  # Controlls all paths

# Own modules
sys.path.append("/home/lukasschoepf/Documents/ProjWork1_DFC")

"""
Uses different text embeddings
"""

def main(evalFodler, datafolder, use5Scentens=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # define text prompts
    indoor_labels = ["furniture", "carpet", "ceiling", "appliances", "lighting"]
    outdoor_labels = ["sky", "grass", "trees", "horizon", "clouds"]
    
    construction_in_labels = ["scaffolding", "wiring", "drywall", "plumbing", "fixtures"]
    architectural_labels = ["pillars", "facade", "cornice", "vault", "domes"]
    construction_out_labels = ["excavator", "blueprints", "rebar", "hardhat", "bulldozer"]
    urban_labels = ["billboards", "sidewalks", "crosswalk", "subway", "alleys"]
    forest_labels = ["underbrush", "wildflowers", "mushrooms", "forrest", "streams"]

    # 5 sentences to use as text prompt
    prompts = [
        "a photo of a {}.",
        "a picture of a {}.",
        "an image of a {}.",
        "a {} scene.",
        "a picture showing a scene of a {}."
    ]
    
    indoor_sent = [prompt.format(label) for label in indoor_labels for prompt in prompts]
    outdoor_sent = [prompt.format(label) for label in outdoor_labels for prompt in prompts]
    
    construction_in_sent = [prompt.format(label) for label in construction_in_labels for prompt in prompts]
    architectural_sent  = [prompt.format(label) for label in architectural_labels for prompt in prompts]
    construction_out_sent  = [prompt.format(label) for label in construction_out_labels for prompt in prompts]
    urban_sent = [prompt.format(label) for label in urban_labels for prompt in prompts]
    forest_sent  = [prompt.format(label) for label in forest_labels for prompt in prompts]

    print("Clip Models:", clip.available_models())

    resnetModels = get_modelnames()
    accuracy_models = []
    balanced_accuracy_models = []
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
                if use5Scentens:
                    indoor_text = openClipTokenizer(indoor_sent).to(device)
                    outdoor_text = openClipTokenizer(outdoor_sent).to(device)
                    construction_in_text = openClipTokenizer(construction_in_sent).to(device)
                    architectural_text  = openClipTokenizer(architectural_sent).to(device)
                    construction_out_text  = openClipTokenizer(construction_out_sent).to(device)
                    urban_text = openClipTokenizer(urban_sent).to(device)
                    forrest_text  = openClipTokenizer(forest_sent).to(device)

                else:
                    indoor_text = openClipTokenizer(indoor_labels).to(device)
                    outdoor_text = openClipTokenizer(outdoor_labels).to(device)
                    construction_in_text = openClipTokenizer(construction_in_labels).to(device)
                    architectural_text  = openClipTokenizer(architectural_labels).to(device)
                    construction_out_text  = openClipTokenizer(construction_out_labels).to(device)
                    urban_text = openClipTokenizer(urban_labels).to(device)
                    forrest_text  = openClipTokenizer(forest_labels).to(device)
            except:
                model, preprocess = clip.load(modelname, device=device)
                # tokenize text prompts
                if use5Scentens:
                    indoor_text = clip.tokenize(indoor_sent).to(device)
                    outdoor_text = clip.tokenize(outdoor_sent).to(device)
                    construction_in_text = clip.tokenize(construction_in_sent).to(device)
                    architectural_text  = clip.tokenize(architectural_sent).to(device)
                    construction_out_text  = clip.tokenize(construction_out_sent).to(device)
                    urban_text = clip.tokenize(urban_sent).to(device)
                    forrest_text  = clip.tokenize(forest_sent).to(device)
                else:
                    indoor_text = clip.tokenize(indoor_labels).to(device)
                    outdoor_text = clip.tokenize(outdoor_labels).to(device)
                    construction_in_text = clip.tokenize(construction_in_labels).to(device)
                    architectural_text  = clip.tokenize(architectural_labels).to(device)
                    construction_out_text  = clip.tokenize(construction_out_labels).to(device)
                    urban_text = clip.tokenize(urban_labels).to(device)
                    forrest_text  = clip.tokenize(forest_labels).to(device)
                    
            df_pred = get_pred2(datafolder,
                                indoor_text,
                                outdoor_text,
                                construction_in_text,
                                architectural_text,
                                construction_out_text,
                                urban_text,
                                forrest_text,
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

            # accuracy = accuracy_score(IO_true, IO_pred)
            # balanced_accuracy = balanced_accuracy_score(IO_true, IO_pred)
            accuracy = accuracy_score(y_test_s, majority_pred)
            balanced_accuracy = balanced_accuracy_score(y_test_s, majority_pred)
        else:
            accuracy = accuracy_score(y_test, y_pred)
        accuracy_models.append(accuracy)
        balanced_accuracy_models.append(balanced_accuracy)
        print(f'Accuracy: {accuracy:.3f}')

        printAndSaveHeatmap(df, modelname, evalFodler, use5Scentens)
        
        # Classification Report (Bar plot)
        classificationReport = printClassificationReport(df, modelname)
        del classificationReport['accuracy'] # Else visualisation doesnt work
        df_classReport = pd.DataFrame.from_dict(classificationReport,orient='index')
        printAndSaveClassReport(classificationReport,modelname,evalFodler)

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
    balanced_accuracy_models = ['%.3f' % elem for elem in balanced_accuracy_models]
    df_perf_acc["Balanced accuracy"] = balanced_accuracy_models
    # Throughput evaluation
    throughput_model_mean = []
    throughput_model_std = []
    throughput_model_mean_image = []
    throughput_model_std_image = []
    
    for i, modelname in enumerate(tqdm(resnetModels, position=0, desc="Params")):
        try:
            if os.path.exists(ptf.tinyClipModels / f"modelname"):
                print(f"Model {modelname} available")
            model, _, preprocess = open_clip.create_model_and_transforms(
                modelname, device=device, pretrained=str(ptf.tinyClipModels / f"{modelname}-LAION400M.pt"))

            # tokenize text prompts
            openClipTokenizer = open_clip.get_tokenizer(modelname)
            if use5Scentens:
                indoor_text = openClipTokenizer(indoor_sent).to(device)
                outdoor_text = openClipTokenizer(outdoor_sent).to(device)
                construction_in_text = openClipTokenizer(construction_in_sent).to(device)
                architectural_text  = openClipTokenizer(architectural_sent).to(device)
                construction_out_text  = openClipTokenizer(construction_out_sent).to(device)
                urban_text = openClipTokenizer(urban_sent).to(device)
                forrest_text  = openClipTokenizer(forest_sent).to(device)

            else:
                indoor_text = openClipTokenizer(indoor_labels).to(device)
                outdoor_text = openClipTokenizer(outdoor_labels).to(device)
                construction_in_text = openClipTokenizer(construction_in_labels).to(device)
                architectural_text  = openClipTokenizer(architectural_labels).to(device)
                construction_out_text  = openClipTokenizer(construction_out_labels).to(device)
                urban_text = openClipTokenizer(urban_labels).to(device)
                forrest_text  = openClipTokenizer(forest_labels).to(device)
        except:
            model, preprocess = clip.load(modelname, device=device)
            # tokenize text prompts
            if use5Scentens:
                indoor_text = clip.tokenize(indoor_sent).to(device)
                outdoor_text = clip.tokenize(outdoor_sent).to(device)
                construction_in_text = clip.tokenize(construction_in_sent).to(device)
                architectural_text  = clip.tokenize(architectural_sent).to(device)
                construction_out_text  = clip.tokenize(construction_out_sent).to(device)
                urban_text = clip.tokenize(urban_sent).to(device)
                forrest_text  = clip.tokenize(forest_sent).to(device)
            else:
                indoor_text = clip.tokenize(indoor_labels).to(device)
                outdoor_text = clip.tokenize(outdoor_labels).to(device)
                construction_in_text = clip.tokenize(construction_in_labels).to(device)
                architectural_text  = clip.tokenize(architectural_labels).to(device)
                construction_out_text  = clip.tokenize(construction_out_labels).to(device)
                urban_text = clip.tokenize(urban_labels).to(device)
                forrest_text  = clip.tokenize(forest_labels).to(device)
        throughput_mean,throughput_std = get_throughput2(
            datafolder,
            indoor_text,
            outdoor_text,
            construction_in_text,
            architectural_text,
            construction_out_text,
            urban_text,
            forrest_text,
            preprocess, model, use5Scentens)
        throughput_model_mean.append(throughput_mean)
        throughput_model_std.append(throughput_std)
        print(f"\nThrouputs: {throughput_model_mean}")
        
        throughput_mean_image,throughput_std_image = get_throughput_image2(
            datafolder,
            indoor_text,
            outdoor_text,
            construction_in_text,
            architectural_text,
            construction_out_text,
            urban_text,
            forrest_text,
            preprocess, model, use5Scentens)
        throughput_model_mean_image.append(throughput_mean_image)
        throughput_model_std_image.append(throughput_std_image)
        print(f"\nThrouputs Image: {throughput_model_mean_image}")

    throughput_model_mean = ['%.2f' % elem for elem in throughput_model_mean]
    throughput_model_mean_image = ['%.2f' % elem for elem in throughput_model_mean_image]
    
    df_perf_acc["Throughput (it/s)"] = throughput_model_mean
    df_perf_acc["Throughput Image (it/s)"] = throughput_model_mean_image
    df_perf_acc["Throughput std"] = throughput_model_std
    df_perf_acc["Throughput Image std"] = throughput_model_std_image
    
    df_perf_acc.to_csv(csv_path_perforemance, index=False)


if __name__ == "__main__":
    use5Scentens = False
    evalFodler = ptf.evaluationFolder2
    datafolder = ptf.Dataset5Patch
    main(evalFodler, datafolder, use5Scentens)
