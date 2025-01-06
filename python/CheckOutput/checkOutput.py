from checkOutput_utils import get_pred_CPU, get_pred_HEF, getHighestProb, getMaxIndexPerClass,get_trueClass
import pandas as pd
import numpy as np
import os
import sys
import clip

# Own modules
cwd = os.getcwd()
newPath = cwd + "/python"
print(newPath)
sys.path.append(newPath)
import folderManagment.pathsToFolders as ptf  # Controlls all paths
import torch
import open_clip
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

inLabels = ["architectural", "office", "residential", "school", "manufacturing",
              "cellar", "laboratory", "construction site", "mining", "tunnel"]

outlabels = ["construction site", "town", "city",
              "country side", "alley", "parking lot", "forest"]


names2 = ["architectural", "office", "residential", "school", "manufacturing",
            "cellar", "laboratory", "construction site", "mining", "tunnel"]
names3 = ["construction site", "town", "city",
            "country side", "alley", "parking lot", "forest"]

def main():
    
    model_name = "RN101"
    har = "models/Harfiles/CLIP_RN101_compiled_model.har"
    input_folder = str(ptf.Dataset5Patch)
    try:
        openClipTokenizer = open_clip.get_tokenizer(model_name)
        model, preprocess_train, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=f"tinyClipModels/{model_name}-LAION400M.pt"
        )
        # tokenize text prompts
        text1 = openClipTokenizer(["indoor", "outdoor"]).to(device)
        text2 = openClipTokenizer(names2).to(device)
        text3 = openClipTokenizer(names3).to(device)
    except:
        model, preprocess = clip.load(model_name, device=device)
        # tokenize text prompts
        text1 = clip.tokenize(["indoor", "outdoor"]).to(device)
        text2 = clip.tokenize(names2).to(device)
        text3 = clip.tokenize(names3).to(device)
    
    df_path_CPU = f"temp/checkOutput_CPU_ {model_name}.df"
    # df_path_CPU = "temp/checkOutput_CPU_Tiny19.df"
    if os.path.exists(df_path_CPU):
        # df_cpu = pd.read_csv(csv_path_CPU)
        df_cpu = pd.read_pickle(df_path_CPU)
    else:
        df_cpu = get_pred_CPU(input_folder=input_folder,
                            text1=text1,
                            text2=text2,
                            text3=text3,
                            preprocess=preprocess,
                            model=model
                            )
        
        
        # df_cpu.to_csv(csv_path_CPU)
        df_cpu.to_pickle(df_path_CPU)
        
    df_path_HEF = f"temp/checkOutput_HEF_{model_name}.df"
    # df_path_HEF = "temp/checkOutput_HEF_Tiny19_compiled.df"
    if os.path.exists(df_path_HEF):
        # df_hef = pd.read_csv(df_path_HEF)
        df_hef = pd.read_pickle(df_path_HEF)
    else:
        df_hef = get_pred_HEF(input_folder = input_folder,
                            textEmb_json = "models/textEmbeddings/textEmbeddings_RN101.json",
                            preprocess = preprocess,
                            postProcess_onnx = "models/RestOfGraphONNX/RestOf_CLIP_RN101_simplified.onnx",
                            har = har,
                            )
        # df_hef.to_csv(df_path_HEF)
        df_hef.to_pickle(df_path_HEF)
    
    df_cpu =get_trueClass(df_cpu)
    df_hef =get_trueClass(df_hef)
    
    df_cpu["HighestPorb"] = df_cpu.apply(getMaxIndexPerClass, axis=1)
    
    df_hef["HighestPorb"] = df_hef.apply(getMaxIndexPerClass, axis=1)
    
    df_check = df_cpu["HighestPorb"].compare(df_hef["HighestPorb"]) # get missclassified Rows
    print(df_check)
    print(len(df_check))
    
    # ===========
    #    Plots
    # ===========
    
    plotstocheck = 3
    randomNumbers = np.random.randint(0,4000,size = plotstocheck)
    numbers = [1000, 1001, 1002]
    # Update plot settings
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.size": 10
    })
    
    # inLabels
    # outlabels
    labels = [label.replace(" ","\n") for label in  outlabels]
    takeFrom = 'Out_probs' # Out_probs In_probs
    
    # Original
    f, axes = plt.subplots(plotstocheck, 1, figsize= (6,6))
    
    axes[0].set_title('Original')
    for i in range(plotstocheck):
        axes[i].bar(labels,df_cpu.iloc[numbers[i]][takeFrom])
        axes[i].set_ylim([0,1])
    
    f.tight_layout()
    
    plt.savefig("temp/"+ "outprobs_Original", dpi = 600)
    plt.show()
    
    # Compiled
    f, axes = plt.subplots(plotstocheck, 1, figsize= (6,6))
    axes[0].set_title('HEF Compiled')
    for i in range(plotstocheck):
        axes[i].bar(labels,df_hef.iloc[numbers[i]][takeFrom])
        axes[i].set_ylim([0,1])
    
    f.tight_layout()
    
    plt.savefig("temp/" + "outprobs_compiled", dpi = 600)
    plt.show()
    
    # Quantized
    df_path_HEF = "temp/checkOutput_HEF_Tiny19.df"
    if os.path.exists(df_path_HEF):
        # df_hef = pd.read_csv(df_path_HEF)
        df_hef = pd.read_pickle(df_path_HEF)
    plotstocheck = 6
    f, axes = plt.subplots(plotstocheck, 1,figsize= (6,6))
    axes[0].set_title('HEF Quantized')
    for i in range(plotstocheck):
        axes[i].bar(labels,df_hef.iloc[i + 1000][takeFrom])
        # axes[i].bar(labels,df_hef.iloc[numbers[i]][takeFrom])
        axes[i].set_ylim([0,1])
    
    f.tight_layout()
    
    plt.savefig("temp/" + "outprobs_quantized",dpi = 600)
    plt.show()

if __name__ == "__main__":
    main()
