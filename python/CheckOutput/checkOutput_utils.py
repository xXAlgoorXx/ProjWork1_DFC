import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import torch
import time
import pandas as pd
from torcheval.metrics import Throughput
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from torch.nn import Softmax
import matplotlib.pyplot as plt
import seaborn as sns
import clip
import open_clip
import json
from pathlib import Path
from hailo_sdk_client import ClientRunner, InferenceContext
import onnxruntime as ort
from ast import literal_eval

class RestOfGraphOnnx:
    """
    GemmLayer which got cut off
    """

    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)

    def __call__(self, input):
        # input = np.array(list(input.values())[0]).squeeze()
        if input.ndim == 1:
            input = input[np.newaxis, :]
        
        if input.ndim >= 3 and input.shape[0] > 1:
            
            result = []
            for inImage in input:
                res = self.session.run(None, {"/attnpool/Reshape_7_output_0": inImage})
                result.append(res)
                
            result = np.array(result).squeeze((1,2))
        else:
            if input.shape[0] ==1:
                input = input.squeeze(0)
            result = self.session.run(
                None, {"/attnpool/Reshape_7_output_0": input})
            result = np.array(result).squeeze(0)
        return result

    
def loadJson(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        data = json.load(file)

    return data


def getHighestProb(df):
    # pd.DataFrame({'Scene': scene_list,
    #                         'Image': img_list,
    #                         'In_probs': score_in_list,
    #                         'Out_probs': score_out_list,
    #                         'In': in_list,
    #                         'Out': out_list})
    
    df["InOutMax"] = df.idxmax(axis=('In','Out'))
    df["InMax"] = df.idxmax(axis='In_probs')
    df["OutMax"] = df.idxmax(axis='Out_probs')
    
    return df

def getMaxIndexPerClass(row):
    # in_prob = np.array(literal_eval(row['In_probs']))
    # out_probs = np.array(literal_eval(row['Out_probs']))
    # inOut_probs = np.array([literal_eval(row['InOut'])])
    inLabels  = [ 'In_Arch',  'In_Constr']
    outLabels  = [ 'Out_Constr', 'Out_Urban', 'Forest']
    
    inOut_probs = np.array(row['InOut'])
    
    if row['ClassTrue'] in inLabels:   
        probs = np.array(row['In_probs'])
    else:
        probs = np.array(row['Out_probs'])
    
    return [inOut_probs.argmax(),probs.argmax()]


def get_pred_CPU(input_folder, text1, text2, text3, preprocess, model, use5Scentens=False):
    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''
    model.eval()
    # List all files in the input folder
    files = os.listdir(input_folder)

    modelOutput_list = []
    inout_list = []
    score_in_list = []
    score_out_list = []
    scene_list = []
    img_list = []
    # Loop through each file
    for file in tqdm(files, desc="Files", position=1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Read the image
        image_path = os.path.join(input_folder, file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image).detach().numpy()
        ### FIRST DEGREE ###
        text = text1
        probs = evalModel(model, image, text, False)

        score_in_out = probs[0]

        ### SECOND DEGREE (in) ###
        text = text2
        probs = evalModel(model, image, text, use5Scentens)

        score_in_probs = probs[0]

        ### SECOND DEGREE (out) ###
        text = text3
        probs = evalModel(model, image, text, use5Scentens)

        score_out_probs = probs[0]

        modelOutput_list.append(image_features)
        inout_list.append(score_in_out)
        score_in_list.append(score_in_probs)
        score_out_list.append(score_out_probs)

        scene_list.append(int(os.path.basename(
            file).split('.')[0].split('_')[1]))
        img_list.append(int(os.path.basename(
            file).split('.')[0].split('_')[2]))

    df_pred = pd.DataFrame({'Scene': scene_list,
                            'Image': img_list,
                            'ImageEmb':modelOutput_list,
                            'In_probs': score_in_list,
                            'Out_probs': score_out_list,
                            'InOut': inout_list})
    return df_pred


def evalModel(model, image, text, use_5_Scentens=False):
    with torch.no_grad():
        # Encode image and text features separately
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize features for cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity (scaled by 100)
        logits_per_image = 100.0 * (image_features @ text_features.T)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # uncomment when processing 5 sentences
        if use_5_Scentens:
            new_size = probs.shape[1] // 5
            probs = np.array([probs[0, i:i+5].sum()
                             for i in range(0, probs.shape[1], 5)]).reshape(1, new_size)

    return probs


def get_pred_HEF(input_folder, textEmb_json, preprocess, postProcess_onnx,har, use5Scentens=False):
    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''
    chosen_hw_arch = "hailo8l"
    postProcess = RestOfGraphOnnx(postProcess_onnx)
    textEmb = loadJson(textEmb_json)
    # List all files in the input folder
    files = os.listdir(input_folder)
    x_y_pixel = 224
    hefOutput_list = []
    inOut_list = []
    score_in_list = []
    score_out_list = []
    scene_list = []
    img_list = []
    runner = ClientRunner(har=har,
                          hw_arch=chosen_hw_arch)
    # Loop through each file
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        for file in tqdm(files, desc="Files", position=1):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Read the image
            image_path = os.path.join(input_folder, file)
            image = preprocess(Image.open(image_path)).to(device)
            img_transposed = np.transpose(image.numpy(), (1, 2, 0))[np.newaxis,:]
            output_quantized_har = runner.infer(ctx, img_transposed,batch_size=1)
            output_quantized_har = output_quantized_har[:,0,:,:]
            output_Hef = postProcess(output_quantized_har)
            
            ### FIRST DEGREE ###
            text = np.array(textEmb["1"]["embeddings"])
            probs = evalModel_hef(output_Hef, text, use5Scentens)

            score_in_out = probs[0]

            ### SECOND DEGREE (in) ###
            text = np.array(textEmb["2"]["embeddings"])
            probs = evalModel_hef(output_Hef, text, use5Scentens)

            score_in_probs = probs[0]

            ### SECOND DEGREE (out) ###
            text = np.array(textEmb["3"]["embeddings"])
            probs = evalModel_hef(output_Hef, text, use5Scentens)

            score_out_probs = probs[0]

            hefOutput_list.append(output_Hef)
            inOut_list.append(score_in_out)
            score_in_list.append(score_in_probs)
            score_out_list.append(score_out_probs)

            scene_list.append(int(os.path.basename(
                file).split('.')[0].split('_')[1]))
            img_list.append(int(os.path.basename(
                file).split('.')[0].split('_')[2]))

        df_pred = pd.DataFrame({'Scene': scene_list,
                                'Image': img_list,
                                'ImageEmb':hefOutput_list,
                                'In_probs': score_in_list,
                                'Out_probs': score_out_list,
                                'InOut': inOut_list})
    return df_pred


def get_trueClass(df):
    '''
    Function reads and stores the image features and true class in a dataframe
    In: Dataframe with the class probability
    Out: Dataframe with an additional column 'ClassTrue' 
    '''
    df['Image'] = df['Scene'].astype(str) + '_' + df['Image'].astype(str)

    # define second degree classes
    in_arch = [7, 10, 18, 27, 29, 32, 36, 1,
               28, 6, 33, 40, 30, 31, 24]  # [7,18,31]#
    out_constr_res = [8, 16, 22]  # [16]#
    in_constr_res = [9, 13, 39, 12]  # [12]#
    out_urb = [2, 20, 38, 26, 15, 42, 44, 4, 23]  # [15,2,23]#
    out_forest = [17]

    # add second degree classes
    df['ClassTrue'] = np.select(
        [
            df['Scene'].isin(in_arch),
            df['Scene'].isin(out_constr_res),
            df['Scene'].isin(in_constr_res),
            df['Scene'].isin(out_urb),
            df['Scene'].isin(out_forest)
        ],
        ['In_Arch', 'Out_Constr', 'In_Constr', 'Out_Urban', 'Forest'],
        default='Other'
    )

    df.drop(df[df['ClassTrue'] == 'Other'].index, inplace=True)
    df.drop('Scene', axis=1, inplace=True)

    # uncomment for evaluation only on test set
    # df.drop(df[df['Image'].isin(['17_3', '17_4', '17_5', '17_6', '17_7', '17_8', '17_9', '17_10', '17_11',
    #                         '17_12', '17_13', '17_14', '17_15', '17_16', '17_17', '17_18', '17_19',
    #                         '17_20', '17_21', '17_22', '17_23', '17_24'])].index, inplace=True)

    return df


def evalModel_hef(image, text, use_5_Scentens=False):
    with torch.no_grad():
        # Encode image and text features separately
        image_features = torch.Tensor(image).to(dtype=torch.float32)
        text_features = torch.Tensor(text).to(dtype=torch.float32)

        # Normalize features for cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity (scaled by 100)
        logits_per_image = 100.0 * (image_features @ text_features.T)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # uncomment when processing 5 sentences
        if use_5_Scentens:
            new_size = probs.shape[1] // 5
            probs = np.array([probs[0, i:i+5].sum()
                             for i in range(0, probs.shape[1], 5)]).reshape(1, new_size)

    return probs