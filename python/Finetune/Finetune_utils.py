# https://www.marqo.ai/course/fine-tuning-clip-models

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from pathlib import Path


# Define a custom dataset class
class CandolleDataset(Dataset):
    def __init__(self,preprocess):
        
        self.transform = preprocess

    def loadPath(self,path):
        self.files = os.listdir(path)

    def loadList(self, fileList):
        self.files = fileList

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        image = Image.open(item)
        label = getLabel(item)
        return self.transform(image), label

# Modify the model to include a classifier for subcategories 
class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(num_classes, num_classes)
    
    def embedTextfeatures(self, textTokens):
        self.text_features = self.model.encode_text(textTokens)


    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)




def getLabel(imagePath):
    """
    returns the imagelabel as a string
    """
    imageName = str(Path(imagePath).stem)
    nameSplit = imageName.split("_")
    
    return getClass(int(nameSplit[1]))

def getClass(imageClass):
    """
    Class code in image name
    return: 
    """
    # define second degree classes 
    in_arch = [7,10,18,27,29,32,36,1,28,6,33,40,30,31,24]#[7,18,31]#
    out_constr_res = [8,16,22]#[16]#
    in_constr_res = [9,13,39,12] #[12]#
    out_urb = [2,20,38,26,15,42,44,4,23]#[15,2,23]#
    out_forest = [17]
    
    className = {0:"in_arch",
                 1:"out_constr_res",
                 2:"in_constr_res",
                 3:"out_urb",
                 4:"out_forest"}

    classList=[in_arch,out_constr_res,in_constr_res,out_urb,out_forest]
    for i,classlabel in enumerate(classList):
        if imageClass in classlabel:
            return str(className[i])

def getTexttokens(tokenizer,use5Scentens,device):

    # define text prompts
    names2 = ["architectural","office", "residential", "school", "manufacturing",  "cellar", "laboratory","construction site", "mining", "tunnel"]
    names3 = ["construction site", "town","city", "country side","alley","parking lot", "forest"]

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
    text1 = tokenizer(["indoor", "outdoor"]).to(device)
    if use5Scentens:
        text2 = tokenizer(sent2).to(device)
        text3 = tokenizer(sent3).to(device)
    
    else:
        text2 = tokenizer(names2).to(device)
        text3 = tokenizer(names3).to(device)

    return text1,text2,text3




if __name__ == "__main__":
    testString = "panorama_00002_0015_1.jpg"
    label = getLabel(testString)
    print(label)


