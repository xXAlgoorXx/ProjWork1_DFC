from checkOutput_utils import get_pred_CPU, get_pred_HEF, getHighestProb, getMaxIndexPerClass,get_trueClass,RestOfGraphOnnx,evalModel_hef,loadJson,evalModel
import pandas as pd
import numpy as np
import os
import sys
import clip
from hailo_sdk_client import ClientRunner, InferenceContext
from PIL import Image

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "RN50"
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
    
    image_path = "temp/panorama_00002_0014_2.jpg"
    chosen_hw_arch = "hailo8l"
    textEmb_json = "models/textEmbeddings/textEmbeddings_RN50.json"
    preprocess = preprocess
    postProcess_onnx = "models/RestOfGraphONNX/RestOf_CLIP_RN50_simplified.onnx"
    
    use5Scentens = False
    # == CPU ==
    
    
    # Read the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_features = model.encode_image(image).detach().numpy()
    
    text = text3
    probs_CPU = evalModel(model, image, text, False)
    
    textEmb = loadJson(textEmb_json)
    postProcess = RestOfGraphOnnx(postProcess_onnx)
    
    # == Hailo nativ==
    har = "models/Harfiles/CLIP_RN50_hailo_model.har"
    runner = ClientRunner(har=har,
                          hw_arch=chosen_hw_arch)
    
    with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
        image = preprocess(Image.open(image_path)).to(device)
        img_transposed = np.transpose(image.numpy(), (1, 2, 0))[np.newaxis,:]
        output_quantized_har = runner.infer(ctx, img_transposed,batch_size=1)
        output_quantized_har = output_quantized_har[:,0,:,:]
        output_Hef = postProcess(output_quantized_har)
        text = np.array(textEmb["3"]["embeddings"])
        probs_nativ = evalModel_hef(output_Hef, text, use5Scentens)
    
      
    # == Hailo compiled ==
    har = "models/Harfiles/CLIP_RN50_compiled_model.har"
    runner = ClientRunner(har=har,
                          hw_arch=chosen_hw_arch)
    
    with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
        image = preprocess(Image.open(image_path)).to(device)
        img_transposed = np.transpose(image.numpy(), (1, 2, 0))[np.newaxis,:]
        output_quantized_har = runner.infer(ctx, img_transposed,batch_size=1)
        output_quantized_har = output_quantized_har[:,0,:,:]
        output_Hef = postProcess(output_quantized_har)
        text = np.array(textEmb["3"]["embeddings"])
        probs_compiled = evalModel_hef(output_Hef, text, use5Scentens)
    
    # == Hailo Quant ==
    har = "models/Harfiles/CLIP_RN50_quantized_model.har"
    runner = ClientRunner(har=har,
                          hw_arch=chosen_hw_arch)
    
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        image = preprocess(Image.open(image_path)).to(device)
        img_transposed = np.transpose(image.numpy(), (1, 2, 0))[np.newaxis,:]
        output_quantized_har = runner.infer(ctx, img_transposed,batch_size=1)
        output_quantized_har = output_quantized_har[:,0,:,:]
        output_Hef = postProcess(output_quantized_har)
        text = np.array(textEmb["3"]["embeddings"])
        probs_quant = evalModel_hef(output_Hef, text, use5Scentens)
  
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.size": 10
    })
    f,axes = plt.subplots(4,1,figsize = (6,6))
    
    axes[0].bar(outlabels,probs_CPU[0])
    axes[0].set_ylim([0,1])
    axes[0].set_title("CPU")
    
    axes[1].bar(outlabels,probs_nativ[0])
    axes[1].set_ylim([0,1])
    axes[1].set_title("Nativ")
    
    axes[2].bar(outlabels,probs_compiled[0])
    axes[2].set_ylim([0,1])
    axes[2].set_title("Compiled")
    
    axes[3].bar(outlabels,probs_quant[0])
    axes[3].set_ylim([0,1])
    axes[3].set_title("Quantized")
    
    f.tight_layout()
    plt.savefig("temp/"+ f"compareProbs_{model_name}", dpi = 600)
    plt.show()
    
    
if __name__ == "__main__":
    main()