from Evaluation_utils import evalModel, get_max_class_with_threshold, get_pred, get_throughput, get_trueClass, find_majority_element, printAndSaveHeatmap, get_modelnames
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
modelname = "RN50x4"
model, preprocess = clip.load(modelname, device=device)


# define text prompts
names1 = ["indoor", "outdoor"]
names2 = ["architectural", "office", "residential", "school", "manufacturing",
          "cellar", "laboratory", "construction site", "mining", "tunnel"]
names3 = ["construction site", "town", "city",
          "country side", "alley", "parking lot", "forest"]
startNames2 = len(names1)
startNames3 = len(names1) + len(names2)
names = names1 + names2 + names3
print(len(names))
text = clip.tokenize(names).to(device)
print(text.shape)
image_path = "hailoDFC/testImages/panorama_00009_0012_0.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

probs = evalModel(model, image, text, False)

score_in_arch = (probs[0][startNames2 + 0]
                 + probs[0][startNames2 + 1]
                 + probs[0][startNames2 + 2]
                 + probs[0][startNames2 + 3]
                 + probs[0][startNames2 + 4]
                 + probs[0][startNames2 + 5]
                 + probs[0][startNames2 + 6])

score_in_constr = (probs[0][startNames2 + 7]
                   + probs[0][startNames2 + 8]
                   + probs[0][startNames2 + 9])

score_out_constr = probs[0][startNames3]

score_out_urb = (probs[0][startNames3 + 1]
                 + probs[0][startNames3 + 2]
                 + probs[0][startNames3 + 3]
                 + probs[0][startNames3 + 4]
                 + probs[0][startNames3 + 5])
score_out_for = probs[0][startNames3 + 6]


for name, prob in zip(names, probs[0]):
    print(f"{name}: {prob}")
print(f"in: {probs[0][0]}\n"
      f"out: {probs[0][1]}\n"
      f"in_arch: {score_in_arch}\n"
      f"in_constr: {score_in_constr}\n"
      f"out_constr: {score_out_constr}\n"
      f"out_urb: {score_out_urb}\n"
      f"out_for: {score_out_for}\n")
