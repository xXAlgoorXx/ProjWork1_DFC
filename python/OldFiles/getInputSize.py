import clip
import open_clip
import folderManagment.pathsToFolders as ptf


print("Clip Models:",clip.available_models())
# Evaluate every Resnet model
resnetModels = []
for clipmodel in clip.available_models():
    if "RN" in clipmodel:
        resnetModels.append(clipmodel)

for modelname in resnetModels:
    model, _ = clip.load(modelname)
    print(f"{modelname}: {model.visual.input_resolution}")

resnetModels = []
print("Open Clip Models:",open_clip.list_models())
for clipmodel in open_clip.list_models():
    if "ResNet" in clipmodel and "Tiny" in clipmodel:
        resnetModels.append(clipmodel)

for modelname in resnetModels:
    model, *_ = open_clip.create_model_and_transforms(modelname,pretrained=str(ptf.tinyClipModels / f"{modelname}-LAION400M.pt"))
    print(f"{modelname}: {model.visual.image_size}")
