from Evaluation.Evaluation_utils import printAndSaveHeatmap,get_trueClass,get_max_class_with_threshold,find_majority_element
import pandas as pd
from pathlib import Path

# Print Hailo confusion matrix with correct font

modelname = "RN101"
csv_path_predictions = "USBStick/Data/HailoCut/pred_RN101_5patches_5scentens.csv"
outputfolder = Path("Data/hailoConf")

df_5patch = get_trueClass(pd.read_csv(csv_path_predictions))
df = df_5patch.copy()
df['y_predIO'] = df.apply(
    get_max_class_with_threshold, axis=1, threshold=0.4)
# InOutTh_dict = {'RN50':0.63, 'RN50x4':0.52, 'TinyClip19M':0.51, 'RN101':0.78, 'TinyClip30M':0.72,'TinyClip19M16Bit':0.45}
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

printAndSaveHeatmap(df, modelname, outputfolder, use_5_Scentens = True)