import pandas as pd
import os
from PIL import Image, ImageEnhance, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import random
from pathlib import Path
import folderManagment.pathsToFolders as ptf
from tqdm import tqdm

'''
Function to cut the image into patches with fixed width and height
The images are then safed as .jpg in the output folder

In: image path, output folder, number of patches
'''
def process_image(image_path, output_folder, num_patches):

    # open the image and get its dimensions
    img = Image.open(image_path)
    width, height = img.size

    # define the dimensions for the patches
    patch_width = height
    patch_height = height
    
    interval = (width - patch_width)/num_patches
   
    # cut the image into patches
    patches = []
    for i in range(num_patches):
        # calculate a offset for horizontal shifting
        random_offset = random.randint(0, width - patch_width)
        defined_offset = i * interval

        offset = defined_offset ### choose the random or defined offset
        
        # calculate the width of the remaining portion
        remaining_patch_width = patch_width - offset
     
        # crop the remaining portion and the initial portion
        patch = img.crop((offset, 0, width, patch_height))
        remaining_patch = img.crop((0, 0, remaining_patch_width, patch_height))

        # create image patch by stitching the remaining portion to the initial portion
        combined_patch = Image.new('RGB', (patch_width, patch_height))
        combined_patch.paste(patch, (0, 0))
        combined_patch.paste(remaining_patch, (patch.width, 0))

        # randomly flip combined_patch horizontally (left to right)
        if random.choice([True, False]):
            combined_patch = combined_patch.transpose(Image.FLIP_LEFT_RIGHT)

        # randomly adjust saturation and brightness
        #combined_patch = adjust_saturation_and_brightness(combined_patch)

        patches.append(combined_patch)

    # reduce the resolution to 224x224 and save the images
    for idx, patch in enumerate(patches):
        # patch_resized = patch.resize((224, 224), resample=Image.LANCZOS)
        output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_{idx}.jpg")
        patch.save(output_path)

# def main():
    # input and output folders paths
    # Datafolder = Path()
    # input_folder = "/scratch2/liawin/Master_Thesis/03_data/data_hex/candolle_img_all"
    # output_folder = "/scratch2/liawin/Master_Thesis/03_data/data_hex/candolle_test5"
input_folder = ptf.DatasetPanorama
output_folder = ptf.HexagonDataFolder / "candolle_5patch"
# number of patches you want to create
num_patches = 5  

# process each image in the input folder
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder, num_patches)

# if __file__ == "__main__":
#     main()