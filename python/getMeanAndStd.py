import os
import numpy as np
from PIL import Image
from folderManagment import pathsToFolders as ptf
from tqdm import tqdm

# Calculate mean and std for a folder of images

def calculate_mean_std(image_folder):
    """
    Calculate the mean and standard deviation of images in a folder using a rolling approach.

    Args:
        image_folder (str): Path to the folder containing images.

    Returns:
        tuple: Mean and standard deviation of all images.
    """
    mean_sum = np.zeros(3, dtype=np.float64)
    sq_mean_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for filename in tqdm(os.listdir(image_folder)):
        filepath = os.path.join(image_folder, filename)

        try:
            with Image.open(filepath) as img:
                img = img.convert('RGB')  # Ensure 3-channel RGB
                pixels = np.asarray(img, dtype=np.float32).reshape(-1, 3)

                # Update rolling mean and std sums
                mean_sum += np.sum(pixels, axis=0)
                sq_mean_sum += np.sum(pixels ** 2, axis=0)
                total_pixels += pixels.shape[0]
        except Exception as e:
            print(f"Could not process file {filename}: {e}")

    if total_pixels == 0:
        raise ValueError("No valid images found in the folder.")

    # Calculate mean and std
    mean = mean_sum / total_pixels
    variance = (sq_mean_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    return mean, std

# Example usage
if __name__ == "__main__":
    folder_path = str(ptf.Dataset5Patch)

    try:
        mean, std = calculate_mean_std(folder_path)
        print(f"Mean: {mean}")
        print(f"Standard Deviation: {std}")
    except Exception as e:
        print(f"Error: {e}")
