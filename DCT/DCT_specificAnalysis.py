import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import glob
import os 
def apply_dct(image_block):
    # Apply 2D DCT to the image block
    return dct(dct(image_block.T, norm='ortho').T, norm='ortho')

def select_images(corruption, severity):
    # Define the main folder
    main_folder = f'/visinf/projects_students/group8_dlcv/vilab08/ImageNet-3DCC/{corruption}'

    # Define the subfolder based on severity
    subfolder = f'{main_folder}/{severity}'
    subfolder = f'{subfolder}/n02086910'
    # Use glob to get all JPEG files in the subfolder
    image_files = glob.glob(os.path.join(subfolder, '*.JPEG'))
    return image_files

def visualize_dct(corruption, severity):

    image_paths = select_images(corruption, severity)
    coefficient_sum = None 
    for image_path in image_paths:
    # Load the image using Pillow

        img = Image.open(image_path).convert('L')  # Convert to grayscale

        # Convert the image to a NumPy array
        img_array = np.array(img)

        

        # Apply DCT to the image
        dct_coefficients = apply_dct(img_array)
        if coefficient_sum is None:
            coefficient_sum = np.zeros(np.shape(img_array))
        coefficient_sum = np.add(coefficient_sum, dct_coefficients)

    coefficients_average = np.divide(coefficient_sum, len(image_paths))
    # Visualize the original image REPLACE IWTH AVERAGE OF SYNTHETIC IMAGES
    plt.subplot(121)
    plt.imshow(img_array, cmap='gray')
    plt.title('Example Image')
    coefficients_average = np.log(np.abs(coefficients_average))
    coefficients_average = np.clip(coefficients_average, a_min=-5, a_max=None)

    # Visualize the DCT coefficients
    plt.subplot(122)
    plt.imshow(coefficients_average, cmap='viridis', interpolation='nearest',vmin=np.min(coefficients_average),vmax=np.max(coefficients_average))
    plt.title('DCT Coefficients (log scale)')
    plt.colorbar(shrink=0.6)

    # Show the plot
    output_path = f"DCT_Analysis_corrupted_papillon.png"
    plt.savefig(output_path)

if __name__=="__main__":
    path = "/visinf/projects_students/group8_dlcv/vilab08/ImageNet-C/brightness/1/n01440764/ILSVRC2012_val_00000293.JPEG"
    corruption = "near_focus"
    severity = "1"
    visualize_dct(corruption, severity)
