import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.ndimage import zoom
import glob
import os 
import torchvision.transforms as transforms
def apply_dct(image_block):
    # Apply 2D DCT to the image block
    return dct(dct(image_block.T, norm="ortho").T, norm="ortho")

def select_images(path, png=False):
    
    # Use glob to get all JPEG files in the subfolder
    if png:
        image_files = glob.glob(os.path.join(path, '*.png'))
    else: 
        parts = path.split("/")
        clsnum = parts[-1]
        path = "/".join(parts[:-1])+"/"
        image_files = glob.glob(os.path.join(path, f'*_{clsnum}.JPEG'))
    return image_files

def visualize_dct(paths, names):
    preprocessing = transforms.Compose([
                                    transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
    for i, path in enumerate(paths):
        image_paths = select_images(path, (i%2==0))

        coefficient_sum = None 
        for j, image_path in enumerate(image_paths):
        # Load the image using Pillow
            if j >0:
                break 
            img = Image.open(image_path).convert('RGB')  # Convert to grayscale

            # Convert the image to a NumPy array
            img_array = preprocessing(img)
            img_array = transforms.functional.rgb_to_grayscale(img_array)
            img_array = np.array(img_array)
            
            if coefficient_sum is None:
                coefficient_sum = np.zeros(np.shape(img_array))
            #reshape image as imagenet has many different size images 
            des_shape = np.shape(coefficient_sum)
            im_shape = np.shape(img_array)
            scale_factors= np.array(des_shape) / np.array(im_shape)
            if not im_shape == des_shape:
                img_array = zoom(img_array, scale_factors, order=1)
            # Apply DCT to the image
            dct_coefficients = apply_dct(img_array)
            print(img_array)
            print(dct_coefficients)
            coefficient_sum = np.add(coefficient_sum, np.log(np.abs(dct_coefficients)))

        coefficients_average = np.divide(coefficient_sum, len(image_paths))
        #clip for better visualization
        coefficients_average = np.clip(coefficients_average, a_min=-2, a_max=6)
        # Visualize the original image REPLACE IWTH AVERAGE OF SYNTHETIC IMAGES

        # Visualize the DCT coefficients
        plt.subplot(2,2,i+1)
        plt.imshow(coefficients_average[0], cmap='viridis', interpolation='nearest',vmin=np.min(coefficients_average),vmax=np.max(coefficients_average))
        plt.title(f'{names[i]} DCT (log scale)')
        plt.colorbar(shrink=0.6)
    plt.subplots_adjust(hspace=0.35)
    # Show the plot
    output_path = f"DCT_Analysis_Preprocessed.png"
    plt.savefig(output_path)

if __name__=="__main__":
    paths = [   "/visinf/projects_students/group8_dlcv/vilab08/PPCV_SD_Images/512x512/papilon/",
                "/data/vilab09/imagenet-1k/val/n02086910",
                "/visinf/projects_students/group8_dlcv/vilab08/PPCV_SD_Images/512x512/pirate_ship",
                "/data/vilab09/imagenet-1k/val/n03947888"
    ]
    names = ["Papillon SD", "Papillon real", "Pirate_Ship SD", "Pirate_Ship real"]
    visualize_dct(paths, names)
