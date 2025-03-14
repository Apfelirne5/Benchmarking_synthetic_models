import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.ndimage import zoom
from datasets import load_dataset, VerificationMode
import glob
import os 
import torchvision.transforms as transforms
from classes import IMAGENET2012_CLASSES

def apply_dct(image_block):
    # Apply 2D DCT to the image block
    return dct(dct(image_block.T, norm="ortho").T, norm="ortho")
    #return np.fft.fft2(image_block)



def visualize_dct(paths, names):
    ClassLabels = list(IMAGENET2012_CLASSES.values())
    preprocessing = transforms.Compose([
                                    transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
    Coefficients = []
    val_dataset = load_dataset("/visinf/projects_students/group8_dlcv/vilab08/_Workshop/imagenet-1k.py", split="validation",verification_mode=VerificationMode.NO_CHECKS)
    #print(val_dataset[0])
    def select_images(path, png=False):
        # Use glob to get all JPEG files in the subfolder
        if png:
            images = []
            image_files = glob.glob(os.path.join(path, '*.png'))
            for img_path in image_files: 
                img = Image.open(img_path).convert('RGB')
                images.append(img)
        else: 
            classid = IMAGENET2012_CLASSES[path]
            print(classid)
            images = val_dataset.filter(lambda example: ClassLabels[example["label"]]==classid)["image"]
            for j, image in enumerate(images): 
                if image.mode =="L":
                    image = image.convert("RGB")
                    images[j] = image
        return images


    for i, path in enumerate(paths):
        images = select_images(path, (i%2==0))
        #if i>0:break
        coefficient_sum = None 
        print(len(images))
        for j, image in enumerate(images):
        # Load the image using Pillow
              # Convert to grayscale after preprocessing 
            #apply preprocessing
            #if j>0 : break
            img_array = preprocessing(image)
            #transform to grayscale 
            
            img_array = transforms.functional.rgb_to_grayscale(img_array)
            
            if coefficient_sum is None:
                coefficient_sum = np.zeros(np.shape(img_array))
            # Apply DCT to the image
            dct_coefficients = apply_dct(img_array.numpy()[0])
            #print(img_array)
            #print(dct_coefficients)
            #print(np.shape(coefficient_sum))
            coefficient_sum = np.add(coefficient_sum, np.log(np.abs(dct_coefficients)))

        coefficients_average = np.divide(coefficient_sum, len(images))
        Coefficients.append(coefficients_average)
        #clip for better visualization
        coefficients_average = np.clip(coefficients_average, a_min=-5, a_max=2)
        # Visualize the original image REPLACE IWTH AVERAGE OF SYNTHETIC IMAGES

        # Visualize the DCT coefficients
        plt.subplot(2,2,i+1)
        plt.imshow(coefficients_average[0], cmap='viridis', interpolation='nearest',vmin=np.min(coefficients_average),vmax=np.max(coefficients_average))
        plt.title(f'{names[i]} DCT (log scale)')
        plt.colorbar(shrink=0.6)
    plt.subplots_adjust(hspace=0.35)
    # Show the plot
    output_path = f"DCT_Analysis_all_preprocessed.png"
    plt.savefig(output_path)

if __name__=="__main__":
    paths = [   "/visinf/projects_students/group8_dlcv/vilab08/PPCV_SD_Images/512x512/papilon/",
                "n02086910",
                #"/data/vilab09/imagenet-1k/val/n02086910",
                "/visinf/projects_students/group8_dlcv/vilab08/PPCV_SD_Images/512x512/pirate_ship",
                "n03947888",
                #"/data/vilab09/imagenet-1k/val/n03947888"
    ]
    names = ["Papillon SD", "Papillon real", "Pirate_Ship SD", "Pirate_Ship real"]
    visualize_dct(paths, names)
