import os
from PIL import Image
import matplotlib.pyplot as plt

def display_images(corruptions, severity_levels, classname, base_dir):
    num_severities = len(severity_levels)
    num_corruptions = len(corruptions)
    fig = plt.figure(layout="constrained")
    subfigs = fig.subfigures(nrows=num_corruptions, ncols=1)
    for j, subfigure in enumerate(subfigs):
        corruption_type = corruptions[j]
        subfigure.suptitle(corruption_type)
        axs = subfigure.subplots(1,num_severities)
        for i, severity in enumerate(severity_levels):
            folder_path = os.path.join(base_dir, corruption_type, str(severity), classname)
            image_name = "ILSVRC2012_val_00003817.JPEG"  # Replace with the actual image name or adapt the script to handle multiple images
            # Elster Magpie img ILSVRC2012_val_00010501.JPEG and class n01582220
            # Waffle maker with people and depth /n04542943/ILSVRC2012_val_00003817.JPEG
            image_path = os.path.join(folder_path, image_name)
            
            
            # Create a grid of subplots
            if os.path.exists(image_path):
                img = Image.open(image_path)
                # Display the image in the corresponding subplot
                axs[i].imshow(img)
                axs[i].set_title(f"Severity {severity}")
                axs[i].axis("off")
            else:
                print(f"Image not found: {image_path}")
        
    output_path = "output_plot.png"
    plt.savefig(output_path)
    plt.close()
#/visinf/projects_students/group8_dlcv/vilab08/ImageNet-3DCC/near_focus/1/n01440764/ILSVRC2012_val_00009346.JPEG
if __name__ == "__main__":
    # Set the path to the base directory containing the corrupted images
    base_directory = "/visinf/projects_students/group8_dlcv/vilab08/ImageNet-3DCC"
    #"flash","iso_noise", "bit_error""near_focus", "iso_noise","xy_motion_blur" "frost", "gaussian_noise", "contrast"
    # Specify the corruption type and severity levels
    corruptions = ["near_focus", "xy_motion_blur", "iso_noise"]
    severity_levels = [1, 2, 3, 4, 5]  # Adjust as needed
    classname = "n04542943"

    display_images(corruptions, severity_levels, classname, base_directory)
