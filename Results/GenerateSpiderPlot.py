import matplotlib.pyplot as plt
import numpy as np

ImageNet1k_SD = {   "Near Focus":[30.6,27.2,24.9,23,21],
                    "Far Focus": [32.6,	25.7, 21.5,	18.2, 16.3],
                    "XY Motion Blur": [12.6,	7.4,	4.7,	3.4,	2.9],
                    "Fog 3D":[33.4,	24,	17.5,	12.9,	10.1],
                    "name": "ImageNet-1K-SD"
}
ResNet50 = {   "Near Focus":[64,	60.1,	55.7,	52.7,	49.6],
                "Far Focus": [64.8,	58.1,	51.5,	46.1,	41.6,	53],
                "XY Motion Blur": [48.6,	37.1,	27.9,	21.4,	17.2],
                "Fog 3D":[62.8,	48.8,	37.7,	29.4,	23.3],
                "name": "ResNet50"
}
ConvNext_Base = {   "Near Focus":[74.5,	71.7,	68.7,	65.8,	63.1],
                    "Far Focus": [75.1,	69.5,	64.2,	58.9,	55],
                    "XY Motion Blur": [67,	58.4,	49.2,	40.6,	32.9],
                    "Fog 3D":[76.5,	69,	58.1,	48.9,	40],
                    "name": "ConvNeXt_Base"
}
DinoV2_Base = {     "Near Focus":[84.5,	78.3,	75.8,	74.3,	72.7,	71],
                    "Far Focus": [78.3,	73.6,	69.5,	66.5,	63.4],
                    "XY Motion Blur": [72.2,	65,	57.7,	49.2,	42.4],
                    "Fog 3D":[79.2,	72.5,	61.5,	51.9,	43.3],
                    "name": "DinoV2_Base"
}
DinoV2_Small = {     "Near Focus":[72.7,	69.9,	67.2,	64.3,	61.4],
                    "Far Focus": [73.4,	68.1,	62.6,	58.5,	54.9],
                    "XY Motion Blur": [63.5,	52.8,	42.3,	34.6,	28],
                    "Fog 3D":[75.6,	66.4,	55,	45.5,	36.8],
                    "name": "DinoV2_Small"
}
ConvNext_Tiny = {     "Near Focus":[71,	68.3,	65.3,	62.2,	58.7],
                    "Far Focus": [72,	65.2,	60,	54.8,	50.1],
                    "XY Motion Blur": [61.9,	51.9,	41.4,	32.2,	25.7],
                    "Fog 3D":[73.7,	66.3,	55.2,	45.1,	37.2],
                    "name": "ConvNeXt_Tiny"
}
Models = [ImageNet1k_SD, ResNet50, ConvNext_Tiny, DinoV2_Small]




def GenerateSpiderPlot(corruptions,severity_levels):
    # Number of severity levels
    num_severity_levels = len(severity_levels)
    num_corruptions = len(corruptions)

    # Calculate evenly spaced angles for each severity level
    angles = np.linspace(0, 2*np.pi, num_severity_levels*num_corruptions, endpoint=False)

    # Close the plot by repeating the first angle at the end
    angles = np.concatenate((angles, [angles[0]]))
    # rotate the plot to point upwards:
    #starting_angle = (np.pi /2)
    #angles = np.add(angles,starting_angle)

    # Create the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#ed2868","#006794","green","orange"]
    # Plot the accuracies for each corruption and severity level
    for j, model in enumerate(Models): 
        accs = []
        for i, corruption in enumerate(corruptions):
        # Concatenate the accuracies to close the plot
            for severity in severity_levels:
                value = model[corruption][severity-1]
                accs.append(value)


        accs.append(accs[0])
        ax.plot(angles, accs, label=model["name"],color=colors[j])

        # Fill the area under the lines
        ax.fill(angles, accs, alpha=0.1,color=colors[j])

    # Set the labels for each severity level
    labels = []
    for corruption in corruptions:
        for severity in severity_levels:
            labels.append(corruption+" "+str(severity))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set the y-axis label and add a title
    ax.set_rlabel_position(90)
    ax.set_yticklabels([])
    ax.set_title('Top-1 Accuracy for selected 3D Common Corruptions Severity [1,3,5]', size=16)
    ax.set_yticks(np.linspace(0,100,6))
    ax.set_yticklabels(["0%","20%","40%","60%","80%",""])
    # Add a legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

    # Save the plot
    output_path = "SpiderPlot_out.png"
    plt.savefig(output_path)


if __name__ == "__main__":
    #"flash","iso_noise", "bit_error""near_Focus", "iso_noise","xy_motion_blur" "frost", "gaussian_noise", "contrast"
    # Specify the corruption type and severity levels
    corruptions = ["Near Focus", "Far Focus", "XY Motion Blur","Fog 3D"]
    severity_levels = [1, 3, 5]  # Adjust as needed

    GenerateSpiderPlot(corruptions, severity_levels)
