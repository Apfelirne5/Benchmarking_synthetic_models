import matplotlib.pyplot as plt
import numpy as np

# Sample accuracy values
accuracies = [
    [28.7, 20.8,13.96, 10.4],  # Accuracy values for the first graph
    [98.6, 85.7, 64.0, 65.1]   # Accuracy values for the second graph
]

# Labels for each bar
labels = ['ImageNet-1K-SD', 'ResNet50', 'ConvNeXt_Tiny', 'DinoV2_Small']
accuracies = np.round(np.subtract(100,accuracies),decimals=1)
# Bar positions
x = np.arange(len(labels))

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

# Plot the first graph
axes[0].bar(x, accuracies[0], color=['#ed2868', '#006794',"green", 'orange'])
axes[0].set_title('Relative Retained Accuracy: Near_focus Severity 1')
axes[0].set_ylabel('Rel. Acc (%)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_yticks(np.arange(0, 101, 20))

# Add data values on top of the bars for the first graph
for i, value in enumerate(accuracies[0]):
    axes[0].text(i, value + 1, f'{value}%', ha='center', va='bottom')

# Plot the second graph
axes[1].bar(x, accuracies[1], color=['#ed2868', '#006794',"green", 'orange'])
axes[1].set_title('Relative Retained Accuracy: Iso_noise Severity 5')
axes[1].set_ylabel('Rel. Acc (%)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_yticks(np.arange(0, 101, 20))

# Add data values on top of the bars for the second graph
for i, value in enumerate(accuracies[1]):
    axes[1].text(i, value + 1, f'{value}%', ha='center', va='bottom')

# Adjust layout
plt.tight_layout()

# Display the plot
output_path = "BarGraphRevert_out.png"
plt.savefig(output_path)
