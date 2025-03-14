import pandas as pd
import shutil
import os

# Path to the CSV file and the folder containing the images
csv_path = '../FACET_Data_set_uncom/annotations/annotations.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path, sep=',')

# List files in the specified folder
folder_path = 'imgs_bbox_all_new_faces'
image_files = os.listdir(folder_path)

# Strip the ".jpg" extension from image_files list
image_files = [filename.split('.')[0] for filename in image_files]

# Convert person_id column in DataFrame to string
df['person_id'] = df['person_id'].astype(str)

# Filter DataFrame to include only entries corresponding to image files in the folder
filtered_df = df[df['person_id'].isin(image_files)]

# Mapping column names to gender values
gender_mapping = {
    'gender_presentation_masc': 'masc',
    'gender_presentation_fem': 'fem',
    'gender_presentation_non_binary': 'non-binary',
    'gender_presentation_na': 'na'
}


# Write class1 for each image to a text file
output_file = 'image_classes.txt'
with open(output_file, 'w') as f:
    for index, row in filtered_df.iterrows():
        #filename = row['filename']
        filename = row['person_id']
        class1 = row['class1']
        # Find the gender based on the column with value 1
        gender = next((gender_mapping[column] for column in gender_mapping if row[column] == 1), 'na')
        f.write(f"{filename}\t{class1}\t{gender}\n")
        #f.write(f"{filename}\t{class1}\n")


from sklearn.model_selection import train_test_split

# Read the text file and load the data into a list of tuples (filename, class1)
data = []
with open("image_classes.txt", 'r') as f:
    for line in f:
        filename, class1 = line.strip().split('\t')
        data.append((filename, class1))

# Split data into trainval and test sets (90% trainval, 10% test)
trainval_data, test_data = train_test_split(data, test_size=0.1, stratify=[item[1] for item in data], random_state=42)

# Split trainval data into train and validation sets (90% train, 10% validation)
train_data, val_data = train_test_split(trainval_data, test_size=0.1, stratify=[item[1] for item in trainval_data], random_state=42)

# Write data to txt files for each split variant
def write_to_txt(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item[0]}\t{item[1]}\n")

write_to_txt(train_data, 'images_variant_train.txt')
write_to_txt(val_data, 'images_variant_val.txt')
write_to_txt(trainval_data, 'images_variant_trainval.txt')
write_to_txt(test_data, 'images_variant_test.txt')

print("Data split and saved to txt files successfully.")

import pandas as pd
import os

# Load the test set filenames and class1 labels
test_data = []
with open('class1_split/images_variant_test.txt', 'r') as f:
    for line in f:
        filename, class1 = line.strip().split('\t')
        test_data.append((filename, class1))

# Read the annotations.csv file into a DataFrame
df = pd.read_csv('../FACET_Data_set_uncom/annotations/annotations.csv')

# Filter DataFrame to include only entries corresponding to images in the test set
#test_filenames = [item[0] + '.jpg' for item in test_data]

# Strip the ".jpg" extension from image_files list
test_filenames = [filename[0].split('.')[0] for filename in test_data]


import pandas as pd
import os

# Load the test set filenames and class1 labels
test_data = []
with open('class1_split/images_variant_test.txt', 'r') as f:
    for line in f:
        filename, class1 = line.strip().split('\t')
        test_data.append((filename, class1))

# Read the annotations.csv file into a DataFrame
df = pd.read_csv('../FACET_Data_set_uncom/annotations/annotations.csv')

# Filter DataFrame to include only entries corresponding to images in the test set
#test_filenames = [item[0] + '.jpg' for item in test_data]

# Strip the ".jpg" extension from image_files list
test_filenames = [filename[0].split('.')[0] for filename in test_data]

# Convert person_id column in DataFrame to string
df['person_id'] = df['person_id'].astype(str)

# Filter DataFrame to include only entries corresponding to image files in the folder
#filtered_df = df[df['person_id'].isin(image_files)]

test_df = df[df['person_id'].isin(test_filenames)]


# Specify the directory to save the files
output_directory = 'class1_split/test_class1_split_gender'
os.makedirs(output_directory, exist_ok=True)

# Iterate through each gender type and class1 combination and create a text file
for gender_type in ['masc', 'fem', 'non_binary', 'na']:
    for class1 in test_df['class1'].unique():
        gender_class_output_file = os.path.join(output_directory, f'images_variant_{gender_type}_{class1}.txt')
        with open(gender_class_output_file, 'w') as f:
            for index, row in test_df.iterrows():
                filename = row['person_id']
                row_class1 = row['class1']
                if row[f'gender_presentation_{gender_type}'] == 1 and row_class1 == class1:
                    f.write(f"{filename}\t{class1}\n")

        print(f"Text file '{gender_class_output_file}' has been created for gender presentation '{gender_type}' and class1 '{class1}'.")

# Calculate the sum of skin tones in the range 1-5
test_df['1-5_count'] = test_df[['skin_tone_1', 'skin_tone_2', 'skin_tone_3', 'skin_tone_4', 'skin_tone_5']].sum(
    axis=1)
# Count occurrences of each skin tone category for each image
test_df['6-10_count'] = test_df[['skin_tone_6', 'skin_tone_7', 'skin_tone_8', 'skin_tone_9', 'skin_tone_10']].sum(
    axis=1)

# Determine skin tone category based on majority vote, skip if equal
test_df['skin_tone_category'] = test_df.apply(lambda row: '1-5' if row['1-5_count'] > row['6-10_count'] else (
    '6-10' if row['1-5_count'] < row['6-10_count'] else None), axis=1)


# Specify the directory to save the files
output_directory = 'class1_split/test_class1_split_skintone'
os.makedirs(output_directory, exist_ok=True)

# Iterate through each skin tone category and class1 combination and create a text file
for skin_tone_category in ['1-5', '6-10']:
    for class1 in test_df['class1'].unique():
        skin_tone_output_file = os.path.join(output_directory, f'images_variant_{skin_tone_category}_{class1}.txt')
        with open(skin_tone_output_file, 'w') as f:
            for index, row in test_df.iterrows():
                filename = row['person_id']
                row_class1 = row['class1']
                if row['skin_tone_category'] == skin_tone_category and row_class1 == class1:
                    f.write(f"{filename}\t{class1}\n")

        print(
            f"Text file '{skin_tone_output_file}' has been created for skin tone category '{skin_tone_category}' and class1 '{class1}'.")

# Specify the directory to save the files
output_directory = 'class1_split/test_class1_split_skintone'
os.makedirs(output_directory, exist_ok=True)

# Split test_df by skin tone category
for skin_tone_category in ['1-5', '6-10']:
    skin_tone_df = test_df[test_df['skin_tone_category'] == skin_tone_category]
    skin_tone_output_file = os.path.join(output_directory, f'images_variant_{skin_tone_category}.txt')
    with open(skin_tone_output_file, 'w') as f:
        for index, row in skin_tone_df.iterrows():
            filename = row['person_id']
            class1 = row['class1']
            f.write(f"{filename}\t{class1}\n")

    print(
        f"Text file '{skin_tone_output_file}' has been created for skin tone category '{skin_tone_category}'.")

# Specify the directory to save the files
output_directory_gender = 'class1_split/test_class1_split_gender'
os.makedirs(output_directory_gender, exist_ok=True)

# Iterate through each gender type and class1 combination and create a text file
for gender_type in ['masc', 'fem']:
    gender_output_file = os.path.join(output_directory_gender, f'images_variant_{gender_type}.txt')
    with open(gender_output_file, 'w') as f:
        for index, row in test_df.iterrows():
            filename = row['person_id']
            row_gender_type = f'gender_presentation_{gender_type}'
            row_class1 = row['class1']
            if row[row_gender_type] == 1:
                f.write(f"{filename}\t{row_class1}\n")

    print(f"Text file '{gender_output_file}' has been created for gender presentation '{gender_type}'.")