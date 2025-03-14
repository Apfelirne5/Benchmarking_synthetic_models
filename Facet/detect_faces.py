import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from PIL import Image
import os
from tqdm import tqdm

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
#img = cv2.imread('/data/vilab10/FACET_Dataset/imgs_bbox_all_single/sa_582.jpg')
#img = ins_get_image('t1')
#print("type(img): ", type(img))
#print("img.shape: ", img.shape)
#faces = app.get(img)
#print("len(faces): ", len(faces))


# Directory paths
input_dir = '/data/vilab10/FACET_Dataset/imgs_bbox_all_new/'
output_dir_single = '/data/vilab10/FACET_Dataset/imgs_bbox_all_new_faces/'
output_dir_multi = '/data/vilab10/FACET_Dataset/imgs_bbox_all_new_faces_multi/'

# Ensure output directory exists
os.makedirs(output_dir_single, exist_ok=True)
os.makedirs(output_dir_multi, exist_ok=True)

# List all files in the input directory
image_files = [file for file in os.listdir(input_dir) if file.endswith('.jpg')]

# Loop through each image
for image_file in tqdm(image_files, desc='Processing images', unit='image'):
    # Construct the full path of the input image
    input_path = os.path.join(input_dir, image_file)
    #print(input_path)

    #img = cv2.imread("imgs_bbox_all_single/sa_1596555.jpg")
    img_pil = Image.open(input_path)

    # Convert the Pillow image to a NumPy array (RGB order)
    img_np_pil = np.array(img_pil)

    # Convert the NumPy array to BGR order to match cv2.imread
    img_np = img_np_pil[:, :, ::-1]

    # Read the image
    #img = cv2.imread(input_path)

    # Get faces
    faces = app.get(img_np)

    # Process each detected face
    if len(faces)==1:
        for idx, face_info in enumerate(faces):
            # Extract bounding box coordinates
            bbox = face_info['bbox']
            #print("bbox: ", bbox)
            x, y, x_2, y_2 = map(int, bbox)

            # Ensure x, y are within valid range
            x = max(0, min(x, img_np.shape[1]))
            y = max(0, min(y, img_np.shape[0]))

            # Ensure x_2, y_2 are within valid range
            x_2 = max(0, min(x_2, img_np.shape[1]))
            y_2 = max(0, min(y_2, img_np.shape[0]))

            #print("x, y, x_2, y_2: ", x, y, x_2, y_2)

            # Extract the face region from the original image
            face_region = img_np[y:y_2, x:x_2]
            #print("face_region.shape: ", face_region.shape)

            # Save the extracted face region
            output_path = os.path.join(output_dir_single, f"{image_file}")
            cv2.imwrite(output_path, face_region)
            #print(f"Face saved at: {output_path}")

    elif len(faces)>1:
        for idx, face_info in enumerate(faces):
            # Extract bounding box coordinates
            bbox = face_info['bbox']
            #print("bbox: ", bbox)
            x, y, x_2, y_2 = map(int, bbox)

            # Ensure x, y are within valid range
            x = max(0, min(x, img_np.shape[1]))
            y = max(0, min(y, img_np.shape[0]))

            # Ensure x_2, y_2 are within valid range
            x_2 = max(0, min(x_2, img_np.shape[1]))
            y_2 = max(0, min(y_2, img_np.shape[0]))

            #print("x, y, x_2, y_2: ", x, y, x_2, y_2)

            # Extract the face region from the original image
            face_region = img_np[y:y_2, x:x_2]
            #print("face_region.shape: ", face_region.shape)

            image_name = os.path.splitext(image_file)[0]

            # Save the extracted face region
            output_path = os.path.join(output_dir_multi, f"{image_name}_{idx + 1}.jpg")
            #print(output_path)
            cv2.imwrite(output_path, face_region)


# Process each detected face
#for idx, face_info in enumerate(faces):
    # Extract bounding box coordinates
    #bbox = face_info['bbox']
    #print("bbox: ", bbox)
    #x, y, x_2, y_2 = map(int, bbox)
    #print("x, y, w, h: ", x, y, x_2, y_2)

    # Extract the face region from the original image
    #face_region = img[y:y_2, x:x_2]

    #print("face_region.shape: ", face_region.shape)

    # Save the extracted face region
    #output_path = f"/visinf/home/vilab10/facet/imgs_bbox_all_single_faces/sa_582.jpg"
    #output_path = f"/visinf/home/vilab10/facet/imgs_bbox_all_single_faces/t1_output_{idx + 1}.jpg"
    #cv2.imwrite(output_path, face_region)
    #print(f"Face saved at: {output_path}")

#rimg = app.drawx_on(img, faces)
#cv2.imwrite("./sa_582_output.jpg", rimg)
#cv2.imwrite("imgs_bbox_all_single_faces/sa_582.jpg", rimg)