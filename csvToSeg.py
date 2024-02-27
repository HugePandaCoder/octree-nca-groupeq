#%%
# import os
# import csv

# preprocessedSeg_path = r'/home/jkalkhof_locale/Downloads/physionet.org/files/chexmask-cxr-segmentation-data/0.3/Preprocessed/'

# csv_path = ""

# for root, dirs, files in os.walk(preprocessedSeg_path):
#     for file in files:
#         if file.endswith('.csv'):
#             file_path = os.path.join(root, file)
#             csv_path = file_path
#             print(f"Contents of {file_path}:")
#             with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
#                 csvreader = csv.reader(csvfile)
#                 for i, row in enumerate(csvreader):
#                     if i >= 1:
#                         break
#                     print(row)

# exit()
import pandas as pd
import numpy as np
import cv2
import nibabel as nib
import os
import pydicom
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_mask_from_RLE(rle, height, width, new_height=256, new_width=256):
    mask = np.zeros(height * width, dtype=np.uint8)
    if rle != '':  # Check if RLE string is not empty
        runs = np.array([int(x) for x in rle.split()])
        starts = runs[::2] - 1
        ends = starts + runs[1::2]
        for start, end in zip(starts, ends):
            mask[start:end] = 1
    mask = mask.reshape((height, width))
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return resized_mask

def save_nifti(masks, filename):
    # Stack the masks along a new dimension to create a 3D array
    combined_mask = np.stack(masks, axis=-1)
    nifti_img = nib.Nifti1Image(combined_mask.astype(np.uint8), affine=np.eye(4))
    nib.save(nifti_img, filename)

def read_dicom_and_resize(path, new_height=256, new_width=256):
    if not os.path.exists(path):
        print(f"File does not exist: {path}")
        return None
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array
    #print(image.shape)
    #plt.imshow(image, cmap='gray')
    #plt.show()
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return image

def read_png_and_resize(path, new_height=256, new_width=256):
    # Load the image from the specified path
    if not os.path.exists(path):
        #print(f"File does not exist: {path}")
        return None

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    # Resize the image to the specified dimensions
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    if len(resized_image.shape) == 3:
        resized_image = resized_image[:, :, 0]
    return resized_image

def batch_process(csv_path, record_list_path, output_dir_images, output_dir_labels, tag, column, data_dir, datatype, record_for_path=False, start_row=0, end_row=500, format='nifti'):
    df = pd.read_csv(csv_path)

    print(df.columns)

    df_filtered = df[df['Dice RCA (Mean)'] >= 0.7]

    df = df_filtered

    if record_for_path:
        record_df = pd.read_csv(record_list_path)
    
    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)
    if not os.path.exists(output_dir_labels):
        os.makedirs(output_dir_labels)
    
    count = 0

    for index, row in tqdm(df.iloc[start_row:end_row+1000000].iterrows()):
        dicom_id = row[column]  # Assuming 'ImageID' matches 'dicom_id'
        #print(row)
        if record_for_path:
            matching_record = record_df[record_df[column] == dicom_id].iloc[0]
        else:
            matching_record = {}
            matching_record['path'] = os.path.join(data_dir, dicom_id)
        

        if datatype == 'nifti':
            dicom_path = os.path.join('/mnt/share/2.0.0/', matching_record['path'])  # Update with actual path prefix
            dicom_image = read_dicom_and_resize(dicom_path)
        elif datatype == 'png':
            dicom_image = read_png_and_resize(matching_record['path'])
            #print(dicom_image.shape)
        else:
            raise ValueError("Invalid datatype. Please specify 'nifti' or 'png'.")
        
        # Skip the iteration if the image could not be loaded
        if dicom_image is None:
            continue

        left_lung_mask = get_mask_from_RLE(row['Left Lung'], row['Height'], row['Width'])
        right_lung_mask = get_mask_from_RLE(row['Right Lung'], row['Height'], row['Width'])
        heart_mask = get_mask_from_RLE(row['Heart'], row['Height'], row['Width'])
        
        if format == 'nifti':
            # Save DICOM as NIfTI
            dicom_nifti_filename = f"{dicom_id}.nii"
            dicom_nifti_img = nib.Nifti1Image(dicom_image, affine=np.eye(4)) #.astype(np.uint8)
            nib.save(dicom_nifti_img, os.path.join(output_dir_images, dicom_nifti_filename))
            
            # Combine the masks and save as NIfTI
            masks_filename = f"{dicom_id}.nii"
            save_nifti([left_lung_mask, right_lung_mask, heart_mask], os.path.join(output_dir_labels, masks_filename))
        else:
            # MENT FOR MOBILE TRAINING

            # Save DICOM image as PNG
            dicom_png_filename = f"{dicom_id}.png"
            #print(dicom_image.shape, os.path.join(output_dir_images, dicom_png_filename))

            dicom_image = dicom_image - np.min(dicom_image)
            dicom_image = dicom_image / np.max(dicom_image)

            cv2.imwrite(os.path.join(output_dir_images, dicom_png_filename), dicom_image *255)
            
            merged_lung_mask = np.clip(left_lung_mask + right_lung_mask, 0, 1)

            # Save masks as PNG
            cv2.imwrite(os.path.join(output_dir_labels, f"{dicom_id}_label.png"), merged_lung_mask*255)
            #cv2.imwrite(right_lung_mask, os.path.join(output_dir_labels, f"{dicom_id}_right_lung.png"))
            #cv2.imwrite(heart_mask, os.path.join(output_dir_labels, f"{dicom_id}_heart.png"))

        count += 1
        if count >= end_row:
            break



tag = "Padchest"
#tag = "MIMIC-CXR-JPG"
#tag = "ChestX-Ray8"
image_record = None
data_dir = None

if tag == "MIMIC-CXR-JPG":
    csv_path = '/home/jkalkhof_locale/Downloads/physionet.org/files/chexmask-cxr-segmentation-data/0.3/Preprocessed/MIMIC-CXR-JPG.csv'  # Update this path accordingly
    image_record = '/mnt/share/2.0.0/cxr-record-list.csv'
    record_for_path = True
    column = 'dicom_id'
    datatype = 'nifti'
      # Define where you want to save the masks
elif tag == "ChestX-Ray8":
    csv_path = '/home/jkalkhof_locale/Downloads/physionet.org/files/chexmask-cxr-segmentation-data/0.3/OriginalResolution/ChestX-Ray8.csv'
    record_for_path = False
    column = 'Image Index'
    data_dir = '/home/jkalkhof_locale/Downloads/cxr8/images_01/images/'
    datatype = 'png'
elif tag == "Padchest":
    csv_path = '/home/jkalkhof_locale/Downloads/physionet.org/files/chexmask-cxr-segmentation-data/0.3/OriginalResolution/Padchest.csv'
    record_for_path = False
    column = 'ImageID'
    data_dir = '/home/jkalkhof_locale/Downloads/Padchest/images/'
    datatype = 'png'

output_dir = os.path.join('/home/jkalkhof_locale/Documents/Data/MICCAI24/Padchest', tag)



# Call the batch processing function
batch_process(csv_path, image_record, os.path.join(output_dir, "images"), os.path.join(output_dir, "labels"), tag, column, data_dir, datatype, record_for_path, end_row=2000, format='nifti')


exit()

# Path to your CSV file
#csv_path = '/mnt/data/example.csv'  # Update this path accordingly

# Read the CSV file
df = pd.read_csv(csv_path)

# Assuming you want to visualize for a specific ID, let's say for the first row
# Make sure to adjust these if your IDs or indexing needs are different
row_index = 0  # Example: First row

left_lung_rle = df.iloc[row_index]['Left Lung']
right_lung_rle = df.iloc[row_index]['Right Lung']
heart_rle = df.iloc[row_index]['Heart']

height, width = 1024, 1024  # Adjust these based on your image dimensions

# Convert RLEs to masks
left_lung_mask = get_mask_from_RLE(left_lung_rle, height, width)
right_lung_mask = get_mask_from_RLE(right_lung_rle, height, width)
heart_mask = get_mask_from_RLE(heart_rle, height, width)

# Visualize the masks
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(left_lung_mask, cmap='gray')
ax[0].set_title('Left Lung')
ax[0].axis('off')

ax[1].imshow(right_lung_mask, cmap='gray')
ax[1].set_title('Right Lung')
ax[1].axis('off')

ax[2].imshow(heart_mask, cmap='gray')
ax[2].set_title('Heart')
ax[2].axis('off')

# %%
