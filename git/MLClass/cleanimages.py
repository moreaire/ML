
import SimpleITK as sitk
import os
import pydicom
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
# --- conda activate d2l --- 


# --- filepaths ---
patient_num = '099'

## Filepaths from p-drive for accessibility
mri_folder = "/config/workspace/projects/local/MLproject/data/ZZ.OM.MA.P099_ZZ.OM.MA.P099_MR_2016-10-12_122337_MRI.Brain.w.wo.Contrast_Align_n448__00000"
rtstruct_folder = "/config/workspace/projects/local/MLproject/data/ZZ.OM.MA.P099_ZZ.OM.MA.P099_RTst_2016-10-12_122337_MRI.Brain.w.wo.Contrast_mask_n1__00000"

# --- Load MRI series ---
reader = sitk.ImageSeriesReader()
series_IDs = reader.GetGDCMSeriesIDs(mri_folder)
series_file_names = reader.GetGDCMSeriesFileNames(mri_folder, series_IDs[0])
reader.SetFileNames(series_file_names)
mri_image = reader.Execute()

print("MRI shape (z, y, x):", sitk.GetArrayFromImage(mri_image).shape)
print("MRI spacing:", mri_image.GetSpacing())
print("MRI origin:", mri_image.GetOrigin())

# --- Load RTSTRUCT ---
rt_file = [os.path.join(rtstruct_folder, f) for f in os.listdir(rtstruct_folder) if f.endswith(".dcm")][0]
rtstruct = pydicom.dcmread(rt_file)

def rtstruct_to_mask(rtstruct, reference_image):
    mask_array = np.zeros(sitk.GetArrayFromImage(reference_image).shape, dtype=np.uint8)

    # pick first ROI
    contours = rtstruct.ROIContourSequence[0].ContourSequence

    spacing = reference_image.GetSpacing()
    origin = reference_image.GetOrigin()
    direction = np.array(reference_image.GetDirection()).reshape(3,3)

    for contour in contours:
        coords = np.array(contour.ContourData).reshape(-1, 3)
        z = coords[0, 2]

        slice_index = int(round((z - origin[2]) / spacing[2]))

        # convert physical coordinates to pixel coordinates
        ij = []
        for x, y, _ in coords:
            phys = np.array([x, y, z])
            voxel = np.linalg.inv(direction) @ (phys - origin)
            i = voxel[0] / spacing[0]
            j = voxel[1] / spacing[1]
            ij.append((i, j))

        img = Image.new('L', (mask_array.shape[2], mask_array.shape[1]), 0)
        ImageDraw.Draw(img).polygon(ij, outline=1, fill=1)
        mask_array[slice_index] = np.maximum(mask_array[slice_index], np.array(img))

    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(reference_image)
    return mask_image

mask_image = rtstruct_to_mask(rtstruct, mri_image)
mask_array = sitk.GetArrayFromImage(mask_image)

print("Mask shape (z, y, x):", mask_array.shape)

# --- Print Mask slices with Tumor Volume ---
contains_tumor = []
for slice_idx in range(mask_array.shape[0]):
    if np.any(mask_array[slice_idx] == 1):  # check if any voxel is 1
        contains_tumor.append(slice_idx)
print(f"Slices that contain tumor: {contains_tumor}")       

# --- Visualize slices with tumor ---
for x in contains_tumor: 
    slice_idx = x
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(sitk.GetArrayFromImage(mri_image)[slice_idx], cmap='gray')
    plt.title(f"MRI Slice {slice_idx}")
    plt.subplot(1,2,2)
    plt.imshow(mask_array[slice_idx], cmap='Reds', alpha=0.5)
    plt.title(f"Mask Slice {slice_idx}")
    plt.show()


# --- Select Slices ---
'''
Only want to keep tumor slices and equal number of non tumor slices for data to be balanced
randomize non-tumor slices

COPY THIS INTO TERMINAL TO DETUMRNINE MR CUTTOFF SLICES
MR_slice = 290
plt.figure(figsize=(12,6))
plt.imshow(sitk.GetArrayFromImage(mri_image)[MR_slice], cmap='gray')
plt.title(f"MRI Slice {MR_slice}")
plt.show()
'''

mri_array = sitk.GetArrayFromImage(mri_image)
mask_array = sitk.GetArrayFromImage(mask_image)
num_tumor_slices = len(contains_tumor)
num_mr_slices = mri_array.shape[0]
random_nontumor_slices = []
for x in range(num_tumor_slices):
    random_int = random.randint(50,280) #change per patient to only include slices with brain
    random_nontumor_slices.append(random_int)


# --- Save specific dicoms ---
output_mri = f"/config/workspace/projects/local/MLproject/data/ZZ.DS.OMMA.p{patient_num}_MRI_slices"
output_mask = f"/config/workspace/projects/local/MLproject/data/ZZ.DS.OMMA.p{patient_num}_Mask_slices"
os.makedirs(output_mri, exist_ok=True)
os.makedirs(output_mask, exist_ok=True)

mri_array = sitk.GetArrayFromImage(mri_image)
mask_array = sitk.GetArrayFromImage(mask_image)

selected_slices = sorted(list(set(contains_tumor + random_nontumor_slices)))

# --- Original spacing, origin, direction ---
spacing = mri_image.GetSpacing()
origin = mri_image.GetOrigin()
direction = mri_image.GetDirection()

# --- Loop over slices ---
for slice_idx in selected_slices:
    tumor_flag = 1 if slice_idx in contains_tumor else 0

    mri_slice = mri_array[slice_idx]
    mask_slice = mask_array[slice_idx]

    slice_mri_img = sitk.GetImageFromArray(mri_slice[np.newaxis, ...])
    slice_mask_img = sitk.GetImageFromArray(mask_slice[np.newaxis, ...])

    spacing = list(mri_image.GetSpacing())
    origin = list(mri_image.GetOrigin())
    direction = mri_image.GetDirection()

    origin[2] = origin[2] + slice_idx * spacing[2]

    for img in [slice_mri_img, slice_mask_img]:
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        img.SetDirection(direction)

    mri_filename = os.path.join(output_mri, f"patient{patient_num}_slice_{slice_idx:03d}_tumor_{tumor_flag}.dcm")
    mask_filename = os.path.join(output_mask, f"patient{patient_num}_slice_{slice_idx:03d}_tumor_{tumor_flag}.dcm")

    sitk.WriteImage(slice_mri_img, mri_filename)
    sitk.WriteImage(slice_mask_img, mask_filename)



"""
Export tumor slices (with mask)
Export matche dnumber of normal slices (with mask)
- randomize normal slices (probably subtract first 20 and last 20 so no blank slices)

Possibly new script:
- Import patients and their masks
- flip/mirror for every other slice of a patient
- randomize order (keep association/labeling consistent with mask)

"""