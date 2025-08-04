import os
import ants
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from scipy.io import loadmat

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
os.environ["ANTS_RANDOM_SEED"] = "3"

# Base directory should point to your BIDS dataset 
base_dir = "/path/to/your/bids_dataset"
sub = "sub-01"  # BIDS-style subject ID, e.g., sub-01

anat_dir = os.path.join(base_dir, sub, "anat")
dwi_dir = os.path.join(base_dir, sub, "dwi")

t1_path = os.path.join(anat_dir, f"{sub}_T1w.nii.gz")
brain_extracted_prefix = os.path.join(base_dir, f"{sub}_t1")

# Run ANTs brain extraction if not already present
if not os.path.isfile(f"{brain_extracted_prefix}_BrainExtractionBrain.nii.gz"):
    os.system(
        f"antsBrainExtraction.sh -d 3 "
        f"-a {t1_path} "
        f"-e {base_dir}/Ants_templates/T_template0.nii.gz "
        f"-m {base_dir}/Ants_templates/T_template0_BrainCerebellumProbabilityMask.nii.gz "
        f"-f {base_dir}/Ants_templates/T_template0_BrainCerebellumRegistrationMask.nii.gz "
        f"-o {brain_extracted_prefix}_"
    )

# Load T1 brain and mask
t1_brain = ants.image_read(f"{brain_extracted_prefix}_BrainExtractionBrain.nii.gz")
t1_mask = ants.image_read(f"{brain_extracted_prefix}_BrainExtractionMask.nii.gz")

# K-means segmentation
mask = ants.get_mask(t1_brain)
segs = ants.kmeans_segmentation(t1_brain, k=3, kmask=mask)

for i, img in enumerate(segs["probabilityimages"]):
    ants.image_write(img, os.path.join(base_dir, f"{sub}_kmeans_seg_prob{i}.nii.gz"))

# Save white matter binary mask (thresholded)
wm_prob = os.path.join(base_dir, f"{sub}_kmeans_seg_prob2.nii.gz")
wm_mask = os.path.join(base_dir, f"{sub}_wm_bin_mask.nii.gz")
os.system(f"fslmaths {wm_prob} -thr 0.10 -bin {wm_mask}")

# Load QTI maps
param_names = ["FA", "miso", "msqanison", "ufa", "vison"]
param_images = {}

for name in param_names:
    img_path = os.path.join(dwi_dir, f"dtd_{name}.nii.gz")
    param_images[name] = ants.image_read(img_path)

# Register QTI maps to T1
for name, img in param_images.items():
    print(f"Registering {name}...")

    # Perform registration
    reg_out_prefix = os.path.join(base_dir, f"{sub}_{name}_to_t1")
    reg = ants.registration(fixed=t1_brain, moving=img, type_of_transform="SyN", outprefix=reg_out_prefix)

    warp = f"{reg_out_prefix}1Warp.nii.gz"
    affine = f"{reg_out_prefix}0GenericAffine.mat"
    param_file = os.path.join(dwi_dir, f"dtd_{name}.nii.gz")
    output_native = os.path.join(base_dir, f"{sub}_{name}_to_t1_brain.nii.gz")
    output_masked = os.path.join(base_dir, f"{sub}_{name}_wm_t1nativespace.nii.gz")

    os.system(
        f"antsApplyTransforms -d 3 -e 3 "
        f"-i {param_file} "
        f"-r {t1_brain.fn} "
        f"-t {warp} "
        f"-t {affine} "
        f"-o {output_native}"
    )

    os.system(f"fslmaths {output_native} -mul {wm_mask} {output_masked}")

    print(f"{name} registered and masked.")





