import os
import glob
import pandas as pd
import nibabel as nib
import numpy as np
import ants

# Base BIDS directory and subject/group identifiers
bids_root = "/path/to/folder"
group = "controls"
sub = "sub-01"  # BIDS-compliant subject ID
group_idx = 0   # Replace with appropriate group ID depending on the subject

base_dir = os.path.join(bids_root, group)
anat_dir = os.path.join(base_dir, sub, "anat")
dwi_dir = os.path.join(base_dir, sub, "dwi")
roi_dir = os.path.join(base_dir, sub, "rois")
os.makedirs(roi_dir, exist_ok=True)

# Load T1 brain image and MNI template
t1_brain_path = os.path.join(anat_dir, f"{sub}_t1_BrainExtractionBrain.nii.gz")
t1_brain = ants.image_read(t1_brain_path)
template_path = os.path.join(bids_root, "MNI152_T1_2mm_brain.nii.gz")
template = ants.image_read(template_path)

# Perform registration (MNI → T1)
reg_prefix = os.path.join(roi_dir, "MNI_to_t1")
mnitot1 = ants.registration(fixed=t1_brain, moving=template, type_of_transform="SyN", outprefix=reg_prefix)
ants.image_write(mnitot1['warpedmovout'], f"{reg_prefix}.nii.gz")

# Transformation files
tran1 = f"{reg_prefix}1Warp.nii.gz"
tran2 = f"{reg_prefix}0GenericAffine.mat"

# ROI mask sources
mask_rois = sorted(glob.glob("/path/to/pitt_atlas_by_region/masks/thr_50/*mask*"))
tdm_rois = sorted(glob.glob("/path/to/volz_cross_masks/*TDM*"))

# Apply transformation and resample all ROI masks
def transform_and_resample(roi_list, suffix_len):
    for roi in roi_list:
        roi_name = os.path.basename(roi)[:-suffix_len]
        transformed_roi = os.path.join(roi_dir, f"{roi_name}_to_t1_brain.nii.gz")
        resampled_roi = os.path.join(roi_dir, f"{roi_name}_to_t1_brain_2mm.nii.gz")

        print(f"Transforming ROI mask: {roi_name}")
        os.system(f"antsApplyTransforms -d 3 -e 3 -i {roi} -r {t1_brain_path} -t {tran1} -t {tran2} -o {transformed_roi} -n NearestNeighbor")
        os.system(f"flirt -in {transformed_roi} -ref {transformed_roi} -out {resampled_roi} -applyisoxfm 2")
        os.system(f"fslmaths {resampled_roi} -thrP 50 -bin {resampled_roi}")

transform_and_resample(tdm_rois, suffix_len=7)
transform_and_resample(mask_rois, suffix_len=12)

# Load and resample diffusion parameter images
diffusion_param_names = ["FA", "miso", "msqanison", "ufa", "vison"]
diffusion_param_files = sorted([
    f for f in glob.glob(f"{base_dir}/{sub}_*wm_t1nativespace.nii.gz")
    if any(param in f for param in diffusion_param_names)
])

for param in diffusion_param_files:
    param_base = os.path.basename(param)[:-7]
    print(f"Resampling: {param_base}")
    resampled_param = os.path.join(base_dir, f"{param_base}_2mm.nii.gz")
    os.system(f"flirt -in {param} -ref {param} -out {resampled_param} -applyisoxfm 2")

# Load 2mm resampled masks
roi_masks = sorted(glob.glob(os.path.join(roi_dir, "*thr_50*2mm*")))
dtdm_masks = sorted(glob.glob(os.path.join(roi_dir, "*dTDM*2mm*")))
roi_masks_data = [nib.load(p).get_fdata() for p in roi_masks]
dtdm_masks_data = [nib.load(p).get_fdata() for p in dtdm_masks]

# Load diffusion maps
diffusion = {
    param: os.path.join(base_dir, f"{sub}_{param}_wm_t1nativespace_2mm.nii.gz")
    for param in diffusion_param_names
}
diffusion_data = {param: nib.load(path).get_fdata() for param, path in diffusion.items()}

# Extract voxelwise data
rows = []

for roi_idx, roi_data in enumerate(roi_masks_data):
    print(f"Processing ROI {roi_idx}")

    # Mask diffusion data
    masked = {param: diffusion_data[param] * roi_data for param in diffusion_param_names}
    nonzero_voxels = np.argwhere(masked["FA"] > 0)

    for i, j, k in nonzero_voxels:
        # Determine dTDM ID
        tdm_id = -1
        for idx, dtdm_data in enumerate(dtdm_masks_data):
            if dtdm_data[i, j, k] > 0:
                tdm_id = idx
                break

        # Extract values
        values = []
        for param in diffusion_param_names:
            val = diffusion_data[param][i, j, k]
            if param == "miso":
                val *= 1e10
            values.append(val)

        rows.append([sub, group_idx, roi_idx, tdm_id, [i, j, k]] + values)

# Save to CSV
df = pd.DataFrame(rows, columns=["subject_id", "group_id", "roi_id", "tdm_id", "voxel_coord"] + diffusion_param_names)
output_csv = os.path.join(base_dir, f"{sub}_voxel_data.csv")
df.to_csv(output_csv, index=False)
