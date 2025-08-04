import pymc as pm
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az


roi_brains = ["Arcuate Fasciculus L", "Arcuate Fasciculus R", "Cingulum Frontal Parahippocampal L", "Cingulum Frontal Parahippocampal R", 
              "Cingulum Frontal Parietal L", "Cingulum Frontal Parietal R", "Cingulum Parahippocampal L", "Cingulum Parahippocampal Parietal L", 
              "Cingulum Parahippocampal Parietal R", "Cingulum Parahippocampal R", "Cingulum Rarolfactory L", "Cingulum Rarolfactory R", 
              "Corpus Callosum Body", "Corpus Callosum Forceps Major", "Corpus Callosum Forceps Minor", "Corpus Callosum Tapetum", 
              "Corticobulbar Tract L", "Corticobulbar Tract R", "Corticopontine Tract Frontal L", "Corticopontine Tract Frontal R", 
              "Corticopontine Tract Occipital L", "Corticopontine Tract Occipital R", "Corticopontine Tract Parietal L", 
              "Corticopontine Tract Parietal R", "Corticospinal Tract L", "Corticospinal Tract R", "Corticostriatal Tract Anterior L", 
              "Corticostriatal Tract Anterior R", "Corticostriatal Tract Posterior L", "Corticostriatal Tract Posterior R", 
              "Corticostriatal Tract Superior L", "Corticostriatal Tract Superior R", "Fornix L", "Fornix R", "Frontal Aslant Tract L", 
              "Frontal Aslant Tract R", "Inferior Fronto Occipital Fasciculus L", "Inferior Fronto Occipital Fasciculus R", 
              "Inferior Longitudinal Fasciculus L", "Inferior Longitudinal Fasciculus R", "Middle Longitudinal Fasciculus L", 
              "Middle Longitudinal Fasciculus R", "Optic Radiation L", "Optic Radiation R", "Parietal Aslant Tract L", "Parietal Aslant Tract R", 
              "Reticulospinal Tract L", "Reticulospinal Tract R", "Superior Longitudinal Fasciculus1 L", "Superior Longitudinal Fasciculus1 R", 
              "Superior Longitudinal Fasciculus2 L", "Superior Longitudinal Fasciculus2 R", "Superior Longitudinal Fasciculus3 L", 
              "Superior Longitudinal Fasciculus3 R", "Thalamic Radiation Anterior L", "Thalamic Radiation Anterior R", 
              "Thalamic Radiation Posterior L", "Thalamic Radiation Posterior R", "Thalamic Radiation Superior L", "Thalamic Radiation Superior R", 
              "Uncinate Fasciculus L", "Uncinate Fasciculus R", "Vertical Occipital Fasciculus L", "Vertical Occipital Fasciculus R"]

roi_abbr = [
    "A_Fasc_L", "A_Fasc_R",
    "Cing_F_PHip_L", "Cing_F_PHip_R",
    "Cing_FP_L", "Cing_FP_R",
    "Cing_PHip_L", "CingPHip_P_L", "Cing_PHip_P_R", "Cing_PHip_R",
    "Cing_R_L", "Cing_R_R",
    "CC_Body", "CC_FMaj", "CC_FMin", "CC_Tap",
    "CBT_L", "CBT_R",
    "CPT_F_L", "CPT_F_R", "CPT_O_L", "CPT_O_R", "CPT_P_L", "CPT_P_R",
    "CST_L", "CST_R",
    "CSTr_A_L", "CSTr_A_R", "CSTr_P_L", "CSTr_P_R", "CSTr_S_L", "CSTr_S_R",
    "Fornix_L", "Fornix_R",
    "FAT_L", "FAT_R",
    "IFO_Fasc_L", "IFO_Fasc_R",
    "IL_Fasc_L", "IL_Fasc_R",
    "ML_Fasc_L", "ML_Fasc_R",
    "O_Rad_L", "O_Rad_R",
    "PAT_L", "PAT_R",
    "RST_L", "RST_R",
    "SL_Fasc1_L", "SL_Fasc1_R",
    "SL_Fasc2_L", "SL_Fasc2_R",
    "SL_Fasc3_L", "SL_Fasc3_R",
    "T_Rad_A_L", "T_Rad_A_R",
    "T_Rad_P_L", "T_Rad_P_R",
    "T_Rad_S_L", "T_Rad_S_R",
    "U_Fasc_L", "U_Fasc_R",
    "VO_Fasc_L", "VO_Fasc_R"
]

###PLOTS FOR MDM SIZE PARAMETERS###

model = 'roi'
diffusion_params = ['miso', 'vison'] 
title_names= [r'$D_{iso}$', r'$V_{ison}$']
summary_dfs = {}

# Loop through each diffusion parameter to calculate summary statistics
for param in diffusion_params:
    summary_stats = []
    df = pd.read_csv('/path/to/'+model+'_model/'+param+'/posteriors.csv')
    # Filter the main DataFrame for the current diffusion parameter
    for roi in df['Bx_dim_0'].unique():
        # Extract the data for the current ROI and diffusion parameter
        roi_data = df[(df['Bx_dim_0'] == roi)]['Bx'].to_numpy()
        
        # Calculate 94% HDI and IQR within that HDI
        hdi_94 = az.hdi(roi_data, hdi_prob=0.94)
        hdi_data = roi_data[(roi_data >= hdi_94[0]) & (roi_data <= hdi_94[1])]
        q1, q3 = np.percentile(hdi_data, [25, 75])  # IQR within the 94% HDI
        
        # Determine if the HDI includes 0 (credible if HDI excludes 0)
        credible = (hdi_94[0] > 0) or (hdi_94[1] < 0)
        
        summary_stats.append({
            'ROI': roi,
            'Median':np.median(roi_data),
            'IQR_Lower': q1,
            'IQR_Upper': q3,
            'HDI_Lower': hdi_94[0],
            'HDI_Upper': hdi_94[1],
            'Credible': credible
        })
    
    # Convert summary statistics to a DataFrame for plotting and add to the dictionary
    summary_df = pd.DataFrame(summary_stats)
    summary_df['ROI_Label'] = [roi_abbr[int(roi)] for roi in summary_df['ROI']]
    summary_dfs[param] = summary_df  # Store the summary DataFrame for this parameter
    
    
def is_left_or_cc(label):
    return label.endswith('_L') or label.startswith('CC')

def is_right(label):
    return label.endswith('_R')

# Create LEFT hemisphere + CC plot
fig_left, axes_left = plt.subplots(1, 2, figsize=(18, 22), sharey=True)

for idx, param in enumerate(diffusion_params):
    ax = axes_left[idx]
    summary_df = summary_dfs[param]
    
    # Filter for left hemisphere and CC regions
    left_df = summary_df[summary_df['ROI_Label'].apply(is_left_or_cc)].copy()
    left_df['Sort_Label'] = left_df['ROI_Label'].str.removesuffix('_L')
    cc_rows = left_df[left_df['Sort_Label'].str.startswith('CC')]
    non_cc_rows = left_df[~left_df['Sort_Label'].str.startswith('CC')]
    left_df = pd.concat([non_cc_rows, cc_rows], ignore_index=True)
    left_df = left_df.iloc[::-1].reset_index(drop=True)
    
    for i, row in left_df.iterrows():
        line_width = 2 if row['Credible'] else 1
        line_color = 'black' if row['Credible'] else 'gray'

        ax.plot([row['HDI_Lower'], row['HDI_Upper']], [i, i], color=line_color, lw=line_width, zorder=1)
        ax.fill_betweenx(
            y=[i - 0.2, i + 0.2],
            x1=row['IQR_Lower'],
            x2=row['IQR_Upper'],
            color='hotpink' if row['Credible'] else 'lightgray',
            edgecolor='hotpink' if row['Credible'] else 'lightgray',
            linewidth=line_width,
            zorder=2
        )
        ax.scatter(row['Median'], i, color='black', zorder=3)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='x', labelsize=27)
    ax.set_xlim(-6, 6.1)
    ax.set_xlabel(r'$\beta_r$', fontsize=45)
    ax.set_title(f"{title_names[idx]}", fontsize=45)

    if idx == 0:
        ax.set_yticks(range(len(left_df)))
        ax.set_yticklabels(left_df['Sort_Label'], fontsize=32)
    ax.set_ylim(-1, len(left_df))

fig_left.suptitle("Left Hemisphere + Corpus Callosum", fontweight='bold', fontsize=45, x=0.58, y=0.95)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('//path/to/size_lefthem.tiff', dpi=300, bbox_inches='tight')


# Create RIGHT hemisphere plot
fig_right, axes_right = plt.subplots(1, 2, figsize=(18, 22), sharey=True)

for idx, param in enumerate(diffusion_params):
    ax = axes_right[idx]
    summary_df = summary_dfs[param]

    # Filter for right hemisphere regions
    right_df = summary_df[summary_df['ROI_Label'].apply(is_right)].copy()
    right_df['Sort_Label'] = right_df['ROI_Label'].str.removesuffix('_R')
    right_df = right_df.iloc[::-1].reset_index(drop=True)

    for i, row in right_df.iterrows():
        line_width = 2 if row['Credible'] else 1
        line_color = 'black' if row['Credible'] else 'gray'

        ax.plot([row['HDI_Lower'], row['HDI_Upper']], [i, i], color=line_color, lw=line_width, zorder=1)
        ax.fill_betweenx(
            y=[i - 0.2, i + 0.2],
            x1=row['IQR_Lower'],
            x2=row['IQR_Upper'],
            color='hotpink' if row['Credible'] else 'lightgray',
            edgecolor='hotpink' if row['Credible'] else 'lightgray',
            linewidth=line_width,
            zorder=2
        )
        ax.scatter(row['Median'], i, color='black', zorder=3)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='x', labelsize=27)
    ax.set_xlim(-6, 6.1)
    ax.set_xlabel(r'$\beta_r$', fontsize=45)
    ax.set_title(f"{title_names[idx]}", fontsize=45)

    if idx == 0:
        ax.set_yticks(range(len(right_df)))
        ax.set_yticklabels(right_df['Sort_Label'], fontsize=32)
    ax.set_ylim(-1, len(right_df))

fig_right.suptitle("Right Hemisphere", fontweight='bold', fontsize=45, x=0.58, y=0.95)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/path/to/size_righthem.tiff', dpi=300, bbox_inches='tight')



###PLOTS FOR MDM SHAPE PARAMETERS###

model = 'roi'
diffusion_params = ['ufa', 'msqanison', 'FA']
title_names= [r"$\mu FA$", r'$D_{anison}^{2}$', r"$FA$"]
summary_dfs = {}

# Loop through each diffusion parameter to calculate summary statistics
for param in diffusion_params:
    summary_stats = []
    df = pd.read_csv('bayes_out/'+model+'_model/'+param+'/posteriors.csv')
    # Filter the main DataFrame for the current diffusion parameter
    for roi in df['Bx_dim_0'].unique():
        # Extract the data for the current ROI and diffusion parameter
        roi_data = df[(df['Bx_dim_0'] == roi)]['Bx'].to_numpy()
        
        # Calculate 94% HDI and IQR within that HDI
        hdi_94 = az.hdi(roi_data, hdi_prob=0.94)
        hdi_data = roi_data[(roi_data >= hdi_94[0]) & (roi_data <= hdi_94[1])]
        q1, q3 = np.percentile(hdi_data, [25, 75])  # IQR within the 94% HDI
        
        # Determine if the HDI includes 0 (credible if HDI excludes 0)
        credible = (hdi_94[0] > 0) or (hdi_94[1] < 0)
        
        summary_stats.append({
            'ROI': roi,
            'Median':np.median(roi_data),
            'IQR_Lower': q1,
            'IQR_Upper': q3,
            'HDI_Lower': hdi_94[0],
            'HDI_Upper': hdi_94[1],
            'Credible': credible
        })
    
    # Convert summary statistics to a DataFrame for plotting and add to the dictionary
    summary_df = pd.DataFrame(summary_stats)
    summary_df['ROI_Label'] = [roi_abbr[int(roi)] for roi in summary_df['ROI']]
    summary_dfs[param] = summary_df  # Store the summary DataFrame for this parameter
    
def is_left_or_cc(label):
    return label.endswith('_L') or label.startswith('CC')

def is_right(label):
    return label.endswith('_R')

# Create LEFT hemisphere + CC plot
fig_left, axes_left = plt.subplots(1, 3, figsize=(22, 22), sharey=True)

for idx, param in enumerate(diffusion_params):
    ax = axes_left[idx]
    summary_df = summary_dfs[param]
    
    # Filter for left hemisphere and CC regions
    left_df = summary_df[summary_df['ROI_Label'].apply(is_left_or_cc)].copy()
    left_df['Sort_Label'] = left_df['ROI_Label'].str.removesuffix('_L')
    cc_rows = left_df[left_df['Sort_Label'].str.startswith('CC')]
    non_cc_rows = left_df[~left_df['Sort_Label'].str.startswith('CC')]
    left_df = pd.concat([non_cc_rows, cc_rows], ignore_index=True)
    left_df = left_df.iloc[::-1].reset_index(drop=True)
    
    for i, row in left_df.iterrows():
        line_width = 2 if row['Credible'] else 1
        line_color = 'black' if row['Credible'] else 'gray'

        ax.plot([row['HDI_Lower'], row['HDI_Upper']], [i, i], color=line_color, lw=line_width, zorder=1)
        ax.fill_betweenx(
            y=[i - 0.2, i + 0.2],
            x1=row['IQR_Lower'],
            x2=row['IQR_Upper'],
            color='hotpink' if row['Credible'] else 'lightgray',
            edgecolor='hotpink' if row['Credible'] else 'lightgray',
            linewidth=line_width,
            zorder=2
        )
        ax.scatter(row['Median'], i, color='black', zorder=3)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='x', labelsize=27)
    ax.set_xlim(-11, 11)
    ax.set_xlabel(r'$\beta_r$', fontsize=45)
    ax.set_title(f"{title_names[idx]}", fontsize=45)

    if idx == 0:
        ax.set_yticks(range(len(left_df)))
        ax.set_yticklabels(left_df['Sort_Label'], fontsize=32)
    ax.set_ylim(-1, len(left_df))

fig_left.suptitle("Left Hemisphere + Corpus Callosum", fontweight='bold', fontsize=45, x=0.55, y=0.95)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/path/to/shape_lefthem.tiff', dpi=300, bbox_inches='tight')


# Create RIGHT hemisphere plot
fig_right, axes_right = plt.subplots(1, 3, figsize=(22, 22), sharey=True)

for idx, param in enumerate(diffusion_params):
    ax = axes_right[idx]
    summary_df = summary_dfs[param]

    # Filter for right hemisphere regions
    right_df = summary_df[summary_df['ROI_Label'].apply(is_right)].copy()
    right_df['Sort_Label'] = right_df['ROI_Label'].str.removesuffix('_R')
    right_df = right_df.iloc[::-1].reset_index(drop=True)

    for i, row in right_df.iterrows():
        line_width = 2 if row['Credible'] else 1
        line_color = 'black' if row['Credible'] else 'gray'

        ax.plot([row['HDI_Lower'], row['HDI_Upper']], [i, i], color=line_color, lw=line_width, zorder=1)
        ax.fill_betweenx(
            y=[i - 0.2, i + 0.2],
            x1=row['IQR_Lower'],
            x2=row['IQR_Upper'],
            color='hotpink' if row['Credible'] else 'lightgray',
            edgecolor='hotpink' if row['Credible'] else 'lightgray',
            linewidth=line_width,
            zorder=2
        )
        ax.scatter(row['Median'], i, color='black', zorder=3)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='x', labelsize=27)
    ax.set_xlim(-11, 11)
    ax.set_xlabel(r'$\beta_r$', fontsize=45)
    ax.set_title(f"{title_names[idx]}", fontsize=45)

    if idx == 0:
        ax.set_yticks(range(len(right_df)))
        ax.set_yticklabels(right_df['Sort_Label'], fontsize=32)
    ax.set_ylim(-1, len(right_df))

fig_right.suptitle("Right Hemisphere", fontsize=45, fontweight='bold', x=0.55, y=0.95)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/path/to/shape_righthem.tiff', dpi=300, bbox_inches='tight')



