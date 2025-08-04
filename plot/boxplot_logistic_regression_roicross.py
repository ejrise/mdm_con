import pymc as pm
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import glob

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

model = 'roicross'
combinations_names = pd.read_csv("/path/to/roicross_labels.csv")
#combinations_names = combinations_names[combinations_names['TDM ID'] != 2]
diffusion_params = ['miso', 'vison'] 
#diffusion_params = ['ufa', 'FA', 'msqanison']
#title_names= [r"$\mu FA$", r"$FA$", r'$D_{anison}^{2}$']
title_names= [r'$D_{iso}$', r'$V_{ison}$']
summary_dfs = {}


# Loop through each diffusion parameter to calculate summary statistics
for param in diffusion_params:
    summary_stats = []
    df = pd.read_csv(f'path/to/{model}_model/{param}/posteriors.csv')
    
    # Merge df with combinations_names to get TDM ID and filter as needed
    df = df.merge(combinations_names, left_on='Bx_dim_0', right_on=['Unnamed: 0'])

    # Calculate summary statistics for each ROI and TDM ID combination
    for (roi, tdm), group_df in df.groupby(['roi_id', 'tdm_id']):
        # Extract data
        roi_data = group_df['Bx'].to_numpy()
        
        # Calculate 94% HDI and IQR within that HDI
        hdi_94 = az.hdi(roi_data, hdi_prob=0.94)
        hdi_data = roi_data[(roi_data >= hdi_94[0]) & (roi_data <= hdi_94[1])]
        q1, q3 = np.percentile(hdi_data, [25, 75])
        
        # Determine if the HDI includes 0
        credible = (hdi_94[0] > 0) or (hdi_94[1] < 0)
        
        summary_stats.append({
            'ROI': roi,
            'TDM': tdm,
            'IQR_Lower': q1,
            'IQR_Upper': q3,
            'HDI_Lower': hdi_94[0],
            'HDI_Upper': hdi_94[1],
            'Credible': credible,
            'Median': np.round(np.median(roi_data),3),
            'Odds Ratio': np.median(np.exp(roi_data)),
            'HDI': '['+str(round(hdi_94[0],3))+', '+str(round(hdi_94[1],3))+']'
        })

    # Store summary DataFrame
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('results/region_summ_roicross_' + param + '.csv')
    summary_df['ROI_Label'] = [roi_brains[int(roi)] for roi in summary_df['ROI']]
    summary_df.to_csv('results/region_summ_roicross_' + param + '.csv')
    summary_dfs[param] = summary_df

# Create a 1x6 grid of subplots
fig, axes = plt.subplots(1, 4, figsize=(24, 18), sharey=True)

# Loop through each diffusion parameter, creating pairs of subplots
for idx, param in enumerate(diffusion_params):
    # Set up the two axes for TDM 0 and TDM 1 for this parameter
    ax_tdm0 = axes[2 * idx]       # Left panel for TDM 0
    ax_tdm1 = axes[2 * idx + 1]   # Right panel for TDM 1
    summary_df = summary_dfs[param]
    summary_df = summary_df[summary_df['TDM'] != 2]
    
    # Draw boxplots for TDM 0 and TDM 1 separately
    y_ticks = []
    y_tick_labels = []
    for i, (roi, roi_data) in enumerate(summary_df.groupby('ROI')):
        y_ticks.append(i)
        y_tick_labels.append(roi_brains[int(roi)])
        
        # Plot for each TDM ID
        for j, (_, row) in enumerate(roi_data.iterrows()):
            if row['TDM'] == 0:
                ax = ax_tdm0
                y_position = i - 0.2
            else:
                ax = ax_tdm1
                y_position = i + 0.2
            
            # Line color and width based on credibility
            line_width = 2 if row['Credible'] else 1
            line_color = 'black' if row['Credible'] else 'gray'
            
            # Plot HDI whiskers
            ax.plot([row['HDI_Lower'], row['HDI_Upper']], [y_position, y_position], color=line_color, lw=line_width, zorder=1)
            
            # Plot IQR
            
            if row['TDM'] == 0:
                ax.fill_betweenx(
                    y=[y_position - 0.1, y_position + 0.1],
                    x1=row['IQR_Lower'],
                    x2=row['IQR_Upper'],
                    color='cornflowerblue' if row['Credible'] else 'lightgray',
                    edgecolor='cornflowerblue' if row['Credible'] else 'lightgray',
                    linewidth=line_width,
                    zorder=2
                )
            else:
                ax.fill_betweenx(
                    y=[y_position - 0.1, y_position + 0.1],
                    x1=row['IQR_Lower'],
                    x2=row['IQR_Upper'],
                    color='salmon' if row['Credible'] else 'lightgray',
                    edgecolor='salmon' if row['Credible'] else 'lightgray',
                    linewidth=line_width,
                    zorder=2
                )
            
            # Plot median
            ax.scatter(row['Median'], y_position, color='black' if row['Credible'] else 'gray', zorder=3)

    # Add vertical line at x=0 and configure axis properties for each TDM panel
    for ax, tdm_label in zip([ax_tdm0, ax_tdm1], ["0-Cross", "1-Cross"]):
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        if idx == 0:
            ax.set_xlim(-8, 13)
        if idx == 1:
            ax.set_xlim(-8, 8)
        if idx == 2:
            ax.set_xlim(-8, 9)
        ax.set_ylim(-1, len(y_tick_labels)) 
        ax.set_xlabel(r'$\beta_{r,c}$', fontsize=14)
        ax.set_title(tdm_label, fontsize=16)
        if idx == 0:  # y-axis labels only for the first set of panels
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels, fontsize=14)

    # Add super title straddling the TDM 0 and TDM 1 panels for each parameter
    if idx == 0:
        fig.text(
        0.37, 0.92, title_names[0], ha='center', va='center', 
        fontsize=22, fontweight='bold')
    elif idx == 1:
        fig.text(
        0.79, 0.92, title_names[1], ha='center', va='center', 
        fontsize=22, fontweight='bold')        
    elif idx == 2:
        fig.text(
        0.86, 0.92, title_names[2], ha='center', va='center', 
        fontsize=22, fontweight='bold')        

# Adjust layout and display plot
plt.tight_layout(rect=[0, 0.03, 1, 0.9])
#plt.show()

plt.savefig('/path/to/size_regioncross_fig2.png', dpi=300, bbox_inches='tight')

