import pymc as pm
import numpy as np
import pandas as pd
import os
import glob 
from scipy.stats import zscore


# Find all CSV files in the folder and subfolders
csv_files = glob.glob('/path/to/*voxel_data.csv', recursive=True)
csv_files = [file for file in csv_files if not any(sub_id in file for sub_id in exclude_subject_ids)]
age_sex_file = '/path/to/age_sex.csv'
age_sex = pd.read_csv(age_sex_file) 
age_sex['sex_binary'] = age_sex['sex'].map({'M': 1, 'F': 0})

# Concatenate all dataframes
df = pd.concat( 
    map(pd.read_csv, csv_files))

df = df[df['tdm_id'] != -1]

# Step 1: Count the number of rows per subject for each (roi_id, tdm_id) combination
count_df = df.groupby(['sub_id', 'roi_id', 'tdm_id']).size().reset_index(name='count')

# Step 2: Filter combinations where each subject has at least 10 rows
# First, filter counts to keep only combinations with counts >= 10
count_df = count_df[count_df['count'] >= 10]

# Step 3: Identify combinations where all subjects meet the criterion
# Count the number of unique subjects per (roi_id, tdm_id) combination
subjects_per_combination = count_df.groupby(['roi_id', 'tdm_id'])['sub_id'].nunique().reset_index(name='subject_count')

# Get total number of subjects
total_subjects = df['sub_id'].nunique()

# Keep combinations where subject_count equals total_subjects
valid_combinations = subjects_per_combination[subjects_per_combination['subject_count'] == total_subjects][['roi_id', 'tdm_id']]

# Step 4: Merge with the original df to keep only the valid combinations
df_filtered = pd.merge(df, valid_combinations, on=['roi_id', 'tdm_id'], how='inner')

# Step 5: Re-apply the per-subject row count filter
# Ensure each subject has at least 10 rows for these combinations
count_df_filtered = df_filtered.groupby(['sub_id', 'roi_id', 'tdm_id']).size().reset_index(name='count')
valid_sub_combinations = count_df_filtered[count_df_filtered['count'] >= 10][['sub_id', 'roi_id', 'tdm_id']]

df_filtered = pd.merge(df_filtered, valid_sub_combinations, on=['sub_id', 'roi_id', 'tdm_id'], how='inner')

# Step 6: Calculate the median for each remaining combination
df_avg = df_filtered.groupby(['sub_id', 'roi_id', 'tdm_id', 'group_id'])[['FA', 'miso', 'msqanison', 'op', 'ufa', 'vison']].median().reset_index()

# Step 7: Merge with age and sex data
df_avg = pd.merge(df_avg, age_sex, on='sub_id', how='left')
df_avg = df_avg.sort_values(by=['roi_id','tdm_id','sub_id']).reset_index(drop=True)

#make idx
subs_per_param = df_avg.groupby(['roi_id', 'tdm_id'])['sub_id'].nunique().reset_index(name='sub_count')
idx = np.repeat(np.arange(len(subs_per_param)), subs_per_param['sub_count'])
n_param = len(np.unique(idx))

qti = 'ufa' #replace with appropriate parameter

X = zscore(df_avg[qti].values.astype(float))
y = df_avg['group_id'].values.astype(int)
age = zscore(df_avg['age'].values)
sex = df_avg['sex_binary'].values.astype(int)

with pm.Model() as logistic_model:
    B0 = pm.Normal('B0', mu=0, sigma=10,shape=n_param)
    Bx = pm.Normal('Bx', mu=0, sigma=10,shape=n_param)
    Bage = pm.Normal('Bage', mu=0, sigma=10,shape=n_param)
    Bsex = pm.Normal('Bsex', mu=0, sigma=10,shape=n_param)
    
    mu = B0[idx] + Bx[idx]*X + Bage[idx]*age + Bsex[idx]*sex
    
    p = pm.Deterministic('p', pm.math.sigmoid(mu))
    
    lf=pm.Bernoulli("lf", p, observed=y)
    
    # Sampling
    trace = pm.sample(5000, tune=5000)
    
import arviz as az

w = pm.waic(trace)

tracedf = az.extract(trace, var_names=["Bx"]).to_dataframe()
tracedf.to_csv(f'/path/to/{qti}/posteriors.csv')

import matplotlib.pyplot as plt
import seaborn as sns

# Extract samples for the parameter of interest
#bX_samples = trace.get_values('Bx', combine=True)
bX_samples = trace.posterior['Bx'].values

# Map parameter indices to roi_id and tdm_id
roi_names = pd.read_csv("/path/to/mask_labels_roicross_base.csv")

combinations = subs_per_param[['roi_id', 'tdm_id']].reset_index(drop=True)
combinations_names = pd.merge(combinations, roi_names, on=['roi_id', 'tdm_id'], how='inner')
combinations_names.to_csv("/path/to/roicross_labels.csv")

# Get unique rois and tdm levels
unique_rois = combinations['roi_id'].unique()
tdm_levels = combinations['tdm_id'].unique()

# Color palette for up to 3 tdm levels
palette = sns.color_palette("colorblind", n_colors=3)
tdm_colors = dict(zip(tdm_levels, palette[:len(tdm_levels)]))
 
# Set up plotting
fig, axes = plt.subplots(8, 8, figsize=(1.5*11.69, 1.5*8.27),sharex=True)
axes = axes.flatten()

for idx, roi in enumerate(unique_rois):
    ax = axes[idx]
    roi_params = combinations[combinations['roi_id'] == roi].index
    for i in roi_params:
        tdm = combinations.loc[i, 'tdm_id']
        samples = bX_samples[:, :, i].flatten()
        hdi = az.hdi(samples, hdi_prob=0.94)
        alpha = 1.0 if hdi[0] * hdi[1] > 0 else 0.3
        sns.kdeplot(samples, ax=ax, shade=True, color=tdm_colors[tdm], alpha=alpha, label=f'TDM {tdm}')
    ax.axvline(0, color='k', linestyle='-', alpha=1)
    ax.set_title(f'ROI {roi}', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    sns.despine(ax=ax)
    ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig('/path/to/'+qti+'_roicross_parameters_posterior.png', dpi=300, bbox_inches='tight')

# get values in CSV files

sigs=[]
region_beta = []
region_HDIs = []
region_sig_text = []
odds_ratio = []
odds_text = []
percent = []
brains = []
tdms = []

#Get beta values and odds ratios and 94% HDI 
for idx, roi in enumerate(unique_rois):
    roi_params = combinations[combinations['roi_id'] == roi].index
    brain = combinations_names.loc[i, 'mask_name']
    for i in roi_params:
        tdm = combinations.loc[i, 'tdm_id']
        tdms.append(tdm)
        brains.append(brain)
        samples = bX_samples[:, :, i].flatten()
        #Compute HDI for each ROI's posterior
        hdi = pm.hdi(samples, hdi_prob=0.94)
        hdi_text =  '['+str(round(hdi[0],3))+', '+str(round(hdi[1],3))+']'
        region_HDIs.append(hdi_text)
        
        beta = np.median(samples)
        region_beta.append(round(beta,3))
        
        if np.prod(hdi)>0:
            sigs.append(1)
            sig_text = '***'
            region_sig_text.append(sig_text)
        else:
            sigs.append(0)
            sig_text = 'ns'
            region_sig_text.append(sig_text)
        
        lb, ub = np.percentile(samples, 2.5), np.percentile(samples, 97.5)
        lb, ub = np.exp(lb), np.exp(ub)
        odds_med = np.median(np.exp(samples))
        per_effect = 100 * (odds_med - 1)
        text = f'P({lb:.3f} < Odds Ratio < {ub:.3f}) = 0.95'
        odds_text.append(text) 
        odds_ratio.append(odds_med)
        percent.append(per_effect)
        
region_df = pd.DataFrame({'cross_id':tdms, 'B1':region_beta, 'HDI':region_HDIs, 'Odds Ratio': odds_ratio, 'Percent Effect': percent, 'Credible':region_sig_text}, index=sorted(brains)
region_df.to_csv('/path/to/region_summ_roicross_' + qti + '.csv')

