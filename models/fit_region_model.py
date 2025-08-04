import pymc as pm
import numpy as np
import pandas as pd
import os
import glob 
from scipy.stats import zscore

# Find all CSV files in the folder and subfolders
csv_files = glob.glob('/path/to/directory/*voxel_data.csv', recursive=True)
csv_files = [file for file in csv_files if not any(sub_id in file for sub_id in exclude_subject_ids)]
age_sex_file = '/path/to/age_sex.csv'
age_sex = pd.read_csv(age_sex_file) 
age_sex['sex_binary'] = age_sex['sex'].map({'M': 1, 'F': 0})

# Concatenate all dataframes
df = pd.concat( 
    map(pd.read_csv, csv_files))

df = df[df['tdm_id'] != -1]

df_avg = df.groupby(['sub_id', 'roi_id', 'group_id'])[['FA', 'miso', 'msqanison', 'op', 'ufa', 'vison']].median().reset_index()
df_avg = pd.merge(df_avg, age_sex, on='sub_id', how='left')

df_avg = df_avg.sort_values(by=['roi_id', 'sub_id'])
idx = df_avg['roi_id'].values.astype(int)

n_param = len(df_avg.groupby('roi_id').count())
unique_rois = df_avg['roi_id'].unique()

qti = 'miso' #replace with parameter of choice

X = zscore(df_avg[qti].values).astype(float)
y = df_avg['group_id'].values.astype(int)
age = zscore(df_avg['age'].values)
sex = df_avg['sex_binary'].values.astype(int)

with pm.Model() as logistic_model:
    # Priors for coefficients
    B0 = pm.Normal('B0', mu=0, sigma=10,shape=n_param)
    Bx = pm.Normal('Bx', mu=0, sigma=10,shape=n_param)
    Bage = pm.Normal('Bage', mu=0, sigma=10,shape=n_param)
    Bsex = pm.Normal('Bsex', mu=0, sigma=10,shape=n_param)
    
    mu = B0[idx] + Bx[idx]*X + Bage[idx]*age + Bsex[idx]*sex
    
    p = pm.Deterministic('p', pm.math.sigmoid(mu))
    
    lf=pm.Bernoulli("lf", p, observed=y)
    
    # Sampling
    trace = pm.sample(5000, tune=5000)
    
w = pm.waic(trace)

import arviz as az

tracedf = az.extract(trace, var_names=["Bx"]).to_dataframe()
tracedf.to_csv(f'/path/to/{qti}/posteriors.csv')

#plot to look at data (not in Rizor et al. manuscript)

import matplotlib.pyplot as plt
import seaborn as sns

#Extract samples for each parameter
bX_samples = trace.posterior['Bx'].values

#Iterate over parameters and ROIs
fig, axes = plt.subplots(8, 8, figsize=(11.69, 8.27))
axes = axes.flatten()

for i, roi in enumerate(unique_rois):
    # Compute HDI for each ROI's posterior
    samples = bX_samples[:, :, i].flatten()
    hdi = pm.hdi(samples, hdi_prob=0.94)
    strong_alpha = 1.0 if np.prod(hdi) > 0 else 0.3  # High opacity if HDI does not contain zero
    
    #Plot distribution with adjusted alpha
    sns.kdeplot(samples, ax=axes[i], shade=True, color='blue', alpha=strong_alpha)
    median = np.median(samples)
    
   # Highlight median with a red dashed line
    axes[i].axvline(median, color='red', linestyle='--')
    axes[i].set_title(f'ROI_{i}', fontsize=8)
    axes[i].tick_params(axis='both', which='major', labelsize=6)
    sns.despine(ax=axes[i])
 
# Adjust layout for each parameter plot
plt.tight_layout()
plt.savefig('/path/to/'+qti+'_roi_parameters_posterior.png', dpi=300, bbox_inches='tight')
plt.show()

brains = ["Arcuate Fasciculus L", "Arcuate Fasciculus R", "Cingulum Frontal Parahippocampal L", "Cingulum Frontal Parahippocampal R", 
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
              
sigs=[]
region_beta = []
region_HDIs = []
region_sig_text = []
odds_ratio = []
odds_text = []
percent = []

#Get beta values and odds ratios and 94% HDI 
for i, roi in enumerate(unique_rois):
    samples = bX_samples[:, :, i].flatten()
    brain = brains[i]
    # Compute HDI for each ROI's posterior
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

region_df = pd.DataFrame({'B1':region_beta, 'HDI':region_HDIs, 'Odds Ratio': odds_ratio, 'Percent Effect': percent, 'Credible':region_sig_text}, index=brains)
region_df.to_csv('/path/to/region_summ_roi_' + qti + '.csv')


