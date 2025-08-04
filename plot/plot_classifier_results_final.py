import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PARAMETERS   = ['FA','miso','msqanison', 'ufa','vison']
Y_LIM = (0.4, 0.90) 

COLOR_FA    = "black"
COLOR_PARAM = "blue"

# --- ROI Label Processing ---
mask_labels = pd.read_csv('mask_labels.csv')
mask_labels.columns = mask_labels.columns.str.strip()

roi_label_map = mask_labels.drop_duplicates(subset=['roi_id']).set_index('roi_id')['mask_name'].to_dict()

def clean_label(name):
    if pd.isna(name): 
        return name
    name = name.replace("MNI_space_thr_50_", "")
    name = name.split("_mask")[0]
    return name

def abbreviate_label(name):
    parts = name.split('_')
    # Determine if the last part is a side indicator
    if parts[-1] in ['L', 'R']:
        side = parts[-1]
        main_parts = parts[:-1]
    else:
        side = ''
        main_parts = parts
    
    # Abbreviate the first part to 6 letters and remaining parts to 4 letters
    if main_parts:
        abbrev_main = [main_parts[0][:6]] + [part[:4] for part in main_parts[1:]]
    else:
        abbrev_main = []

    # Reassemble the label, preserving the side indicator if present
    if side:
        return '_'.join(abbrev_main) + '_' + side
    else:
        return '_'.join(abbrev_main)

roi_label_map = {roi: abbreviate_label(clean_label(name)) for roi, name in roi_label_map.items()}

table_df = pd.read_csv('/path/to/classifier/results')

p_threshold = 0.05 / 4

### Console print out summary ###
for param in PARAMETERS[1:]:
    print(f'\n{param}')
    for agg in ['Per-ROI', 'ROI×TDM']:
        if agg == 'ROI×TDM':
            for tdm_val in [0, 1]:
                group_mask = (
                    (table_df.Aggregator == agg) & 
                    (table_df.Parameter == param) & 
                    (table_df['TDM index'] == tdm_val)
                )
                group_data = table_df[group_mask]
                mu1 = group_data["AUC"].mean()
                mu2 = group_data["FA_analogue AUC"].mean()
                p_vals = group_data["ttest_pvalue"].unique()
                p_val = p_vals[0] if len(p_vals) > 0 else np.nan
                sig = '*' if not np.isnan(p_val) and p_val < p_threshold else ''
                print(f'Across ROIs(TDM={tdm_val}): AUC={mu1}, vs FA_AUC={mu2}, p={p_val}{sig}')
                boot_sigs = np.sum(group_data.boot_pvalue<.05)
                #print(f'TDM={tdm_val}, bootstrap sigs = {boot_sigs}')
        else:
            group_mask = (table_df.Aggregator == agg) & (table_df.Parameter == param)
            group_data = table_df[group_mask]
            mu1 = group_data["AUC"].mean()
            mu2 = group_data["FA_analogue AUC"].mean()
            min1 = group_data["AUC"].min()
            min2 = group_data["FA_analogue AUC"].min()
            max1 = group_data["AUC"].max()
            max2 = group_data["FA_analogue AUC"].max()
            p_vals = group_data["ttest_pvalue"].unique()
            p_val = p_vals[0] if len(p_vals) > 0 else np.nan
            sig = '*' if not np.isnan(p_val) and p_val < p_threshold else ''
            print(f'Across ROIs(TDM collaps): AUC={mu1}; range={min1}-{max1}, vs FA_AUC={mu2}; range={min2}-{max2}, p={p_val}{sig}')
            boot_sigs = np.sum(group_data.boot_pvalue<.05)
            #print(f'TDM collapsed, bootstrap sigs = {boot_sigs}')
            

### TTEST RESULTS BARPLOTS ###

# Adjusted p-value threshold for 12 comparisons (Bonferroni)

#T-test barplot for per-ROI only 

agg = 'Per-ROI' #other option is ROIxTDM

# Collect summary statistics 
group_mask = (table_df.Aggregator == agg)
group_data = table_df[group_mask]

params = ['miso','msqanison','ufa','vison']
plot_order = ['ufa', 'msqanison', 'miso', 'vison']

# Create a list of AUC value lists for each parameter
auc_data = [group_data[group_data['Parameter'] == param]['AUC'].values for param in params]
group_data = group_data[group_data['Parameter'].isin(params)].copy()

# Set up base plot
plt.figure(figsize=(12, 6))

# First draw split violins
df_long = pd.melt(
    group_data,
    id_vars=['Parameter'],
    value_vars=['AUC', 'FA_analogue AUC'],
    var_name='Type',
    value_name='Value'
)

sns.violinplot(
    data=df_long,
    x='Parameter',
    y='Value',
    hue='Type',
    split=True,
    inner=None,
    palette='Set2',
    cut=0,
    order=plot_order
)


# Swarm on top
sns.swarmplot(
    data=df_long,
    x='Parameter',
    y='Value',
    hue='Type',
    dodge=True,
    color='k',
    size=3,
    alpha=0.6,
    order=plot_order
)


# Clean up duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
my_labels = ['MDM', 'FA']

n = len(set(df_long['Type']))
plt.legend(handles[:n], my_labels, loc='upper left', fontsize=15)

# Labels and style
plt.ylabel('AUC', size=22, fontweight='bold')
plt.xlabel('MDM Parameter', size=22, fontweight='bold')
plt.yticks(fontsize=14)
plt.ylim(0.55, 0.8)

sns.despine(top=True, right=True)
plt.xticks(
    ticks=range(len(params)),
    labels=[r"$\mu FA$", r'$D_{anison}^{2}$', r'$D_{iso}$', r'$V_{ison}$'],
    size=18
)

# Threshold for significance
alpha = 0.05 / 4

# Add stars above violins
for i, param in enumerate(plot_order):
    # Get the p-value for this param
    p = group_data.loc[group_data['Parameter'] == param, 'ttest_pvalue'].values
    if len(p) > 0 and p[0] < alpha:
        # Get y max for positioning the star
        y_max = df_long[df_long['Parameter'] == param]['Value'].max()
        plt.text(i, y_max + 0.01, '*', ha='center', va='bottom', fontsize=18, color='black')
        
plt.tight_layout()
plt.savefig('/path/to/folder/', dpi=300, bbox_inches='tight')





