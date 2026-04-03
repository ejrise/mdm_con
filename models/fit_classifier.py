import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

PARAMETERS   = ['FA','miso','msqanison', 'ufa','vison']
#COVARIATES   = []
COVARIATES   = ['age','sex_binary'] 
LABEL_COL    = 'group_id'
N_SPLITS     = 4
SEED         = 42

DATA_GLOB_PATTERN = '/Users/elizabethrizor/Desktop/mtbi/mtbi_code/*voxel_data.csv'
AGE_SEX_CSV       = '/Users/elizabethrizor/Desktop/mtbi/mtbi_code/age_sex.csv'

COLOR_FA    = "black"
COLOR_PARAM = "blue"

all_csvs = [f for f in glob.glob(DATA_GLOB_PATTERN, recursive=True)]
df_as = pd.read_csv(AGE_SEX_CSV)
df_as['sex_binary'] = df_as['sex'].map({'M':1, 'F':0})

df_raw = pd.concat(map(pd.read_csv, all_csvs), ignore_index=True)

## Kick out the -1s and 2s
df_raw = df_raw[(df_raw['tdm_id'] != -1) & (df_raw['tdm_id'] != 2)]

cnt = df_raw.groupby(['sub_id','roi_id','tdm_id']).size().reset_index(name='count')
cnt = cnt[cnt['count'] >= 10]
sc_ = cnt.groupby(['roi_id','tdm_id'])['sub_id'].nunique().reset_index(name='subject_count')
tot_sub = df_raw['sub_id'].nunique()
valid_pairs = sc_[sc_['subject_count']==tot_sub][['roi_id','tdm_id']]
df_m = pd.merge(df_raw, valid_pairs, on=['roi_id','tdm_id'], how='inner')

cnt2 = df_m.groupby(['sub_id','roi_id','tdm_id']).size().reset_index(name='count')
valid_sub = cnt2[cnt2['count']>=10][['sub_id','roi_id','tdm_id']]
df_m = pd.merge(df_m, valid_sub, on=['sub_id','roi_id','tdm_id'], how='inner')

df_m = pd.merge(df_m, df_as[['sub_id','age','sex_binary']], on='sub_id', how='left')
df_m = df_m.sort_values(['sub_id','roi_id','tdm_id']).reset_index(drop=True)
print(COVARIATES)

# Global aggregator
df_global = df_m.groupby(['sub_id','group_id'])[PARAMETERS].median().reset_index()
df_global = pd.merge(df_global, df_as[['sub_id','age','sex_binary']], on='sub_id', how='left')
for p in PARAMETERS:
    df_global.rename(columns={p: f"{p}_GLOBAL"}, inplace=True)

# Per-ROI aggregator
df_roi = df_m.groupby(['sub_id','roi_id','group_id'])[PARAMETERS].median().reset_index()
df_roi = pd.merge(df_roi, df_as[['sub_id','age','sex_binary']], on='sub_id', how='left')
df_roi_piv = df_roi.pivot_table(index=['sub_id','group_id','age','sex_binary'],
                                columns='roi_id', values=PARAMETERS, aggfunc='median')
df_roi_piv.columns = [f"{param}_roi{roi}_GLOBAL" for param, roi in df_roi_piv.columns]
df_roi_piv = df_roi_piv.reset_index()

# Per-TDM aggregator
df_tdm = df_m.groupby(['sub_id','tdm_id','group_id'])[PARAMETERS].median().reset_index()
df_tdm = pd.merge(df_tdm, df_as[['sub_id','age','sex_binary']], on='sub_id', how='left')
df_tdm_piv = df_tdm.pivot_table(index=['sub_id','group_id','age','sex_binary'],
                                columns='tdm_id', values=PARAMETERS, aggfunc='median')
df_tdm_piv.columns = [f"{param}_tdm{tdm}_GLOBAL" for param, tdm in df_tdm_piv.columns]
df_tdm_piv = df_tdm_piv.reset_index()

# Full ROI×TDM pivot
df_combo_piv = df_m.pivot_table(
    index=['sub_id','group_id','age','sex_binary'],
    columns=['roi_id','tdm_id'],
    values=PARAMETERS,
    aggfunc='median'
)
df_combo_piv.columns = [f"{param}_roi{roi}_tdm{tdm}" for param, roi, tdm in df_combo_piv.columns]
df_combo_piv = df_combo_piv.reset_index()

# Merge all aggregations
df_wide = df_global.merge(df_roi_piv, on=['sub_id','group_id','age','sex_binary'], how='outer')
df_wide = df_wide.merge(df_tdm_piv, on=['sub_id','group_id','age','sex_binary'], how='outer')
df_wide = df_wide.merge(df_combo_piv, on=['sub_id','group_id','age','sex_binary'], how='outer')
df_wide = df_wide.sort_values('sub_id').reset_index(drop=True)
print("df_wide shape:", df_wide.shape)

## Normalize
scaler = StandardScaler()
df_wide.iloc[:,2:] = scaler.fit_transform(df_wide.iloc[:, 2:])

def crossval_auc_for_column(df, col, covariates=COVARIATES, label=LABEL_COL, n_splits=N_SPLITS):
    needed = [col] + covariates + [label]
    data = df[needed].dropna()
    if data.empty:
        return np.nan, None, None
    Xs = data[[col] + covariates].astype(float).to_numpy()
    y = data[label].astype(int).to_numpy()

    #skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=10, random_state=SEED)

    all_preds = []
    all_trues = []
    for tr, te in skf.split(Xs, y):
        if len(np.unique(y[tr])) < 2:
            continue
        model = LogisticRegression(max_iter=300, solver='lbfgs')
        model.fit(Xs[tr], y[tr])
        probs = model.predict_proba(Xs[te])[:,1]
        all_preds.extend(probs)
        all_trues.extend(y[te])
    if len(np.unique(all_trues))<2:
        return np.nan, None, None
    fpr, tpr, _ = roc_curve(all_trues, all_preds)
    return auc(fpr, tpr), np.array(all_preds), np.array(all_trues)

def find_analogous_fa(col_name, results_list):
    pattern = re.sub(r'^([A-Za-z0-9]+)', 'FA', col_name)
    for r in results_list:
        if r[0] == pattern:
            return r
    return None

## ACTUAL CODE ### 
results = {}
for col in df_wide.columns:
    if col in ['sub_id','group_id','age','sex_binary']:
        continue
    A_val, preds_, trues_ = crossval_auc_for_column(df_wide, col)
    results[col] = ((col, A_val, preds_, trues_))


output_rows = []
# Loop over each non-FA column in results
for col_name, (col, auc_val, preds, trues) in ((k,v) for k,v in results.items() if not k.startswith("FA")):
    # Determine aggregator type based on column naming pattern
    if col_name.endswith("_GLOBAL") and "roi" not in col_name and "tdm" not in col_name:
        aggregator = "Global"
    elif "roi" in col_name and col_name.endswith("_GLOBAL"):
        aggregator = "Per-ROI"
    elif "tdm" in col_name and col_name.endswith("_GLOBAL"):
        aggregator = "Per-TDM"
    elif "roi" in col_name and "tdm" in col_name and not col_name.endswith("_GLOBAL"):
        aggregator = "ROI×TDM"
    else:
        aggregator = "Unknown"

    # Extract parameter from column name (first part before underscore)
    parameter = col_name.split("_")[0]

    # Initialize ROI and TDM indices as NaN
    roi_index = np.nan
    tdm_index = np.nan

    # Parse ROI and TDM indices based on aggregator type
    if aggregator == "Per-ROI":
        m = re.search(r'_roi(\d+)_', col_name)
        if m:
            roi_index = int(m.group(1))
    elif aggregator == "Per-TDM":
        m = re.search(r'_tdm(\d+)_', col_name)
        if m:
            tdm_index = int(m.group(1))
    elif aggregator == "ROI×TDM":
        m = re.search(r'_roi(\d+)_tdm(\d+)', col_name)
        if m:
            roi_index = int(m.group(1))
            tdm_index = int(m.group(2))

    # Find the analogous FA column entry for this feature
    fa_entry = find_analogous_fa(col_name, list(results.values()))
    if fa_entry is None:
        continue  # Skip if no analogous FA found

    # Extract FA AUC value
    fa_auc = fa_entry[1]

    # Append the row with collected info including ROI and TDM indices
    output_rows.append({
        "Parameter": parameter,
        "Aggregator": aggregator,
        "ROI index": roi_index,
        "TDM index": tdm_index,
        "AUC": auc_val,
        "FA_analogue AUC": fa_auc,
    })

# Create DataFrame from the collected rows and display it
table_df = pd.DataFrame(output_rows).sort_values(by=['Parameter','Aggregator']).reset_index(drop=True)
print(table_df)


print("\nAll done.")


from scipy.stats import ttest_rel

# Initialize new column for t-test p-values
table_df["ttest_pvalue"] = np.nan

for param in table_df["Parameter"].unique():
    # Per-ROI aggregator t-test
    group_mask = (table_df["Parameter"] == param) & (table_df["Aggregator"] == "Per-ROI")
    group_data = table_df[group_mask]
    if not group_data.empty:
        _, pval = ttest_rel(group_data["AUC"], group_data["FA_analogue AUC"], nan_policy='omit')
        table_df.loc[group_mask, "ttest_pvalue"] = pval

    # ROI×TDM aggregator t-tests for TDM = 0 and TDM = 1
    for tdm_val in [0, 1]:
        group_mask = (
            (table_df["Parameter"] == param) &
            (table_df["Aggregator"] == "ROI×TDM") &
            (table_df["TDM index"] == tdm_val)
        )
        group_data = table_df[group_mask]
        if not group_data.empty:
            _, pval = ttest_rel(group_data["AUC"], group_data["FA_analogue AUC"], nan_policy='omit')
            table_df.loc[group_mask, "ttest_pvalue"] = pval

print(table_df)
            
for param in PARAMETERS[1:]:
    print(f'\n{param}')
    for agg in ['Per-ROI', 'ROI×TDM']:
        if agg == 'ROI×TDM':
            for tdm_val in [0, 1]:
                # Filter data for specific aggregator, parameter, and TDM value
                group_mask = (
                    (table_df.Aggregator == agg) & 
                    (table_df.Parameter == param) & 
                    (table_df['TDM index'] == tdm_val)
                )
                group_data = table_df[group_mask]
                
                # Compute means
                mu1 = group_data["AUC"].mean()
                mu2 = group_data["FA_analogue AUC"].mean()
                
                # Retrieve p-value(s) for the group; handle cases with no data
                p_vals = group_data["ttest_pvalue"].unique()
                p_val = p_vals[0] if len(p_vals) > 0 else np.nan

                # Determine significance
                sig = '*' if not np.isnan(p_val) and p_val < 0.05 else ''
                print(f'Across ROIs(TDM={tdm_val}): AUC={mu1}, vs FA_AUC={mu2}, p={p_val}{sig}')
        else:  # For 'Per-ROI'
            group_mask = (table_df.Aggregator == agg) & (table_df.Parameter == param)
            group_data = table_df[group_mask]
            
            mu1 = group_data["AUC"].mean()
            mu2 = group_data["FA_analogue AUC"].mean()
            p_vals = group_data["ttest_pvalue"].unique()
            p_val = p_vals[0] if len(p_vals) > 0 else np.nan

            sig = '*' if not np.isnan(p_val) and p_val < 0.05 else ''
            print(f'Across ROIs(TDM collaps): AUC={mu1}, vs FA_AUC={mu2}, p={p_val}{sig}')


table_df.to_csv(f"classifier_results_{'_'.join(COVARIATES)}.csv", index=False)





