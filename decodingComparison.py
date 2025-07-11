import os
import locale
import pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.ioff()
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = [10, 8]
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

# Set paths
homeDir = '/media/headmodel/Elements/AWM4/'
os.chdir(homeDir)  # change current directory

# Load in meta information
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample == 1, 'Subject'])

locale.setlocale(locale.LC_ALL, "en_US.utf8")

# Paths and file names
data_path = homeDir + '/AWM4_data/raw/'
corrected_data = homeDir + '/AWM4_data/raw/correctTriggers/'
allFiles = metaInfo['MEG_Name']
noiseFiles = metaInfo['Noise_Measurement']
block = metaInfo['Block']

# Set up paths for results
results_path = str(homeDir) + '/AWM4_data/processed/timepoints'
stimuli_list = ['S1', 'S2']
feature_list = ['pitch', 'location']

# Prepare a list to store p-values for all time points
p_values = []
feature_stimuli_combinations = [f'{f}_{s}' for f in feature_list for s in stimuli_list]


# Loop through each feature and stimuli combination
for i, (feature, stimuli) in enumerate([(f, s) for f in feature_list for s in stimuli_list]):
    values_path = os.path.join(results_path, feature + stimuli, 'slidingWindow')
    
    # Check if it's a location feature
    if feature == 'location':
        timescores_filename = os.path.join(values_path, f'all_location{stimuli}_mean_timescores_SVM.txt')
    else:
        timescores_filename = os.path.join(values_path, f'all_encoding{stimuli}_mean_timescores_SVM.txt')
    
    # Read the .txt files
    all_mean_timescores = np.loadtxt(timescores_filename)
    mean_scores = all_mean_timescores[:, 0:50]  # Taking the first 50 columns
    
    # Loop over each feature and stimuli combination and store all_mean_timescores for each combination
    if feature == 'location' and stimuli == 'S1':
        location_S1 = mean_scores
    elif feature == 'pitch' and stimuli == 'S1':
        pitch_S1 = mean_scores
    elif feature == 'location' and stimuli == 'S2':
        location_S2 = mean_scores
    elif feature == 'pitch' and stimuli == 'S2':
        pitch_S2 = mean_scores 

subject_ids = np.arange(1, 24)  # Assuming 23 subjects
condition_names = feature_stimuli_combinations

# Calculate difference lines for pitchS1-locationS1 and pitchS2-locationS2
diffS1 = pitch_S1 - location_S1  # Shape (23, 50)
diffS2 = pitch_S2 - location_S2  # Shape (23, 50)

# Compare diffS1 for each subject at each timepoint compared to 0
p_values_diffS1 = []
for t in range(50):  # Loop over the 50 time points
    # Paired t-test comparing diffS1 to 0
    t_stat, p_value = stats.ttest_1samp(diffS1[:, t], 0)
    p_values_diffS1.append(p_value)


# run cluster-based permutation test for diffS1
# Set up parameters for the cluster-based permutation test
n_permutations = 1000
threshold = 0.05
cluster_stats = np.zeros((n_permutations, 50))
cluster_p_values = np.zeros(50)


# Bonferroni correction for multiple comparisons
_, p_values_diffS1_corrected, _, _ = multipletests(p_values_diffS1, method='bonferroni')

# Identify significant time points for diffS1
significant_timepoints_diffS1 = np.where(np.array(p_values_diffS1_corrected) < 0.05)[0]

# Save p-values and significant time points to a text file
output_txt_file = os.path.join(results_path, 'thesisPlots/decodingComparison_diffS1.txt')
with open(output_txt_file, 'w') as f:
    f.write("Timepoint\tP-Value\tCorrected P-Value\tSignificant\n")
    for t, (p, p_corr) in enumerate(zip(p_values_diffS1, p_values_diffS1_corrected)):
        sig = 'Yes' if p_corr < 0.05 else 'No'
        f.write(f"{t}\t{p:.6f}\t{p_corr:.6f}\t{sig}\n")

# Save p-values and significant time points to an Excel file
output_excel_file = os.path.join(results_path, 'thesisPlots/decodingComparison_diffS1.xlsx')
df_p_values_diffS1 = pd.DataFrame({
    'Timepoint': np.arange(50),
    'P-Value': p_values_diffS1,
    'Corrected P-Value': p_values_diffS1_corrected,
    'Significant': ['Yes' if p < 0.05 else 'No' for p in p_values_diffS1_corrected]
})
df_p_values_diffS1.to_excel(output_excel_file, index=False)

# Plot diffS1 and mark significant time points with vertical lines
fig, ax = plt.subplots()
ax.plot(np.linspace(0, 0.5, 50), diffS1.mean(axis=0), color='blue', label='Pitch S1 - Location S1')
ax.fill_between(np.linspace(0, 0.5, 50), diffS1.mean(axis=0) - stats.sem(diffS1, axis=0), 
                diffS1.mean(axis=0) + stats.sem(diffS1, axis=0), color='blue', alpha=0.2)
ax.axhline(0, color='black', linestyle='--')
for tp in significant_timepoints_diffS1:
    ax.axvline(x=tp * 0.01, color='red', linestyle='--')  # Timepoints in seconds (50 points over 0.5s)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Accuracy Difference')
ax.set_title('Difference Between Pitch and Location for S1')
plt.savefig(os.path.join(results_path, 'thesisPlots/diffS1.png'))
plt.close()

# Compare diffS2 for each subject at each timepoint compared to 0
p_values_diffS2 = []
for t in range(50):  # Loop over the 50 time points
    # Paired t-test comparing diffS2 to 0
    t_stat, p_value = stats.ttest_1samp(diffS2[:, t], 0)
    p_values_diffS2.append(p_value)

# Bonferroni correction for multiple comparisons
_, p_values_diffS2_corrected, _, _ = multipletests(p_values_diffS2, method='bonferroni')

# Identify significant time points for diffS2
significant_timepoints_diffS2 = np.where(np.array(p_values_diffS2_corrected) < 0.05)[0]

# Save p-values and significant time points to a text file
output_txt_file = os.path.join(results_path, 'thesisPlots/decodingComparison_diffS2.txt')
with open(output_txt_file, 'w') as f:
    f.write("Timepoint\tP-Value\tCorrected P-Value\tSignificant\n")
    for t, (p, p_corr) in enumerate(zip(p_values_diffS2, p_values_diffS2_corrected)):
        sig = 'Yes' if p_corr < 0.05 else 'No'
        f.write(f"{t}\t{p:.6f}\t{p_corr:.6f}\t{sig}\n")

# Save p-values and significant time points to an Excel file
output_excel_file = os.path.join(results_path, 'thesisPlots/decodingComparison_diffS2.xlsx')
df_p_values_diffS2 = pd.DataFrame({
    'Timepoint': np.arange(50),
    'P-Value': p_values_diffS2,
    'Corrected P-Value': p_values_diffS2_corrected,
    'Significant': ['Yes' if p < 0.05 else 'No' for p in p_values_diffS2_corrected]
})
df_p_values_diffS2.to_excel(output_excel_file, index=False)

# Plot diffS2 and mark significant time points with vertical lines
fig, ax = plt.subplots()
ax.plot(np.linspace(0, 0.5, 50), diffS2.mean(axis=0), color='blue', label='Pitch S2 - Location S2')
ax.fill_between(np.linspace(0, 0.5, 50), diffS2.mean(axis=0) - stats.sem(diffS2, axis=0), 
                diffS2.mean(axis=0) + stats.sem(diffS2, axis=0), color='blue', alpha=0.2)
ax.axhline(0, color='black', linestyle='--')
for tp in significant_timepoints_diffS2:
    ax.axvline(x=tp * 0.01, color='red', linestyle='--')  # Timepoints in seconds (50 points over 0.5s)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Accuracy Difference')
ax.set_title('Difference Between Pitch and Location for S2')
plt.savefig(os.path.join(results_path, 'thesisPlots/diffS2.png'))
plt.close()