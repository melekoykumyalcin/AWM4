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
VP = np.unique([int(s[1:]) for s in metaInfo['SubjectCode']])

# Set up paths for results
results_path = str(homeDir) + '/AWM4_data/processed/timepoints'
stimuli_list = ['S1', 'S2']
feature_list = ['pitch', 'location']

# Prepare a list to store p-values for all time points
p_values = []
feature_stimuli_combinations = [f'{f}_{s}' for f in feature_list for s in stimuli_list]

subject_ids = np.arange(1, 24)  # Assuming 23 subjects
condition_names = feature_stimuli_combinations

behAcc_S1 = np.loadtxt(str(results_path)+'/mean_Accuracy_S1.txt', skiprows=1)
# get only the second column
behAcc_S1 = behAcc_S1[:, 1]
behAcc_S2 = np.loadtxt(str(results_path)+'/mean_Accuracy_S2.txt', skiprows=1)
# get only the second column
behAcc_S2 = behAcc_S2[:, 1]

# Loop through each feature and stimuli combination
for i, (feature, stimuli) in enumerate([(f, s) for f in feature_list for s in stimuli_list]):
    values_path = os.path.join(results_path, feature + stimuli, 'slidingWindow')
    
    # Check if it's a location feature
    if feature == 'location':
        timescores_filename = os.path.join(values_path, f'all_location{stimuli}_mean_timescores_SVM.txt')
    else:
        timescores_filename = os.path.join(values_path, f'all_encoding{stimuli}_mean_timescores_SVM.txt')
    
    # Read the .txt files
    all_mean_acrossalltimes_timescores = np.loadtxt(timescores_filename)
    mean_scores = all_mean_acrossalltimes_timescores[:, 0:50]  # Taking the first 50 columns
    mean_scores = mean_scores*100

    # Loop over each feature and stimuli combination and store all_mean_timescores for each combination
    if feature == 'location' and stimuli == 'S1':
        location_S1 = mean_scores
    elif feature == 'pitch' and stimuli == 'S1':
        pitch_S1 = mean_scores
    elif feature == 'location' and stimuli == 'S2':
        location_S2 = mean_scores
    elif feature == 'pitch' and stimuli == 'S2':
        pitch_S2 = mean_scores 

# Calculate the correlation between the behavioral accuracy and the mean scores
# for each feature and stimuli combination
# Initialize dictionaries to store correlation coefficients and p-values for each combination
correlation_dict = {}
p_value_dict = {}

# Loop over each feature and stimuli combination
for feature, stimuli in [(f, s) for f in feature_list for s in stimuli_list]:
    
    # Determine which data to use based on feature and stimuli
    if feature == 'location' and stimuli == 'S1':
        feature_data = location_S1  # shape: (n_samples, n_timepoints)
        beh_data = behAcc_S1
    elif feature == 'pitch' and stimuli == 'S1':
        feature_data = pitch_S1  # shape: (n_samples, n_timepoints)
        beh_data = behAcc_S1
    elif feature == 'location' and stimuli == 'S2':
        feature_data = location_S2  # shape: (n_samples, n_timepoints)
        beh_data = behAcc_S2
    elif feature == 'pitch' and stimuli == 'S2':
        feature_data = pitch_S2  # shape: (n_samples, n_timepoints)
        beh_data = behAcc_S2
    else:
        continue
    
    # Create key for the current feature-stimuli combination
    key = f"{feature}_{stimuli}"
    
    # Initialize lists to store correlation coefficients and p-values for this combination
    correlation_dict[key] = []
    p_value_dict[key] = []
    
    # For each timepoint, calculate correlation with behavioral accuracy
    n_timepoints = feature_data.shape[1]  # Assuming feature_data is (n_samples, n_timepoints)
    
    for timepoint in range(n_timepoints):
        # Extract data for this timepoint
        timepoint_data = feature_data[:, timepoint]
        
        # Calculate Pearson correlation between behavioral accuracy and this timepoint data
        corr, p = stats.pearsonr(beh_data, timepoint_data)
        
        # Append the results to the appropriate list for this feature-stimuli combination
        correlation_dict[key].append(corr)
        p_value_dict[key].append(p)


fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(correlation_dict['location_S1'])
axs[0, 0].set_title('behAcc_S1 and location_S1')
axs[0, 0].set_xlabel('Timepoints')
axs[0, 0].set_ylabel('Correlation Coefficient')

axs[0, 1].plot(correlation_dict['pitch_S1'])
axs[0, 1].set_title('behAcc_S1 and pitch_S1')
axs[0, 1].set_xlabel('Timepoints')
axs[0, 1].set_ylabel('Correlation Coefficient')

axs[1, 0].plot(correlation_dict['location_S2'])
axs[1, 0].set_title('behAcc_S2 and location_S2')
axs[1, 0].set_xlabel('Timepoints')
axs[1, 0].set_ylabel('Correlation Coefficient')

axs[1, 1].plot(correlation_dict['pitch_S2'])
axs[1, 1].set_title('behAcc_S2 and pitch_S2')
axs[1, 1].set_xlabel('Timepoints')
axs[1, 1].set_ylabel('Correlation Coefficient')

plt.tight_layout()
plt.savefig(os.path.join(results_path, 'thesisPlots/corr_subplots.pdf'))
plt.close()

import numpy as np
from scipy import stats

# Assuming you have 'behAcc' (behavioral accuracy) and 'feature_data' (e.g., location_S1, pitch_S1)
# corr_data contains the correlations you calculated for each timepoint

def cluster_permutation_test(beh_data, feature_data, n_permutations=1000, threshold=0.05):
    n_samples, n_timepoints = feature_data.shape
    real_corrs = np.zeros(n_timepoints)
    p_values = np.zeros(n_timepoints)
    
    # Calculate the real correlations
    for t in range(n_timepoints):
        real_corrs[t], p_values[t] = stats.pearsonr(beh_data, feature_data[:, t])
    
    # Apply the threshold to find "clusters" of timepoints
    significant_points = np.where(p_values < threshold)[0]
    clusters = np.split(significant_points, np.where(np.diff(significant_points) != 1)[0] + 1)
    
    # Compute the real cluster statistic (e.g., sum of correlation coefficients in a cluster)
    real_cluster_stats = np.array([np.sum(real_corrs[cluster]) for cluster in clusters])
    
    # Now, perform the permutations
    max_cluster_stats = np.zeros(n_permutations)
    
    for perm in range(n_permutations):
        # Randomly shuffle the behavioral accuracy
        permuted_behAcc = np.random.permutation(beh_data)
        
        # Calculate correlations for the permuted data
        perm_corrs = np.zeros(n_timepoints)
        for t in range(n_timepoints):
            perm_corrs[t], _ = stats.pearsonr(permuted_behAcc, feature_data[:, t])
        
        # Find clusters in the permuted data
        perm_significant_points = np.where(np.abs(perm_corrs) > np.percentile(np.abs(perm_corrs), 95))[0]
        perm_clusters = np.split(perm_significant_points, np.where(np.diff(perm_significant_points) != 1)[0] + 1)
        
        if perm_clusters:  # Only compute cluster stats if there are clusters
            perm_cluster_stats = np.array([np.sum(perm_corrs[cluster]) for cluster in perm_clusters])
            max_cluster_stats[perm] = np.max(perm_cluster_stats)  # Record the largest cluster
    
    # Now, compare the real cluster stats to the permuted cluster stats
    real_cluster_p_values = np.array([np.mean(max_cluster_stats >= cluster_stat) for cluster_stat in real_cluster_stats])
    
    return real_corrs, real_cluster_stats, real_cluster_p_values, max_cluster_stats

# Run the cluster-based permutation test for each feature-stimuli combination
n_permutations = 1000
threshold = 0.05
cluster_stats = {}
cluster_p_values = {}

for feature, stimuli in [(f, s) for f in feature_list for s in stimuli_list]:
    if feature == 'location' and stimuli == 'S1':
        beh_data = behAcc_S1
        feature_data = location_S1
    elif feature == 'pitch' and stimuli == 'S1':
        beh_data = behAcc_S1
        feature_data = pitch_S1
    elif feature == 'location' and stimuli == 'S2':
        beh_data = behAcc_S2
        feature_data = location_S2
    elif feature == 'pitch' and stimuli == 'S2':
        beh_data = behAcc_S2
        feature_data = pitch_S2
    else:
        continue
    
    real_corrs, real_cluster_stats, real_cluster_p_values, max_cluster_stats = cluster_permutation_test(beh_data, feature_data, n_permutations, threshold)
    
    key = f"{feature}_{stimuli}"
    cluster_stats[key] = real_cluster_stats
    cluster_p_values[key] = real_cluster_p_values

    # save the results
    np.savetxt(os.path.join(results_path, f'cluster_stats_{feature}_{stimuli}.txt'), real_cluster_stats)
    np.savetxt(os.path.join(results_path, f'cluster_p_values_{feature}_{stimuli}.txt'), real_cluster_p_values)
    