#plotting 

import os
import locale
from tqdm import tqdm
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
matplotlib.use('Qt5Agg')
plt.ioff()
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = [10, 8]
import pandas as pd

colors = dict()
colors['pitch'] = '#1c686b'
colors['location'] = '#cb6a3e'

# Set paths
homeDir = '/media/headmodel/Elements/AWM4/'
os.chdir(homeDir) # change current directory

# Load in meta information
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs       = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
#Subs = Subs[20:21] #for testing purposes !!!
locale.setlocale(locale.LC_ALL, "en_US.utf8")

# paths and file names
data_path   = homeDir +'/AWM4_data/raw/'
corrected_data = homeDir +'/AWM4_data/raw/correctTriggers/'
allFiles    = metaInfo['MEG_Name']
noiseFiles  = metaInfo['Noise_Measurement']
block = metaInfo['Block']
# for the Subs[:7]
corrected_files = [f.split('.')[0] + '_correct_triggers.fif' for f in allFiles]
corrected_files_series = pd.Series(corrected_files)
# saving the results
results_path = str(homeDir)+'/AWM4_data/processed/timepoints'
stimuli_list = ['S1', 'S2']
feature_list = ['pitch', 'location']

timepoints = np.arange(0.0, 0.5, 0.01)
#timepoints = np.arange(0, 500, 1)


fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
axes = axes.flatten()

all_mean_timescores_dict = {}

# Loop through each feature and stimuli combination
for i, (feature, stimuli) in enumerate([(f, s) for f in feature_list for s in stimuli_list]):
    values_path = os.path.join(results_path, feature+stimuli, 'slidingWindow')
    
    # Check if it's a location feature
    if feature == 'location':
        timescores_filename = os.path.join(values_path, f'all_location{stimuli}_mean_timescores_SVM.txt')
        acrossalltimes_filename = os.path.join(values_path, f'all_location{stimuli}_mean_acrossalltimes_SVM.txt')
    else:  
        timescores_filename = os.path.join(values_path, f'all_encoding{stimuli}_mean_timescores_SVM.txt')
        acrossalltimes_filename = os.path.join(values_path, f'all_encoding{stimuli}_mean_acrossalltimes_SVM.txt')
    
    # Read the .txt files
    all_mean_timescores = np.loadtxt(timescores_filename)
    all_mean_timescores_dict[feature+stimuli] = all_mean_timescores
    all_mean_acrossalltimes = np.loadtxt(acrossalltimes_filename)
    #mean_scores = all_mean_timescores[:, 1:]
    #there are 95 columns, get from index 0 to 48 
    mean_scores = all_mean_timescores[:, 0:50]
    mean_scores = mean_scores*100
    # mean scores have 23 rows and 50 columns, average across the rows
    mean_scores = np.mean(mean_scores, axis=1)


    # Read the p_values.csv file
    p_values_path = os.path.join(values_path, 'p_values.csv')
    p_values_df = pd.read_csv(p_values_path)
    significant_points = p_values_df[p_values_df['p_values'] < 0.05].index.tolist()
    significant_points = [tp for tp in significant_points if tp < 50]
 
    # Plot the average accuracy over time
    mean_accuracy = np.mean(mean_scores, axis=0)
    std_accuracy = np.std(mean_scores, axis=0)

    #find the 95% confidence interval
    confidence = 0.95
    n = len(mean_scores)
    h = std_accuracy * 1.96 / np.sqrt(n)
    lower = mean_accuracy - h
    upper = mean_accuracy + h
    
    ax = axes[i]
    ax.plot(timepoints, mean_accuracy, color=colors[feature])
    ax.fill_between(timepoints, lower, upper, color=colors[feature], alpha=0.2)
    ax.set_xticks(np.arange(0.0, 0.6, 0.1))
    ax.set_xlim(0.0, 0.5)  
    # Plot significant time points
    # if significant_points:
    #     earliest_tp = timepoints[min(significant_points)]
    #     latest_tp = timepoints[max(significant_points)]
    #     ax.hlines(y=0.6, xmin=earliest_tp, xmax=latest_tp, color=colors[feature], linestyle='-', linewidth=2)
    ax.axhline(y=0.5, color='black', linestyle='--')
    ax.vlines(x=0.5, ymin=0.49, ymax=0.5, color='black', linestyle='-')
    # Labels and title for the subplot
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{feature.capitalize()} {stimuli}')

    #subtract 50 from mean scores 
    stats_mean = mean_scores - 0.5

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(stats_mean, n_permutations=1000000, threshold=None,tail=1,out_type = 'mask')  
    #save clusters and cluster_p_values into the same txt file as two columns with their feature+stimuli
    #make clusters into an array 
    clusters_converted = np.array([[cl.start, cl.stop] for cl in [c[0] for c in clusters]])
    # add clusters converted and cluster_p_values into the same array
    results = np.column_stack((clusters_converted, cluster_p_values))
    output_file = os.path.join(results_path, f'clusters_{feature}{stimuli}.txt')
    np.savetxt(output_file, results, fmt='%d %d %f', header="start stop p_value", comments='')

    # here you will check which clusters are greater than .05  
    sigPs = np.argwhere(cluster_p_values<.05)[:,0]

    sigTOIs = np.zeros((sigPs.shape[0],stats_mean.shape[1]),dtype=bool)
    significant_points = []
    for xy in range(sigPs.shape[0]):
        actInd = np.array(np.linspace(0,stats_mean.shape[1]-1,num=stats_mean.shape[1])[clusters[sigPs[xy]]],dtype=int)
        sigTOIs[xy,actInd] = True
        sigTimes = np.array(range(stats_mean.shape[1]))[sigTOIs[xy,:]]
        # add the significant points to the list
        significant_points.extend(sigTimes)
        ax.plot([timepoints[sigTimes[0]-1],timepoints[sigTimes[-1]]],[.6,.6], color=colors[feature], linestyle='-', linewidth=2)
        # make the mean accuracx line with a linewidth of 2 only for the significant points 
        ax.plot(timepoints, mean_accuracy, color=colors[feature], linewidth=2)

    # print done for (feature + stimuli)
    print(f'{feature} {stimuli} done')

plt.tight_layout()
plt.savefig(os.path.join(results_path, 'thesisPlots/decodingAccuracySubplots_cluster.pdf'))
plt.close()



# Latency & Amplitude analysis jackknife procedure Miller

# get all combinations of all_mean_timescores_dict.keys()
import itertools
import scipy

all_combinations = list(itertools.combinations(all_mean_timescores_dict.keys(), 2))

t_value_dict = {}
p_value_dict = {}
mean_laten_onset_diff_dict = {}
sem_latency_onset_diff_dict = {}
latency_onset_dict = {}




for tmp_comb in all_combinations:

    above_chance_all_mean_acrossalltimes_a = all_mean_timescores_dict[tmp_comb[0]] - 0.5
    #above_chance_all_mean_acrossalltimes_b = all_mean_timescores_dict[tmp_comb[1]] - 0.5

    mean_above_chance_all_mean_acrossalltimes_a = np.mean(above_chance_all_mean_acrossalltimes_a, axis=0)
    #mean_above_chance_all_mean_acrossalltimes_b = np.mean(above_chance_all_mean_acrossalltimes_b, axis=0)

    max_amplitude_a = np.max(mean_above_chance_all_mean_acrossalltimes_a[:50])
    half_max_a = max_amplitude_a / 2
    #max_amplitude_b = np.max(mean_above_chance_all_mean_acrossalltimes_b[:50])
    #half_max_b = max_amplitude_b / 2

    # Find the index where the amplitude first reaches 50% of the maximum
    index_a = np.where(mean_above_chance_all_mean_acrossalltimes_a >= half_max_a)[0][0]
    #index_b = np.where(mean_above_chance_all_mean_acrossalltimes_b >= half_max_b)[0][0]

    grand_average_onset_a = timepoints[index_a]
    #grand_average_onset_b = timepoints[index_b]

    #grand_average_onset_diff = grand_average_onset_a - grand_average_onset_b

    # Initialize an empty list to store the latency onset for each resample
    latency_onsets_a = []
    latency_onsets_b = []

    # Perform the jackknife procedure
    for i in range(len(above_chance_all_mean_acrossalltimes_a[:,:50])):
        # Create a new array that excludes the i-th observation
        resample_a = np.delete(above_chance_all_mean_acrossalltimes_a[:,:50], i, axis=0)
        #resample_b = np.delete(above_chance_all_mean_acrossalltimes_b[:,:50], i, axis=0)

        # mean across subjects
        resample_a = np.mean(resample_a, axis=0)
        #resample_b = np.mean(resample_b, axis=0)

        # Calculate the maximum amplitude and the 50% of this maximum
        max_amplitude_a = np.max(resample_a)
        half_max_a = max_amplitude_a / 2

        #max_amplitude_b = np.max(resample_b)
        #half_max_b = max_amplitude_b / 2

        # Find the index where the amplitude first reaches 50% of the maximum
        index_a = np.where(resample_a >= half_max_a)[0][0]
        #index_b = np.where(resample_b >= half_max_b)[0][0]

        # Get the corresponding time and add it to the list
        latency_onsets_a.append(timepoints[index_a])
        #latency_onsets_b.append(timepoints[index_b])
        latency_onset_dict[tmp_comb] = latency_onsets_a


    # take the difference between the two lists
    latency_onset_diff = np.array(latency_onsets_a) - np.array(latency_onsets_b)

    # get the standard error of the mean of latency onset
    x = 0
    J = np.mean(latency_onset_diff)
    for left_out_sub in range(len(latency_onset_diff)):
        x += (latency_onset_diff[left_out_sub] - J)**2  
    sem_latency_onset_diff = np.sqrt((len(latency_onset_diff) - 1) / len(latency_onset_diff) * x)
    
    mean_laten_onset_diff = np.mean(latency_onset_diff)

    t_value = abs(mean_laten_onset_diff / sem_latency_onset_diff)
    p_value = 1 - scipy.stats.t.cdf(t_value, len(latency_onset_diff) - 1)

    t_value_dict[tmp_comb] = t_value
    p_value_dict[tmp_comb] = p_value
    mean_laten_onset_diff_dict[tmp_comb] = mean_laten_onset_diff
    sem_latency_onset_diff_dict[tmp_comb] = sem_latency_onset_diff

    #save the results in a txt file within results_path
    with open(results_path+'/t_value.txt', 'w') as file:
        file.write(str(t_value_dict))

    with open(results_path+'/p_value.txt', 'w') as file:
        file.write(str(p_value_dict))

    with open(results_path+'/mean_laten_onset_diff.txt', 'w') as file:
        file.write(str(mean_laten_onset_diff_dict))

    with open(results_path+'/sem_latency_onset_diff.txt', 'w') as file:
        file.write(str(sem_latency_onset_diff_dict))
    all_combinations = list(itertools.combinations(all_mean_timescores_dict.keys(), 2))
    # name the columns of the txt file
    onset_results = np.column_stack((t_value_dict, p_value_dict, mean_laten_onset_diff_dict))
    # save the results in a xlxs file
    with open(results_path+'/onset_results.xlsx', 'w') as file:
        file.write(str(onset_results))


# to get subsampled values 
latency_onset_dict = {}  # Dictionary to store onset latencies for each feature + stimuli

# Iterate over all feature and stimuli pairs
for i, (feature, stimuli) in enumerate([(f, s) for f in feature_list for s in stimuli_list]):
    
    # Get the above-chance values for the current feature + stimuli combination
    above_chance_all_mean_acrossalltimes = all_mean_timescores_dict[feature+stimuli] - 0.5

    # Calculate the mean of above-chance values across all times
    mean_above_chance_all_mean_acrossalltimes = np.mean(above_chance_all_mean_acrossalltimes, axis=0)

    # Find the maximum amplitude and half-max value
    max_amplitude = np.max(mean_above_chance_all_mean_acrossalltimes[:50])
    half_max = max_amplitude / 2

    # Find the index where amplitude first reaches 50% of the maximum
    index = np.where(mean_above_chance_all_mean_acrossalltimes >= half_max)[0][0]

    # Get the onset time for the grand average
    grand_average_onset = timepoints[index]

    # Initialize a list to store onset latencies from the jackknife procedure
    latency_onsets = []

    # Perform jackknifing by leaving one subject out at a time
    for j in range(len(above_chance_all_mean_acrossalltimes[:,:50])):
        # Create a resample excluding the j-th observation
        resample = np.delete(above_chance_all_mean_acrossalltimes[:,:50], j, axis=0)

        # Compute the mean of the resample
        resample = np.mean(resample, axis=0)

        # Find the maximum amplitude and half-max for the resample
        max_amplitude = np.max(resample)
        half_max = max_amplitude / 2

        # Find the index where amplitude first reaches 50% of the maximum
        index = np.where(resample >= half_max)[0][0]

        # Get the corresponding time and add it to the list of onsets
        latency_onsets.append(timepoints[index])

    # Store the jackknifed onset latencies for this feature + stimuli combination
    latency_onset_dict[feature+stimuli] = latency_onsets
    # Save the results in a txt file
    with open(results_path+'/latency_onset_dict.txt', 'w') as file:
        file.write(str(latency_onset_dict))








# do the same for the peak amplitude instead of the latency onset

t_value_dict_peak = {}
p_value_dict_peak = {}
mean_peak_diff_dict = {}
sem_peak_diff_dict = {}

for tmp_comb in all_combinations:
    
        above_chance_all_mean_acrossalltimes_a = all_mean_timescores_dict[tmp_comb[0]] - 0.5
        above_chance_all_mean_acrossalltimes_b = all_mean_timescores_dict[tmp_comb[1]] - 0.5
    
        mean_above_chance_all_mean_acrossalltimes_a = np.mean(above_chance_all_mean_acrossalltimes_a, axis=0)
        mean_above_chance_all_mean_acrossalltimes_b = np.mean(above_chance_all_mean_acrossalltimes_b, axis=0)
    
        max_amplitude_a = np.max(mean_above_chance_all_mean_acrossalltimes_a[:50])
        max_amplitude_b = np.max(mean_above_chance_all_mean_acrossalltimes_b[:50])
    
        grand_average_peak_diff = max_amplitude_a - max_amplitude_b
    
        # Initialize an empty list to store the peak amplitude for each resample
        peak_amplitudes_a = []
        peak_amplitudes_b = []
    
        # Perform the jackknife procedure
        for i in range(len(above_chance_all_mean_acrossalltimes_a[:,:50])):
            # Create a new array that excludes the i-th observation
            resample_a = np.delete(above_chance_all_mean_acrossalltimes_a[:,:50], i, axis=0)
            resample_b = np.delete(above_chance_all_mean_acrossalltimes_b[:,:50], i, axis=0)
    
            # mean across subjects
            resample_a = np.mean(resample_a, axis=0)
            resample_b = np.mean(resample_b, axis=0)
    
            # Calculate the maximum amplitude
            max_amplitude_a = np.max(resample_a)
            max_amplitude_b = np.max(resample_b)
    
            # Get the corresponding time and add it to the list
            peak_amplitudes_a.append(max_amplitude_a)
            peak_amplitudes_b.append(max_amplitude_b)
    


        # take the difference between the two lists
        peak_diff = np.array(peak_amplitudes_a) - np.array(peak_amplitudes_b)
    
        # get the standard error of the mean of peak amplitude
    
        x = 0
        J = np.mean(peak_diff)
        for left_out_sub in range(len(peak_diff)):
            x += (peak_diff[left_out_sub] - J)**2  
        sem_peak_diff = np.sqrt((len(peak_diff) - 1) / len(peak_diff) * x)

        mean_peak_diff = np.mean(peak_diff)
        t_value_peak = abs(mean_peak_diff / sem_peak_diff)
        p_value_peak = 1 - scipy.stats.t.cdf(t_value_peak, len(peak_diff) - 1)

        t_value_dict_peak[tmp_comb] = t_value_peak
        p_value_dict_peak[tmp_comb] = p_value_peak
        mean_peak_diff_dict[tmp_comb] = mean_peak_diff
        sem_peak_diff_dict[tmp_comb] = sem_peak_diff

        #save the results in a txt file
        with open(results_path+'/t_value_peak.txt', 'w') as file:
            file.write(str(t_value_dict_peak))
        
        with open(results_path+'/p_value_peak.txt', 'w') as file:
            file.write(str(p_value_dict_peak))

        with open(results_path+'/mean_peak_diff.txt', 'w') as file:
            file.write(str(mean_peak_diff_dict))

        with open(results_path+'/sem_peak_diff.txt', 'w') as file:
            file.write(str(sem_peak_diff_dict))
        all_combinations = list(itertools.combinations(all_mean_timescores_dict.keys(), 2))
        # name the columns of the txt file
        peak_results = np.column_stack((t_value_dict_peak, p_value_dict_peak, mean_peak_diff_dict))
        # save the results in a xlxs file
        with open(results_path+'/peak_results.xlsx', 'w') as file:
            file.write(str(peak_results))

# subsampled data for pak amplitude
peak_amplitude_dict = {}

# Iterate over all feature and stimuli pairs
for i, (feature, stimuli) in enumerate([(f, s) for f in feature_list for s in stimuli_list]):
    
    # Get the above-chance values for the current feature + stimuli combination
    above_chance_all_mean_acrossalltimes = all_mean_timescores_dict[feature+stimuli] - 0.5

    # Calculate the mean of above-chance values across all times
    mean_above_chance_all_mean_acrossalltimes = np.mean(above_chance_all_mean_acrossalltimes, axis=0)

    # Find the maximum amplitude
    max_amplitude = np.max(mean_above_chance_all_mean_acrossalltimes[:50])

    # Initialize a list to store peak amplitudes from the jackknife procedure
    peak_amplitudes = []

    # Perform jackknifing by leaving one subject out at a time
    for j in range(len(above_chance_all_mean_acrossalltimes[:,:50])):
        # Create a resample excluding the j-th observation
        resample = np.delete(above_chance_all_mean_acrossalltimes[:,:50], j, axis=0)

        # Compute the mean of the resample
        resample = np.mean(resample, axis=0)

        # Find the peak amplitude for the resample
        resample_max_amplitude = (np.max(resample) + 0.5)  # Add 0.5 to get the actual amplitude

        # Append to the list of peak amplitudes
        peak_amplitudes.append(resample_max_amplitude)

    # Store the peak amplitudes for this feature + stimuli combination
    peak_amplitude_dict[feature+stimuli] = peak_amplitudes
    # Save the results in a txt file
    with open(results_path+'/peak_amplitude_dict.txt', 'w') as file:
        file.write(str(peak_amplitude_dict))
    # Save the results in a xlsx file
    with open(results_path+'/peak_amplitude_dict.xlsx', 'w') as file:
        file.write(str(peak_amplitude_dict))


# LAtency & Amplitude analysis for every condition separately

mean_onset_latency_dict = {}
mean_amplitude_peak_dict = {}

for feature in feature_list:
    for stimuli in ['S1','S2']:

        above_chance_all_mean_acrossalltimes = all_mean_timescores_dict[feature+stimuli] - 0.5
        mean_above_chance_all_mean_acrossalltimes = np.mean(above_chance_all_mean_acrossalltimes, axis=0)

        max_amplitude = np.max(mean_above_chance_all_mean_acrossalltimes[:50])
        # when you're halfing it, bring mroe precision to the number
        half_max = max_amplitude / 2 
        #half_max = max_amplitude/ 2

        # Find the index where the amplitude first reaches 50% of the maximum
        index = np.where(mean_above_chance_all_mean_acrossalltimes >= half_max)[0][0]
        grand_average_onset = timepoints[index]
        mean_onset_latency_dict[feature+stimuli] = grand_average_onset

        # do the same for the peak amplitude instead of the latency onset
        index = np.where(mean_above_chance_all_mean_acrossalltimes >= max_amplitude)[0][0]
        grand_average_peak = timepoints[index]
        mean_amplitude_peak_dict[feature+stimuli] = max_amplitude
        #save the results in a txt file
        with open(results_path+'/mean_onset_latency.txt', 'w') as file:
            file.write(str(mean_onset_latency_dict))
        
        with open(results_path+'/mean_amplitude_peak.txt', 'w') as file:
            file.write(str(mean_amplitude_peak_dict))


#%# ANOVA

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Assuming you have 'feature' and 'stimulus_position' as factors, and 'latency_onset_diff' or 'peak_diff' as the dependent variable.

# Create an empty list to store the data
data = []

# For each combination
for tmp_comb in all_combinations:
    
    # Perform the calculations for latency_onset_diff or peak_diff
    # This would be where you calculate `latency_onset_diff` or `peak_diff`
    
    latency_onset_diff = np.array(latency_onsets_a) - np.array(latency_onsets_b)
    peak_diff = np.array(peak_amplitudes_a) - np.array(peak_amplitudes_b)
    
    # Append the results along with the feature and stimulus position to the data list
    for i in range(len(latency_onset_diff)):
        data.append([tmp_comb[0], 'latency', latency_onset_diff[i]])

    for i in range(len(peak_diff)):
        data.append([tmp_comb[0], 'peak', peak_diff[i]])

# Convert the list to a DataFrame for ANOVA
df = pd.DataFrame(data, columns=['Feature', 'Stimulus_Position', 'Value'])

# Run the two-way ANOVA using statsmodels
model = ols('Value ~ C(Feature) + C(Stimulus_Position) + C(Feature):C(Stimulus_Position)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)
