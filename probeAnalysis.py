#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Individual Condition ERPs Analysis
Show ERPs for all four probe conditions separately
"""

#%% Setup ####################################################################

import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from mne.stats import permutation_cluster_1samp_test

# Set paths
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
PROBE_EPOCHS_PATH = HOME_DIR + '/AWM4_data/processed/ProbeEpochs/'
OUTPUT_PATH = HOME_DIR + '/AWM4_data/processed/IndividualConditions/'

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(HOME_DIR)

# Load meta information
metaInfo = pd.read_excel(META_FILE)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
#Subs = Subs[21:]  # Use same subject selection

# Define all four conditions
conditions = {
    'Probe/SpNM/LoNM': 150,  # Voice Non-match, Location Non-match
    'Probe/SpNM/LoM': 151,   # Voice Non-match, Location Match  
    'Probe/SpM/LoNM': 160,   # Voice Match, Location Non-match
    'Probe/SpM/LoM': 161     # Voice Match, Location Match
}

condition_labels = {
    'Probe/SpNM/LoNM': 'Voice NoMatch + Location NoMatch',
    'Probe/SpNM/LoM': 'Voice NoMatch + Location Match',
    'Probe/SpM/LoNM': 'Voice Match + Location NoMatch',
    'Probe/SpM/LoM': 'Voice Match + Location Match'
}

print(f"Processing {len(Subs)} subjects for individual condition analysis...")

#%% Step 1: Load and compute individual subject ERPs for each condition

all_condition_evokeds = {cond: [] for cond in conditions.keys()}
all_subjects = []

for subject in Subs:
    try:
        print(f"\nProcessing subject {subject}...")
        
        # Load probe epochs
        probe_epochs = mne.read_epochs(PROBE_EPOCHS_PATH + f'ProbeEpochs_VP{subject}-epo-dropped.fif', verbose=False)
        
        # Print trial counts
        print(f"  Trial counts:")
        for cond in conditions.keys():
            count = len(probe_epochs[cond])
            print(f"    {cond}: {count}")
                    
        # Compute evoked responses for each condition
        for cond in conditions.keys():
            evoked = probe_epochs[cond].average()
            all_condition_evokeds[cond].append(evoked.copy())
        
        all_subjects.append(subject)
        print(f"  ✓ Processed subject {subject}")
        
    except Exception as e:
        print(f"  Error processing subject {subject}: {str(e)}")
        continue

print(f"\nSuccessfully processed {len(all_subjects)} subjects")

#%% Step 2: Compute grand averages for each condition

if len(all_subjects) > 0:
    # Compute grand averages
    grand_averages = {}
    for cond in conditions.keys():
        grand_averages[cond] = mne.grand_average(all_condition_evokeds[cond])
        print(f"Grand average for {cond}: {len(all_condition_evokeds[cond])} subjects")
    
    #%% Step 3: Comprehensive Visualization - Individual Conditions
    
    # 1. Butterfly plots for all four conditions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'orange', 'green', 'red']
    
    for i, (cond, label) in enumerate(condition_labels.items()):
        grand_averages[cond].plot(axes=axes[i], spatial_colors=True, show=False)
        axes[i].set_title(f'{label}\n(N={len(all_condition_evokeds[cond])} subjects)')
        axes[i].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
        axes[i].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[i].legend()
        
        # Add some stats
        data = grand_averages[cond].data
        peak_amp = np.max(np.abs(np.mean(data, axis=0)))
        peak_time = grand_averages[cond].times[np.argmax(np.abs(np.mean(data, axis=0)))]
        axes[i].text(0.02, 0.98, f'Peak: {peak_amp:.1e}T at {peak_time:.3f}s', 
                    transform=axes[i].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'individual_conditions_butterfly.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    #%% Step 4: ERP Comparison - All conditions on same plot
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. All conditions averaged across magnetometers
    times = grand_averages[list(conditions.keys())[0]].times
    
    for i, (cond, label) in enumerate(condition_labels.items()):
        erp_avg = np.mean(grand_averages[cond].data, axis=0)  # Average across magnetometers
        axes[0,0].plot(times, erp_avg, label=label, linewidth=2, color=colors[i])
    
    axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[0,0].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('ERP Amplitude (T)')
    axes[0,0].set_title('All Conditions - ERPs (Averaged Across Magnetometers)')
    axes[0,0].legend(fontsize=9)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Voice effect comparison (Match vs Non-match, split by location)
    voice_nm_lonm = np.mean(grand_averages['Probe/SpNM/LoNM'].data, axis=0)
    voice_nm_lom = np.mean(grand_averages['Probe/SpNM/LoM'].data, axis=0)
    voice_m_lonm = np.mean(grand_averages['Probe/SpM/LoNM'].data, axis=0)
    voice_m_lom = np.mean(grand_averages['Probe/SpM/LoM'].data, axis=0)
    
    axes[0,1].plot(times, voice_nm_lonm, label='Voice NoMatch + Loc NoMatch', linewidth=2, color='blue')
    axes[0,1].plot(times, voice_m_lonm, label='Voice Match + Loc NoMatch', linewidth=2, color='green')
    axes[0,1].plot(times, voice_nm_lom, label='Voice NoMatch + Loc Match', linewidth=2, color='orange', linestyle='--')
    axes[0,1].plot(times, voice_m_lom, label='Voice Match + Loc Match', linewidth=2, color='red', linestyle='--')
    
    axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0,1].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('ERP Amplitude (T)')
    axes[0,1].set_title('Voice Effect (solid=Loc NoMatch, dashed=Loc Match)')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Location effect comparison (Match vs Non-match, split by voice)
    axes[1,0].plot(times, voice_nm_lonm, label='Loc NoMatch + Voice NoMatch', linewidth=2, color='blue')
    axes[1,0].plot(times, voice_nm_lom, label='Loc Match + Voice NoMatch', linewidth=2, color='orange')
    axes[1,0].plot(times, voice_m_lonm, label='Loc NoMatch + Voice Match', linewidth=2, color='green', linestyle='--')
    axes[1,0].plot(times, voice_m_lom, label='Loc Match + Voice Match', linewidth=2, color='red', linestyle='--')
    
    axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1,0].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('ERP Amplitude (T)')
    axes[1,0].set_title('Location Effect (solid=Voice NoMatch, dashed=Voice Match)')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Difference waves
    voice_effect = (voice_m_lonm + voice_m_lom) / 2 - (voice_nm_lonm + voice_nm_lom) / 2
    location_effect = (voice_nm_lom + voice_m_lom) / 2 - (voice_nm_lonm + voice_m_lonm) / 2
    
    axes[1,1].plot(times, voice_effect, label='Voice Effect (Match - NoMatch)', linewidth=3, color='blue')
    axes[1,1].plot(times, location_effect, label='Location Effect (Match - NoMatch)', linewidth=3, color='orange')
    axes[1,1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1,1].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Difference Wave Amplitude (T)')
    axes[1,1].set_title('Difference Waves: Relevant vs Irrelevant Features')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'condition_comparisons.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    #%% Step 5: Topographical comparison at key time points
    
    key_times = [0.1, 0.2, 0.3, 0.5]  # Adjust based on what you see
    
    fig, axes = plt.subplots(len(conditions), len(key_times), figsize=(4*len(key_times), 4*len(conditions)))
    
    for i, (cond, label) in enumerate(condition_labels.items()):
        for j, time_point in enumerate(key_times):
            grand_averages[cond].plot_topomap(times=time_point, axes=axes[i,j], show=False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'topographical_all_conditions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    #%% Step 6: Peak Analysis and Statistics
    
    print("\n" + "="*60)
    print("PEAK ANALYSIS FOR EACH CONDITION")
    print("="*60)
    
    # Analyze peaks for each condition
    peak_analysis = {}
    
    for cond, label in condition_labels.items():
        erp_avg = np.mean(grand_averages[cond].data, axis=0)
        
        # Find positive and negative peaks
        pos_peak_idx = np.argmax(erp_avg)
        neg_peak_idx = np.argmin(erp_avg)
        
        pos_peak_amp = erp_avg[pos_peak_idx]
        neg_peak_amp = erp_avg[neg_peak_idx]
        pos_peak_time = times[pos_peak_idx]
        neg_peak_time = times[neg_peak_idx]
        
        peak_analysis[cond] = {
            'pos_peak_amp': pos_peak_amp,
            'pos_peak_time': pos_peak_time,
            'neg_peak_amp': neg_peak_amp,
            'neg_peak_time': neg_peak_time
        }
        
        print(f"\n{label}:")
        print(f"  Positive peak: {pos_peak_amp:.2e} T at {pos_peak_time:.3f}s")
        print(f"  Negative peak: {neg_peak_amp:.2e} T at {neg_peak_time:.3f}s")
    
    # Save individual condition evoked responses
    for cond in conditions.keys():
        grand_averages[cond].save(OUTPUT_PATH + f'{cond.replace("/", "_")}_grand_avg-ave.fif', overwrite=True)
    
    # Save peak analysis
    peak_df = pd.DataFrame(peak_analysis).T
    peak_df.to_csv(OUTPUT_PATH + 'peak_analysis.csv')
    
    print(f"\nAnalysis complete! Results saved to: {OUTPUT_PATH}")
    print(f"Individual condition grand averages saved as .fif files")
    print(f"Peak analysis saved as peak_analysis.csv")
    
else:
    print("No valid subjects processed!")


# %%
# Crop evoked data to start at -100 ms
for condition, evokeds in all_condition_evokeds.items():
    for i, evoked in enumerate(evokeds):
        evokeds[i] = evoked.copy().crop(tmin=-0.1)  # Crop to start at -100 ms

# Plot the evoked responses
mne.viz.plot_compare_evokeds(
    all_condition_evokeds,
    picks="mag",
    colors=dict(LoM=0, LoNM=1),  # Adjust colors as needed
    linestyles=dict(SpM="solid", SpNM="dashed"),  # Adjust linestyles as needed
    time_unit="ms",
)

# %%
regions = {
'frontal_left': [
'MLF11-3609', 'MLF12-3609', 'MLF13-3609', 'MLF14-3609',
'MLF21-3609', 'MLF22-3609', 'MLF23-3609', 'MLF24-3609', 'MLF25-3609',
'MLF31-3609', 'MLF32-3609', 'MLF33-3609', 'MLF34-3609', 'MLF35-3609',
'MLF41-3609', 'MLF42-3609', 'MLF43-3609', 'MLF44-3609', 'MLF45-3609', 'MLF46-3609',
'MLF51-3609', 'MLF52-3609', 'MLF53-3609', 'MLF54-3609', 'MLF55-3609', 'MLF56-3609',
'MLF61-3609', 'MLF62-3609', 'MLF63-3609', 'MLF64-3609', 'MLF65-3609', 'MLF66-3609', 'MLF67-3609'
],

'frontal_right': [
    'MRF11-3609', 'MRF12-3609', 'MRF13-3609', 'MRF14-3609',
    'MRF21-3609', 'MRF22-3609', 'MRF23-3609', 'MRF24-3609', 'MRF25-3609',
    'MRF31-3609', 'MRF32-3609', 'MRF33-3609', 'MRF34-3609', 'MRF35-3609',
    'MRF41-3609', 'MRF42-3609', 'MRF43-3609', 'MRF44-3609', 'MRF45-3609', 'MRF46-3609',
    'MRF51-3609', 'MRF52-3609', 'MRF53-3609', 'MRF54-3609', 'MRF55-3609', 'MRF56-3609',
    'MRF61-3609', 'MRF62-3609', 'MRF63-3609', 'MRF64-3609', 'MRF65-3609', 'MRF66-3609', 'MRF67-3609'
],

'frontal_midline': [
    'MZF01-3609', 'MZF02-3609', 'MZF03-3609'
],

'temporal_left': [
    'MLT11-3609', 'MLT12-3609', 'MLT13-3609', 'MLT14-3609', 'MLT15-3609', 'MLT16-3609',
    'MLT21-3609', 'MLT22-3609', 'MLT23-3609', 'MLT24-3609', 'MLT25-3609', 'MLT26-3609', 'MLT27-3609',
    'MLT31-3609', 'MLT32-3609', 'MLT33-3609', 'MLT34-3609', 'MLT35-3609', 'MLT36-3609', 'MLT37-3609',
    'MLT41-3609', 'MLT42-3609', 'MLT43-3609', 'MLT44-3609', 'MLT45-3609', 'MLT46-3609', 'MLT47-3609',
    'MLT51-3609', 'MLT52-3609', 'MLT53-3609', 'MLT54-3609', 'MLT55-3609', 'MLT56-3609', 'MLT57-3609'
],

'temporal_right': [
    'MRT11-3609', 'MRT12-3609', 'MRT13-3609', 'MRT14-3609', 'MRT15-3609', 'MRT16-3609',
    'MRT21-3609', 'MRT22-3609', 'MRT23-3609', 'MRT24-3609', 'MRT25-3609', 'MRT26-3609', 'MRT27-3609',
    'MRT31-3609', 'MRT32-3609', 'MRT33-3609', 'MRT34-3609', 'MRT35-3609', 'MRT36-3609', 'MRT37-3609',
    'MRT41-3609', 'MRT42-3609', 'MRT43-3609', 'MRT44-3609', 'MRT45-3609', 'MRT46-3609', 'MRT47-3609',
    'MRT51-3609', 'MRT52-3609', 'MRT53-3609', 'MRT54-3609', 'MRT55-3609', 'MRT56-3609', 'MRT57-3609'
],

'central_left': [
    'MLC11-3609', 'MLC12-3609', 'MLC13-3609', 'MLC14-3609', 'MLC15-3609', 'MLC16-3609', 'MLC17-3609',
    'MLC21-3609', 'MLC22-3609', 'MLC23-3609', 'MLC24-3609', 'MLC25-3609',
    'MLC31-3609', 'MLC32-3609', 'MLC41-3609', 'MLC42-3609',
    'MLC51-3609', 'MLC52-3609', 'MLC53-3609', 'MLC54-3609', 'MLC55-3609',
    'MLC61-3609', 'MLC62-3609', 'MLC63-3609'
],

'central_right': [
    'MRC11-3609', 'MRC12-3609', 'MRC14-3609', 'MRC15-3609', 'MRC16-3609', 'MRC17-3609',
    'MRC21-3609', 'MRC22-3609', 'MRC23-3609', 'MRC24-3609', 'MRC25-3609',
    'MRC31-3609', 'MRC32-3609', 'MRC41-3609', 'MRC42-3609',
    'MRC51-3609', 'MRC52-3609', 'MRC53-3609', 'MRC54-3609', 'MRC55-3609',
    'MRC61-3609', 'MRC62-3609', 'MRC63-3609'
],

'central_midline': [
    'MZC01-3609', 'MZC02-3609', 'MZC03-3609', 'MZC04-3609'
],

'parietal_left': [
    'MLP11-3609', 'MLP12-3609', 'MLP21-3609', 'MLP22-3609', 'MLP23-3609',
    'MLP31-3609', 'MLP32-3609', 'MLP33-3609', 'MLP34-3609', 'MLP35-3609',
    'MLP41-3609', 'MLP42-3609', 'MLP43-3609', 'MLP44-3609', 'MLP45-3609',
    'MLP51-3609', 'MLP52-3609', 'MLP53-3609', 'MLP54-3609', 'MLP55-3609', 'MLP56-3609', 'MLP57-3609'
],

'parietal_right': [
    'MRP11-3609', 'MRP12-3609', 'MRP21-3609', 'MRP22-3609', 'MRP23-3609',
    'MRP31-3609', 'MRP32-3609', 'MRP33-3609', 'MRP34-3609', 'MRP35-3609',
    'MRP41-3609', 'MRP42-3609', 'MRP43-3609', 'MRP44-3609', 'MRP45-3609',
    'MRP51-3609', 'MRP52-3609', 'MRP53-3609', 'MRP54-3609', 'MRP55-3609', 'MRP56-3609', 'MRP57-3609'
],

'parietal_midline': [
    'MZP01-3609'
],

'occipital_left': [
    'MLO11-3609', 'MLO12-3609', 'MLO13-3609', 'MLO14-3609',
    'MLO21-3609', 'MLO22-3609', 'MLO23-3609', 'MLO24-3609',
    'MLO31-3609', 'MLO32-3609', 'MLO33-3609', 'MLO34-3609',
    'MLO42-3609', 'MLO43-3609', 'MLO44-3609',
    'MLO51-3609', 'MLO52-3609', 'MLO53-3609'
],

'occipital_right': [
    'MRO11-3609', 'MRO12-3609', 'MRO13-3609', 'MRO14-3609',
    'MRO21-3609', 'MRO22-3609', 'MRO23-3609', 'MRO24-3609',
    'MRO31-3609', 'MRO32-3609', 'MRO33-3609', 'MRO34-3609',
    'MRO41-3609', 'MRO42-3609', 'MRO43-3609', 'MRO44-3609',
    'MRO51-3609', 'MRO52-3609', 'MRO53-3609'
],

'occipital_midline': [
    'MZO01-3609', 'MZO02-3609', 'MZO03-3609'
]

}

# %%
temporal_channels = regions['temporal_left'] + regions['temporal_right']

# %%
mne.viz.plot_compare_evokeds(
    all_condition_evokeds,
    picks=temporal_channels,
    colors=dict(LoM=0, LoNM=1),  # Adjust colors as needed
    linestyles=dict(SpM="solid", SpNM="dashed"),  # Adjust linestyles as needed
    time_unit="ms",
)


location_match = probe_epochs['LoM'].average() 
location_nonmatch = probe_epochs['LoNM'].average()
# %%
difference_wave = mne.combine_evoked([location_match, location_nonmatch], weights=[1, -1])


# %%
location_evoked = []
# %%
location_evoked.append(location_match.copy())
location_evoked.append(location_nonmatch.copy())

# %%
mne.viz.plot_compare_evokeds(
    location_evoked,
    picks=temporal_channels,
    colors=dict(LoM=0, LoNM=1),  # Adjust colors as needed
    #linestyles=dict(SpM="solid", SpNM="dashed"),  # Adjust linestyles as needed
    time_unit="ms",
)