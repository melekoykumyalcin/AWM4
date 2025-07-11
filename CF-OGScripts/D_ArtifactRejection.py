#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:38:24 2020

@author: Cora
"""

# MEG Python script of Cora - Analysis of Context MEG
# Step C: Artifact identification (ECG and EOG) based on ICA
# interactive script - has to be executed manually

#%% Setup ####################################################################

# Import libraries
import os
import locale
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 8]
import numpy as np#, h5py
#import scipy
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import pandas as pd
from autoreject import get_rejection_threshold
import scipy.stats
from pynput.keyboard import Key, Controller
keyboard = Controller()

# Set paths
homeDir = '/home/headmodel/Schreibtisch/AWM4/'
os.chdir(homeDir) # change current directory

# Load in meta information
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs       = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
#Subs       = Subs[Subs != 3] #exclude sub 3 for no, need noise file
locale.setlocale(locale.LC_ALL, "en_US.utf8")

# create function for plotting
def show_images(images, rows = 3, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    rows (Default = 3): Number of rows in figure (number of columns is 
                        set to np.ceil(n_images/float(rows))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, np.ceil(n_images/float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    return fig

#%% Artifact rejection based on ICA ###########################################

# paths and file names
data_path   = homeDir +'/AWM4_data/raw/'
decim = 4

# Ask for subjects & session:
# (as interaction with the data is required, no loop is used here)
actSubj = Subs[int(input("Subject no [1-26]: "))-1] 
        
# load cut epochs
CombTrials = mne.read_epochs(str(homeDir)+'/processed/CutEpochs/CutData_VP'+str(actSubj)+'-epo.fif')
raw        = mne.io.read_raw_fif(str(homeDir)+'/processed/CutEpochs/CutData_VP'+str(actSubj)+'-raw.fif',preload=True)

# load ICA
ica   = mne.preprocessing.ica.read_ica(str(homeDir)+'/processed/ICAs/VP'+str(actSubj)+'_fastica-ica.fif')
picks = mne.pick_types(CombTrials.info, meg=True, eeg=True, stim=False, 
                    eog=False, ref_meg=False, include=[], exclude='bads')
title = 'Sources related to %s artifacts (red)'      
  
# generate ECG epochs use detection via phase statistics
n_max_ecg = 10
ecg_epochs = create_ecg_epochs(raw, ch_name = 'EEG059-3609',tmin=-.5, tmax=.5, picks=picks) # find ecg epochs based on raw data
ecg_epochs.decimate(decim).apply_baseline((None,None))
ecg_inds, scores = ica.find_bads_ecg(ecg_epochs)

# detect EOG by correlation
n_max_eog = 20
 
eog_epochs1 = create_eog_epochs(raw, tmin=-.2, tmax=.2,ch_name = 'EEG060-3609', picks=picks,l_freq=.5,h_freq=20) # changed from plug A (57) to D (60)
eog_epochs2 = create_eog_epochs(raw, tmin=-.2, tmax=.2,ch_name = 'EEG058-3609', picks=picks,l_freq=.5,h_freq=20)
eog_epochs1.decimate(decim).apply_baseline((None, None))
eog_epochs2.decimate(decim).apply_baseline((None, None))
#if eog_epochs is empty...
if not eog_epochs1:
    eog_inds1=[]
    scores_eog1=[]
else:
    eog_inds1, scores_eog1 = ica.find_bads_eog(eog_epochs1,ch_name = 'EEG060-3609')
if not eog_epochs2:
    eog_inds2=[]
    scores_eog2=[]
else:    
    eog_inds2, scores_eog2 = ica.find_bads_eog(eog_epochs2,ch_name = 'EEG058-3609')

raw_sources = ica.get_sources(raw)
raw_sources = raw_sources.get_data()
raw_ecg = raw.get_data(picks='EEG059-3609')
raw_eog1 = raw.get_data(picks='EEG060-3609')
raw_eog2 = raw.get_data(picks='EEG058-3609')
ecg_Rs = []
eog1_Rs = []
eog2_Rs = []
for sc in range(raw_sources.shape[0]):
    ecg_corr = scipy.stats.linregress(raw_sources[sc,:],raw_ecg)
    eog1_corr = scipy.stats.linregress(raw_sources[sc,:],raw_eog1)
    eog2_corr = scipy.stats.linregress(raw_sources[sc,:],raw_eog2)
    ecg_Rs.append(ecg_corr.rvalue)
    eog1_Rs.append(eog1_corr.rvalue)
    eog2_Rs.append(eog2_corr.rvalue)

#%% detect ecg components - do until you're happy with the result
ica.plot_scores(scores, exclude=ecg_inds, title=title % 'ecg', labels='ecg')
show_picks = np.abs(scores).argsort()[::-1][:n_max_ecg]
ica.plot_sources(raw, show_picks, title=title % 'ecg',stop=60)
ica.plot_components(show_picks, title=title % 'ecg', colorbar=True)
ica.plot_sources(ecg_epochs.average(), title='ECG average')
ica.plot_overlay(ecg_epochs.average(), exclude=ecg_inds, picks='mag')
ans = input("Do you agree? [y/n] ")
if ans != "y":
    ecg_inds = list(map(int, input("Please enter the ecg components (as: x y z): ").split())) 
plt.close('all')


#%% create overview figure
tmpfig = ica.plot_components(ecg_inds, colorbar=True)
plt.savefig(str(homeDir)+'/processed/tmp/tmp1.png')
tmpfig = ica.plot_sources(raw, ecg_inds,stop=60)
print('ADJUST Y AXIS MANUALLY BEFORE SAVING') #what should I do here? 
plt.savefig(str(homeDir)+'/processed/tmp/tmp2.png')
tmpfig = ica.plot_overlay(ecg_epochs.average(), exclude=ecg_inds, picks='mag')
plt.savefig(str(homeDir)+'/processed/tmp/tmp3.png')
image = []
titles = []
for im in range(3):
    image.append(plt.imread(str(homeDir)+'/processed/tmp/tmp' +str(im+1)+'.png'))
    if im == 0:
        titles.append('ECG components')
    elif im == 1:
        titles.append('ECG sources')
    else:
        titles.append('Exclusion result')    
plt.close('all')

 
#%% EOG no 1 (horizontal)
if eog_epochs1:
    ica.plot_scores(scores_eog1, exclude=eog_inds1, title=title % 'eog1', labels='eog1') 
    show_picks1 = np.abs(scores_eog1).argsort()[::-1][:n_max_eog]
    ica.plot_sources(raw, show_picks1, title=title % 'eog1')
    ica.plot_components(show_picks1, title=title % 'eog1', colorbar=True)
    ica.plot_sources(eog_epochs1.average(), title='EOG 1 average')
    ica.plot_overlay(eog_epochs1.average(), exclude=eog_inds1, picks='mag')

print("Suggested components: " +str(eog_inds1))
ans = input("Do you agree? [y/n] ")
if ans != "y":
    eog_inds1 = list(map(int, input("Please enter the eog1 components (as: x y z): ").split())) 
plt.close('all')

#%% create overview figure
if not eog_inds1:
    tmpfig = plt.plot()
else:
    tmpfig = ica.plot_components(eog_inds1, colorbar=True)
plt.savefig(str(homeDir)+'/processed/tmp/tmp4.png')
if not eog_inds1:
    tmpfig = plt.plot()
else:
    tmpfig = ica.plot_sources(raw, eog_inds1,stop=120)
print('ADJUST Y AXIS MANUALLY BEFORE SAVING') #what's happening here
plt.savefig(str(homeDir)+'/processed/tmp/tmp5.png')
if eog_epochs1: 
    tmpfig = ica.plot_overlay(eog_epochs1.average(), exclude=eog_inds1, picks='mag')
else:
    tmpfig = plt.plot()
plt.savefig(str(homeDir)+'/processed/tmp/tmp6.png')

for im in range(3):
    image.append(plt.imread(str(homeDir)+'/processed/tmp/tmp' +str(im+4)+'.png'))
    if im == 0:
        titles.append('EOG1 components')
    elif im == 1:
        titles.append('EOG1 sources')
    else:
        titles.append('Exclusion result')    
plt.close('all')

#%% EOG no 2 (vertical)
ica.plot_scores(scores_eog2, exclude=eog_inds2, title=title % 'eog2', labels='eog2')
show_picks2 = np.abs(scores_eog2).argsort()[::-1][:n_max_eog]
ica.plot_sources(raw, show_picks2, title=title % 'eog2')
ica.plot_components(show_picks2, title=title % 'eog2', colorbar=True)
ica.plot_sources(eog_epochs2.average(), title='EOG2 average')
ica.plot_overlay(eog_epochs2.average(), exclude=eog_inds2, picks='mag')

print("Suggested components: " +str(eog_inds2))
ans = input("Do you agree? [y/n] ")
if ans != "y":
    eog_inds2 = list(map(int, input("Please enter the eog2 components (as: x y z): ").split())) 
plt.close('all')

#%% create overview figure
tmpfig = ica.plot_components(eog_inds2, colorbar=True)
plt.savefig(str(homeDir)+'/processed/tmp/tmp7.png')
tmpfig = ica.plot_sources(raw, eog_inds2,stop=120)
print('ADJUST Y AXIS MANUALLY BEFORE SAVING')
plt.savefig(str(homeDir)+'/processed/tmp/tmp8.png')
tmpfig = ica.plot_overlay(eog_epochs2.average(), exclude=eog_inds2, picks='mag')
plt.savefig(str(homeDir)+'/processed/tmp/tmp9.png')

for im in range(3):
    image.append(plt.imread(str(homeDir)+'/processed/tmp/tmp' +str(im+7)+'.png'))
    if im == 0:
        titles.append('EOG2 components')
    elif im == 1:
        titles.append('EOG2 sources')
    else:
        titles.append('Exclusion result')    
#sumfig = show_images(image, rows = 1, titles = titles)
plt.close('all')

sumfig = show_images(image, rows = 3, titles = titles)
plt.savefig(str(homeDir)+'/processed/ICAs/ICA_Exclusions_VP'+str(actSubj)+'.pdf')
compfig = ica.plot_components(range(20), title='First 20 Components', colorbar=True)
plt.savefig(str(homeDir)+'/processed/ICAs/ICA_Comps_VP'+str(actSubj)+'.pdf')
plt.close('all')

#%% Combine all exclusions

all_inds = list(np.unique((ecg_inds + eog_inds1 + eog_inds2)))
ica.exclude = all_inds
ica_raw = raw.copy()
ica.apply(ica_raw)
# compare original data to cleaned data:
#raw.plot()
#ica_raw.plot()
#ica_trials = CombTrials.copy()
#ica.apply(ica_trials)

#%% Save result
ica.save(str(homeDir)+'/processed/ICAs/CleanVP'+str(actSubj)+'-ica.fif')
plt.close('all')
