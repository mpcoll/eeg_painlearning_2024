'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
 # @ Date: 2023
 # @ Description:
 '''

import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from bids import BIDSLayout
from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp
from scipy import stats

###############################
# Parameters
###############################
inpath = 'source'
outpathall = 'derivatives'


# Get BIDS layout
layout = BIDSLayout(inpath)

# Load participants
part = pd.read_csv(opj(inpath, 'participants.tsv'), sep='\t')
# Exlcude participants
part = part[part['excluded'] == 0]['participant_id'].tolist()

# Outpath for analysis
outpath = opj(outpathall, 'statistics/erps_modelfree')
if not os.path.exists(outpath):
    os.makedirs(outpath)

param = {
    # Njobs for permutations
    'njobs': 20,
    # Number of permutations
    'nperms': 5000,
    # Random state to get same permutations each time
    'random_state': 23,
    # Downsample to this frequency prior to analysis
    'testresampfreq': 1024,
    'cluster_threshold': 0.01,

}


###########################################################################
# Load and stack data
###########################################################################
# Epoched data
conditions = ['CS-1', 'CS-2', 'CS+', 'CS-E']

data = dict()
gavg = dict()
for cond in conditions:
    data[cond] = list()

for p in part:
    outdir = opj(outpathall,  p, 'eeg', 'erps')
    epo = mne.read_epochs(opj(outpathall,  p, 'eeg', 'erps',
                              p + '_task-fearcond_cues_singletrials-epo.fif')).drop_channels(['M1', 'M2']).set_montage('biosemi64')

    goodtrials = np.where(epo.metadata['badtrial'] == 0)[0]
    epo = epo[goodtrials]

    for cond in conditions:

        epo[epo.metadata['trial_cond4'] == cond]

        data[cond].append(epo[epo.metadata['trial_cond4'] == cond].average())

#####################################################################
# Statistics - T-test on the difference between CS+ vs CS-E and CS-1 vs CS-2
#####################################################################

anova_data = list()
# Loop conditions
for idxc, cond in enumerate(conditions):
    cond_data = []
    # Loop participants
    for idxp, p in enumerate(part):
        # Get data for this part
        pdat = data[cond][idxp].copy()

        # Resample if needed
        if pdat.info['sfreq'] != param['testresampfreq']:
            pdat = pdat.resample(param['testresampfreq'], npad='auto')
        # Append to list
        cond_data.append(np.swapaxes(pdat.data, axis1=1, axis2=0))
    anova_data.append(np.stack(cond_data))
# Put all in a single array
anova_data = np.stack(anova_data)


# # Take difference of interest for each part
csplusvscs1 = np.empty((1,) + anova_data.shape[1:])
csevscs2 = np.empty((1,) + anova_data.shape[1:])
csplusvscse = np.empty((1,) + anova_data.shape[1:])
csplusvscse2 = np.empty((1,) + anova_data.shape[1:])

# Calculate differences
for s in range(anova_data.shape[1]):
    csplusvscs1[0, s, ::] = (anova_data[2, s, :] - anova_data[0, s, :])
    csevscs2[0, s, ::] = (anova_data[3, s, :] - anova_data[1, s, :])
    csplusvscse[0, s, ::] = ((anova_data[2, s, :] - anova_data[3, s, :]) -
                             (anova_data[0, s, :] - anova_data[1, s, :]))

# REmove extra dimension
csplusvscs1 = np.squeeze(csplusvscs1)
csevscs2 = np.squeeze(csevscs2)
csplusvscse = np.squeeze(csplusvscse)


####################################################################
# Cluster based permutation test
####################################################################

# Get channels connectivity
connect, names = mne.channels.read_ch_adjacency('biosemi64')


if type(param['cluster_threshold']) is not dict:
    # Get cluster entering treshold
    p_thresh = param['cluster_threshold'] / 2  # two tailed
    param['cluster_threshold'] = -stats.t.ppf(p_thresh, len(part) - 1)


# Run cluster permutation test
tval, clusters, cluster_p_values, _ = perm1samp(csplusvscse,
                                                n_jobs=param["njobs"],
                                                threshold=param['cluster_threshold'],
                                                adjacency=connect,
                                                n_permutations=param['nperms'],
                                                buffer_size=None)

# Arrange pvals in a data array
pvals = np.ones_like(tval)
for c, p_val in zip(clusters, cluster_p_values):
    pvals[c] = p_val

# Save statistics
np.save(opj(outpath, 'csplusvscse_ttest_pvals.npy'), pvals)
np.save(opj(outpath, 'csplusvscse_ttest_tvals.npy'), tval)
np.save(opj(outpath, 'resamp_times.npy'), pdat.times)

# Same for other contrasts

tval, clusters, cluster_p_values, _ = perm1samp(csplusvscs1,
                                                n_jobs=param["njobs"],
                                                threshold=param['cluster_threshold'],
                                                adjacency=connect,
                                                n_permutations=param['nperms'],
                                                buffer_size=None)


pvals = np.ones_like(tval)
for c, p_val in zip(clusters, cluster_p_values):
    pvals[c] = p_val

np.save(opj(outpath, 'csplusvscs1_ttest_pvals.npy'), pvals)
np.save(opj(outpath, 'csplusvscs1_ttest_tvals.npy'), tval)


tval, clusters, cluster_p_values, _ = perm1samp(csevscs2,
                                                n_jobs=param["njobs"],
                                                threshold=param['cluster_threshold'],
                                                adjacency=connect,
                                                n_permutations=param['nperms'],
                                                buffer_size=None)


pvals = np.ones_like(tval)
for c, p_val in zip(clusters, cluster_p_values):
    pvals[c] = p_val

np.save(opj(outpath, 'csevscs2_ttest_pvals.npy'), pvals)
np.save(opj(outpath, 'csevscs2_ttest_tvals.npy'), tval)
