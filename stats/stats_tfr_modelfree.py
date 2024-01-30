# -*- coding: utf-8  -*-
"""
Author: michel-pierre.coll
Date: 2022-04-12
Description: TFR modelfree statistical analyses for pain conditioning task
"""
import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from bids import BIDSLayout
from functools import partial
from mne.stats import ttest_1samp_no_p
from mne.time_frequency import read_tfrs
import scipy
from mne.stats import permutation_cluster_1samp_test as perm1samp
from scipy import stats
from tqdm import tqdm

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

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpath for analysis
outpath = opj(outpathall, 'statistics/tfr_modelfree')
if not os.path.exists(outpath):
    os.makedirs(outpath)

param = {  # Njobs for permutations
    'njobs': -1,
    # Number of permutations
    'nperms': 5000,
    # Random state to get same permutations each time
    'random_state': 23,
    'baselinemode': 'logratio',
    'baselinetime': (-0.5, -0.2),
    'cluster_threshold': 0.01
}

###########################################################################
# Load and stack data
###########################################################################

conditions = ['CS-1', 'CS-2', 'CS+', 'CS-E', ]

part.sort()
data = dict()
for cond in conditions:
    data[cond] = []
gavg = dict()
for pidx, p in tqdm(enumerate(part)):
    print('Loading part ' + p)
    # Load trials
    sdata = read_tfrs(opj(outpathall, p,
                          'eeg', 'tfr',
                          p + '_task-fearcond_'
                          + 'epochs-tfr.h5'))[0].drop_channels(['M1', 'M2'])

    # Baseline correct
    sdata = sdata.apply_baseline(mode=param['baselinemode'],
                                 baseline=param['baselinetime'])

    # Remove bad trials
    goodtrials = np.where(sdata.metadata['badtrial'] == 0)[0]
    sdata = sdata[goodtrials]

    # Crop in ROI
    sdata = sdata.crop(tmin=-0.2, tmax=1, fmax=50)

    if pidx == 0:
        mock_data = sdata.copy()

    # Average for each condition
    for cond in conditions:
        # Average and append
        data[cond].append(
            sdata[sdata.metadata['trial_cond4'] == cond].average())
        data[cond][-1].save(opj(outpathall, p,
                                'eeg', 'tfr',
                                p + '_task-fearcond_' +
                                cond + '_avg-tfr.h5'), overwrite=True)

# Put part grand averages for all conditions in a single array
anova_data = []
for cond in conditions:
    pdat = []
    gavg[cond] = mne.grand_average(data[cond])
    gavg[cond].save(opj(outpath, 'task-fearcond_' +
                    cond + '_gavg-tfr.h5'), overwrite=True)
    for idx, p in enumerate(part):
        pdat.append(np.float32(data[cond][idx].data))

    anova_data.append(np.stack(pdat))
anova_data = np.stack(anova_data)
np.save(opj(outpath, 'anova_data.npy'), anova_data)


# Take difference of interest for each part
csplusvscs1 = np.empty((1,) + anova_data.shape[1:])
csevscs2 = np.empty((1,) + anova_data.shape[1:])
csplusvscse = np.empty((1,) + anova_data.shape[1:])

# Calculate differences
for s in range(anova_data.shape[1]):
    csplusvscs1[0, s, ::] = (anova_data[2, s, :] - anova_data[0, s, :])
    csevscs2[0, s, ::] = (anova_data[3, s, :] - anova_data[1, s, :])
    csplusvscse[0, s, ::] = ((anova_data[2, s, :] - anova_data[3, s, :])
                             - (anova_data[0, s, :] - anova_data[1, s, :]))

csplusvscs1 = np.squeeze(csplusvscs1)
csevscs2 = np.squeeze(csevscs2)
csplusvscse = np.squeeze(csplusvscse)
# Save for test
np.save(opj(outpath, 'csplusvscs1.npy'), anova_data)
np.save(opj(outpath, 'csevscs2.npy'), anova_data)
np.save(opj(outpath, 'csplusvscse.npy'), anova_data)


###############################################################
# Cluster to test interaction (difference of difference)
###############################################################

# Get cluster entering threshold
if type(param['cluster_threshold']) is not dict:
    # Get cluster entering treshold
    p_thresh = param['cluster_threshold'] / 2
    param['cluster_threshold'] = -stats.t.ppf(p_thresh, len(part) - 1)


# Find connectivity structure
mock_data.info.set_montage('biosemi64')
chan_connect, _ = mne.channels.find_ch_adjacency(mock_data.info,
                                                 'eeg')

# Create adjacency matrix for clustering in space x freq x time
adjacency = mne.stats.combine_adjacency(
    chan_connect, len(mock_data.freqs), len(mock_data.times))


assert adjacency.shape[0] == adjacency.shape[1] == \
    len(mock_data.ch_names) * len(mock_data.freqs) * len(mock_data.times)


for diff_data, savename in zip([csplusvscs1, csevscs2, csplusvscse],
                               ['csplusvscs1', 'csevscs2', 'csplusvscse']):

    tval, clusters, cluster_p_values, H0 = perm1samp(diff_data,
                                                     n_jobs=param['njobs'],
                                                     threshold=param['cluster_threshold'],
                                                     adjacency=adjacency,
                                                     n_permutations=param['nperms'])

    # Pval out as vector, reshape in chan x time x freq for plots
    # pval = pval.reshape(tval.shape)
    pvals = np.ones_like(tval)
    for c, p_val in zip(clusters, cluster_p_values):
        pvals[c] = p_val
    ###############################################################
    # Save for plots
    np.save(opj(outpath, 'cuesdiff_tfr_ttest_tvals' + savename + '.npy'),
            tval)
    np.save(opj(outpath, 'cuesdiff_tfr_ttest_pvals' + savename + '.npy'),
            pvals)
