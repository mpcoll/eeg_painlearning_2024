# -*- coding: utf-8  -*-
"""
Author: michel-pierre.coll
Date: 2022-04-12
Description: TFR modelbased statistical analyses for pain conditioning task
"""
import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from bids import BIDSLayout
from mne.time_frequency import read_tfrs
import scipy
from mne.stats import permutation_cluster_1samp_test as clust_1s_ttest
from mne.decoding import Scaler
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
outpath = opj(outpathall, 'statistics/tfr_modelbased')
if not os.path.exists(outpath):
    os.makedirs(outpath)

param = {
    # Njobs for permutations
    'njobs': 15,
    # Alpha Threshold
    'alpha': 0.05,
    # Number of permutations
    'nperms': 5000,
    # Random state to get same permutations each time
    'random_state': 23,
    'ignoreshocks': True,
    'baselinemode': 'logratio',
    'baselinetime': (-0.5, -0.2),
    'cluster_threshold': 0.01
}


# ########################################################################
# Run linear models at the first level
###########################################################################
# Load model data
mod_data = pd.read_csv(opj(outpathall, 'task-fearcond_alldata.csv'))

regvars = ['vhat']
regvarsnames = ['Expectation']

# Loop participants and load single trials file
allbetasnp, all_epos = [], []
betas = [[] for i in range(len(regvars))]
part.sort()
for p in part:

    # Get model data for this part
    df = mod_data[mod_data['sub'] == p].reset_index(drop=True)

    # Load single epochs file (cotains one epoch/trial)
    epo = read_tfrs(opj(outpathall,  p, 'eeg', 'tfr',
                        p + '_task-fearcond_epochs-tfr.h5'))[0].drop_channels(['M1', 'M2'])

    # Removed shocked trials
    df = df[df['cond'] != 'CS++'].reset_index(drop=True)

    # drop bad trials
    goodtrials = np.where(epo.metadata['badtrial'] == 0)[0]
    df = df.iloc[goodtrials]
    epo = epo[goodtrials]

    # Baseline
    epo = epo.apply_baseline(mode=param['baselinemode'],
                             baseline=param['baselinetime'])
    # Crop to ROI
    epo.crop(tmin=-0.2, tmax=1, fmax=50)
    # Scale
    scale = Scaler(scalings='mean')  # Says mean but is z score, see docs

    # Consider time*frequency as a single vector for mass univariate
    tfdata = epo.data.reshape(epo.data.shape[0], epo.data.shape[1],
                              epo.data.shape[2]*epo.data.shape[3])

    tfdata_z = mne.EpochsArray(scale.fit_transform(tfdata),
                               info=epo.info)

    betasnp = []
    for idx, regvar in enumerate(regvars):
        # Standardize data
        df[regvar + '_z'] = scipy.stats.zscore(df[regvar])

        epo.metadata = df.assign(Intercept=1)  # Add an intercept

        # Perform regression
        names = ["Intercept"] + [regvar + '_z']
        res = mne.stats.linear_regression(tfdata_z, epo.metadata[names],
                                          names=names)
        # Reshape back to tfr
        betaout = mne.time_frequency.AverageTFR(info=epo.info,
                                                data=np.reshape(res[regvar
                                                                    + '_z'].beta.data,
                                                                epo.data.shape[1:]),
                                                times=epo.times,
                                                freqs=epo.freqs,
                                                nave=1)
        # Collect betas
        betas[idx].append(betaout)
        betasnp.append(betaout.data)
    allbetasnp.append(np.stack(betasnp))

# Stack all data
allbetas = np.stack(allbetasnp)

# Save for test
np.save(opj(outpath, 'ols_2ndlevel_allbetas.npy'), allbetas)
np.save(opj(outpath, 'resamp_times.npy'), epo.times)
np.save(opj(outpath, 'resamp_freqs.npy'), epo.freqs)

# # #########################################################################
# # Perform second level test on betas
# ##########################################################################
regvars = ['vhat']

allbetas = np.load(opj(outpath, 'ols_2ndlevel_allbetas.npy'))
epo = read_tfrs(opj(outpathall,  part[0], 'eeg', 'tfr',
                    part[0] + '_task-fearcond_epochs-tfr.h5'))[0]

epo.crop(tmin=-0.2, tmax=1, fmax=50)

# Find connectivity structure
epo = epo.drop_channels(['M1', 'M2'])
epo.info.set_montage('biosemi64')
chan_connect, _ = mne.channels.find_ch_adjacency(epo.info, 'eeg')

# Create adjacency matrix for clustering
adjacency = mne.stats.combine_adjacency(
    chan_connect, len(np.zeros(len(epo.freqs))), len(epo.times))

assert adjacency.shape[0] == adjacency.shape[1] == \
    len(epo.ch_names) * len(epo.freqs) * len(epo.times)


# Get cluster entering threshold
if type(param['cluster_threshold']) is not dict:
    # Get cluster entering treshold
    p_thresh = param['cluster_threshold'] / 2
    param['cluster_threshold_test'] = -stats.t.ppf(p_thresh, len(part) - 1)


tvals, pvalues = [], []
for idx, regvar in enumerate(regvars):

    # Test each predictor
    betas_cluster = allbetas[:, idx, ::]

    # Cluster test
    tval, clusters, cluster_p_values, _ = clust_1s_ttest(betas_cluster,
                                                         n_permutations=param['nperms'],
                                                         threshold=param['cluster_threshold_test'],
                                                         adjacency=adjacency,
                                                         n_jobs=param['njobs'],
                                                         seed=param['random_state'])

    pvals = np.ones_like(tval)
    for c, p_val in zip(clusters, cluster_p_values):
        pvals[c] = p_val

    tvals.append(tval)
    pvalues.append(pvals)

    np.save(opj(outpath, 'ols_2ndlevel_tfr_pvals_' + regvar + '_.npy'),
            pvalues[-1])
    np.save(opj(outpath, 'ols_2ndlevel_tfr_tvals_' + regvar + '_.npy'),
            tvals[-1])

tvals = np.stack(tvals)
pvals = np.stack(pvalues)

np.save(opj(outpath, 'ols_2ndlevel_tfr_pvals.npy'), pvals)
np.save(opj(outpath, 'ols_2ndlevel_tfr_tvals.npy'), tvals)


# ########################################################################
# Same for reinfroced trials
###########################################################################
mod_data = pd.read_csv(opj(outpathall, 'task-fearcond_alldata.csv'))

regvars = ['ratings', 'nfr_auc']
regvarsnames = ['ratings', 'nfr_auc']


# Loop participants and load single trials file
allbetasnp, all_epos = [], []
betas = [[] for i in range(len(regvars))]
part.sort()
for p in part:

    # Get model data for this part
    df = mod_data[mod_data['sub'] == p].reset_index(drop=True)

    # Load single epochs file (cotains one epoch/trial)
    epo = read_tfrs(opj(outpathall,  p, 'eeg', 'tfr',
                        p + '_task-fearcond_epochsreinforced-tfr.h5'))[0].drop_channels(['M1', 'M2'])

    # Removed shocked trials
    df = df[df['cond'] == 'CS++'].reset_index(drop=True)

    # drop bad trials
    goodtrials = np.where(epo.metadata['badtrial'] == 0)[0]
    df = df.iloc[goodtrials]
    epo = epo[goodtrials]

    # Baseline
    epo = epo.apply_baseline(mode=param['baselinemode'],
                             baseline=param['baselinetime'])
    # Crop to ROI
    epo.crop(tmin=-0.2, tmax=1, fmax=50)
    # Scale
    scale = Scaler(scalings='median')  # Robust scale

    # Consider time*frequency as a single vector for mass univariate
    tfdata = epo.data.reshape(epo.data.shape[0], epo.data.shape[1],
                              epo.data.shape[2]*epo.data.shape[3])

    tfdata_z = mne.EpochsArray(scale.fit_transform(tfdata),
                               info=epo.info)

    betasnp = []
    for idx, regvar in enumerate(regvars):
        # Standardize data
        df[regvar + '_z'] = scipy.stats.zscore(df[regvar])

        epo.metadata = df.assign(Intercept=1)  # Add an intercept

        # Perform regression
        names = ["Intercept"] + [regvar + '_z']
        res = mne.stats.linear_regression(tfdata_z, epo.metadata[names],
                                          names=names)
        # Reshape back to tfr
        betaout = mne.time_frequency.AverageTFR(info=epo.info,
                                                data=np.reshape(res[regvar
                                                                    + '_z'].beta.data,
                                                                epo.data.shape[1:]),
                                                times=epo.times,
                                                freqs=epo.freqs,
                                                nave=1)
        # Collect betas
        betas[idx].append(betaout)
        betasnp.append(betaout.data)
    allbetasnp.append(np.stack(betasnp))

# Stack all data
allbetas = np.stack(allbetasnp)

# Save for test
np.save(opj(outpath, 'ols_2ndlevel_allbetas_reinforced.npy'), allbetas)
np.save(opj(outpath, 'resamp_times_reinforced.npy'), epo.times)
np.save(opj(outpath, 'resamp_freqs_reinforced.npy'), epo.freqs)


# # #########################################################################
# # Perform second level test on betas
# ##########################################################################

allbetas = np.load(opj(outpath, 'ols_2ndlevel_allbetas_reinforced.npy'))
epo = read_tfrs(opj(outpathall,  part[0], 'eeg', 'tfr',
                    part[0] + '_task-fearcond_epochsreinforced-tfr.h5'))[0]

epo.crop(tmin=-0.2, tmax=1, fmax=50)

# Find connectivity structure
epo = epo.drop_channels(['M1', 'M2'])
epo.info.set_montage('biosemi64')
chan_connect, _ = mne.channels.find_ch_adjacency(epo.info, 'eeg')

# Create adjacency matrix for clustering
adjacency = mne.stats.combine_adjacency(
    chan_connect, len(np.zeros(len(epo.freqs))), len(epo.times))

assert adjacency.shape[0] == adjacency.shape[1] == \
    len(epo.ch_names) * len(epo.freqs) * len(epo.times)


# Get cluster entering threshold
if type(param['cluster_threshold']) is not dict:
    # Get cluster entering treshold
    p_thresh = param['cluster_threshold'] / 2
    param['cluster_threshold_test'] = -stats.t.ppf(p_thresh, len(part) - 1)


tvals, pvalues = [], []
for idx, regvar in enumerate(regvars):

    # Test each predictor
    betas_cluster = allbetas[:, idx, ::]

    # Cluster test
    tval, clusters, cluster_p_values, _ = clust_1s_ttest(betas_cluster,
                                                         n_permutations=param['nperms'],
                                                         threshold=param['cluster_threshold_test'],
                                                         adjacency=adjacency,
                                                         n_jobs=param['njobs'],
                                                         seed=param['random_state'])

    pvals = np.ones_like(tval)
    for c, p_val in zip(clusters, cluster_p_values):
        pvals[c] = p_val

    tvals.append(tval)
    pvalues.append(pvals)

    np.save(opj(outpath, 'ols_2ndlevel_tfr_pvals_' + regvar + '_.npy'),
            pvalues[-1])
    np.save(opj(outpath, 'ols_2ndlevel_tfr_tvals_' + regvar + '_.npy'),
            tvals[-1])

tvals = np.stack(tvals)
pvals = np.stack(pvalues)

np.save(opj(outpath, 'ols_2ndlevel_tfr_pvals_reinforced.npy'), pvals)
np.save(opj(outpath, 'ols_2ndlevel_tfr_tvals_reinforced.npy'), tvals)
