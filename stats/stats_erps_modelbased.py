'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
 # @ Date: 2023
 # @ Description:
 '''


import mne
from os.path import join as opj
import pandas as pd
import numpy as np
import os
from mne.decoding import Scaler
import scipy
from bids import BIDSLayout
from mne.stats import spatio_temporal_cluster_1samp_test as st_clust_1s_ttest
from scipy import stats

###############################
# Parameters
###############################
inpath = 'source'
outpathall = 'derivatives'

# Get BIDS layoutaa
layout = BIDSLayout(inpath)

# Load participants
part = pd.read_csv(opj(inpath, 'participants.tsv'), sep='\t')
# Exlcude participants
part = part[part['excluded'] == 0]['participant_id'].tolist()


# Silence pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Parameters
param = {
    # Njobs for permutations
    'njobs': 20,
    # Number of permutations
    'nperms': 5000,
    # Random state to get same permutations each time
    'random_state': 23,
    # Downsample to this frequency prior to analysis
    'testresampfreq': 1024,
    # clustering threshold
    'cluster_threshold': 0.01
}


# Outpath for analysis
outpath = opj(outpathall, 'statistics')
if not os.path.exists(outpath):
    os.mkdir(outpath)

outpath = opj(outpath, 'erps_modelbased')
if not os.path.exists(outpath):
    os.mkdir(outpath)

########################################################################
# Mass univariate regression vhat ~ ERP
##########################################################################

# Load model data
mod_data = pd.read_csv(opj(outpathall, 'task-fearcond_alldata.csv'))

# Regressors
regvars = ['vhat', 'nfr_auc', 'ratings']
regvarsnames = ['Expectation', 'NFR', 'Ratings']

betas, betasnp = [], []

# Loop participants and load single trials file
all_epos = [[] for i in range(len(regvars))]
allbetasnp = []
betas = [[] for i in range(len(regvars))]
part.sort()
for p in part:
    # Get external data for this part
    df = mod_data[mod_data['sub'] == p]

    # Load single epochs file (cotains one epoch/trial)
    epo = mne.read_epochs(opj(outpathall,  p, 'eeg', 'erps',
                              p + '_task-fearcond_cues_singletrials-epo.fif'))

    # downsample if necessary
    if epo.info['sfreq'] != param['testresampfreq']:
        epo = epo.resample(param['testresampfreq'])

    # Drop bad trials
    goodtrials = np.where(epo.metadata['badtrial'] == 0)[0]
    df = df.iloc[goodtrials]
    epo = epo[goodtrials]

    # Robust standardize data before regression
    scale = Scaler(scalings='mean')
    epo_z = mne.EpochsArray(scale.fit_transform(epo.get_data()),
                            epo.info)

    betasnp = []
    for idx, regvar in enumerate(regvars):

        # Keep only rows with values on regressor
        keep = np.where(~np.isnan(df[regvar]))[0]
        df_reg = df.iloc[keep]
        epo_reg = epo_z.copy()[keep]
        epo_keep = epo.copy()[keep]

        # Standardize data
        df_reg[regvar + '_z'] = stats.zscore(df_reg[regvar])

        # Add an intercept to the matrix
        epo_keep.metadata = df_reg.assign(Intercept=1)
        epo_reg.metadata = df_reg.assign(Intercept=1)

        # Perform regression
        names = ["Intercept"] + [regvar + '_z']
        res = mne.stats.linear_regression(epo_reg, epo_reg.metadata[names],
                                          names=names)

        # Collect betas
        betas[idx].append(res[regvar + '_z'].beta)
        betasnp.append(res[regvar + '_z'].beta.data)
        all_epos[idx].append(epo_keep)
    allbetasnp.append(np.stack(betasnp))


# Stack all data
allbetas = np.stack(allbetasnp)


# Grand average
beta_gavg = []
for idx, regvar in enumerate(regvars):
    beta_gavg.append(mne.grand_average(betas[idx]))

# _________________________________________________________________
# Second level test on betas

# Get channels connectivity
connect, names = mne.channels.find_ch_adjacency(epo.info, ch_type='eeg')


# Get cluster entering threshold
if type(param['cluster_threshold']) is not dict:
    # Get cluster entering treshold
    p_thresh = param['cluster_threshold'] / 2  # two sided
    n_samples = allbetas.shape[0]
    param['cluster_threshold'] = -stats.t.ppf(p_thresh, n_samples - 1)


# Perform test for each regressor
tvals, pvalues = [], []
for idx, regvar in enumerate(regvars):
    # Reshape sub x time x vertices
    testdata = np.swapaxes(allbetas[:, idx, :, :], 2, 1)
    # data is (n_observations, n_times, n_vertices)
    tval, clusters, cluster_p_values, _ = st_clust_1s_ttest(testdata,
                                                            n_jobs=param["njobs"],
                                                            threshold=param['cluster_threshold'],
                                                            adjacency=connect,
                                                            n_permutations=param['nperms'],
                                                            buffer_size=None)

    # Reshape p-values to match data
    pvals = np.ones_like(tval)
    for c, p_val in zip(clusters, cluster_p_values):
        pvals[c] = p_val

    # In a list for each regressor
    tvals.append(tval)
    pvalues.append(pvals)
    # Save for each regressor in case crash/stop
    np.save(opj(outpath, 'ols_2ndlevel_tval_' + regvar + '.npy'), tvals[-1])
    np.save(opj(outpath, 'ols_2ndlevel_pval_' + regvar + '.npy'), pvalues[-1])

# Stack and save
tvals = np.stack(tvals)
pvals = np.stack(pvalues)

np.save(opj(outpath, 'ols_2ndlevel_tvals.npy'), tvals)
np.save(opj(outpath, 'ols_2ndlevel_pvals.npy'), pvalues)
np.save(opj(outpath, 'ols_2ndlevel_betas.npy'), allbetas)

for idx, regvar in enumerate(regvars):
    epo_save = mne.concatenate_epochs(all_epos[idx])
    epo_save.save(opj(outpath, 'ols_2ndlevel_allepochs-epo_' + regvar + '.fif'),
                  overwrite=True)
np.save(opj(outpath, 'ols_2ndlevel_betasavg.npy'), beta_gavg)
