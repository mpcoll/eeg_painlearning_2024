'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
 # @ Date: 2023
 # @ Description:
 '''

import pandas as pd
import numpy as np
import os
from os.path import join as opj
from bids import BIDSLayout
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns

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


# Parameters
param = {
    # Epoch length (in samples)
    'epochlen': 200,
    # Epoch boundaries for AUC measure
    'latencyauc': [90, 180],

}

###########################################################################
# Load data, plot and measure AUC
###########################################################################
fig_global, ax = plt.subplots(nrows=6, ncols=6, figsize=(20, 16))
axis_rat = ax.flatten()

ratings_all = []
for idx, p in enumerate(part):
    print(p)
    newp = 'sub-' + str(idx + 1).zfill(2)
    nfrout = pd.DataFrame(index=range(54))

    outpath = opj(outpathall, p, 'emg')
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Load physio data
    physdat = [f for f in os.listdir(opj(inpath, p, 'eeg'))
               if 'physio' in f][0]
    physdat = pd.read_csv(opj(inpath, p, 'eeg', physdat), sep='\t')
    physdat.columns

    # Get ratings
    events = pd.read_csv(opj(inpath, p, 'eeg',
                             [f for f in os.listdir(opj(inpath,
                                                        p, 'eeg'))
                              if 'events' in f][0]), sep='\t')

    ratings = np.asarray(events['painrating'].dropna())
    part_rate = pd.DataFrame(dict(sub=newp, rating=ratings))
    ratings_all.append(part_rate)

    # Find shocks triggers
    trig_ons = np.where(physdat['events'] == 'shock')[0]

    # Create epochs
    epochs = []
    for t in trig_ons:
        epochs.append(np.asarray(physdat['rmsemg'])[t:t + param['epochlen']])

    epochs = np.stack(epochs)

    # Get AUC and plot all epochs
    fig, ax = plt.subplots(nrows=7, ncols=8, figsize=(20, 16))
    ax = ax.flatten()
    nfr_auc = []
    for i in range(epochs.shape[0]):
        # Get AUC
        nfr_auc.append(np.trapz(y=epochs[i,
                                         param['latencyauc'][0]:
                                             param['latencyauc'][1]]))
        # Plot
        ax[i].plot(epochs[i, :])
        ax[i].set_title('Shock ' + str(i))
        ax[i].set_xlabel('Time from trigger (ms)')

    fig.tight_layout()
    fig.savefig(opj(outpath, p + '_rmsemg_plot.png'), dpi=600)

    # Get AUC measure
    nfr_auc_z = zscore(nfr_auc)
    ratings_z = zscore(ratings)
    nfrout['nfr_auc_z'] = nfr_auc_z
    nfrout['nfr_auc'] = nfr_auc
    nfrout['ratings_z'] = ratings_z
    nfrout['ratings'] = ratings

    # Plot correlation between rating and nfr
    sns.regplot(nfr_auc_z, ratings_z, ax=axis_rat[idx])
    axis_rat[idx].set_xlabel('Z scored NFR (AUC of RMS EMG 90-180 ms)')
    axis_rat[idx].set_ylabel('Z scored pain rating')
    axis_rat[idx].set_title(p)

    nfrout.to_csv(opj(outpath, p + '_task-fearcond_nfrauc.csv'))

fig_global.tight_layout()
fig_global.savefig(opj(outpathall, 'figures',
                       'nfr_rating_correlation.png'), dpi=600)
