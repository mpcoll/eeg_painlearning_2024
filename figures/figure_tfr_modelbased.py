'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
 # @ Date: 2023
 # @ Description:
 '''

import seaborn as sns
import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
import matplotlib.pyplot as plt
from bids import BIDSLayout
from mne.time_frequency import read_tfrs
from scipy.stats import zscore
import pickle

###############################
# Parameters
##############################
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
# Outpath for figures
outfigpath = opj(outpathall, 'figures/tfr_modelbased')


if not os.path.exists(outpath):
    os.mkdir(outpath)
if not os.path.exists(outfigpath):
    os.mkdir(outfigpath)

param = {
    # Alpha Threshold
    'alpha': 0.05/3,
    # Random state to get same permutations each time
    'random_state': 23,
    # Font sizez in plot
    'titlefontsize': 12,
    'labelfontsize': 12,
    'ticksfontsize': 11,
    'legendfontsize': 10,
    # Downsample to this frequency prior to analysis
    'testresampfreq': 256,
    'baselinemode': 'logratio',
    'baselinetime': (-0.5, -0.2),
    # Color palette
    'palette': ['#4C72B0', '#0d264f', '#55a868', '#c44e52']

}


# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Liberation Sans Narrow'


###############################
# Load data
##############################
betas = np.load(opj(outpath, 'ols_2ndlevel_allbetas.npy'))

pvals = np.load(opj(outpath, 'ols_2ndlevel_tfr_pvals.npy'))
tvals = np.load(opj(outpath, 'ols_2ndlevel_tfr_tvals.npy'))
part.sort()
# Mock  info
epo = read_tfrs(opj(outpathall,  part[0], 'eeg', 'tfr',
                    part[0] + '_task-fearcond_epochs-tfr.h5'))[0].drop_channels(['M1', 'M2'])

epo.apply_baseline(mode=param['baselinemode'],
                   baseline=param['baselinetime'])

epo.crop(tmin=-0.2, tmax=1, fmax=50)

regvarsnames = ['Expectation']

# ###########################################################################
# Make plot
###############################################################################

chans_to_plot = ['Cz', 'POz']
for idx, regvar in enumerate(regvarsnames):

    betas_plot = np.average(betas[:, idx, ...], axis=0)

    pvals_plot = pvals[idx, ...]
    pvals_mask = np.where(pvals_plot < param['alpha'], 1, 0)

    pvals_tfr = mne.time_frequency.AverageTFR(info=epo.info,
                                              data=pvals[idx, ...],
                                              times=epo.times,
                                              freqs=epo.freqs,
                                              nave=1)

    pvals_plot = mne.time_frequency.AverageTFR(info=epo.info,
                                               data=pvals_mask,
                                               times=epo.times,
                                               freqs=epo.freqs,
                                               nave=1)

    beta_gavg_plot = mne.time_frequency.AverageTFR(info=epo.info,
                                                   data=betas_plot,
                                                   times=epo.times,
                                                   freqs=epo.freqs,
                                                   nave=1)

    for chan in chans_to_plot:

        pick = epo.ch_names.index(chan)
        fig, ax = plt.subplots(figsize=(2, 2.5), sharey=False)

        beta_gavg_plot.plot(picks=[pick],
                            tmin=-0.2, tmax=1,
                            show=False,
                            cmap='Greys',
                            colorbar=False,
                            title='',
                            vmin=-0.08, vmax=0.08,
                            axes=ax
                            )
        powsig = pvals_plot.copy()

        powsig.data = np.where(pvals_plot.data == 1, beta_gavg_plot.data,
                               np.nan)
        fig3 = powsig.plot(picks=[pick],
                           tmin=-0.2, tmax=1,
                           show=False,
                           cmap='viridis',
                           vmin=-0.08, vmax=0.08,
                           title='',
                           axes=ax,
                           colorbar=False,
                           )

        ax.set_title(chan,
                     fontsize=param['titlefontsize'])
        ax.set_ylabel("Frequency (Hz)",
                      fontsize=param['labelfontsize'])

        ax.set_xlabel('Time (ms)',
                      fontdict={'fontsize': param['labelfontsize']})

        ax.set_xticks(ticks=np.arange(-0.2, 1.2, 0.4))
        ax.set_xticklabels(labels=[str(i) for i in np.arange(-200, 1200, 400)])
        ax.set_yticks(ticks=np.arange(5, 55, 10))
        ax.tick_params(axis="both",
                       labelsize=param['ticksfontsize']-1, pad=0.1)
        plt.tight_layout()
        fig.savefig(opj(outfigpath, 'TF_plots_oslbetas_' + regvar
                        + "_" + chan + '.svg'),
                    bbox_inches='tight', dpi=600)

    from mne.viz import plot_topomap
    # Topo plot
    fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(2.5, 1.5))
    titles = [
        '400 ms',
        '600 ms',
        '800 ms']

    chankeep = [True if c not in ['M1', 'M2']
                else False for c in beta_gavg_plot.ch_names]
    chankeepname = [
        c for c in beta_gavg_plot.ch_names if c not in ['M1', 'M2']]
    beta_gavg_plot = beta_gavg_plot.pick_channels(chankeepname)
    pvals_plot = pvals_plot.pick_channels(chankeepname)
    for idx, times in enumerate([
        [0.4],
        [0.6],
            [0.8]]):

        plt_dat_alpha = np.mean(beta_gavg_plot.copy().crop(fmin=8, fmax=13,
                                                           tmin=times[0], tmax=times[0]).data, axis=1).squeeze()

        plt_dat_beta = np.mean(beta_gavg_plot.copy().crop(fmin=15, fmax=30,
                                                          tmin=times[0], tmax=times[0]).data, axis=1).squeeze()

        plot_topomap(plt_dat_alpha,
                     beta_gavg_plot.info,
                     show=False,
                     cmap='viridis',
                     vlim=(-0.08, 0.08),
                     outlines='head',
                     extrapolate='head',
                     axes=ax1[idx],
                     sensors=False,
                     contours=False)

        plot_topomap(plt_dat_beta,
                     beta_gavg_plot.info,
                     show=False,
                     cmap='viridis',
                     vlim=(-0.08, 0.08),
                     outlines='head',
                     extrapolate='head',
                     axes=ax2[idx],
                     sensors=False,
                     contours=False)

        ax1[idx].set_title(titles[idx],
                           fontsize=param["labelfontsize"]-3, pad=0.1)

    ax1[0].set_ylabel('Alpha\n(8-13 Hz)', fontsize=param["labelfontsize"]-3)
    ax2[0].set_ylabel('Beta\n(15-30 Hz)', fontsize=param["labelfontsize"]-3)
    plt.tight_layout()

    fig.savefig(opj(outfigpath, 'TF_plots_oslbetas_'
                    + regvar + '_topo.svg'), dpi=600, bbox_inches='tight')

fig5, cax = plt.subplots(1, 1, figsize=(0.1, 1))

cbar1 = fig.colorbar(ax2[0].images[0], cax=cax,
                     orientation='vertical', aspect=1)
cbar1.set_label('Beta coefficient', rotation=90,
                labelpad=5,
                fontdict={'fontsize': param['labelfontsize']-3})
cbar1.ax.tick_params(labelsize=param['ticksfontsize']-3)
fig5.savefig(opj(outfigpath, 'topo_colorbar_alpha.svg'), dpi=600,
             bbox_inches='tight')


###############################
# Same with reinforced
##############################
betas = np.load(opj(outpath, 'ols_2ndlevel_allbetas_reinforced.npy'))

pvals = np.load(opj(outpath, 'ols_2ndlevel_tfr_pvals_reinforced.npy'))
tvals = np.load(opj(outpath, 'ols_2ndlevel_tfr_tvals_reinforced.npy'))

# Mock  info
epo = read_tfrs(opj(outpathall,  part[0], 'eeg', 'tfr',
                    part[0] + '_task-fearcond_epochsreinforced-tfr.h5'))[0].drop_channels(['M1', 'M2'])

epo.apply_baseline(mode=param['baselinemode'],
                   baseline=param['baselinetime'])

epo.crop(tmin=-0.2, tmax=1, fmax=50)

regvarsnames = ['Ratings', 'NFR']

# ###########################################################################
# Make plot
###############################################################################

chans_to_plot = ['Cz', 'Oz', 'POz', 'Pz']
for ridx, regvar in enumerate(regvarsnames):

    if ridx == 0:
        cmap = 'plasma'
    elif ridx == 1:
        cmap = 'cividis'

    betas_plot = np.average(betas[:, ridx, ...], axis=0)

    pvals_plot = pvals[ridx, ...]
    pvals_mask = np.where(pvals_plot < param['alpha'], 1, 0)

    pvals_tfr = mne.time_frequency.AverageTFR(info=epo.info,
                                              data=pvals[ridx, ...],
                                              times=epo.times,
                                              freqs=epo.freqs,
                                              nave=1)

    pvals_plot = mne.time_frequency.AverageTFR(info=epo.info,
                                               data=pvals_mask,
                                               times=epo.times,
                                               freqs=epo.freqs,
                                               nave=1)

    beta_gavg_plot = mne.time_frequency.AverageTFR(info=epo.info,
                                                   data=betas_plot,
                                                   times=epo.times,
                                                   freqs=epo.freqs,
                                                   nave=1)

    for chan in chans_to_plot:

        pick = epo.ch_names.index(chan)
        fig, ax = plt.subplots(figsize=(2, 2.5), sharey=False)

        beta_gavg_plot.plot(picks=[pick],
                            tmin=-0.2, tmax=1,
                            show=False,
                            cmap=cmap,
                            colorbar=False,
                            title='',
                            vmin=-0.05,
                            vmax=0.05,
                            axes=ax
                            )
        powsig = pvals_plot.copy()

        powsig.data = np.where(pvals_plot.data == 1, beta_gavg_plot.data,
                               np.nan)
        fig3 = powsig.plot(picks=[pick],
                           tmin=-0.2, tmax=1,
                           show=False,
                           cmap=cmap,
                           vmin=-0.05,
                           vmax=0.05,
                           title='',
                           axes=ax,
                           colorbar=False,
                           )

        ax.set_title(chan,
                     fontsize=param['titlefontsize'])
        ax.set_ylabel("Frequency (Hz)",
                      fontsize=param['labelfontsize'])
        ax.set_xlabel("Time (ms)",
                      fontsize=param['labelfontsize'])
        ax.tick_params(axis="both",
                       labelsize=param['ticksfontsize'])

        ax.set_xticks(ticks=np.arange(-0.2, 1.2, 0.2))
        ax.set_xticklabels(labels=[str(i) for i in np.arange(-200, 1200, 200)])
        ax.set_yticks(ticks=np.arange(5, 55, 10))

        fig.savefig(opj(outfigpath, 'TF_plots_oslbetas_reinforced_' + regvar
                        + "_" + chan + '.svg'),
                    bbox_inches='tight', dpi=600)

        # Generate colorbars
        fig4, cax = plt.subplots(1, 1, figsize=(0.2, 1))

        cbar1 = fig.colorbar(ax.images[2], cax=cax,
                             orientation='vertical', aspect=1)
        cbar1.set_label('Beta coefficient', rotation=270,
                        labelpad=12,
                        fontdict={'fontsize': param['labelfontsize']-1})
        cbar1.ax.tick_params(labelsize=param['ticksfontsize']-2)
        fig4.tight_layout()
        fig4.savefig(opj(outfigpath, 'chan_beta_colorbar_reinforced_' + str(idx) + '.svg'), dpi=600,
                     bbox_inches='tight')

        # Topo plot
    fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(2.5, 1.5))
    titles = [
        '400 ms',
        '600 ms',
        '800 ms']

    chankeep = [True if c not in ['M1', 'M2']
                else False for c in beta_gavg_plot.ch_names]
    chankeepname = [
        c for c in beta_gavg_plot.ch_names if c not in ['M1', 'M2']]
    beta_gavg_plot = beta_gavg_plot.pick_channels(chankeepname)
    pvals_plot = pvals_plot.pick_channels(chankeepname)
    for idx, times in enumerate([
        [0.4],
        [0.6],
            [0.8]]):

        plt_dat_alpha = np.mean(beta_gavg_plot.copy().crop(fmin=8, fmax=13,
                                                           tmin=times[0], tmax=times[0]).data, axis=1).squeeze()

        plt_dat_beta = np.mean(beta_gavg_plot.copy().crop(fmin=15, fmax=30,
                                                          tmin=times[0], tmax=times[0]).data, axis=1).squeeze()

        plot_topomap(plt_dat_alpha,
                     beta_gavg_plot.info,
                     show=False,
                     cmap=cmap,
                     vlim=(-0.05, 0.05),
                     outlines='head',
                     extrapolate='head',
                     axes=ax1[idx],
                     sensors=False,
                     contours=False)

        plot_topomap(plt_dat_beta,
                     beta_gavg_plot.info,
                     show=False,
                     cmap=cmap,
                     vlim=(-0.05, 0.05),
                     outlines='head',
                     extrapolate='head',
                     axes=ax2[idx],
                     sensors=False,
                     contours=False)

        ax1[idx].set_title(titles[idx],
                           fontsize=param["labelfontsize"]-3, pad=0.1)

    ax1[0].set_ylabel('Alpha\n(8-13 Hz)', fontsize=param["labelfontsize"]-3)
    ax2[0].set_ylabel('Beta\n(15-30 Hz)', fontsize=param["labelfontsize"]-3)
    plt.tight_layout()

    fig.savefig(opj(outfigpath, 'TF_plots_oslbetas_'
                    + regvar + '_topo.svg'), dpi=600, bbox_inches='tight')

    fig5, cax = plt.subplots(1, 1, figsize=(0.1, 1))

    cbar1 = fig.colorbar(ax2[0].images[0], cax=cax,
                         orientation='vertical', aspect=1)
    cbar1.set_label('Beta coefficient', rotation=90,
                    labelpad=5,
                    fontdict={'fontsize': param['labelfontsize']-3})
    cbar1.ax.tick_params(labelsize=param['ticksfontsize']-3)
    fig5.savefig(opj(outfigpath, 'topo_colorbar_alpha_reinforced' + str(ridx) + '.svg'), dpi=600,
                 bbox_inches='tight')
