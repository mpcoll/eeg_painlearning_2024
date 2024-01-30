'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
 # @ Date: 2023
 # @ Description:
 '''
from collections import OrderedDict
import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
import matplotlib.pyplot as plt
from bids import BIDSLayout
from mne.time_frequency import read_tfrs
import ptitprince as pt
import seaborn as sns
from mne.viz import plot_topomap
from scipy.stats import zscore

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
outpath = opj(outpathall, 'statistics/tfr_modelfree')
# Outpath for figures
outfigpath = opj(outpathall, 'figures/tfr_modelfree')

if not os.path.exists(outpath):
    os.mkdir(outpath)
if not os.path.exists(outfigpath):
    os.mkdir(outfigpath)

param = {
    # Alpha Threshold
    'alpha': 0.05/3,
    # Font sizez in plot
    'titlefontsize': 12,
    'labelfontsize': 12,
    'ticksfontsize': 11,
    'legendfontsize': 10,
    # Color palette
    'palette': ['#4C72B0', '#0d264f', '#55a868', '#c44e52'],
    # range on colormaps
    'pwrv': [-0.3, 0.3],
    'baselinemode': 'logratio',
    'baselinetime': (-0.5, -0.2)

}

# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Liberation Sans Narrow'

# Load data
mock_data = read_tfrs(opj(outpathall, 'sub-23',
                          'eeg', 'tfr',
                          'sub-23' + '_task-fearcond_' +
                          'CS-1' + '_avg-tfr.h5'))[0].crop(tmin=-0.2, fmax=50)


anova_data = np.load(opj(outpath,
                         'anova_data.npy'))


gavg = dict()
for cond in ['CS-1', 'CS-2', 'CS+', 'CS-E', ]:
    gavg[cond] = read_tfrs(opj(outpath, 'task-fearcond_'
                               + cond + '_gavg-tfr.h5'))[0].crop(tmin=-0.2,
                                                                 fmax=50)

# Calculate differences
csplusvscs1 = np.empty((1,) + anova_data.shape[1:])
csevscs2 = np.empty((1,) + anova_data.shape[1:])
csplusvscse = np.empty((1,) + anova_data.shape[1:])

for s in range(anova_data.shape[1]):

    csplusvscs1[0, s, ::] = (anova_data[2, s, :] - anova_data[0, s, :])
    csevscs2[0, s, ::] = (anova_data[3, s, :] - anova_data[1, s, :])
    csplusvscse[0, s, ::] = ((anova_data[2, s, :] - anova_data[3, s, :]) -
                             (anova_data[0, s, :] - anova_data[1, s, :]))


# ###########################################################################
# Make plot
###############################################################################

# _________________________________________________________________
# Differences plot

for diff_data, savename, title, ylabel in zip([csplusvscs1, csevscs2,
                                               csplusvscse],
                                              ['csplusvscs1', 'csevscs2',
                                                  'csplusvscse'],
                                              ['Acquisition',
                                               'Memory',
                                               'Extinction'],
                                              [True, True, True]):
    pvals = np.load(opj(outpath,
                        'cuesdiff_tfr_ttest_pvals' + savename + '.npy'))
    tvals = np.load(opj(outpath,
                        'cuesdiff_tfr_ttest_tvals' + savename + '.npy'))

    # Plot difference
    for chan in ['POz']:

        fig, ax = plt.subplots(figsize=(2, 2))

        p_plot_fwe = mock_data.copy().crop(tmin=-0.2, fmax=50)
        p_plot_fwe.data = pvals
        p_plot_fwe.data = np.where(p_plot_fwe.data < param['alpha'], 1, 0)

        pltdat = mock_data.copy().crop(tmin=-0.2, fmax=50)
        pltdat.data = np.mean(diff_data[0, ::], axis=0)

        pick = pltdat.ch_names.index(chan)
        ch_mask = np.asarray([1 if c == chan else 0
                              for c in pltdat.ch_names])

        fig2 = pltdat.plot(picks=[pick],
                           tmin=-0.2, tmax=1,
                           show=False,
                           cmap='Greys',
                           vmin=param['pwrv'][0],
                           vmax=param['pwrv'][1],
                           title='',
                           axes=ax,
                           colorbar=False,
                           )

        powsig = pltdat.copy()

        powsig.data = np.where(p_plot_fwe.data == 1, pltdat.data, np.nan)
        # powsig.data = np.mean(diff_data[0, ::], axis=0)

        fig3 = powsig.plot(picks=[pick],
                           tmin=-0.2, tmax=1,
                           show=False,
                           cmap='viridis',
                           vmin=param['pwrv'][0],
                           vmax=param['pwrv'][1],
                           title='',
                           axes=ax,
                           colorbar=False,
                           )

        ax.set_xlabel('Time (ms)',
                      fontdict={'fontsize': param['labelfontsize']-1})

        ax.set_xticks(ticks=np.arange(-0.2, 1.2, 0.4))
        ax.set_xticklabels(labels=[str(i) for i in np.arange(-200, 1200, 400)])

        if ylabel:
            ax.set_ylabel('Frequency (Hz)',
                          fontdict={'fontsize': param['labelfontsize']-1})
        else:
            ax.set_ylabel('',
                          fontdict={'fontsize': param['labelfontsize']-1})
        ax.set_yticks(ticks=np.arange(5, 55, 10))

        # ax.set_ylabel('',
        #         fontdict={'fontsize': param['labelfontsize']})

        ax.tick_params(axis="y", labelsize=param['ticksfontsize']-1)
        ax.tick_params(axis="x", labelsize=param['ticksfontsize']-1)

        ax.set_title(title, fontdict={
                     "fontsize": param['titlefontsize']-1}, pad=0.1)

        plt.tight_layout()

        plt.savefig(opj(outfigpath, 'TF_plots_diff_' + chan + '_'
                        + savename + '.svg'),
                    bbox_inches='tight', dpi=600)

# Generate colorbars
fig3, cax = plt.subplots(2, 1, figsize=(1, 0.25))

cbar1 = fig3.colorbar(ax.images[2], cax=cax[1],
                      orientation='horizontal', aspect=2)
cbar1.set_label('Power (difference)', rotation=0,
                labelpad=0,
                fontdict={'fontsize': param['labelfontsize']-2})
cbar1.ax.tick_params(labelsize=param['ticksfontsize']-1)


cbar2 = fig3.colorbar(ax.images[0], cax=cax[0],
                      orientation='horizontal', aspect=2)
cbar2.set_label('', rotation=-90,
                labelpad=15,
                fontdict={'fontsize': param['labelfontsize']-5})
cbar2.ax.tick_params(size=0, labelsize=0)
fig3.tight_layout()
fig3.savefig(opj(outfigpath, 'diff_colorbar.svg'), dpi=600,
             bbox_inches='tight')


fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(2.5, 1.5))

for diff_data, savename, title, ylabel, axidx in zip([csplusvscs1, csplusvscse, csevscs2],
                                                     ['csplusvscs1',
                                                         'csplusvscse', 'csevscs2'],
                                                     ['Acquisition',
                                                      'Extinction',
                                                      'Memory'],
                                                     [True, False, False],
                                                     [0, 1, 2]):
    pvals = np.load(opj(outpath,
                        'cuesdiff_tfr_ttest_pvals' + savename + '.npy'))
    tvals = np.load(opj(outpath,
                        'cuesdiff_tfr_ttest_tvals' + savename + '.npy'))

    # Topo plots
    for band, foi, lims, lab, axtopo in zip(['alpha', 'beta'], [[8, 13], [15, 30]],
                                            [[-0.1, 0.1], [-0.1, 0.1]],
                                            ['8-13 Hz', '15-30 Hz'],
                                            [ax1, ax2]):
        # fig, axtopo = plt.subplots(figsize=(4, 4))
        time = [0.6, 0.8]
        fidx = np.arange(np.where(gavg['CS-1'].freqs == foi[0])[0],
                         np.where(gavg['CS-1'].freqs == foi[1])[0])

        times = gavg['CS-1'].times
        tidx = np.arange(np.argmin(np.abs(times - time[0])),
                         np.argmin(np.abs(times - time[1])))
        ddata = np.mean(diff_data[0, ::], axis=0)
        plt_dat = np.squeeze(
            np.mean(ddata[:, fidx, :][:, :, tidx], axis=(1, 2)))
        p_dat = np.squeeze(np.mean(pvals[:, fidx, :][:, :, tidx], axis=(1, 2)))

        mask = np.where(p_dat < param['alpha'], 1, 0)

        chankeep = [True if c not in ['M1', 'M2']
                    else False for c in gavg['CS-1'].ch_names]
        chankeepname = [
            c for c in gavg['CS-1'].ch_names if c not in ['M1', 'M2']]

        plot_topomap(plt_dat[chankeep],
                     pltdat.copy().pick(chankeepname).info,
                     show=False,
                     cmap='viridis',
                     vlim=(lims[0], lims[1]),
                     # mask_params=dict(markersize=8),
                     outlines='head',
                     extrapolate='head',
                     # mask=mask[chankeep],
                     axes=axtopo[axidx],
                     sensors=False,
                     contours=False)
        ax1[axidx].set_title(title,
                             fontsize=param["labelfontsize"]-3, pad=0.1)

ax1[0].set_ylabel('Alpha\n(8-13 Hz)', fontsize=param["labelfontsize"]-3)
ax2[0].set_ylabel('Beta\n(15-30 Hz)', fontsize=param["labelfontsize"]-3)

plt.tight_layout()

plt.savefig(opj(outfigpath, 'TF_diff_topo.svg'), bbox_inches='tight',
            dpi=600)

# Generate a colorbar
fig3, cax = plt.subplots(1, 1, figsize=(1, 0.1))

cbar1 = fig3.colorbar(ax1[0].images[0], cax=cax,
                      orientation='horizontal', aspect=2)

cbar1.set_label('Mean power difference\n(500-1000 ms)', rotation=0,
                labelpad=2,
                fontdict={'fontsize': param['labelfontsize']-3})
cbar1.ax.tick_params(labelsize=param['ticksfontsize']-3)
fig3.tight_layout()
fig3.savefig(opj(outfigpath, 'topo_colorbar.svg'), dpi=600,
             bbox_inches='tight')


###################################################################
# Plot power in each condition
###################################################################
param['pwrv'] = [-0.3, 0.3]
for chan in ['POz']:

    for idx, c in enumerate(['CS-1', 'CS-2', 'CS+', 'CS-E']):
        fig, ax = plt.subplots(figsize=(2, 2))

        pltdat = gavg[c]
        pick = pltdat.ch_names.index(chan)

        pltdat.plot(picks=[pick],
                    tmin=-0.5, tmax=2,
                    show=False,
                    cmap='viridis',
                    vmin=param['pwrv'][0],
                    vmax=param['pwrv'][1],
                    title='',
                    axes=ax,
                    colorbar=False,
                    )

        ax.set_xlabel('Time (ms)',
                      fontdict={'fontsize': param['labelfontsize']-1})
        ax.tick_params(axis="x",
                       labelsize=param['ticksfontsize']-1)
        ax.set_xticks(ticks=np.arange(-0.2, 1.2, 0.4))
        ax.set_xticklabels(labels=[str(i) for i in np.arange(-200, 1200, 400)])
        ax.set_yticks(ticks=np.arange(5, 55, 10))

        ax.set_ylabel('Frequency (Hz)',
                      fontdict={'fontsize': param['labelfontsize']-1})

        ax.tick_params(axis="y", labelsize=param['ticksfontsize']-1)

        ax.set_title(
            c, fontdict={"fontsize": param['titlefontsize']-1}, pad=0.1)

        plt.tight_layout()

        plt.savefig(opj(outfigpath, 'TF_plots_' + chan
                        + '_' + c + '.svg'),
                    bbox_inches='tight', dpi=600)

fig3, cax = plt.subplots(1, 1, figsize=(1, 0.125))

cbar1 = fig3.colorbar(ax.images[1], cax=cax,
                      orientation='horizontal', aspect=2)
cbar1.set_label('Power (logratio)', rotation=0,
                labelpad=0,
                fontdict={'fontsize': param['labelfontsize']-2})
cbar1.ax.tick_params(labelsize=param['ticksfontsize']-1)


fig3.tight_layout()
fig3.savefig(opj(outfigpath, 'nodiff_colorbar.svg'), dpi=600,
             bbox_inches='tight')

# Topo plots for each condition
time = [0.5, 1]

for band, foi, lim, lab in zip(['alpha', 'beta'], [[8, 13], [15, 30]],
                               [[0.3], [0.25]], ['8-13 Hz', '15-30 Hz']):
    for idx, c in enumerate(['CS-1', 'CS-2', 'CS+', 'CS-E']):
        fig, ax = plt.subplots(figsize=(4, 4))

        fidx = np.arange(np.where(gavg['CS-1'].freqs == foi[0])[0],
                         np.where(gavg['CS-1'].freqs == foi[1])[0])

        times = gavg['CS-1'].times
        tidx = np.arange(np.argmin(np.abs(times - time[0])),
                         np.argmin(np.abs(times - time[1])))
        plt_dat = np.average(gavg[c].data[:, :, tidx], 2)
        plt_dat = np.average(plt_dat[:, fidx], 1)

        chankeep = [True if c not in ['M1', 'M2'] else False
                    for c in gavg[c].ch_names]
        chankeepname = [c for c in gavg[c].ch_names if c not in
                        ['M1', 'M2']]

        fig2 = plot_topomap(plt_dat[chankeep],
                            gavg[c].copy().pick_channels(chankeepname).info,
                            show=False,
                            cmap='viridis',
                            vlim=(-lim[0], 0.1),
                            outlines='head',
                            extrapolate='head',
                            axes=ax,
                            contours=False)
        ax.set_title(c, fontdict={'fontsize': param['titlefontsize']})

        plt.savefig(opj(outfigpath, 'TF_topo_'
                        + c + '_' + band + '.svg'),
                    bbox_inches='tight', dpi=600)

    # Generate a colorbar
    fig3, cax = plt.subplots(figsize=(2, 0.25))

    cbar1 = fig3.colorbar(ax.images[0], cax=cax,
                          orientation='horizontal', aspect=2)

    cbar1.set_label('Power (500-1000 ms, ' + lab + ')', rotation=0,
                    labelpad=10,
                    fontdict={'fontsize': param['labelfontsize']-5})
    cbar1.ax.tick_params(labelsize=param['ticksfontsize']-4)

    fig3.savefig(opj(outfigpath, 'topo_colorbar' + band + '.svg'), dpi=600,
                 bbox_inches='tight')


mod_data = pd.read_csv(opj(outpathall, 'task-fearcond_alldata.csv'))

# Loop participants and load single trials file
allbetasnp, all_epos = [], []

times = [0.5, 1]
chan = ['POz', 'Cz']

# Loop freqbands

# Loop for part
all_meta = []
for p in part:
    # Get external data for this part
    df = mod_data[mod_data['sub'] == p]

    # Remove shocked
    df = df[df['cond'] != 'CS++']

    # Load single epochs file (cotains one epoch/trial)
    epo = read_tfrs(opj(outpathall,  p, 'eeg', 'tfr',
                        p + '_task-fearcond_epochs-tfr.h5'))[0]

    # Baseline
    epo = epo.apply_baseline(mode=param['baselinemode'],
                             baseline=param['baselinetime'])

    for freqs, name in zip([[8, 13], [15, 30]], ['alpha', 'beta']):
        for chan in ['POz', 'Cz', 'CPz', 'Pz']:
            epoc = epo.copy().pick(chan)
            # Extract data in time and frequency
            dat = np.squeeze(epoc.crop(tmin=times[0],
                                       tmax=times[1],
                                       fmin=freqs[0],
                                       fmax=freqs[1]).data)

            # Average in time and frequency
            dat = zscore(np.average(dat, axis=(1, 2)))
            df[name + '_power_strial_' + chan] = dat
    df['subject_id'] = p
    all_meta.append(df)

cols = []
for name in ['alpha', 'beta']:
    for chan in ['POz']:
        cols.append(name + '_power_strial_' + chan)

all_meta_all = pd.concat(all_meta)
# Drop shocked trials
all_meta_clean = all_meta_all[all_meta_all['cond'] != 'CS++'].reset_index()
# Save to restart straight to figure
all_meta_clean.to_csv(opj(outpathall, 'task-fearcond_tfralldata_.csv'))


# Reread to start here if necessary
all_meta_clean = pd.read_csv(opj(outpathall, 'task-fearcond_tfralldata_.csv'))


for idx, col in enumerate(cols):

    all_meta_clean = all_meta_all.copy()
    fig, line_axis = plt.subplots(figsize=(4, 1))
    data_avg_all = all_meta_clean.groupby(['condblock_join',
                                           'block'])[col].mean().reset_index()

    # Standard error across all conditons
    # Get sd
    data_se_all = all_meta_clean.groupby(['condblock_join',
                                          'block'])[col].std().reset_index()
    npart = len(set(all_meta_clean['participant_id']))
    data_se_all[col] = data_se_all[col]/np.sqrt(npart)

    off = 0.1  # Dots offset to avoid overlap

    # Create a new df with colors, styles, etc.
    for cond in data_avg_all['condblock_join']:
        if cond[0:3] == 'CS+':
            label = 'CS+ / CSE'
            marker = 'o'
            color = '#d53e4f'
            linestyle = '-'
            condoff = 0.25
            off2 = 0
            off1 = 0.375
        else:
            label = 'CS-1 / CS-2'
            marker = '^'
            color = '#3288bd'
            linestyle = '--'
            condoff = -0.25
            off1 = 0.25
            off2 = -0.25
        dat_plot = data_avg_all[data_avg_all.condblock_join
                                == cond].reset_index()
        dat_plot_se = data_se_all[data_se_all.condblock_join == cond]

        if len(dat_plot) > 1:
            line_axis.errorbar(x=[dat_plot.block[0]+off1,
                                  dat_plot.block[1]+off2],
                               y=dat_plot[col],
                               yerr=dat_plot_se[col], label=label,
                               marker=marker, color=color, ecolor=color,
                               linestyle=linestyle, markersize=4, linewidth=1,
                               rasterized=True)
        else:
            line_axis.errorbar(x=[dat_plot.block[0]+off1],
                               y=dat_plot[col],
                               yerr=dat_plot_se[col], label=label,
                               marker=marker, color=color, ecolor=color,
                               linestyle=linestyle, markersize=4, linewidth=1,
                               rasterized=True)
    for line in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
        line_axis.axvline(x=line, linestyle=':', color='k', rasterized=True)

    line_axis.set_ylabel('Mean ' + col.split('_')[0] + ' power\n500-1000 ms (Z scored)',
                         fontsize=param['labelfontsize']-2)
    line_axis.set_xlabel('Block',
                         fontsize=param['labelfontsize'])

    line_axis.set_xticks([1, 2, 3, 4, 5, 6, 7])
    line_axis.tick_params(labelsize=param['ticksfontsize'])
    handles, labels = line_axis.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    if idx == 0:
        line_axis.legend(by_label.values(), by_label.keys(), fontsize=param["legendfontsize"]-2,
                         frameon=True, ncol=2, loc=(0.2, 1))

    # fig.tight_layout()
    fig.savefig(opj(outfigpath, 'figure_strials_' + col + '.svg'),
                bbox_inches='tight', dpi=600)
