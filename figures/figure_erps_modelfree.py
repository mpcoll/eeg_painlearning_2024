'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
 # @ Date: 2023
 # @ Description:
 '''

import mne
import scipy.stats
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from collections import OrderedDict
import matplotlib.pyplot as plt
from bids import BIDSLayout

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
outpath = opj(outpathall, 'statistics/erps_modelfree')
# Outpath for figures
outfigpath = opj(outpathall, 'figures/erps_modelfree')

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
    # Downsample to this frequency prior to analysis
    'testresampfreq': 1024,
}


# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Liberation Sans Narrow'


###############################
# Load data
##############################
chan_to_plot = ['POz', 'CPz', 'Fz', 'Pz', 'Oz']

# Epoched data
conditions = ['CS-1', 'CS-2', 'CS+', 'CS-E']

all_meta = pd.read_csv(opj(outpathall, 'task-fearcond_erpsmeta.csv'))
data = dict()
gavg = dict()
all_meta_clean = []
for cond in conditions:
    data[cond] = list()
for p in part:
    outdir = opj(outpathall,  p, 'eeg', 'erps')
    epo = mne.read_epochs(opj(outpathall,  p, 'eeg', 'erps',
                              p + '_task-fearcond_cues_singletrials-epo.fif')).drop_channels(['M1', 'M2']).set_montage('biosemi64')

    goodtrials = np.where(epo.metadata['badtrial'] == 0)[0]
    epo = epo[goodtrials]

    all_meta_sub = all_meta[all_meta.participant_id == p]
    all_meta_clean.append(all_meta_sub.iloc[goodtrials])

    for cond in conditions:

        epo[epo.metadata['trial_cond4'] == cond]

        data[cond].append(epo[epo.metadata['trial_cond4'] == cond].average())

all_meta_clean = pd.concat(all_meta_clean)

for cond in conditions:
    gavg[cond] = mne.grand_average(data[cond])

# Get difference for each part

conditions_diff = {'Acquisition': ['CS+', 'CS-1', 'csplusvscs1'],
                   'Extinction': ['CS+', 'CS-E', 'csplusvscse'],
                   'Memory': ['CS-E', 'CS-2', 'csevscs2']}

data_pairs = dict()
gavg_pairs = dict()
tpairs = dict()
ppairs = dict()
pys = [[-1, -1.2], [-1.4, -1.6], [-1.8, -2]]
for condd, vals in conditions_diff.items():
    data_pairs[condd] = list()
    for idx, p in enumerate(part):
        dat = data['CS-1'][idx].copy()
        dat.data = data[vals[0]][idx].data - data[vals[1]][idx].data
        data_pairs[condd].append(dat)

    tpairs[condd] = np.load(opj(outpath, vals[2] + '_ttest_tvals.npy'))
    ppairs[condd] = np.load(opj(outpath, vals[2] + '_ttest_pvals.npy'))
    gavg_pairs[condd] = mne.grand_average(data_pairs[condd])

    mock = gavg_pairs[list(conditions_diff.keys())[0]]

    # Plot topo
    mock_nomast = mock.copy()
    gavg_nomast = gavg_pairs[condd].copy()
    chankeep = [True if c not in ['M1', 'M2']
                else False for c in mock.ch_names]
    plot_times = [0.6]
    times_pos = [np.abs(mock.times - t).argmin() for t in plot_times]
    for t in times_pos:
        fig, ax = plt.subplots(figsize=(0.7, 0.7))
        plot_data = gavg_nomast.data[:, t] * 1000000  # to get microvolts
        im, _ = mne.viz.plot_topomap(plot_data,
                                     pos=mock_nomast.info,
                                     mask=ppairs[condd][t,
                                                        chankeep] < param['alpha'],
                                     cmap='viridis',
                                     show=False,
                                     vlim=(-3, 3),
                                     mask_params=dict(marker='o',
                                                      markerfacecolor='w',
                                                      markeredgecolor='k',
                                                      linewidth=0,
                                                      markersize=0.5),
                                     axes=ax,
                                     outlines='head',
                                     extrapolate='head',
                                     contours=0,
                                     sensors=False)
        ax.set_title(condd, fontsize=param['labelfontsize']-6, pad=0.1)
        # plt.tight_layout()
        fig.savefig(opj(outfigpath, 'fig_topo_pairs_' + condd + str(t) + '.svg'),
                    dpi=800, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(2, 0.1))
cbar1 = fig.colorbar(im, cax=ax,
                     orientation='horizontal', aspect=1)
cbar1.set_label('Amplitude difference (uV)\n at 600 ms', rotation=0,
                labelpad=5, fontdict={'fontsize': param["labelfontsize"]-4})
cbar1.ax.tick_params(labelsize=param['ticksfontsize']-5)
fig.savefig(opj(outfigpath, 'fig_topo_pairs_colorbar.svg'),
            dpi=600, bbox_inches='tight')

# 2 conditions
clrs = ['#4C72B0', '#0d264f', '#c44e52', '#55a868']
for chan in chan_to_plot:
    # Add ERP line plots
    fig, line_axis = plt.subplots(figsize=(4, 2.5))
    pick = mock.ch_names.index(chan)
    for cidx, val, in enumerate(conditions_diff.items()):
        cond = val[0]
        # Calculate standard error for shading
        sub_avg = []
        for s in range(len(part)):
            sub_avg.append(data_pairs[cond][s].data[pick, :])
        sub_avg = np.stack(sub_avg)
        # Get standard error
        sem = scipy.stats.sem(sub_avg, axis=0)*1000000
        mean = gavg_pairs[cond].data[pick, :]*1000000

        line_axis.plot(mock.times*1000, mean,
                       label=cond,
                       color=clrs[cidx])
        line_axis.fill_between(mock.times*1000,
                               mean-sem, mean+sem, alpha=0.3,
                               facecolor=clrs[cidx])
        # Get p values
        p_vals = np.squeeze(ppairs[cond][:, pick])

        ymin = pys[cidx][0]
        ymax = pys[cidx][1]
        for tidx, t in enumerate(gavg[conditions[0]].times*1000):
            if p_vals[tidx] < param['alpha']:
                line_axis.vlines(t, ymin=ymin,
                                 ymax=ymax,
                                 linestyle="-",
                                 colors=clrs[cidx],
                                 alpha=0.2)

    line_axis.hlines(0, xmin=line_axis.get_xlim()[0],
                     xmax=line_axis.get_xlim()[1],
                     linestyle="--",
                     colors="gray")
    line_axis.vlines(0, ymin=-1,
                     ymax=1,
                     linestyle="--",
                     colors="gray")
    line_axis.legend(frameon=False, fontsize=param['legendfontsize'],
                     loc='upper left')
    # line_axis[0].set_title('Grand average ERP at ' + chan_to_plot[0],
    #                        fontdict={'size': 14})
    line_axis.set_xlabel('Time (ms)',
                         fontdict={'size': param['labelfontsize']})
    line_axis.set_ylabel('Amplitude (uV)',
                         fontdict={'size': param['labelfontsize']})

    line_axis.set_xticks(np.arange(-200, 1100, 200))
    # line_axis.set_xticklabels(np.arange(0, 900, 100))
    line_axis.tick_params(axis='both', which='major',
                          labelsize=param['ticksfontsize'],
                          length=5, width=1, direction='out', color='k')

    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'fig_lineplot_pairwise_' + chan + '.svg'),
                dpi=600, bbox_inches='tight')


###############################
# Plot line plots
##############################
# 4 conditions
clrs = ['#4C72B0', '#0d264f', '#c44e52', '#55a868']
for chan in chan_to_plot:
    # Add ERP line plots
    fig, line_axis = plt.subplots(figsize=(4, 2.5))
    pick = gavg[conditions[0]].ch_names.index(chan)
    for cidx, cond in enumerate(conditions):
        # Calculate standard error for shading
        sub_avg = []
        for s in range(len(part)):
            sub_avg.append(data[cond][s].data[pick, :])
        sub_avg = np.stack(sub_avg)
        # Get standard error
        sem = scipy.stats.sem(sub_avg, axis=0)*1000000
        mean = gavg[cond].data[pick, :]*1000000

        line_axis.plot(gavg[conditions[0]].times*1000, mean, label=cond,
                       color=clrs[cidx])
        line_axis.fill_between(gavg[conditions[0]].times*1000,
                               mean-sem, mean+sem, alpha=0.3,
                               facecolor=clrs[cidx])

    line_axis.hlines(0, xmin=line_axis.get_xlim()[0],
                     xmax=line_axis.get_xlim()[1],
                     linestyle="--",
                     colors="gray")
    line_axis.vlines(0, ymin=-1,
                     ymax=1,
                     linestyle="--",
                     colors="gray")
    line_axis.legend(frameon=False, fontsize=param['legendfontsize'])
    # line_axis[0].set_title('Grand average ERP at ' + chan_to_plot[0],
    #                        fontdict={'size': 14})
    line_axis.set_xlabel('Time (ms)',
                         fontdict={'size': param['labelfontsize']})
    line_axis.set_ylabel('Amplitude (uV)',
                         fontdict={'size': param['labelfontsize']})

    line_axis.set_xticks(np.arange(-200, 1100, 200))
    # line_axis.set_xticklabels(np.arange(0, 900, 100))
    line_axis.tick_params(axis='both', which='major',
                          labelsize=param['ticksfontsize'],
                          length=5, width=1, direction='out', color='k')
    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'fig_lineplot_4cond_' + chan + '.svg'),
                dpi=600, bbox_inches='tight')


# Single trials line plot
lppcols = [c for c in list(all_meta_clean.columns.values) if 'amp_' in c]

for col in lppcols:
    fig, line_axis = plt.subplots(figsize=(4, 2.5))
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
            label = 'CS+/CSE'
            marker = 'o'
            color = '#d53e4f'
            linestyle = '-'
            condoff = 0.25
            off2 = 0
            off1 = 0.375
        else:
            label = 'CS-1/CS-2'
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
        line_axis.axvline(x=line, linestyle=':', color='k', rasterized=True,
                          alpha=0.4)

    line_axis.set_ylabel('Mean amplitude\n400-800 ms (Z scored)',
                         fontsize=param['labelfontsize'])
    line_axis.set_xlabel('Block',
                         fontsize=param['labelfontsize'])

    line_axis.set_xticks([1, 2, 3, 4, 5, 6, 7])
    line_axis.tick_params(labelsize=param['ticksfontsize'])
    handles, labels = line_axis.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    line_axis.legend(by_label.values(), by_label.keys(), ncol=2,
                     loc=(0.05, 0.85), fontsize=param["legendfontsize"],
                     frameon=True)
    line_axis.set_ylim([-0.3, 0.6])

    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'figure_strials_' + col + '.svg'),
                bbox_inches='tight', dpi=600)
