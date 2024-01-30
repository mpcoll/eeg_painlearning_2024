'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
 # @ Date: 2023
 # @ Description:
 '''

from os.path import join as opj
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from bids import BIDSLayout
import ptitprince as pt
from scipy.io import loadmat
###############################
# Parameters
###############################

inpath = 'source'
outpath = 'derivatives'
outfigpath = opj(outpath, 'figures/comp_modelsscr')


# Get BIDS layout
layout = BIDSLayout(inpath)

# Load participants
part = pd.read_csv(opj(inpath, 'participants.tsv'), sep='\t')
# Exlcude participants
part = part[part['excluded'] == 0]['participant_id'].tolist()


# Outpath for figures
if not os.path.exists(outfigpath):
    os.makedirs(outfigpath)

param = {
    # Font sizez in plot
    'titlefontsize': 12,
    'labelfontsize': 12,
    'ticksfontsize': 11,
    'legendfontsize': 10,
}

# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Liberation Sans Narrow'

#  ################################################################
# Figure X SCR plot
#################################################################


# Winning model
win = 'RW_intercue'

# Load data
data = pd.read_csv(opj(outpath, 'task-fearcond_alldata.csv'))

# Remove shocks
data_ns = data.copy()
data_ns = data[data['cond'] != 'CS++']

data_pred = data[~np.isnan(data['pred'])]
mae = np.average(np.abs(data_pred['scr'] - data_pred['pred']))

# Get average SCR/cond/block
data_avg_all = data_ns.groupby(['cond_plot',
                                'block'])['scr',
                                          'pred'].mean().reset_index()

# Get SD
data_se_all = data_ns.groupby(['cond_plot',
                               'block'])['scr',
                                         'pred'].std().reset_index()
# Divide by sqrt(n)
data_se_all.scr = data_se_all.scr / np.sqrt(len(set(data_ns['sub'])))
data_se_all.pred = data_se_all.pred / np.sqrt(len(set(data_ns['sub'])))


# Init figure
fig, ax = plt.subplots(figsize=(4, 2.5))

off = 0.1  # Dots offset to avoid overlap

for cond in data_avg_all['cond_plot']:
    if cond[0:3] == 'CS+':
        label = 'CS+/CSE'
        marker = 'o'
        color = "#C44E52"
        linestyle = '-'
        condoff = 0.05
    else:
        label = 'CS-1/CS-2'
        marker = '^'
        color = '#4C72B0'
        linestyle = '--'
        condoff = -0.025
    dat_plot = data_avg_all[data_avg_all.cond_plot == cond].reset_index()
    dat_plot_se = data_se_all[data_se_all.cond_plot == cond]

    # len(dat_plot)
    if len(dat_plot) > 1:
        ax.errorbar(x=[dat_plot.block[0] + off, dat_plot.block[1] + condoff],
                    y=dat_plot.scr,
                    yerr=dat_plot_se.scr, label=label,
                    marker=marker, color=color, ecolor=color,
                    linestyle=linestyle, markersize=6, linewidth=2,
                    rasterized=True)
    else:
        ax.errorbar(x=[dat_plot.block[0] - off],
                    y=dat_plot.scr,
                    yerr=dat_plot_se.scr, label=label,
                    marker=marker, color=color, ecolor=color,
                    linestyle=linestyle, markersize=6, linewidth=2,
                    rasterized=True)

for line in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    ax.axvline(x=line, linestyle=':', color='k', alpha=0.5)
ax.set_ylabel('SCR (beta estimate)', fontsize=param['labelfontsize'])
ax.set_xlabel('Block', fontsize=param['labelfontsize'])
# ax1[0].set_ylim([0.1, 0.26])
ax.tick_params(labelsize=param['ticksfontsize'])
handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
          loc='best', fontsize=param["legendfontsize"],
          frameon=False)

ax.set_xticks([1, 2, 3, 4, 5, 6, 7])

fig.tight_layout()
fig.savefig(opj(outfigpath, 'scr_average.svg'), dpi=600, bbox_inches='tight')

# SAME WITH PRED
# Init figure
fig, ax = plt.subplots(figsize=(4, 2.5))

countcs1 = 0
countcsp = 0
for cond in data_avg_all['cond_plot']:
    if cond[0:3] == 'CS+':
        if countcsp == 0:
            label = 'CS+/CSE'
        else:
            label = '_nolegend_'
        marker = 'o'
        color = "#C44E52"
        linestyle = '-'
        condoff = 0.1
        countcsp += 1
    else:
        if countcs1 == 0:
            label = 'CS-1/CS-2'
        else:
            label = '_nolegend_'
        marker = '^'
        color = '#4C72B0'
        linestyle = '--'
        condoff = -0.05
        countcs1 += 1

    dat_plot = data_avg_all[data_avg_all.cond_plot == cond].reset_index()
    dat_plot_se = data_se_all[data_se_all.cond_plot == cond]

    # len(dat_plot)
    if len(dat_plot) > 1:
        if label != '_nolegend_':
            labelo = label + ' (Pred.)'
        else:
            labelo = '_nolegend_'
        ax.errorbar(x=[dat_plot.block[0] + off, dat_plot.block[1] + condoff],
                    y=dat_plot.pred,
                    yerr=dat_plot_se.pred, label=labelo,
                    marker=marker, color='gray', ecolor='gray',
                    linestyle=linestyle, markersize=6, linewidth=2,
                    rasterized=True)
        if label != '_nolegend_':
            labelo = label + ' (Obs.)'
        else:
            labelo = '_nolegend_'
        ax.errorbar(x=[dat_plot.block[0] + off, dat_plot.block[1] + condoff],
                    y=dat_plot.scr,
                    yerr=dat_plot_se.pred, label=labelo,
                    marker=marker, color=color, ecolor=color,
                    linestyle=linestyle, markersize=6, linewidth=2,
                    rasterized=True)
    else:
        ax.errorbar(x=[dat_plot.block[0] - off],
                    y=dat_plot.pred,
                    yerr=dat_plot_se.pred, label='_nolegend_',
                    marker=marker, color='gray', ecolor='gray',
                    linestyle=linestyle, markersize=6, linewidth=2,
                    rasterized=True)
        ax.errorbar(x=[dat_plot.block[0] - off],
                    y=dat_plot.scr,
                    yerr=dat_plot_se.pred, label='_nolegend_',
                    marker=marker, color=color, ecolor=color,
                    linestyle=linestyle, markersize=6, linewidth=2,
                    rasterized=True)

for line in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    ax.axvline(x=line, linestyle=':', color='k', alpha=0.3, rasterized=True)
ax.set_ylabel('SCR (beta estimate)', fontsize=param['labelfontsize'])
ax.set_xlabel('Block', fontsize=param['labelfontsize'])
ax.set_ylim([-0.15, 0.20])
ax.tick_params(labelsize=param['ticksfontsize'])
ax.legend(ncol=2, loc=(0.05, 0.8), fontsize=param["legendfontsize"])
ax.set_xticks(np.arange(1, 8))
# fig.tight_layout()
fig.savefig(opj(outfigpath, 'pred_scr_average.svg'), dpi=800,
            bbox_inches='tight')


fig, ax = plt.subplots(figsize=(4, 2.5))

# Actual vs predicted /trial
deep_pal = sns.color_palette('deep')

data_ns['cond2'] = 0
data_ns['cond2'] = np.where(data_ns['cond'] == 'CS++',
                            "CS+", data_ns['cond2'])
data_ns['cond2'] = np.where(data_ns['cond'] == 'CS-1',
                            'CS-1', data_ns['cond2'])
data_ns['cond2'] = np.where(data_ns['cond'] == 'CS-2',
                            'CS-2', data_ns['cond2'])
data_ns['cond2'] = np.where(data_ns['cond'] == 'CS+',
                            "CS+", data_ns['cond2'])

data_ns['cond2'] = np.where(data_ns['cond'] == 'CS-E',
                            "CS-E", data_ns['cond2'])

data_avg_all = data_ns.groupby(['block',
                                'trial_within_wb',
                                'cond'])['scr', 'pred',
                                         'vhat'].mean().reset_index()

dotsize = 15

ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-1'],
           y=data_avg_all.scr[data_avg_all.cond == 'CS-1'], s=dotsize,
           facecolors='none',
           color='#4C72B0',
           alpha=1,
           label='CS-1', rasterized=True)
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-2'],
           y=data_avg_all.scr[data_avg_all.cond == 'CS-2'], s=dotsize,
           facecolors='none',
           color='#0d264f',
           alpha=1,
           label='CS-2', rasterized=True)

ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS+'],
           y=data_avg_all.scr[data_avg_all.cond == 'CS+'], s=dotsize,
           label='CS+',
           facecolors='none',
           color="#C44E52",
           alpha=1, rasterized=True)
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-E'],
           y=data_avg_all.scr[data_avg_all.cond == 'CS-E'], s=dotsize,
           label='CS-E',
           facecolors='none',
           color="#55A868",
           alpha=1, rasterized=True)
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-1'],
           y=data_avg_all.pred[data_avg_all.cond == 'CS-1'], s=dotsize,
           color='#4C72B0',
           alpha=0.8,
           label='CS-1', rasterized=True)
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-2'],
           y=data_avg_all.pred[data_avg_all.cond == 'CS-2'], s=dotsize,
           color='#0d264f',
           alpha=0.8,
           label='CS-2', rasterized=True)
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS+'],
           y=data_avg_all.pred[data_avg_all.cond == 'CS+'], s=dotsize,
           color="#C44E52",
           alpha=0.8,
           label='CS+', rasterized=True)
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-E'],
           y=data_avg_all.pred[data_avg_all.cond == 'CS-E'], s=dotsize,
           color="#55A868",
           alpha=0.8,
           label='CS-E', rasterized=True)


# Find trials where new block begins
lines = []
for idx in range((len(data_avg_all.block) - 1)):
    if data_avg_all.block[idx + 1] != data_avg_all.block[idx]:
        lines.append(data_avg_all.trial_within_wb[idx] + 0.5)

for line in lines:
    ax.axvline(x=line, linestyle=':', color='k', alpha=0.5)


ax.set_ylabel('Observed / Predicted SCR', fontsize=param['labelfontsize'])
ax.set_xlabel('Trials within condition and block',
              fontsize=param['labelfontsize'])

ax.tick_params(labelsize=param['ticksfontsize'])
handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax.set_ylim(-0.25, 0.2)
ax.legend(by_label.values(), by_label.keys(),
          loc=(0.1, 0.9), fontsize=param['legendfontsize'],
          frameon=True, handletextpad=0.0, ncol=4, columnspacing=0.8)

# fig.tight_layout()
fig.savefig(opj(outfigpath, 'pred_scr_bytrial.svg'), dpi=600,
            bbox_inches='tight')

# Estimated quantities throught time
data_ns['cond2'] = 0
data_ns['cond2'] = np.where(data_ns['cond'] == 'CS++',
                            "CS+", data_ns['cond'])


data_avg_all = data_ns.groupby(['block',
                                'trial_within_wb_wcs',
                                'cond_plot2',
                                'cond2'])['scr',
                                          'pred',
                                          'vhat'].mean().reset_index()

xlabels = [r'Expected value $(\hat{\mu}_1)$',
           r'Irreducible uncertainty $(\hat{\sigma}_1)$',
           r'Estimation uncertainty $(\hat{\sigma}_2)$']
for ucue in data_avg_all['cond_plot2'].unique():
    selected = data_avg_all[data_avg_all.cond_plot2 == ucue].reset_index()

for idx, to_plot in enumerate(['vhat']):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    for ucue in data_avg_all['cond_plot2'].unique():
        selected = data_avg_all[data_avg_all.cond_plot2 == ucue].reset_index()

        if selected.cond_plot2.loc[0][0:3] == 'CS-':
            color1 = '#4C72B0'
            color2 = '#0d264f'
            leg1 = 'CS-1'
            leg2 = 'CS-2'
        else:
            color1 = '#c44e52'
            color2 = '#55a868'
            leg1 = 'CS+'
            leg2 = 'CS-E'

        sns.lineplot(x=selected.trial_within_wb_wcs,
                     y=selected[to_plot],
                     color=color1,
                     alpha=1,
                     ax=ax,
                     label=leg1)

        if selected.block.unique().shape[0] > 1:

            selected2 = selected[selected.block
                                 == selected.block.unique()[1]]
            sns.lineplot(x=selected2.trial_within_wb_wcs,
                         y=selected2[to_plot],
                         color=color2,
                         alpha=1,
                         ax=ax,
                         label=leg2)

    ax.set_ylabel(xlabels[idx], fontsize=param['labelfontsize'])
    ax.set_xlabel('Trials', fontsize=param['labelfontsize'])

    ax.tick_params(labelsize=param['ticksfontsize'])
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='best', fontsize=param["legendfontsize"]-6, frameon=True)

    for line in lines:
        ax.axvline(x=line, linestyle=':', color='k', alpha=0.5)

    # fig.tight_layout()
    fig.savefig(opj(outfigpath, 'traj_bytrial_' + to_plot + '.svg'), dpi=600,
                bbox_inches='tight')


# ################################################################
# Parameters plot
##################################################################

fig, ax = plt.subplots(1, 5, figsize=(4, 2.5))
pal = sns.color_palette("deep", 5)
labels = [r'$\alpha$', r'$v_0$', r'$\beta_0$', r'$\beta_1$', r'$\zeta$']
for idx, var in enumerate(['al', 'v_0', 'be0', 'be1', 'ze']):

    data_param = data.groupby(['sub'])[var].mean().reset_index()

    dplot = data_param.melt(['sub'])

    pt.half_violinplot(x='variable', y="value", data=dplot, inner=None,
                       color=pal[idx], width=0.6,
                       offset=0.17, cut=1, ax=ax[idx],
                       linewidth=1, alpha=0.6, zorder=19)
    sns.stripplot(x='variable', y="value", data=dplot,
                  jitter=0.08, ax=ax[idx],
                  linewidth=1, alpha=0.6, color=pal[idx], zorder=1)
    sns.boxplot(x='variable', y="value", data=dplot,
                color=pal[idx], whis=np.inf, linewidth=1, ax=ax[idx],
                width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                whiskerprops={'zorder': 10, 'alpha': 1},
                medianprops={'zorder': 11, 'alpha': 0.5})
    ax[idx].set_xticklabels([labels[idx]], fontsize=param['labelfontsize'])
    if idx == 0:
        ax[idx].set_ylabel('Value', fontsize=param['labelfontsize'])
    else:
        ax[idx].set_ylabel('')
    ax[idx].set_xlabel('')
    ax[idx].tick_params('y', labelsize=param['ticksfontsize']-4)
    ax[idx].tick_params('x', labelsize=param['ticksfontsize'])

    fig.tight_layout(pad=0.05)
    fig.savefig(opj(outfigpath, 'model_parameters.svg'), dpi=800)


# ################################################################
# Model comparison plots
##################################################################

# Compare families
famcomp = loadmat(opj(outpath, 'computational_models/',
                      'comp_families_VBA_model_comp.mat'))

modnames = ['RW\ncue-specific', 'PH\ncue-specific',
            'HGF2\ncue-specific', 'RW\ninter-cue', 'PH\ninter-cue',
            'HGF2\ninter-cue']

modnames.append('Family\ncue specific')
modnames.append('Family\ninter-cue')

ep = list(famcomp['out']['ep'][0][0][0])
ef = [float(ef)*100 for ef in famcomp['out']['Ef'][0][0]]

ef_fam = famcomp['out']['families'][0][0][0][0][4]
ep_fam = famcomp['out']['families'][0][0][0][0][6]

ep.append(ep_fam[0][0])
ep.append(ep_fam[0][1])

ef.append(float(ef_fam[0])*100)
ef.append(float(ef_fam[1])*100)


ep = np.asarray(ep)
ef = np.asarray(ef)
fig, host = plt.subplots(figsize=(4, 2.5))

par1 = host.twinx()
color1 = '#1D497B'
color2 = '#B4783C'

x = np.arange(0.5, (len(ep))*0.75, 0.75)
x2 = [c + 0.25 for c in x]
p1 = host.bar(x, ep, width=0.25, color=color1, linewidth=1, edgecolor='k')
p2 = par1.bar(x2, ef, width=0.25, color=color2, linewidth=1, edgecolor='k')

host.set_ylim(0, 1)
par1.set_ylim(0, 100)


# host.set_xlabel("Distance")
host.set_ylabel("Exceedance probability",
                fontsize=param["labelfontsize"])
par1.set_ylabel("Model Frequency (%)",  fontsize=param["labelfontsize"])


for ax in [par1]:
    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    plt.setp(ax.spines.values(), visible=False)
    ax.spines["right"].set_visible(True)

host.yaxis.label.set_color(color1)
par1.yaxis.label.set_color(color2)

host.spines["left"].set_edgecolor(color1)
par1.spines["right"].set_edgecolor(color2)
host.axvline(2.5, linestyle='--', color='gray')
host.axvline(4.75, linestyle='--', color='gray')

host.set_xticks([i+0.125 for i in x])
host.set_xticklabels(modnames, size=param['ticksfontsize'])

host.tick_params(axis='x', labelsize=param['labelfontsize'] - 7)

host.tick_params(axis='y', colors=color1, labelsize=param['labelfontsize'])
par1.tick_params(axis='y', colors=color2, labelsize=param['labelfontsize'])
fig.tight_layout()
fig.savefig(opj(outfigpath, 'model_comparison_families.svg'), dpi=800)


# Compare intercues
famcomp = loadmat(opj(outpath, 'computational_models/',
                      'comp_intercue_VBA_model_comp.mat'))

modnames = [str(m[0]) for m in famcomp['out']['options'][0][0][0][0][0][0]]
modnames = ['Null', 'RW', 'RW/PH', 'HGF2']


ep = famcomp['out']['pep'][0][0][0]
ef = [float(ef)*100 for ef in famcomp['out']['Ef'][0][0]]

fig, host = plt.subplots(figsize=(4, 2.5))

par1 = host.twinx()
color1 = '#1D497B'
color2 = '#B4783C'

x = np.arange(0.5, (len(ep))*0.75, 0.75)
x2 = [c + 0.25 for c in x]
p1 = host.bar(x, ep, width=0.25, color=color1, linewidth=1, edgecolor='k')
p2 = par1.bar(x2, ef, width=0.25, color=color2, linewidth=1, edgecolor='k')

host.set_ylim(0, 1)
par1.set_ylim(0, 100)


# host.set_xlabel("Distance")
host.set_ylabel("Protected exceedance probability",
                fontsize=param["labelfontsize"])
par1.set_ylabel("Model Frequency (%)",  fontsize=param["labelfontsize"])


for ax in [par1]:
    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    plt.setp(ax.spines.values(), visible=False)
    ax.spines["right"].set_visible(True)

host.yaxis.label.set_color(color1)
par1.yaxis.label.set_color(color2)

host.spines["left"].set_edgecolor(color1)
par1.spines["right"].set_edgecolor(color2)

host.set_xticks([i+0.125 for i in x])
host.set_xticklabels(modnames, size=param['ticksfontsize'])

host.tick_params(axis='x', labelsize=param['labelfontsize'])

host.tick_params(axis='y', colors=color1, labelsize=param['labelfontsize'])
par1.tick_params(axis='y', colors=color2, labelsize=param['labelfontsize'])
fig.tight_layout()
fig.savefig(opj(outfigpath, 'model_comparison_intercues.svg'), dpi=800)
