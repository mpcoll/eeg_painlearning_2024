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
import matplotlib.pyplot as plt
import seaborn as sns
from bids import BIDSLayout
from statsmodels.stats.multitest import multipletests
import pingouin as pg
###############################
# Parameters
# a

inpath = 'source'
outpathall = 'derivatives'


# Get BIDS layout
layout = BIDSLayout(inpath)

# Load participants
part = pd.read_csv(opj(inpath, 'participants.tsv'), sep='\t')
# Exlcude participants
part = part[part['excluded'] == 0]['participant_id'].tolist()


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


outfigpath = opj(outpathall, 'figures/pain_responses')
outpath = opj(outpathall, 'statistics/pain_responses')
if not os.path.exists(outpath):
    os.mkdir(outpath)
if not os.path.exists(outfigpath):
    os.mkdir(outfigpath)


#  ################################################################
# Figure pain ratings
#################################################################


# Load model data
mod_data = pd.read_csv(opj(outpathall, 'task-fearcond_alldata.csv'))
# Test ratings

mod_data.columns
pain_data = mod_data[~np.isnan(mod_data['ratings'])]
ratings = pain_data.groupby(['sub', 'block']).mean().reset_index()


# Remove 1st block no ratings
ratings = ratings[ratings['block'] != 1]


fig = plt.figure(figsize=(3, 3))
sns.pointplot(x='block', y='ratings', data=ratings,
              ci=95, color='#C44E52', alpha=0.05)
sns.swarmplot(x='block', y='ratings', data=ratings,
              color='gray', size=5, alpha=0.5)
plt.ylim((0, 80))
plt.xlabel('Blocks', fontsize=param['labelfontsize'])
plt.ylabel('Pain ratings', fontsize=param['labelfontsize'])
plt.xticks(fontsize=param['ticksfontsize'])
plt.yticks(fontsize=param['ticksfontsize'])
plt.tight_layout()
plt.savefig(opj(outfigpath, 'pain_ratings.svg'), dpi=600)

out = pg.rm_anova(data=ratings, dv='ratings', within='block', subject='sub')
df1 = out['ddof1']*out['eps'][0]
df2 = out['ddof2']*out['eps'][0]

ratings_all = ratings.groupby(['sub']).mean().reset_index()
ratings_all['ratings'].describe()

pain_data['trials'] = list(np.arange(1, 55))*len(np.unique(pain_data['sub']))

trial_plot = pain_data.groupby(['sub', 'trials']).mean().reset_index()


fig = plt.figure(figsize=(10, 3))
sns.pointplot(x='trials', y='ratings', data=trial_plot,
              ci=95, color='#C44E52', alpha=0.05)

for i in [9.5, 18.5, 27.5, 36.5, 45.5]:
    plt.axvline(i, color='gray', linestyle='--', alpha=0.5)
plt.ylim((0, 80))
plt.xlabel('Reinforced trials', fontsize=param['labelfontsize'])
plt.ylabel('Pain ratings', fontsize=param['labelfontsize'])
plt.xticks(fontsize=param['ticksfontsize'])
plt.yticks(fontsize=param['ticksfontsize'])
plt.tight_layout()
plt.savefig(opj(outfigpath, 'pain_ratings_trials.svg'), dpi=600)

out = pg.rm_anova(data=pain_data, dv='ratings', within='trials', subject='sub')
df1 = out['ddof1']*out['eps'][0]
df2 = out['ddof2']*out['eps'][0]


#  ################################################################
# Figure relationship matrix
#################################################################

t_out = np.load(opj(outpath, 'slopes_tvals.npy'))
p_out = np.load(opj(outpath, 'slopes_pvals.npy'))
var_names = list(np.load(opj(outpath, 'slopes_varnames.npy')))


fig, ax = plt.subplots(figsize=(4, 4))

# Create masks for upper diagnoal and significance
sqsize = int(np.sqrt(t_out.shape))
mask = np.triu(np.ones_like(t_out.reshape((sqsize, sqsize))))

# Get unique pvals
iv_mask = ~mask.astype(bool)
pvals_unique = p_out.copy().reshape((sqsize, sqsize))[iv_mask]

# Get significance (with correction)
p_corrected = multipletests(pvals_unique, alpha=0.05, method='holm')[1]

# Put corrected values in matrix
p_out_corrected = p_out.reshape((sqsize, sqsize))
p_out_corrected[iv_mask] = p_corrected


sig = np.where(p_out_corrected < 0.05, 0, 1)
mask_sig = np.where((sig + mask) == 0, 0, 1)
mask_notsig = np.where(mask_sig == 0, 1, 0) + mask
mask_diag = np.ones(sqsize**2).reshape(sqsize, sqsize)
mask_diag[np.diag_indices(sqsize)] = 0


t_out.reshape((sqsize, sqsize))[np.diag_indices(sqsize)] = 3
t_out_annot = np.round(t_out, 3).astype(str).reshape((sqsize, sqsize))


t_out = np.round(t_out, 4).reshape((sqsize, sqsize))
im1 = sns.heatmap(t_out.reshape((sqsize, sqsize)), square=True, vmin=0, vmax=6,
                  annot=True,
                  cmap='viridis', mask=mask, axes=ax,
                  annot_kws=dict(fontsize=param['ticksfontsize']),
                  cbar_kws={"shrink": .75})

im2 = sns.heatmap(t_out.reshape((sqsize, sqsize)), square=True, vmin=0, vmax=6,
                  cmap='Greys', mask=mask_notsig, axes=ax, annot=True, cbar=False,
                  annot_kws=dict(fontsize=param['ticksfontsize']))
im = sns.heatmap(mask_diag, square=True, vmin=-10, vmax=10, center=0,
                 cmap='seismic', mask=mask_diag, axes=ax, annot=False,
                 cbar=False)

# Correct axis
left, right = ax.get_xlim()
ax.set_xlim(left, right-1)
ax.tick_params(bottom=False, left=False)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=param['ticksfontsize'], bottom=False, left=False)
cax.set_ylabel('Multilevel t-value for slope ',
               fontsize=param['labelfontsize'], labelpad=10)
var_names_y = list(var_names)
var_names_y[0] = ''
ax.set_yticklabels(var_names_y,
                   rotation=0, fontsize=param['labelfontsize'], va="center")
ax.set_xticklabels(var_names, fontsize=param['labelfontsize'])

fig.savefig(opj(outfigpath, 'pain_corrmatrix.svg'),
            dpi=600, bbox_inches='tight')
