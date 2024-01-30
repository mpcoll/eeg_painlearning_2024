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
import matplotlib.pyplot as plt
import scipy.io as sio
import json


# Where to create the bids dataset
bidsout = 'source'

# Where to create the bids dataset
derivativesout = 'derivatives'

if not os.path.exists(bidsout):
    os.mkdir(bidsout)


if not os.path.exists(derivativesout):
    os.mkdir(derivativesout)
    os.mkdir(opj(derivativesout, 'figures'))

# Helperunction to write to json files


def writetojson(outfile, path, content):
    """
    Helper to write a dictionnary to json
    """
    data = os.path.join(path, outfile)
    if not os.path.isfile(data):
        with open(data, 'w') as outfile:
            json.dump(content, outfile)
    else:
        print('File ' + outfile + ' already exists.')


# sub to run
part = pd.read_csv(opj(bidsout, 'participants.tsv'), sep='\t')

for sub in part.participant_id:

    outpath = opj(derivativesout, sub)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
        os.mkdir(opj(outpath, 'scr'))

    # Load physdata
    phys = pd.read_csv(opj(bidsout, sub, 'eeg',
                           sub + '_task-fearcond_physio.tsv'),
                       sep="\t")

    events = pd.read_csv(opj(bidsout, sub, 'eeg',
                             sub + '_task-fearcond_events.tsv'),
                         sep="\t")

    # Extract data and onsets
    scrdat = np.asarray(phys['scr'])
    trigdat = np.asarray(phys['trigger'])
    cue_ons_acq = np.asarray(phys['sample'][phys['events'] == 'cue'])
    fix_ons_acq = np.asarray(phys['sample'][phys['events'] == 'fix'])
    shock_ons_acq = np.asarray(phys['sample'][phys['events'] == 'shock'])
    rat_ons_acq = np.asarray(phys['sample'][phys['events'] == 'vas'])
    p_ons_acq = np.asarray(phys['sample'][phys['events'] == 'pause'])
    pause_dur = np.asarray(phys['duration_samp'][phys['events'] == 'pause'])
    rat_dur = np.asarray(phys['duration_samp'][phys['events'] == 'vas'])
    fix_dur = np.asarray(phys['duration_samp'][phys['events'] == 'fix'])

    cue = list(events['trial_cues'].dropna())
    len(cue)
    cue = [c.replace('.jpg', '') for c in cue]
    cuenum = [int(c[1:]) for c in cue]

    # Create epochs to plot
    epochlength = 10000
    screpochs = []

    if sub == 'sub-25':
        # Pad with zero because recording stopped too soon
        scrdat = np.append(scrdat, np.zeros(epochlength))
    for c in cue_ons_acq:
        screpochs.append(scrdat[c:c+epochlength])

    screpochs = np.stack(screpochs)

    conditions = list(phys['condition'].dropna())
    # Make an average for each condition and plot
    avg = []
    for cond in list(set(conditions)):
        ep = screpochs[np.argwhere(np.asarray(conditions) == cond), :]
        avg.append(np.average(np.squeeze(ep), 0))

    np.squeeze(screpochs[np.argwhere(conditions == cond), :])
    fig = plt.figure(figsize=(10, 10))
    plt.plot(np.swapaxes(avg, 1, 0))
    plt.legend(labels=list(set(conditions)))
    plt.xlabel('0-10 seconds from cue onset (samples) ')
    plt.ylabel('SCR')
    plt.title('Avg SCR for sub ' + sub + ', srate = ' + str(1000) + ' Hz')
    plt.savefig(opj(outpath, 'scr', 'import_average.png'), dpi=300)

    # Events timing over data and ACQ trigger
    # plot zscored data
    scrplot = (scrdat - np.mean(scrdat))/np.std(scrdat)
    # emgplot = (emgdat - np.mean(emgdat))/np.std(emgdat)
    trim = fix_ons_acq[0]
    fig = plt.figure(figsize=(20, 10))
    plt.plot(scrplot[trim:], 'k', label='raw scr')
    # plt.plot(emgplot[trim:], 'g', label='RMS emg')
    plt.plot(trigdat[trim:], 'b', label='raw trig')
    plt.plot(shock_ons_acq-trim, [5]*len(shock_ons_acq), marker='D',
             linestyle='',
             color='r', markersize=5, label='marked shocks')
    plt.plot(cue_ons_acq-trim, [6]*len(cue_ons_acq), marker='D',
             linestyle='',
             color='g', markersize=1, label='marked cues')
    plt.plot(fix_ons_acq-trim, [5.5]*len(fix_ons_acq), marker='D',
             linestyle='',
             color='b', markersize=1, label='marked fixations')
    plt.plot(rat_ons_acq-trim, [4.5]*len(rat_ons_acq), marker='D',
             linestyle='',
             color='y', markersize=5, label='marked ratings')
    for idx, p in enumerate(p_ons_acq):
        plt.axvline(p-trim, label='pause' + str(idx))
    plt.xlabel('Time (samples)')
    plt.ylabel('Raw SCR (Z scored)')
    plt.legend(loc='lower left')
    plt.title('Markers for sub ' + sub + ', srate = ' + str(1000) + ' Hz')
    plt.savefig(opj(opj(outpath, 'scr'), 'import_onsets.png'), dpi=300)
    plt.clf()
    plt.close('all')

    # ___________________________________________________________________
    # Save in a matlab stucture to process with PSPM
    impdata = {}
    impdata['scrdata'] = scrdat
    impdata['sub'] = sub
    impdata['srate'] = 1000
    impdata['emgsrate'] = 1000
    impdata['trigsrate'] = 1000
    impdata['rmsemgsrate'] = 1000
    impdata['cue_onsets'] = cue_ons_acq
    impdata['rat_onsets'] = rat_ons_acq
    impdata['rat_durations'] = rat_dur
    impdata['pause_onsets'] = p_ons_acq
    impdata['pause_durations'] = pause_dur
    impdata['fix_onsets'] = fix_ons_acq
    impdata['fix_durations'] = fix_dur
    impdata['shock_onsets'] = shock_ons_acq
    impdata['conditions'] = list(conditions)
    impdata['trial'] = range(1, 469)
    impdata['cue'] = cue
    impdata['cuenum'] = cuenum

    sio.savemat(opj(outpath, 'scr', sub + '_scr_pspm.mat'),
                impdata)
    np.savetxt(opj(outpath, 'scr',  sub + '_scr_data.txt'), scrdat)
