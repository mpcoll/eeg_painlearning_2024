# '''
#  # @ : -*- coding: utf-8 -*-
#  # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
#  # @ Date: 2023
#  # @ Description:
#  '''

from mne.report import Report
import pprint
import mne
import os
from os.path import join as opj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from mne.time_frequency import tfr_morlet
from bids import BIDSLayout
from tqdm import tqdm
###############################
# Parameters
###############################
inpath = '/media/mp/Crucial X8/2023_painlerarning_validate_R2Rpain/source'
outpath = '/media/mp/Crucial X8/2023_painlerarning_validate_R2Rpain/derivatives'
# Get BIDS layout
layout = BIDSLayout(inpath)

# Load participants
part = pd.read_csv(opj(inpath, 'participants.tsv'), sep='\t')
# Exlcude participants
part = part[part['excluded'] == 0]['participant_id'].tolist()


param = {
    # Length of epochs
    'erpbaseline': -0.20,  # Used for trial rejection
    'erpepochend': 1,
    'tfrbaseline': -0.50,  # Used for trial rejection
    'tfrcropend': 1,
    'tfrepochstart': -2,  # Used for TFR transform
    'tfrepochend': 2,
    'ttfreqs': np.arange(4, 101, 1),  # Frequencies
    'n_cycles': 0.5*np.arange(4, 101, 1),  # Wavelet cycles
    'testresampfreq': 256,  # Sfreq to downsample to
    'njobs': -1,  # N cpus to run TFR
    # Removed shocked trails
    'ignoreshocks': False,
}

##############################################################################
# EPOCH AND TF transform
##############################################################################

part.sort()
removed_frame = pd.DataFrame(index=part)
removed_frame['percleft_cue'] = 999
removed_frame['percleft_shock'] = 999
percleft_cue = []
percleft_shock = []
percremoved_cue_comperp = []
mod_data = pd.read_csv(opj(outpath, 'task-fearcond_alldata.csv'))

for p in tqdm(part):

    df = mod_data[mod_data['sub'] == p].reset_index(drop=True)
    df_s = df[df.trial_type == 'CS+S'].reset_index(drop=True)

    # ______________________________________________________
    # Make out dir
    indir = opj(outpath,  p, 'eeg')
    outdir = opj(outpath,  p, 'eeg', 'tfr')
    outdirerp = opj(outpath,  p, 'eeg', 'erps')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # ______________________________________________________
    # Load cleaned raw file
    raw = mne.io.read_raw_fif(opj(indir,
                                  p + '_task-fearcond_cleanedeeg_raw.fif'),
                              preload=True)

    # Load trial info in scr data
    events = pd.read_csv(layout.get(subject=p[-2:], extension='tsv',
                                    suffix='events',
                                    return_type='filename')[0], sep='\t')

    # Update samples stamp of events because some files were resampled
    evsamples = mne.find_events(raw)[:, 0][mne.find_events(raw)[:, 2] < 1000]
    events['sample'] = evsamples

    # Get erps metadata
    erps = mne.read_epochs(
        opj(outdirerp, p + '_task-fearcond_cues_singletrials-epo.fif'))
    meta = erps.metadata
    allbad = np.sum(meta.badtrial)

    # ________________________________________________________
    # Epoch according to condition

    # Drop unused channels
    chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
                                 'STI 014', 'Status'] if c in raw.ch_names]
    raw.drop_channels(chans_to_drop)

    events['empty'] = 0
    events_c = events[events['trial_type'].notna()]

    # # ________________________________________________________
    # # Epoch according to condition
    events_id = {
        'CS-1': 1,
        'CS-2': 2,
        'CS+':  3,
        'CS-E': 4,
    }

    events_c['cue_num'] = [events_id[s] for s in events_c.trial_cond4]
    events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])

    # Epoch for TFR
    tf_cues_strials = mne.Epochs(
        raw,
        events=events_cues,
        event_id=events_id,
        tmin=param['tfrepochstart'],
        baseline=None,
        metadata=meta,
        tmax=param['tfrepochend'],
        preload=True,
        verbose=False)

    # # TFR single trials
    strials = tfr_morlet(tf_cues_strials,
                         freqs=param['ttfreqs'],
                         n_cycles=param['n_cycles'],
                         return_itc=False,
                         use_fft=True,
                         decim=int(1024/param["testresampfreq"]),
                         n_jobs=param['njobs'],
                         average=False)

    # Clear for memory
    tf_cues_strials = None

    # Remove unused part
    strials.crop(tmin=param['tfrbaseline'],
                 tmax=param['tfrcropend'])

    # Check drop statistics
    percleft_cue.append(
        (len(strials) - np.sum(meta.badtrial))/len(strials)*100)
    percremoved_cue_comperp.append(100-((468 - allbad)/468*100))

    # Remove shocked trials and save different file
    strials_shocked = strials.copy()[df['trial_type'] == 'CS+S']
    strials = strials[df['trial_type'] != 'CS+S']

    strials.save(opj(outdir,  p + '_task-fearcond_'
                     + 'epochs-tfr.h5'), overwrite=True)
    strials_shocked.save(opj(outdir,  p + '_task-fearcond_'
                             + 'epochsreinforced-tfr.h5'), overwrite=True)

    strials = None  # Clear for memory
    strials_shocked = None  # Clear for memory
    # Same process but for TFR response to shock
    events_s = events[events.trigger_info == 'shock']
    events_s['cue_num'] = 255
    events_s['empty'] = 0
    events_shocks = np.asarray(events_s[['sample', 'empty', 'cue_num']])
    events_id = {
        'shock': 255,
    }

    erps = mne.read_epochs(
        opj(outdirerp, p + '_task-fearcond_shock_singletrials-epo.fif'))
    meta = erps.metadata

    tf_shock_strials = mne.Epochs(
        raw,
        events=events_shocks,
        event_id=events_id,
        tmin=param['tfrepochstart'],
        baseline=None,
        metadata=meta,
        tmax=param['tfrepochend'],
        preload=True,
        verbose=False)

    percleft_shock.append(
        (len(tf_shock_strials) - np.sum(meta.badtrial))/len(tf_shock_strials)*100)

    strials_shock = tfr_morlet(tf_shock_strials,
                               freqs=param['ttfreqs'],
                               n_cycles=param['n_cycles'],
                               return_itc=False,
                               use_fft=True,
                               decim=int(1024/param["testresampfreq"]),
                               n_jobs=param['njobs'],
                               average=False)

    # Remove unused parts
    strials_shock.crop(tmin=param['tfrbaseline'],
                       tmax=param['tfrcropend'])

    strials_shock.save(opj(outdir,  p + '_task-fearcond_' + 'shock_'
                           + 'epochs-tfr.h5'), overwrite=True)

    strials_shock = None

# Save rejection stats (should be the same as erps for cues)
removed_frame['percleft_cue'] = percleft_cue
removed_frame['percleft_shock'] = percleft_shock
removed_frame['percremoved_cue_comperp'] = percremoved_cue_comperp
removed_frame.to_csv(opj(outpath, 'task-fearcond_tfr_rejectionstats.csv'))
