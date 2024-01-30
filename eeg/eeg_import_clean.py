'''
 # @ : -*- coding: utf-8 -*-
 # @ Author: Michel-Pierre Coll (michel-pierre.coll@psy.ulaval.ca)
 # @ Date: 2023
 # @ Description:
 '''

from mne.report import Report
import pprint
import mne
import os
from mne.preprocessing import ICA, create_eog_epochs
from os.path import join as opj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bids import BIDSLayout
from scipy.stats import zscore
from mne_icalabel import label_components

###############################
# Parameters
##############################

inpath = '/media/mp/Crucial X8/2023_painlerarning_validate_R2Rpain/source'
outpath = '/media/mp/Crucial X8/2023_painlerarning_validate_R2Rpain/derivatives'

# Get BIDS layout
layout = BIDSLayout(inpath)

# Load participants
part = pd.read_csv(opj(inpath, 'participants.tsv'), sep='\t')

# Exlcude participants
part = part[part['excluded'] == 0]['participant_id'].tolist()


param = {
    # EOG channels
    'eogchan': ['EXG3', 'EXG4', 'EXG5'],
    # Empty channels to drop
    'dropchan': ['EXG6', 'EXG7', 'EXG8'],
    # Channels to rename
    'renamechan': {'EXG1': 'M1', 'EXG2': 'M2', 'EXG3': 'HEOGL',
                   'EXG4': 'HEOGR', 'EXG5': 'VEOGL'},


    # Montage to use
    'montage': 'standard_1005',
    # High pass filter cutoff
    'hpfilter': 0.1,
    # Low pass filter cutoff
    'lpfilter': 100,
    # Filter to use
    'filtertype': 'fir',
    # Plot for visual inspection (in Ipython, change pyplot to QT5)
    'visualinspect': False,
    # Reference
    'ref': 'average',
    # ICA parameters
    # Decimation factor before running ICA
    'icadecim': 4,
    # Set to get same decomposition each run
    'random_state': 23,
    # How many components keep in PCA
    'n_components': None,
    # Reject trials exceeding this amplitude before ICA
    'erpreject': dict(eeg=500e-6),
    # Algorithm
    'icamethod': 'infomax',
    'icafitparams': dict(extended=True),
    # Visually identified bad channels
    'badchannels': {'23': ['T7', 'F6'],
                    '24': ['F7', 'FC4', 'Fp1', 'AF4', 'FC3'],
                    '25': ['AF7', 'AF8'],
                    '26': ['P9'],
                    '27': ['AF7', 'Fp2'],
                    '28': [],
                    '29': ['F5'],
                    '30': [],
                    '31': ['T8', 'FC5', 'TP8'],  # Exclude, bad SCR
                    '32': ['P9'],
                    '33': ['F6', 'FT8', 'AF7'],
                    '34': ['T7', 'Iz'],
                    '35': [],  # Exclude, bad SCR
                    '36': ['P9', 'FC3'],
                    '37': ['AFz', 'AF8'],
                    '38': ['P10', 'O2'],
                    '39': ['P9'],
                    '40': ['Fpz', 'Fp2'],
                    '41': ['TP7'],
                    '42': ['AF8'],  # Exclude, many bad channels
                    '43': ['FC2', 'FC5'],
                    '44': ['FC2', 'AF8', 'Fp1'],
                    '45': ['T7'],
                    '46': [],
                    '47': ['T8'],
                    '48': [],
                    '49': [],
                    '50': [],
                    '51': ['AF3'],
                    '52': ['P10'],
                    '53': ['P9', 'AF7'],
                    '54': ['AF3', 'Fp1'],
                    # Exclude, ++ bad channels
                    '55': ['Fp2', 'F6', 'AF8', 'FT8'],
                    '56': ['P2', 'P10', 'AF7', 'Iz', 'P10'],
                    '57': ['Oz', 'O1', 'TP8']}}


# Output dir
outdir = opj(outpath)

count = 0
nbic = []
for p in part:
    import warnings
    warnings.simplefilter('ignore')

    ###############################
    # Initialise
    ##############################
    print('Processing participant '
          + p)

    # _______________________________________________________
    # Make fslevel part dir
    pdir = opj(outdir, p, 'eeg')
    if not os.path.exists(pdir):
        os.mkdir(pdir)

    # _______________________________________________________
    # Initialise MNE report
    report = Report(verbose=False, subject=p,
                    title='EEG report for part ' + p)

    # report.add_htmls_to_section(
    #     htmls=part.comments[p], captions='Comments', section='Comments')
    report.add_html(pprint.pformat(param),
                    title='Parameters',
                    section='Parameters')

    # ______________________________________________________
    # Load EEG file
    f = layout.get(subject=p[-2:], extension='bdf', return_type='filename')[0]
    raw = mne.io.read_raw_bdf(f, verbose=True,
                              eog=param['eogchan'],
                              exclude=param['dropchan'],
                              preload=True)

    # Rename external channels
    raw.rename_channels(param['renamechan'])

    events = pd.read_csv(layout.get(subject=p[-2:], extension='tsv',
                                    suffix='events',
                                    return_type='filename')[0], sep='\t')

    # For part with sampling at 2048 Hz, downsample and correct events samples
    print(raw.info['sfreq'])
    if raw.info['sfreq'] == 2048:
        print(p)
        print('Resampling data to 1024 Hz')
        raw.resample(1024)
        events['sample'] = [e[0]
                            for e in mne.find_events(raw)]

    # ______________________________________________________
    # Get events

    # Keep only rows for cues
    events_c = events[events['trial_type'].notna()]  # Cues
    events_s = events[events.trigger_info == 'shock']  # Shcok

    # Get events count
    events_count = events_c.trial_type.value_counts()

    # ReCalculate duration between events to double check
    events_c['time_from_prev2'] = np.insert((np.diff(events_c['sample'].copy()
                                                     / raw.info['sfreq'])),
                                            0, 0)

    events_c.to_csv(opj(pdir, p + '_task-fearcond_events.csv'))
    pd.DataFrame(events_count).to_csv(opj(pdir,
                                          p
                                          + '_task-fearcond_eventscount.csv'))

    # ______________________________________________________
    # Load and apply montage

    raw = raw.set_montage(param['montage'])

    raw.plot_sensors(show_names=True, show=False)
    raw.load_data()  # Load in RAM

    # ________________________________________________________________________
    # Remove bad channels
    if param['visualinspect']:
        raw.plot(
            n_channels=raw.info['nchan'],
            scalings=dict(eeg=0.00020),
            highpass=1,
            lowpass=50,
            block=True)

    raw.info['bads'] = param['badchannels'][p[-2:]]

    # Plot sensor positions and add to report
    plt_sens = raw.plot_sensors(show_names=True, show=False)
    report.add_figure(
        plt_sens,
        title='Sensor positions (bad in red)',
        section='Preprocessing')

    # Rereference to average
    raw.set_eeg_reference(param['ref'], projection=False)

    # _______________________________________________________________________
    # Bandpass filter
    raw_ica = raw.copy()  # Create a copy  to use different filter for ICA

    raw = raw.filter(
        param['hpfilter'],
        None,
        method=param['filtertype'],
        verbose=True)

    # ______________________________________________________________________
    # Plot filtered spectrum
    plt_psdf = raw.plot_psd(
        area_mode='range', tmax=10.0, average=False, show=False)
    report.add_figure(
        plt_psdf, title='Filtered spectrum', section='Preprocessing')

    # ________________________________________________________________________
    # Clean with ICA

    # Make epochs around trial for ICA
    events_c['empty'] = 0
    events_c['triallabel'] = ['trial_' + str(i) for i in range(1, 469)]
    events_c['trialnum'] = range(1, 469)
    events_array_cues = np.asarray(events_c[['sample', 'empty', 'trialnum']])

    alltrialsid_cues = {}
    for idx, name in enumerate(list(events_c['triallabel'])):
        alltrialsid_cues[name] = int(idx + 1)

    # Low pass more agressively for ICA
    raw_ica = raw_ica.filter(l_freq=1, h_freq=100)
    epochs_ICA_cues = mne.Epochs(
        raw_ica,
        events=events_array_cues,
        event_id=alltrialsid_cues,
        tmin=-1,
        baseline=None,
        tmax=2,
        preload=True,
        reject=param['erpreject'],
        verbose=False)

    print('Processing ICA for part ' + p + '. This may take some time.')
    ica = ICA(n_components=param['n_components'],
              method=param['icamethod'],
              random_state=param['random_state'],
              fit_params=param['icafitparams'])

    ica.fit(epochs_ICA_cues, decim=param['icadecim'])

    # Get bad ICA using IClabels
    ica_labels = label_components(epochs_ICA_cues, ica, method='iclabel')

    remove = [1 if ic in ['channel noise', 'eye blink', 'muscle artifact',
                          'line noise', 'heart beat', 'eye movement']
              and prob > 0.70 else 0 for ic,
              prob in zip(ica_labels['labels'], ica_labels['y_pred_proba'])]

    bad_ica = np.argwhere(remove).flatten()
    if p == 'sub-44':
        bad_ica = list(bad_ica) + [8, 11]
    ica.exclude = list(bad_ica)

    report.add_html(pd.DataFrame(ica_labels).to_html(),
                    'ICA labels')

    # Identify which ICA correlate with eye blinks
    chaneog = 'VEOGL'
    eog_averagev = create_eog_epochs(raw_ica, ch_name=chaneog,
                                     verbose=False).average()

    # Find EOG ICA via correlation
    eog_epochsv = create_eog_epochs(
        raw_ica, ch_name=chaneog, verbose=False)  # get single EOG trials
    eog_indsv, scoresr = ica.find_bads_eog(
        eog_epochsv, ch_name=chaneog, verbose=False)  # find correlation

    report.add_ica(ica=ica, inst=epochs_ICA_cues, eog_evoked=eog_epochsv.average(),
                   eog_scores=scoresr,
                   title='ICA components - passive cues')

    # Plot removed ICA and add to report
    fig = ica.plot_sources(eog_averagev,
                           show=False,
                           title='ICA removed on eog epochs')

    report.add_figure(fig, section='ICA',
                      title='Removed components '
                      + 'highlighted')

    report.add_html("Number of removed ICA: " + str(len(ica.exclude)),
                    title="""ICA- Removed""", section='Artifacts')

    nbic.append(len(ica.exclude))

    # Apply ICA
    raw = ica.apply(raw)

    # Interpolate channels
    if raw.info['bads']:
        raw.interpolate_bads(reset_bads=True)
    # ______________________________________________________________________
    # Save cleaned data
    raw.save(opj(pdir, p + '_task-fearcond_cleanedeeg_raw.fif'),
             overwrite=True)

    #  _____________________________________________________________________
    report.save(opj(pdir, p + '_task-fearcond_importclean_report.html'),
                open_browser=False, overwrite=True)

# For manuscript
stats_dict = {}

# Number of bad channels
nbc = []
for p, bc in param['badchannels'].items():
    if 'sub-' + p in part:
        nbc.append(len(bc))


stats_dict = pd.DataFrame()
stats_dict['sub'] = part
stats_dict['n_bad_ica'] = nbic
stats_dict['n_bad_channels'] = nbc
stats_dict.describe().to_csv(opj(outpath,
                                 'task-fearcond_importclean_statsv2.csv'))
