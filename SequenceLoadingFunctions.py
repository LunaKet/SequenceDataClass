#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 12:21:33 2025

@author: loon
"""
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import contextlib
from spatialFilter import spatial_filter

#lab code - built to parse through terabytes of data.
#I have chosen not to containerize this and leave it as is, 
#because the data is not available anyway.
from kettlewell.basic_tools import event_tools
from kettlewell.basic_tools.activity_sequences import load_binary_full
from clmustela.datasets import Ferret, stim_type


# %% Optimized loading routines
# TODO! add docs for these function
def load_PFS_sequences_centered_fast(sequences, templateGroups, t_steps=50, nSkips=0, multiprocess=15):

    df = get_seqCenteredDF(sequences, templateGroups, t_steps=t_steps)
    seqCentered = event_tools.load_PFS_sequences_many(df, pad_pre=0, pad_post=1)
    seqCentered = [seq.T for seq in seqCentered]

    pool = Pool(processes=multiprocess)
    seqCentered = list(tqdm(pool.imap(
        partial(spatial_filter, roi=sequences.roi), seqCentered),
        total=len(seqCentered))
        )
    pool.close()
    seqCentered = np.stack(seqCentered, axis=0)[:,:,sequences.roi]

    seqCenteredBin = load_PFSbin_sequences_many(df, pad_pre=0, pad_post=1)
    seqCenteredBin = np.stack(seqCenteredBin)

    return seqCentered, seqCenteredBin


def load_PFSbin_sequences_many(df, pad_pre=2, pad_post=0):
    '''
    load many sequences with padding on either side
    '''
    ferret = int(df.loc[0].ferret[1:])
    seqID = 'all_may25'

    datasets = Ferret(ferret).datasets_with([stim_type.none_black])
    seqs = [[]]*len(df) #so that we can return the seqs in the same order as df
    for d in datasets:
        if d.series_number in np.unique(df.tser):
            print(d.series_number)
            tseries_idx = d.series_number
            pfs_bin = load_binary_full([d], seqID)[0]
            d_idxs = np.where(df.tser==tseries_idx)[0]
            indices_start = df.loc[d_idxs].indice_start.values - pad_pre -1
            indices_stop = df.loc[d_idxs].indice_stop.values + pad_post - 1
            indices = zip(indices_start, indices_stop)

            seqs_ = [pfs_bin[:, i_start:i_stop] for (i_start, i_stop) in indices]
            if len(indices_stop)>1:
                nSteps = indices_stop[1] - indices_start[1]
            else:
                nSteps = indices_stop[0] - indices_start[0]
            for i in range(len(d_idxs)):
                d_idx_ = d_idxs[i]
                if seqs_[i].T.shape[0] == nSteps:
                    seqs[d_idx_] = seqs_[i].T
                else:
                    seqs[d_idx_] = np.zeros((nSteps, seqs_[i].T.shape[1]))

    return seqs


def get_seqCenteredDF(sequences, templateGroups, t_steps=3):
    #creates a dataframe that has required information for retrieving sequences
    nSeqs = np.sum([len(g) for g in templateGroups.groupIdxs])
    tseries = np.zeros(nSeqs, int)
    ferret = sequences.ferret
    center = np.zeros(nSeqs, int)

    seqs_ = np.concatenate([templateGroups.seqIdx[group] for group in templateGroups.groupIdxs])
    frames_ = np.concatenate([templateGroups.frameIdx[group] for group in templateGroups.groupIdxs])
    for i, (s, fr) in enumerate(zip(seqs_, frames_)):
        tseries[i] = (sequences.df.iloc[s]['tser'])
        center[i] = sequences.df.iloc[s]['indice_start'] + fr + 1

    df_s = pd.DataFrame({
        'tser': tseries,
        'center_idx': center}
        )
    df_s['indice_start'] = df_s.center_idx - t_steps
    df_s['indice_stop']  = df_s.center_idx + t_steps
    df_s['ferret'] = 'F' + str(ferret)

    return df_s


# %% Lab code for parsing through TB of data
def load_PFS_saved(dataset, PFS_file_identifier=False):
    '''
    loads in one at a time. Likely will run with loop, i.e.
    for d in datasets:
        load_PFS_saved(d)
        #do stuff

    it's too big to load any more than 4-5 at a time
    '''

    if PFS_file_identifier:
        fdir = f'DF_F0_PFS_{PFS_file_identifier}/'
    else:
        fdir = 'DF_F0_PFS/'

    filename = f'DF_F0_PFS_tser{dataset.series_number}.npy'

    return np.load(os.path.join(dataset.path_tseries_dir,fdir,filename))


def load_PFS_sequence(df=None, dict_seq=None, pad_pre=2, pad_post=0):
    '''
    load a single sequence with padding on either side
    use a dataframe row for easiest use
    dict_seq must include tseries, indice_start, indice_stop, and ferret
    '''

    if df is not None:
        ferret = int(df.ferret[1:])
        tseries = df.tser
        indice_start = df.indice_start - pad_pre
        indice_stop = df.indice_stop + pad_post

    elif dict_seq:
        ferret = dict_seq['ferret']
        tseries = dict_seq['tseries']
        indice_start = dict_seq['indice_start'] - pad_pre
        indice_stop = dict_seq['indice_stop'] + pad_post

    with contextlib.redirect_stdout(None):
        datasets = Ferret(ferret).datasets_with([stim_type.none_black])
        tseries_list = np.array([d.series_number for d in datasets])

        tseries_idx = np.argwhere(tseries_list == tseries)[0][0]
        dataset = datasets[tseries_idx]

        pfs_sequence = load_PFS_saved(dataset)[:, 99:]
        pfs_sequence = pfs_sequence[:, indice_start:indice_stop]

    return pfs_sequence


def load_PFS_sequences_many(df, pad_pre=2, pad_post=0):
    '''
    load many sequences with padding on either side
    *massive* speedup loading if you have more sequences than datasets
    '''
    ferret = int(df.loc[0].ferret[1:])
    # if df is not None:
    #
    #     tseries = df.tser
    #     indice_start = df.indice_start - pad_pre
    #     indice_stop = df.indice_stop + pad_post

    datasets = Ferret(ferret).datasets_with([stim_type.none_black])
    seqs = [[]]*len(df) #so that we can return the seqs in the same order as df
    for d in datasets:
        if d.series_number in np.unique(df.tser):
            tseries_idx = d.series_number
            pfs_data = load_PFS_saved(d)[:, 99:]
            d_idxs = np.where(df.tser==tseries_idx)[0]
            indices_start = df.loc[d_idxs].indice_start.values - pad_pre
            indices_stop = df.loc[d_idxs].indice_stop.values + pad_post
            indices = zip(indices_start, indices_stop)

            seqs_ = [pfs_data[:, i_start:i_stop] for (i_start, i_stop) in indices]
            nSteps = indices_stop[0] - indices_start[0]

            for i in range(len(d_idxs)):
                d_idx_ = d_idxs[i]
                if seqs_[i].T.shape[0] == nSteps:
                    seqs[d_idx_] = seqs_[i].T
                else:
                    seqs[d_idx_] = np.zeros((nSteps, seqs_[i].T.shape[1]))

    return seqs