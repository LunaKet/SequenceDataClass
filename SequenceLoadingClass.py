#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:46:53 2024

@author: ckettlew
"""
import pickle
import numpy as np
import pandas as pd

from spatialFilter import spatial_filter


# %%Loading classes
class Sequences():
    def __init__(self, SeqFile, animalID=None):
        """
        Parameters
        ----------
        SeqFile : string
            file where sequence data is.
            This is a pickled dictionary, containing the following :
                'roi': binary image of ROI
                'df': dataframe of key metrics
                'event_series': data, as a list of events
                'event_bin': data, but binarized according to a std threshold
          
        Returns
        -------
        None.

        Example loading:
        seqs = Sequences(fdir)
        seqs.removeSeqsThatDontPass(seqs.df.active_pix>500)
        seqs.get_spatialfiltered() 

        Example slicing:
        cluster_1 = Seq335.SeqsSF[[1,50,111,156,257,310,509]]
        first100 = Seq335.SeqsSF[0:100]
        """
        self.animalID = animalID
        self.SeqFile = SeqFile
        self._load_seqs()
        self._get_indexes()

    def summary(self):
        print(f'''
        ferret = {self.animalID}
        fdir = {self.SeqFile.get_seq_data_file()}
        nSeqs = {len(self.Seqs[:])}
        nSeqsBin = {len(self.SeqsBin[:])}
              ''')

    def removeSeqsThatDontPass(self, mask):
        """
        Parameters
        ----------
        mask : bool array
            Same length as nSeqs. True keeps the sequence, False removes.

        Returns
        -------
        None.
        
        Example usage:
            Sequences.removeSeqsThatDontPass(Sequences.df.active_pix>500)

        """
        keep_idx = np.where(mask==True)[0]
        self.df = self.df[mask]
        self.df.reset_index(drop=True, inplace=True)
        self.SeqsBin = SequenceSlices(self.SeqsBin[keep_idx])
        self.Seqs = SequenceSlices(self.Seqs[keep_idx])
        self._update_counts()

    def get_spatialfiltered(self, sig_low=None, sig_high=None):
        sf = [spatial_filter(s, self.roi, sig_low=sig_low, sig_high=sig_high)[:, self.roi].T for s in self.Seqs]
        self.SeqsSF = SequenceSlices(sf)

    def _load_seqs(self):
        with open(self.SeqFile.get_seq_data_file(), 'rb') as f:
            loaded_seq_data = pickle.load(f)

        self.roi = loaded_seq_data['roi']
        self.df = loaded_seq_data['df']
        self.SeqsBin = SequenceSlices(loaded_seq_data['event_bin'])
        self.Seqs = SequenceSlices(loaded_seq_data['event_series'])
        self.nPix = self.Seqs[0].shape[0]
        self._update_counts()

    def _get_indexes(self):
        self.seqIdx = np.zeros(self.nFrames, dtype=int)
        self.frameIdx = np.zeros(self.nFrames, dtype=int)

        pos = 0
        for i in range(self.nSeqs):
            nframes = self.Seqs[i].shape[1]
            self.seqIdx[pos:pos+nframes] = i
            self.frameIdx[pos:pos+nframes] = range(0,nframes)
            pos += nframes

    def _update_counts(self):
        self.nSeqs = len(self.Seqs[:])
        self.nFrames = self.Seqs.stack().shape[0]
        self._get_indexes()


class SequenceSlices():
    '''
    provides intuitive interface for brackets, much like numpy
    '''
    def __init__(self, segmented_data):
        self.segmented_data = segmented_data

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.segmented_data[key]
        if isinstance(key, slice):
            return [self.segmented_data[ii] for ii in range(*key.indices(len(self.segmented_data)))]
        if isinstance(key, np.ndarray):
            key=list(key)
        if isinstance(key, list):
            return [self.segmented_data[ii] for ii in key]
        
        #else
        return IndexError('invalid slice')

    def stack(self):
        return np.hstack(self.segmented_data).T


# %%Centered data
class SequencesCenteredLabeled():
    '''
    Functions for managing sequences loaded around a central "active frame" with a standard temporal window size.
    This class is a container for the centered data.
    This is deeply intertwined with the lab's source code, dependencies, and TB-scale data.
    That is to say, there's no way to demonstrate this functionality outside of the lab environment.
    '''
    def __init__(self, data):
        """
        Parameters
        ----------
        data : Array (nSeqs x nSteps x nPix)
            pre-load data with SequenceLoadingFunction.load_PFS_sequences_centered_fast()

        Returns
        -------
        None.

        """
        self.origData = data
        self.matchedGroupIdxs = range(len(data))
        self.orignSeqs, self.nSteps, self.nPix = data.shape
        self.centerFrame = int(self.nSteps/2)

    def get_frameCollapsed(self):
        return np.reshape(self.data, (self.nSeqs, self.nSteps*self.nPix))

    def get_frameCollapsedGroup(self, idx):
        return np.reshape(self.data[idx,:,:], (self.nSeqs, self.nSteps*self.nPix))

    def get_framestacked(self):
        return np.reshape(self.data, (self.nSeqs*self.nSteps, -1))

    def set_matched_groups(self, idxs=None, groupSize=None):
        if idxs is not None:
            self.matchedGroupIdxs = idxs
        else:
            nGroups = np.unique(self.origGroupLabels)
            groupSize = self._check_group_size(groupSize) 
            newIdxs = []
            for group in nGroups:
                matchedIdx = np.random.permutation(np.where(self.origGroupLabels==group)[0])[0:groupSize]
                newIdxs.append(matchedIdx)
            self.matchedGroupIdxs = np.concatenate(newIdxs).astype(int)

        return self

    def _check_group_size(self, groupSize):
        nGroups = np.unique(self.origGroupLabels)
        minSize = np.min([np.sum(self.origGroupLabels==i) for i in nGroups])
        if groupSize is None:
            groupSize = minSize
        else:
            if groupSize > minSize:
                groupSize = minSize
                print("declared groupSize is bigger than max possible matched groupSize")

        return groupSize

    def set_groups(self, groupLabels):
        self.origGroupLabels = groupLabels

    @property
    def data(self):
        return self.origData[self.matchedGroupIdxs,:,:]

    @property
    def nSeqs(self):
        return len(self.matchedGroupIdxs)

    @property
    def groupLabels(self):
        return self.origGroupLabels[self.matchedGroupIdxs]