#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:46:53 2024

@author: ckettlew
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, leaves_list
import contextlib
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from clmustela.datasets import parse_directory_structure, day_analysis_dir, descriptions
from kettlewell.basic_tools import event_tools
from kettlewell.basic_tools.activity_sequences import load_binary_full

def example_commands():
    Seq335 = Sequences(335)
    Seq335.removeSeqsThatDontPass(Seq335.df.active_pix>500)
    Seq335.get_spatialfiltered() #run after removeSeqs so it loads significantly faster

    #get a group, probably computed from something in another script
    cluster_1 = Seq335.SeqsSF[[1,50,111,156,257,310,509]]
    cluster_1_bin = Seq335.SeqsBin[[1,50,111,156,257,310,509]]

    #or just the first 100
    first100 = Seq335.SeqsSF[0:100]

# %%General classes
class Sequences():
    def __init__(self, ferret, SeqDir=None):
        self.ferret = ferret
        self._get_SeqDir(SeqDir)
        self._load_seqs()
        self._get_indexes()

    def summary(self):
        print(f'''
        ferret = {self.ferret}
        fdir = {self.SeqDir.get_seq_data_file()}
        nSeqs = {len(self.Seqs[:])}
        nSeqsBin = {len(self.SeqsBin[:])}
              ''')

    def removeSeqsThatDontPass(self, mask):
        #mask is the sequences to keep
        #i.e., Sequences.removeSeqsThatDontPass(Sequences.df.active_pix>500)
        keep_idx = np.where(mask==True)[0] #[i for i, mask_ in enumerate(mask) if mask_ is True]
        self.df = self.df[mask]
        self.df.reset_index(drop=True, inplace=True)
        self.SeqsBin = SequenceSlices(self.SeqsBin[keep_idx])
        self.Seqs = SequenceSlices(self.Seqs[keep_idx])
        self._update_counts()

    def get_spatialfiltered(self, sig_low=None, sig_high=None):
        sf = [event_tools.spatial_filter(s, self.roi, sig_low=sig_low, sig_high=sig_high)[:, self.roi].T for s in self.Seqs]
        self.SeqsSF = SequenceSlices(sf)

    def _load_seqs(self):
        with open(self.SeqDir.get_seq_data_file(), 'rb') as f:
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

    def _get_SeqDir(self, SeqDir):
        if SeqDir is None:
            self.SeqDir = SeqDirectory(self.ferret)
        else:
            self.SeqDir = SeqDir


class SequenceSlices():
    #provides intuitive interface for brackets, much like numpy
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

        #if nothing is returned
        return IndexError('invalid slice')

    def stack(self):
        return np.hstack(self.segmented_data).T


class SeqDirectory():
    '''
    ferret: int
    '''
    def __init__(self, ferret):
        #setup
        self.ferret = ferret
        self._PFS_fileID=None
        self._seq_dirID='julai6_ALL'
        self._sigID='julai7'
        self._dfID=None
        self._seq_basefile = 'sequence_dataframe'

        #file operations
        self._date = parse_directory_structure.get_all_dates_for_ferret(ferret)[0]
        self.fdir


    def summary(self):
        print(f'''
        'PFS': {self._PFS_fileID}
        'seq': {self._seq_dirID}
        'sig': {self._sigID}
        'df': {self._dfID}
              ''')

    def set_PFS_fileID(self, file):
        self._PFS_fileID = file

    def set_seq_fileID(self, file):
        self._seq_fileID = file

    def set_sig_dir_id(self, file):
        self._sig_fileID = file

    @property
    def date(self):
        return self._date

    @property
    def fdir(self):
        base_fdir = day_analysis_dir.format(ferret=self.ferret, date=self._date)
        self._fdir = os.path.join(base_fdir, 'Sequences')
        return self._fdir

    def get_seq_data_file(self):

        if self._seq_dirID is not None:
            file =  self._seq_basefile + f'_{self._seq_dirID}'
        file = file + '.pkl'

        return os.path.join(self._fdir, file)

# %%Corr mat indexer and sorter
# TODO!
#     debug and test both functions

class SeqFramesCorrSorter():
    '''
    Class to hold sequence correlation matrix and perform various sorting operation on it.
    Very helpful when trying to maintain accurate indexing across sorts.
    '''
    def __init__(self, seqCentered):
        ### constants ###
        self.nSteps = seqCentered.nSteps
        self.nSeqs = seqCentered.nSeqs
        self.centerFrame = int(self.nSteps/2)
        self.Seqs = seqCentered.data
        self.corrMats #function

        ### mutable variables ###
        self.current_seqs = [i for i in range(self.nSeqs)]
        self.sort_order = np.repeat(np.expand_dims(np.arange(0,self.nSeqs),0), self.nSteps, axis=0).astype(int)
        # self._sort_group()
        self.sort_type = 'None'

    def summary(self):
        print(
        f'''
        Number of sequences: {len(self.seq_list)}
        Frames per sequence: {self.nSteps}
        Sort type: {self.sort_type}
        '''
            )

    @property
    def corrMats(self):
        if not hasattr(self, "_corrMats"):
            corrMats = np.zeros((self.nSteps, self.nSeqs, self.nSeqs))
            for i in range(self.nSteps):
                corrMats[i,:,:] = np.corrcoef(self.Seqs[:,i,:])
            self._corrMats = corrMats

        return self._corrMats

    def sorted_corr_mats(self, sort_type=None, sort_arg=None):
        #values can be called independently, or within this method for convenience
        if sort_type is not None:
            self.compute_sort_order(sort_type, sort_arg)
        corr_mats_ = self.corrMats.copy()

        for i in range(self.nSteps):
            corr_mats_[i,:,:] = corr_mats_[i, self.sort_order[i,:],:][:,self.sort_order[i,:]]

        return corr_mats_

    def set_currentSeqs(self, idxs):
        slices = np.zeros(self.nSeqs, bool)
        slices[idxs] = True
        self.current_seqs = np.where(slices)[0]

    def compute_sort_order(self, sort_type='group', args=None):
        """
        this is the power of this class. do any sorting, add more later, all functionality still available
        arg1 informs the sort order

        Parameters
        ----------
        sort_type : string
            'full': sequences are flattened and full correlations are taken
                args is how many frames to take
            'independent': each time point is hierarchically clustered separately
            'center': see 'frame'
            'frame': a single time point is hierarchically clustered,
                and this is extended to every time point
                args is the timepoint to sort
        """
        self._sortDefault()
        self.sort_type = sort_type

        if (sort_type is None) or (sort_type=='None'):
            pass #defaults to the group ordering, as done two lines above
        elif sort_type == 'full':
            self._sortFull(args)
        elif sort_type == 'independent':
            self._sortIndependent()
        elif sort_type == 'center':
            self._sortFrame(0)
        elif sort_type =='frame':
            self._sortFrame(args)

    #various sorting schemes
    def _sortDefault(self):
        self.sort_order = np.repeat(np.expand_dims(np.arange(0,len(self.current_seqs)),0),
                                    self.nSteps, axis=0).astype(int)

    def _sortFull(self, args):
        data = self.Seqs[:,(self.centerFrame-args):(self.centerFrame+args+1),:]
        data = np.reshape(data, (self.nSeqs, -1))
        dist_mat = 1 - np.corrcoef(data)
        Z = linkage(dist_mat[np.triu_indices(len(dist_mat), 1)], method='ward', optimal_ordering=True)
        res_order = leaves_list(Z)
        for i in range(self.nSteps):
            self.sort_order[i,:] = self.sort_order[i,res_order]

    def _sortIndependent(self):
        corr_mats = self._get_sliced_corr_mats()
        for i in range(self.nSteps):
             dist_mat = 1 - corr_mats[i,:,:]
             Z = linkage(dist_mat[np.triu_indices(len(dist_mat), 1)], method='ward', optimal_ordering=True)
             res_order = leaves_list(Z)
             self.sort_order[i,:] = self.sort_order[i,res_order]

    def _sortFrame(self, arg):
        assert arg == int(arg)
        arg = int(arg)
        dist_mat = 1 - self._get_sliced_corr_mats()[arg+self.centerFrame,:,:]
        Z = linkage(dist_mat[np.triu_indices(len(dist_mat), 1)], method='ward', optimal_ordering=True)
        res_order = leaves_list(Z)
        for i in range(self.nSteps):
            self.sort_order[i,:] = self.sort_order[i,res_order]

    #additional private methods
    def _get_sliced_corr_mats(self):
        sliced_mats = self.corrMats[:, self.current_seqs, :][:, :, self.current_seqs]
        return sliced_mats


class SeqCorrSorterGroups(SeqFramesCorrSorter):
    def __init__(self, seqCentered, Groups, groupTemplates):
        super().__init__(seqCentered)

        self.group_lens = ([len(g) for g in groups.groupIdxs])
        self.group_templates = groupTemplates
        # ->self.groupIdxsPrecomputed = np.concatenate(repeat_groups)
        group_asgn = np.ones(np.sum(self.group_lens), dtype=int)
        start = 0
        for i, idx_ in enumerate(np.cumsum(self.group_lens)):
            group_asgn[start:idx_] = i
            start = idx_
        self.group_idx = group_asgn

        self.corrMats_template2seq #function

        self.current_groups = [1]

    def summary(self):
        print(
        f'''
        Number of sequences: {len(self.seq_list)}
        Frames per sequence: {self.nFrames}
        Number of groups: {len(self.group_lens)}
        Groups: {self.current_groups}
        Group sizes: {self.group_lens}
        Sort type: {self.sort_type}

        '''
            )

    def sorted_corr_mats(self, current_groups=None, sort_type=None, sort_arg=None):
        #values can be called independently, or within this method for convenience
        if current_groups is not None:
            self.set_current_groups(current_groups)
        if sort_type is not None:
            self.compute_sort_order(sort_type, sort_arg)

        corr_mats_ = self._get_sliced_corr_mats()
        group_idx_ =self._get_sliced_group_idx()

        for i in range(self.nFrames):
            corr_mats_[i,:,:] = corr_mats_[i, self.sort_order[i,:],:][:,self.sort_order[i,:]]
            group_idx_[i,:] = group_idx_[i,self.sort_order[i,:]]

        return corr_mats_, group_idx_

    def set_current_groups(self, user_list):
        "operates similar to pandas, in that this value will determine slicing in many other functions"
        if user_list == 'all':
            self.current_groups = [i for i in range(len(self.group_lens))]
        else:
            self.current_groups = user_list
        self._sliceitup()

    def compute_sort_order(self, sort_type='group', args=None):
        """
        this is the power of this class. do any sorting, add more later, all functionality still available
        arg1 informs the sort order

        Parameters
        ----------
        sort_type : string
            'group': sorts into repeat groups. same at each time point
            'group_sorted': sorted into repeat groups, but repeat groups are
                hierarchically sorted by template2template correlation
            'independent': each time point is hierarchically clustered separately
            'center': see 'frame'
            'frame': a single time point is hierarchically clustered, and this is extended to every time point
                arg is the timepoint to sort
            'template': frames are sorted according to their correlation with a template
                [arg]
                    [0] is the group template to compare and
                    [1] optional, frame to compare

        """
        self._sort_order_default()
        self.sort_type = sort_type

        if (sort_type is None) or (sort_type=='None'):
            pass #defaults to the group ordering, as done two lines above
        elif sort_type == 'independent':
            self._sortIndependent()
        elif sort_type == 'center':
            self._sortFrame(0)
        elif sort_type =='frame':
            self._sortFrame(args)
        elif sort_type == 'template':
            self._sortTemplate(args)
        self._sort_group()

        # elif sort_type == 'group_sort':
        #     dist_mat = 1 - corr_mats[i,:,:]
        #     Z = linkage(dist_mat[np.triu_indices(len(dist_mat), 1)], method='ward', optimal_ordering=True)
        #     res_order = hierarchy.leaves_list(Z)
        #     self.sort_order[i,:] = self.sort_order[i,res_order]

    def _sortTemplate(self, args):
        #args[0]: template to compare
        #args[1]: frame to compare
        if isinstance(args, tuple) or isinstance(args, list):
            sort_order_ = np.argsort(self.template2seq_corr_mats[args[1]+self.centerFrame,args[0],self.current_seqs])[::-1].astype(int)
            for i in range(self.nFrames):
                self.sort_order[i,:] = self.sort_order[i,sort_order_]
        else:
            for i in range(self.nFrames):
                sort_order_ = np.argsort(self.template2seq_corr_mats[i,args,self.current_seqs])[::-1].astype(int)
                self.sort_order[i,:] = self.sort_order[i,sort_order_]

    @property
    def corrMats_template2seq(self):
        if not hasattr(self, "_corrMats_template2seq"):
            template2seq_corr_mats = np.zeros((self.nFrames, len(self.group_lens), self.nSeqs))
            for i in range(self.nFrames):
                template2seq_corr_mats[i,:,:] = 1 - cdist(self.group_templates,
                                                          np.corrcoef(self.Seqs[:,i,:]),
                                                          metric='correlation')
            self._corrMats_template2seq = template2seq_corr_mats

        return self._corrMats_template2seq

    @property
    def corrMats_template2template(self):
        if not hasattr(self, "_corrMats_template2template"):
            self._corrMats_template2template = np.corrcoef(self.group_templates)

        return self._corrMats_template2template

    def _sliceitup(self):
        slices = np.zeros(self.nSeqs, bool)
        for g in self.current_groups:
            slices[self.group_idx==g] = True
        self.current_seqs = np.where(slices)[0]

    def _get_sliced_group_idx(self):
        sliced_group_idx = self.group_idx[self.current_seqs]
        return np.repeat(np.expand_dims(sliced_group_idx,0), self.nFrames, axis=0)

    def _sort_group(self):
        #to compare group membership with alternate orderings
        self.group_idx_sorted = self._get_sliced_group_idx()
        print(self.group_idx_sorted.shape)
        print(self.sort_order.shape)
        for i in range(self.nFrames):
            self.group_idx_sorted[i,:] = self.group_idx_sorted[i,self.sort_order[i,:]]


# %%Centered data
#Functions for loading sequences, based on a list of central frames with a standard temporal window size
class SequencesCenteredLabeled():
    #this class is mostly just a container for 'data', which is frames of activity
    #with a specified time window
    #There are additional group options
    def __init__(self, data):
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
            groupSize = self._check_group_size(groupSize) #checks to make sure groupsize <= minimum possible groupSize
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
        #else declared groupSize is valid and remains unchanged

        return groupSize

    def set_groups(self, groupLabels):
        self.origGroupLabels = groupLabels

    #if no groups have been set, matchedGroupIdxs is set to all values
    @property
    def data(self):
        return self.origData[self.matchedGroupIdxs,:,:]

    @property
    def nSeqs(self):
        return len(self.matchedGroupIdxs)

    @property
    def groupLabels(self):
        return self.origGroupLabels[self.matchedGroupIdxs]

# TODO!
#   create a sequence centered class that doesn't rely on groups
# class SequencesCenteredonFrame(Sequences):
#     def __init__(self, df):
#         self.ferret = ferret
#         self._get_SeqDir(SeqDir)
#         self._load_seqs()

#     def _load_seqs(self):
#         pass


#optimized loading routines
# #i chose not to put these in a class since they output data matrices
def load_PFS_sequences_centered_fast(sequences, templateGroups, t_steps=50, nSkips=0, multiprocess=15):

    df = get_seqCenteredDF(sequences, templateGroups, t_steps=t_steps)
    seqCentered = event_tools.load_PFS_sequences_many(df, pad_pre=0, pad_post=1)
    seqCentered = [seq.T for seq in seqCentered]

    # args_pool = zip(this_data_iter, this_mask_iter, isur, container)

    pool = Pool(processes=multiprocess)
    seqCentered = list(tqdm(pool.imap(
        partial(event_tools.spatial_filter, roi=sequences.roi), seqCentered),
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

    datasets = parse_directory_structure.get_all_datasets_for_ferret(ferret)
    datasets = [d for d in datasets if d.matchDesc('stim', descriptions.stim_type.none_black)]

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


#non-optimized loading routines
def load_bin(df, pad_pre, pad_post):
    seqID = 'all_may25'
    ferret = int(df.ferret[1:])
    tseries = df.tser
    indice_start = df.indice_start - pad_pre -1
    indice_stop = df.indice_stop + pad_post-1

    #date = parse_directory_structure.get_all_dates_for_ferret(ferret)[0]
    with contextlib.redirect_stdout(None):
        datasets = parse_directory_structure.get_all_datasets_for_ferret(ferret)
        datasets = [d for d in datasets if d.matchDesc('stim', descriptions.stim_type.none_black)]
        tseries_list = np.array([d.series_number for d in datasets])
        tseries_idx = np.argwhere(tseries_list==tseries)[0][0]
        dataset = datasets[tseries_idx]

        pfs_bin = load_binary_full([dataset], seqID)[0].T
        pfs_bin = pfs_bin[indice_start:indice_stop,:]

    return pfs_bin


def load_PFS_sequences_centered(sequences, templateGroups, t_steps=3):
    #include tseries, indice_start, indice_stop, and ferret

    df = get_seqCenteredDF(sequences, templateGroups, t_steps=t_steps)
    seqCentered = np.zeros((len(df), t_steps*2+1, sequences.Seqs[0].shape[0]))
    seqCenteredBin = np.zeros((len(df), t_steps*2+1, sequences.Seqs[0].shape[0]))
    seqBin = np.zeros((len(df), t_steps*2+1, sequences.Seqs[0].shape[0]), bool)
    for i in range(len(df)):
        seq = event_tools.load_PFS_sequence(df=df.loc[i], pad_pre=0, pad_post=1)
        seq = event_tools.spatial_filter(seq, sequences.roi)
        seqCentered[i,:,:] = seq[:,sequences.roi]
        seqCenteredBin[i,:,:] = load_bin(df=df.loc[i], pad_pre=0, pad_post=1)

        if i%25==0:
            print(f"{i+1} of {len(df)}")
    seqCentered = SequencesCenteredLabeled(seqCentered)
    seqCenteredBin = SequencesCenteredLabeled(seqCenteredBin)

    return [seqCentered, seqCenteredBin]