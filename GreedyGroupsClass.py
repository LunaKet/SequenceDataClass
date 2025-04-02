#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:39:44 2024

@author: ckettlew
"""

import numpy as np
import matplotlib.pyplot as plt

def example():
    pass

class GreedyGroups():
    def __init__(self, corr_mat, threshold, sequences=None):
        self.corrMat = corr_mat
        self.nFrames = self.corrMat.shape[0]
        self.thresholds = threshold
        self.corrMatThresh = self.corrMat>self.thresholds
        self._min_groupsize = 15
        self._groupIdxs = []
        self.groupLeaderIdx = []
        self.nSeqs = corr_mat.shape[0]
        self.method = 'greedy'

        if sequences is not None:
            self.seqIdx = sequences.seqIdx
            self.frameIdx = sequences.frameIdx
            self.nSeqFrames = sequences.df.nFrames.values

    # %%% Setup functions
    def set_min_groupsize(self, val):
        self._min_groupsize = val

    def get_min_groupsize(self, val):
        return self._min_groupsize

    def set_subsetofGroups(self, groupList):
        self.groupLeaderIdx = [self.groupLeaderIdx[i] for i in groupList]
        self._groupIdxs = [self._groupIdxs[i] for i in groupList]
        self._nGroups = len(self._groupIdxs)

    def set_reqActWindow(self, reqActWindow):
        self.reqActWindow = reqActWindow
        req_pre, req_post = reqActWindow

        for i, group in enumerate(self._groupIdxs):
            #pre condition
            keep = np.where(self.frameIdx[group] >= req_pre)[0]
            group = group[keep]

            #post condition
            nFrames_ = self.nSeqFrames[self.seqIdx[group]]
            frameIdx_ = self.frameIdx[group] + 1 #+1 to correctly compare with nFrames
            keep = np.where((nFrames_-frameIdx_)>=req_post)
            group = group[keep]

            self._groupIdxs[i] = group

    def set_noRepeatSeqs(self, verbose=False):
        for i, group in enumerate(self._groupIdxs):
            seq_idx_ = self.seqIdx[group]
            group_leader = self.groupLeaderIdx[i]
            keep_mask = np.ones(len(seq_idx_), dtype=bool)
            for s in np.unique(seq_idx_):
               seqRepeats = np.where(seq_idx_==s)[0]
               if len(seqRepeats)>1:
                   keepSeq = np.argmax(self.corrMat[group_leader,group[seqRepeats]])
                   removeSeqs = list(seqRepeats)
                   removeSeqs.pop(keepSeq)
                   keep_mask[removeSeqs] = 0
                   if verbose:
                       print(f"{s}:\t{seqRepeats}")

            self._groupIdxs[i] = group[keep_mask]

    def set_groups_random(self):
        #needs to loop through repeat_group lists
        for i in range(self._nGroups):
            new_order = self._groupIdxs[i][np.random.permutation(len(self._groupIdxs[i]))]
            new_order = self._set_groupLeader_onTop(new_order, self.groupLeader[i])

    def _set_groupLeader_onTop(self, members, groupLeader):
        updatedIndex = np.vstack((np.argwhere(members==groupLeader), np.argwhere(members!=groupLeader))).flatten()
        members = members[updatedIndex]
        return members

    # %%% Main function
    def get_greedy_groups(self):
        remainingMatches = self.corrMatThresh.copy()
        groupsLargeEnough = True

        while groupsLargeEnough:
            #using greedy approach to find group membership
            mostMatchesIdx = np.argmax(np.sum(remainingMatches, axis=1))
            self.groupLeaderIdx.append(mostMatchesIdx)
            members = np.where(remainingMatches[self.groupLeaderIdx[-1],:])[0]
            members = self._set_groupLeader_onTop(members, self.groupLeaderIdx[-1])
            self._groupIdxs.append(members)

            #setting up next loop
            remainingMatches[members,:] = 0
            remainingMatches[:,members] = 0
            groupsLargeEnough = max(np.sum(remainingMatches, axis=1)) >= self._min_groupsize

        self._nGroups = len(self._groupIdxs)
        self._groupLengths = [len(g) for g in self._groupIdxs]
        self.groupPercentages = [len(g)/self.nSeqs for g in self._groupIdxs]
        self.corrMatGroups = self.corrMat[np.concatenate(self._groupIdxs),:][:,np.concatenate(self._groupIdxs)]


    # %%% Properties
    @property
    def clust_assignments_allFrames(self):
        clust_assign = np.zeros(self.corrMat.shape[0])
        for i, r in enumerate(self.groupIdxs):
            clust_assign[r] = i+1
        return clust_assign

    @property
    def clust_assignments_groupFrames(self):
        return np.concatenate([np.ones(len(self.groupIdxs[i]))*(i+1) for i in range(self.nGroups)] )

    @property
    def groupIdxs(self):
        if self.method=='greedy':
            return self._groupIdxs
        elif self.method=='hier':
            return self.hier.groupIdxs
        elif self.method=='kmeans':
            return self.kmeans.groupIdxs

    @property
    def nGroups(self):
        if self.method=='greedy':
            return self._nGroups
        elif self.method=='hier':
            return self.hier.nGroups
        elif self.method=='kmeans':
            return self.kmeans.nGroups

    @property
    def groupLengths(self):
        if self.method=='greedy':
            return self._groupLengths
        elif self.method=='hier':
            return self.hier.groupLengths
        elif self.method=='kmeans':
            return self.kmeans.groupLengths

    # def plot_process(self):
    #     # notes to self -- can probably just use info from self.groupIdxs to back-compute all these values
    #     fig, ax = plt.subplots(3,3)
    #     ax = ax.flatten()

    #     ax[i].imshow(graph, cmap='Greys')
    #     ax[i].set_xticks([])
    #     ax[i].set_yticks([])

    #     nomembers = np.zeros(nFrames)
    #     nomembers[~members] = np.nan

    #     if i>0:
    #         im = np.zeros_like(graph, dtype='single')
    #         im[nomembers,:] = np.nan
    #         im[:,nomembers] = np.nan
    #         im[members,:] = 1
    #         im[:,members] = 1
    #         ax[i].imshow(im, cmap="Reds_r", vmin=0, vmax=3)
    #         ax[i].set_title(f'group{i}, n={len(members)}', fontsize=8)

    # %%% Alternate clust methods
    def get_hier_groups(self, nclusters=None):
        '''
        hierarchical clustering
        must run greedy clustering first
        function is specifically to compare same events as the greedy clustering for reviewer.
        thus will not run completely independent
        '''

        if nclusters is None:
            nclusters = self._nGroups
        assert nclusters==self._nGroups

        #hierarchical clustering
        clusters = self._get_hier(nclusters)
        self.set_alt_cluster(clusters, 'hier')
        self.set_method('hier')

    def _get_hier(self, nclusters):
        from scipy.cluster.hierarchy import linkage, fcluster

        #only include events that were clustered using the greedy algorithm
        if not hasattr(self, 'hclust_linkage'):
            dist_mat = 1 - self.corrMatGroups
            Z = linkage(dist_mat[np.triu_indices(len(dist_mat), 1)], method='ward', optimal_ordering=True)
            self.hclust_linkage = Z
        clusters = fcluster(self.hclust_linkage, nclusters, criterion='maxclust')

        return clusters

    def get_kmeans_groups(self, seqs_pca=None, nclusters=None):
        '''
        seqs_pca: 2D events x pca_components (len(n_components x nFrames))
        '''

        if not hasattr(self, 'pca_data'):
            self.format_pca(seqs_pca)
        assert hasattr(self, 'pca_data')

        if nclusters is None:
            nclusters = self._nGroups
        assert nclusters==self._nGroups

        clusters = self._get_kmeans(nclusters)

        self.set_alt_cluster(clusters, 'kmeans')
        self.set_method('kmeans')

    def _get_kmeans(self, nclusters):
        from sklearn.cluster import KMeans
        pca_groups = self.pca_data[np.concatenate(self._groupIdxs),:]
        clusters = KMeans(n_clusters=nclusters, init='k-means++', n_init=10).fit(pca_groups)
        clusters = clusters.labels_

        return clusters

    def set_method(self, method):
        self.method = method

    def set_alt_cluster(self, clusters, method):
        '''
        sets up a small class to hold cluster-method-specific variables
        so that multiple methods can be held at the same time
        '''
        clustIdxs = self.clust2list(clusters)

        altmethod = varHolder()
        altmethod.groupIdxs = clustIdxs
        altmethod.groupLengths = [len(r) for r in altmethod.groupIdxs]
        altmethod.nGroups = len(clustIdxs)

        if method=='hier':
            self.hier = altmethod
        elif method=='kmeans':
            self.kmeans = altmethod

    def clust2list(self, clusters):
        groupIdxsOrd = np.concatenate(self._groupIdxs)
        clustIdxs = []
        for i in np.unique(clusters):
            clustIdx_ = groupIdxsOrd[np.where(clusters==i)]
            clustIdxs.append(clustIdx_)

        return clustIdxs

    def format_pca(self, seqs_pca):
        if len(seqs_pca.shape)==3:
            seqs_pca = seqs_pca.reshape((len(seqs_pca), -1))
        self.pca_data = seqs_pca

    # %%%Alt Clust Comparisons
    def multilevel_run(self, levels=None, method=None):
        '''
        levels: tuple
            (pythonic) runs levels from start to end of tuple
        method: the alt method, i.e., 'kmeans', 'hier'
            defaults to the current method

        returns dictionary stored in the altmethod class
        '''
        from sklearn.metrics import silhouette_score as ss

        if method==None:
            method = self.method
        if levels==None:
            levels = (2, self._nGroups+1)

        multi_run = {}
        silhouette_scores = {}
        group_overlap = {}

        for i in range(*levels):
            if method=='kmeans':
                clusters = self._get_kmeans(i)
            elif method=='hier':
                clusters = self._get_hier(i)

            clustIdxs = self.clust2list(clusters)
            multi_run[i] = clustIdxs
            silhouette_scores[i] = ss(1 - self.corrMatGroups, clusters)

        if method=='kmeans':
            self.multi_kmeans_groupIdxs = multi_run
            self.multi_kmeans_silhouette = silhouette_scores
        elif method=='hier':
            self.multi_hier_groupIdxs = multi_run
            self.multi_hier_silhouette = silhouette_scores

    def greedy_silhouette_score(self):
        from sklearn.metrics import silhouette_score as ss

        method = self.method
        self.set_method('greedy')
        self.greedy_silhouette = ss(1 - self.corrMatGroups, self.clust_assignments_groupFrames)
        self.set_method(method)

    def compute_overlap_matrix(self, method=None):
        if method==None:
            method = self.method
        if method == 'kmeans':
            multi_run = self.multi_kmeans_groupIdxs
        elif method == 'hier':
            multi_run = self.multi_hier_groupIdxs

        greedyGroupIdxs = self._groupIdxs
        nGreedyGroups = len(greedyGroupIdxs)

        overlap_mats = {}
        for key in multi_run.keys():
            groupIdxs = multi_run[key]

            overlap_ = np.zeros((nGreedyGroups, key))
            for j in range(nGreedyGroups):
                total = len(greedyGroupIdxs[j])
                for k in range(key):
                    res = len(list(set(greedyGroupIdxs[j]) & set(groupIdxs[k])))
                    overlap_[j,k] = res/total

            overlap_mats[key] = overlap_

        if method == 'kmeans':
            self.multi_kmeans_overlap = overlap_mats
        elif method == 'hier':
            self.multi_hier_overlap = overlap_mats

    def compute_overlap_mat_kmean_hier(self):
        key = list(self.multi_hier_groupIdxs.keys())[-1]
        hier_idxs = self.multi_hier_groupIdxs[key]
        kmeans_idxs = self.multi_kmeans_groupIdxs[key]
        n = len(hier_idxs)

        overlap_ = np.zeros((n,n))
        for j in range(n):
            total = len(hier_idxs[j])
            for k in range(n):
                res = len(list(set(hier_idxs[j]) & set(kmeans_idxs[k])))
                overlap_[j,k] = res/total

        self.overlap_mat_kmeans_hier = overlap_



    # %% Plots
    def plot_multi_silhouette_scores(self):
        fig, ax = plt.subplots(1,1)
        ax.spines[['top', 'right']].set_visible(False)

        x = list(self.multi_kmeans_groupIdxs.keys())

        ax.plot(x, list(self.multi_kmeans_silhouette.values()), marker='s', label='KMeans')
        ax.plot(x, list(self.multi_hier_silhouette.values()), marker='^', label='Hierarchical')
        ax.scatter(x[-1], self.greedy_silhouette, c='r', label='Greedy')
        ax.set_ylabel('Silhouette Score')
        ax.set_xlabel('# Groups')

        ax.legend()

    def plot_multi_group_overlap(self, method):
        if method=='kmeans':
            multi_overlap = self.multi_kmeans_overlap
            title='hierarchical'
        if method=='hier':
            multi_overlap = self.multi_hier_overlap
            title='KMeans'
        keys = list(multi_overlap.keys())
        n = len(keys)

        fig, ax = plt.subplots(1,n, gridspec_kw={'width_ratios': range(2,n+2)},
                               figsize=(15,4))

        for i in range(n):
            cbh = ax[i].imshow(multi_overlap[keys[i]], vmin=0, vmax=1, cmap='Blues')
            ax[i].set_yticks(range(0,n+1), range(1,n+2))
            ax[i].set_xticks(range(0,i+2), range(1,i+3))
            ax[i].set_xlabel(f"nGroups: {i+2}")

            y,x = np.where(multi_overlap[keys[i]]>=0.1)
            vals = multi_overlap[keys[i]][y,x]
            for (x_, y_, s_) in zip(x,y,vals):
                if s_>0.49:
                    ax[i].text(x_,y_,f"{s_:.2g}", horizontalalignment='center', c='white')
                else:
                    ax[i].text(x_,y_,f"{s_:.2g}", horizontalalignment='center', c='black')

        ax[0].set_ylabel('Greedy groups')
        fig.suptitle(f'Group overlap between greedy and {title} method')

        cb_ax = ax[-1].inset_axes([1.05, 0, 0.1, 1])
        plt.colorbar(cbh, cax=cb_ax)

    def plot_kmeans_hier_overlap(self):
        self.compute_overlap_mat_kmean_hier()
        overlap = self.overlap_mat_kmeans_hier
        n = len(overlap)

        fig, ax = plt.subplots(1,1,
                               figsize=(4,4))

        cbh = ax.imshow(overlap, vmin=0, vmax=1, cmap='Blues')
        ax.set_yticks(range(0,n+1), range(1,n+2))
        ax.set_xticks(range(0,n+1), range(1,n+2))

        y,x = np.where(overlap>=0.1)
        vals = overlap[y,x]
        for (x_, y_, s_) in zip(x,y,vals):
            if s_>0.49:
                ax.text(x_,y_,f"{s_:.2g}", horizontalalignment='center', c='white')
            else:
                ax.text(x_,y_,f"{s_:.2g}", horizontalalignment='center', c='black')

        ax.set_ylabel('Hierarchical groups')
        ax.set_xlabel('KMeans groups')
        fig.suptitle('Group overlap between hierarchical and KMeans method')

        cb_ax = ax.inset_axes([1.05, 0, 0.1, 1])
        plt.colorbar(cbh, cax=cb_ax)


    def plot_greedy_kmeans_hier_overlap(self, hier_order=None, kmeans_order=None, kmh_order=None):
        self.compute_overlap_mat_kmean_hier()
        overlap_kmh = self.overlap_mat_kmeans_hier
        n = len(overlap_kmh)
        overlap_kmeans = self.multi_kmeans_overlap[n]
        overlap_hier = self.multi_hier_overlap[n]
        if hier_order is not None:
            overlap_hier = overlap_hier[:, hier_order]
        if kmeans_order is not None:
            overlap_kmeans = overlap_kmeans[:, kmeans_order]
        if kmh_order is not None:
            overlap_kmh = overlap_kmh[:, kmh_order]


        fig, ax = plt.subplots(1,3,
                               figsize=(12,4))

        cbh = self._plot_overlap(overlap_hier, ax[0])
        ax[0].set_xlabel('Hierarchical clusters')
        ax[0].set_ylabel('Greedy clusters')
        self._plot_overlap(overlap_kmeans, ax[1])
        ax[1].set_xlabel('KMeans clusters')
        ax[1].set_ylabel('Greedy clusters')
        self._plot_overlap(overlap_kmh, ax[2])
        ax[2].set_xlabel('KMeans clusters')
        ax[2].set_ylabel('Hierarchical clusters')

        cb_ax = ax[-1].inset_axes([1.05, 0, 0.1, 1])
        plt.colorbar(cbh, cax=cb_ax)
        cb_ax.set_ylabel('Fraction events in greedy clusters')


    def _plot_overlap(self, overlap, ax):
        n=len(overlap)
        cbh = ax.imshow(overlap, vmin=0, vmax=1, cmap='Blues')
        ax.set_yticks(range(0,n), range(1,n+1))
        ax.set_xticks(range(0,n), range(1,n+1))

        y,x = np.where(overlap>=0.1)
        vals = overlap[y,x]
        for (x_, y_, s_) in zip(x,y,vals):
            if s_>0.49:
                ax.text(x_,y_,f"{s_:.2g}", horizontalalignment='center', c='white')
            else:
                ax.text(x_,y_,f"{s_:.2g}", horizontalalignment='center', c='black')

        return cbh


class varHolder():
    def __init__(self):
        pass


class GroupInfo():
    #a much lighter version of the GreedyGroups class
    #this is here because a lot of other functions depend on it and it would be hard to fix at this point
    def __init__(self,group_labels, nSteps):
        self.groupLabels = group_labels
        self.nGroups = len(np.unique(group_labels))
        self.nSteps = nSteps
        self.center = int(self.nSteps/2)

