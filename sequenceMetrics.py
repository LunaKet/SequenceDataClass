#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:32:00 2022

@author: ckettlew


these functions rely on event_series_thresh, xycom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pickle
import contextlib
from sklearn.decomposition import PCA
from skimage import measure
from scipy import ndimage
from scipy.spatial.distance import cdist

from clmustela.datasets import descriptions, parse_directory_structure, day_analysis_dir
from clmustela.tools import loading, scale, smooth_map
from kettlewell.basic_tools import activity_sequences
from kettlewell.interactive_plots import slider_plot
from kettlewell.propagation import eccentricity


def main(datasets, scale_ds='quarter',
        seq_dir_identifier='may25_all',
        saveData=False, identifier=None):
    '''
    for a little bit faster loading in aggregate calls
    '''

    seq_PFS = activity_sequences.load_sequences(datasets,
                                            seq_type='PFS',
                                            seq_dir_identifier=seq_dir_identifier)

    seq_binarized = activity_sequences.load_sequences(datasets,
                                            seq_type='b',
                                            seq_dir_identifier=seq_dir_identifier)

    tseries_list = activity_sequences.load_tseries_list(datasets, seq_dir_identifier=seq_dir_identifier)
    indices_list = activity_sequences.load_indices_list(datasets, seq_dir_identifier=seq_dir_identifier)
    roi = loading.load_masked_roi(datasets[0])
    if scale_ds=='quarter':
        roi = scale.scale_quarter_bool(roi)
    elif scale_ds=='half':
        roi = scale.scale_half_bool(roi)

    xycom = event_series_centermass(seq_PFS, roi)

    len_series = [e.shape[1] for e in seq_PFS]
    df = pd.DataFrame(len_series, columns=['nFrames'])
    df['tser'] = tseries_list

    df['indice_start'] = [i[0] for i in indices_list]
    df['indice_stop'] = [i[1] for i in indices_list]
    df['isi_minus'], df['isi_plus'] = metric_isi(df['indice_start'], df['indice_stop'])
    df['active_pix'], df['active_ratio']= metric_active_pixels(seq_binarized)
    ecce, orientation = eccentricity.main(seq_binarized, roi)
    df['ecce'] = ecce
    df['ecce_orient'] = orientation
    df['isi_ms'] = df['isi_plus']*.020
    df['duration'] = df['nFrames']*.020

    #in case there are 1 sequence frames these metrics will be nonsense or throw indexing errors
    df_not1 = df[df.nFrames>1]
    not1_idx = df_not1.index
    xcom_not1 = [xycom[0][i] for i in df_not1.index]
    ycom_not1 = [xycom[1][i] for i in df_not1.index]
    xycom_not1 = (xcom_not1, ycom_not1)

    #this sets the default values for several metrics
    #if it requires at least 2 frames, set it as nan
    #for pix growth, there's none, so I have set it as that value for each metric
    df['xdisp'], df['ydisp'], df['displacement'] = np.nan, np.nan, np.nan
    df['distance'], df['tortuousity'], df['active_std'] = np.nan, np.nan, np.nan

    #now compute for all seqs greater than 1 frame
    df_not1['xdisp'], df_not1['ydisp'], df_not1['displacement'] = metric_displacement(xycom_not1)
    df_not1['distance'] = metric_distance(xycom_not1)
    df_not1['tortuousity'] = df.distance/df.displacement
    df_not1['active_std'] = compute_active_std_over_sequence([seq_binarized[i] for i in df_not1.index], roi)

    df[df.nFrames>1]['xdisp'], df[df.nFrames>1]['ydisp'], df[df.nFrames>1]['displacement'] = df_not1['xdisp'], df_not1['ydisp'], df_not1['displacement']
    df[df.nFrames>1]['distance'] = df_not1['distance']
    df[df.nFrames>1]['tortuousity'] = df_not1['tortuousity']
    df[df.nFrames>1]['active_std'] = df_not1['active_std']

    df['pix_growth'], df['stationary_bool'] = compute_pix_growth(seq_binarized, df['active_pix'])
    df['pix_growth_restricted_9pix'] = compute_restricted_pix_growth_area(seq_binarized, roi)

    #upp = descriptions.get_umperpix(datasets[0])
    df['module_count'] = compute_active_modules(seq_PFS, roi)

    #this is honestly just leftover code
    df_linear = df.copy()
    event_series_sorted = [seq_PFS[i] for i in df_linear.index]
    event_series_bin_sorted = [seq_binarized[i] for i in df_linear.index]
    xcom_sorted = [xycom[0][i] for i in df_linear.index]
    ycom_sorted = [xycom[1][i] for i in df_linear.index]
    xycom_sorted = (xcom_sorted, ycom_sorted)

    #I almost never return this function, but these are probably all the variables you'd need:
    dict_events = {'df':df_linear,
                   'roi':roi,
                   'xycom':xycom_sorted,
                   'event_series':event_series_sorted,
                   "event_bin":event_series_bin_sorted}

    if saveData:
        fdir = os.path.join(datasets[0].get_path(day_analysis_dir), 'Sequences')
        if os.path.exists(fdir)==False:
            os.mkdir(fdir)
        if identifier is not None:
            file = os.path.join(fdir, f'sequence_dataframe_{identifier}.pkl')
        else:
            file = os.path.join(fdir, 'sequence_dataframe.pkl')

        if os.path.exists(file):
            print('file exists, exiting')
            return None
        with open(file, 'wb') as f:
            pickle.dump(dict_events, f)

    return dict_events


def load_dataframe(ferret, dict_identifier='all'):
    date = parse_directory_structure.get_all_dates_for_ferret(ferret)[0]
    #dataset = parse_directory_structure.get_all_datasets_for_date(ferret, date)[0]
    fdir = os.path.join(day_analysis_dir.format(ferret=ferret, date=date), 'Sequences')

    #this functionality has been moved to linear_wave_fit
    # if sig:
    #     file = 'sequence_dataframe_sig'
    # else:
    file = 'sequence_dataframe'
    if dict_identifier is not None:
        file = file + f'_{dict_identifier}'
    file = file + '.pkl'

    file = os.path.join(fdir, file)
    with open(file, 'rb') as f:
        pickle_dict = pickle.load(f)

    return pickle_dict


def df_subsampling_plotting(df):

    #pull stationary events
    df_station = df.copy()
    df_station = df_station[
        (df_station['pix_growth']<2)
        ]

    #A few things I sometimes run outside of the loop
    df_station = df_station.sort_values(by=['nFrames', 'active_ratio'], ascending=True)

    event_series_sorted = [seq_PFS[i] for i in df_station.index]
    event_series_bin_sorted = [seq_binarized[i] for i in df_station.index]
    xcom_sorted = [xycom[0][i] for i in df_station.index]
    ycom_sorted = [xycom[1][i] for i in df_station.index]
    xycom_sorted = (xcom_sorted, ycom_sorted)

    sld_stat = slider_plot.wrappers(roi, event_series_sorted, xycom_sorted, df_linear, mode=3)


    #pull linear events
    df_lin = df.copy()
    df_lin = df_lin[
        (df_lin['sig']==1)
        ]

    #A few things I sometimes run outside of the loop
    df_lin = df_lin.sort_values(by=['nFrames', 'active_ratio'], ascending=True)

    event_series_sorted = [seq_PFS[i] for i in df_lin.index]
    event_series_bin_sorted = [seq_binarized[i] for i in df_lin.index]
    xcom_sorted = [xycom[0][i] for i in df_lin.index]
    ycom_sorted = [xycom[1][i] for i in df_lin.index]
    xycom_sorted = (xcom_sorted, ycom_sorted)

    sld_lin = slider_plot.wrappers(roi, event_series_sorted, xycom_sorted, df_linear, mode=3)

    #non-linear
    df_nonlin = df.copy()
    df_nonlin = df_nonlin[
        (df_nonlin['sig']==0) &
        (df_nonlin['nFrames']>4)
        ]

    #A few things I sometimes run outside of the loop
    df_nonlin = df_nonlin.sort_values(by=['nFrames', 'active_ratio'], ascending=True)

    event_series_sorted = [seq_PFS[i] for i in df_nonlin.index]
    event_series_bin_sorted = [seq_binarized[i] for i in df_nonlin.index]
    xcom_sorted = [xycom[0][i] for i in df_nonlin.index]
    ycom_sorted = [xycom[1][i] for i in df_nonlin.index]
    xycom_sorted = (xcom_sorted, ycom_sorted)

    sld_lin = slider_plot.wrappers(roi, event_series_sorted, xycom_sorted, df_linear, mode=3)

    #indeterminate
    df_indet = df_linear.copy()
    df_indet = df_indet[
        (df_indet['pix_growth']>2) &
        (df_indet['nFrames']<4)
        ]


    #pie
    counts = [len(df_station), len(df_indet), len(df_lin), len(df_nonlin)]
    plt.figure()
    plt.pie(counts, labels=['stationary', 'indeterminate', 'linear', 'non-linear'],
            colors=['b', 'y', 'r', 'purple'], autopct='%1.1f%%', shadow=True)
    plt.title(f'F{ferret} all sequences')

    #active_ratio by event type
    plt.figure()
    plt.hist(df_station['active_ratio'], bins=50, histtype='step', density=True, label='Stationary', cumulative=True)
    plt.hist(df_lin['active_ratio'], bins=50, histtype='step', density=True, label='Linear', cumulative=True)
    plt.hist(df_nonlin['active_ratio'], bins=50, histtype='step', density=True, label='Non-linear', cumulative=True)
    plt.hist(df_indet['active_ratio'], bins=50, histtype='step', density=True, label='Indeterminate', cumulative=True)
    plt.xlabel('Active Ratio in FOV')
    plt.ylabel('CDF')
    plt.legend()

    #active ratio be nFrames
    plt.figure()
    sns.ecdfplot(data=df[df.nFrames<11], x='active_ratio', hue='nFrames')


def main_with_roimask(datasets,
        roi_mask = None,
        run_significance=False,
        plot_slider=False,
        std_level=4, minlen=5, minpix=10, dil_ero_steps=2):

    try:
        with contextlib.redirect_stdout(None):

            seq_PFS, seq_binarized, lens = activity_sequences.find_sequences(datasets,
                                               seq_dir_identifier=None,
                                               PFS_file_identifier=None,
                                               std_level=std_level, minlen=minlen, minpix=minpix,
                                               dil_ero_steps=dil_ero_steps,
                                               roi_mask=roi_mask,
                                               saveData=False, fdir=None)
    except IndexError:
        return None

    roi = roi_mask

    xycom = event_series_centermass(seq_PFS, roi)

    len_series = [e.shape[1] for e in seq_PFS]
    df = pd.DataFrame(len_series, columns=['nFrames'])
    df['xdisp'], df['ydisp'], df['displacement'] = metric_displacement(xycom)
    df['distance'] = metric_distance(xycom)
    df['tortuousity'] = df.distance/df.displacement



    df_linear = df.copy()
    df_linear = df[
        (df['nFrames']>4) &
        (df['tortuousity']<100)
        ]

    event_series_sorted = [seq_PFS[i] for i in df_linear.index]
    event_series_bin_sorted = [seq_binarized[i] for i in df_linear.index]
    xcom_sorted = [xycom[0][i] for i in df_linear.index]
    ycom_sorted = [xycom[1][i] for i in df_linear.index]
    xycom_sorted = (xcom_sorted, ycom_sorted)

    #This step takes a long time. Maybe don't run it unless you really need it
    if run_significance:
        df_linear['sig'] = metric_sig(event_series_bin_sorted, roi)
        df_sig = df_linear[df_linear.sig==True]
    if False:
        df_linear['sig'] = np.load(r'/home/naxos2-raid14/ckettlew/Python Scripts/kettlewell/propagation/sig_linear_events_f261.npy')
        df_sig = df_linear[df_linear.sig==True]
    if False:
        event_series_sorted = [seq_PFS[i] for i in df_sig.index]
        event_series_bin_sorted = [seq_binarized[i] for i in df_sig.index]
        xcom_sorted = [xycom[0][i] for i in df_sig.index]
        ycom_sorted = [xycom[1][i] for i in df_sig.index]
        xycom_sorted = (xcom_sorted, ycom_sorted)

    #To visualize the data
    if plot_slider:
        sld = slider_plot.wrappers(roi, event_series_sorted, xycom_sorted, df_linear, mode=2)

    #I almost never return this function, but these are probably all the variables you'd need:
    dict_events = {'df':df_linear,
                   'roi':roi,
                   'xycom':xycom_sorted,
                   'event_series':event_series_sorted,
                   "event_bin":event_series_bin_sorted}


    return dict_events


# %%
def roi_masker(roi, roi_mask):
    labeledROI = roi.astype('int16')
    labeledROI[roi] = np.arange(0, np.sum(roi.flatten()))
    roi_mask_idx = labeledROI[roi_mask]

    return roi_mask_idx

def seq_masker(seq_list, roi_mask_idx):
    seq_PFS = [s[roi_mask_idx,:] for s in seq_PFS]


def event_series_centermass(event_series, roi):
    def bool_centerofmass(im):
        xx, yy = np.where(im)
        x = np.mean(xx)
        y = np.mean(yy)
        return x, y

    def full_centerofmass(im):
        im[im<0] = 0
        im = im**3
        m_x = np.sum(im,1)
        xm_x = m_x * range(len(m_x))
        x = np.sum(xm_x) / np.sum(m_x)

        m_y = np.sum(im,0)
        ym_y = m_y * range(len(m_y))
        y = np.sum(ym_y) / np.sum(m_y)
        return x, y

    if len(np.unique(event_series[0]))==2:
        boolean_arr = True
    else:
        boolean_arr=False

    xcom_all = []
    ycom_all = []
    for e in event_series:
        xcom = []
        ycom = []
        for i in range(e.shape[1]):
            im = roi.astype('single')
            im[roi] = e[:,i]
            if boolean_arr:
                x, y = bool_centerofmass(im)
            else:
                x, y = full_centerofmass(im)
            xcom.append(x)
            ycom.append(y)
        xcom_all.append(xcom)
        ycom_all.append(ycom)

    return (xcom_all, ycom_all)


def metric_active_pixels(event_series):
    active_pix = [np.sum(np.sum(e, axis=1)>0) for e in event_series]
    active_ratio = np.array(active_pix)/event_series[0].shape[0]

    return active_pix, active_ratio


def compute_pix_growth(event_series, active_pix, threshold=1.5):
    active_pix_frame0 = [np.sum(e[:,0]) for e in event_series]
    growth = active_pix/active_pix_frame0
    stationary_bool = growth<threshold

    return growth, stationary_bool


def compute_restricted_pix_growth(event_series, roi, iters=9):
    active_pix_frame0 = [np.sum(e[:,0]) for e in event_series]
    pix_growth = np.zeros(len(event_series))

    for i, e in enumerate(event_series):
        im = roi.copy()
        im[roi] = e[:,0]
        im = ndimage.binary_dilation(im, iterations=iters)

        im_total = roi.copy()
        im_total[roi] = (np.sum(e, axis=1)>0)

        im_difference = np.logical_or(im_total, im)
        im_difference = np.logical_xor(im_difference, im)

        pix_growth[i] = np.sum(im_difference) / active_pix_frame0[i]

    return pix_growth


def compute_restricted_pix_growth_area(event_series, roi, iters=9):
    pix_growth = np.zeros(len(event_series))

    for i, e in enumerate(event_series):
        im = roi.copy()
        im[roi] = e[:,0]
        im = ndimage.binary_dilation(im, iterations=iters)

        im_total = roi.copy()
        im_total[roi] = (np.sum(e, axis=1)>0)

        im_difference = np.logical_or(im_total, im)
        im_difference = np.logical_xor(im_difference, im)

        pix_growth[i] = np.sum(im_difference)

    return pix_growth


def compute_active_std_over_sequence(event_series, roi):

    active_std = np.zeros(len(event_series))

    for i, e in enumerate(event_series):
        active_std[i] = np.std(np.mean(e, axis=0))

    return active_std


def metric_isi(start_list, stop_list):
    '''
    return isi minus and plus.
    They are the same numbers, but shifted by 1.
    Minus denotes time since last event.
    Plus denotes time until next event.
    '''

    if type(start_list)==pd.core.series.Series:
        start_list = np.array(start_list)
        stop_list = np.array(stop_list)

    #isi =

    isin = np.array(np.nan)
    isi_minus = start_list[1:] - stop_list[:-1]
    isi_minus = np.concatenate((isin.reshape(-1,1), isi_minus.reshape(-1,1)), axis=0)
    isi_minus[isi_minus<0] = np.nan

    isi_plus = np.concatenate((isi_minus[1:], isin.reshape(-1,1)))

    return isi_minus, isi_plus


def metric_displacement(xycom):
    xcom = xycom[1]
    ycom = xycom[0]

    xdisp = np.array([(x[-1]-x[0])**2 for x in xcom])
    ydisp = np.array([(y[-1]-y[0])**2 for y in ycom])

    disp = np.sqrt(xdisp+ydisp)

    return np.sqrt(xdisp), np.sqrt(ydisp), disp


def metric_distance(xycom):
    xcom = xycom[1]
    ycom = xycom[0]

    distance = np.zeros(len(xcom))
    for i in range(len(xcom)):
        distance_temp = 0
        xx, yy = xcom[i], ycom[i]
        xx = np.diff(xx)
        yy = np.diff(yy)
        for x, y in zip(xx,yy):
            distance_temp += np.sqrt(x**2+y**2)
        distance[i] = distance_temp

    return distance


def metric_distance(xycom):
    xcom = xycom[1]
    ycom = xycom[0]

    distance = np.zeros(len(xcom))
    for i in range(len(xcom)):
        distance_temp = 0
        xx, yy = xcom[i], ycom[i]
        xx = np.diff(xx)
        yy = np.diff(yy)
        for x, y in zip(xx,yy):
            distance_temp += np.sqrt(x**2+y**2)
        distance[i] = distance_temp

    return distance


def compute_velocity(displacement, nFrames, umperpix, roi):

    #this is the simplest way I can think to identify downsampling post-hoc
    if roi.shape[0] == 68:
        factor = 4
    elif roi.shape[0] == 135:
        factor = 2
    elif roi.shape[0] == 270:
        factor=1
    umperpix *= factor #account for downsampling
    umperpix *= .001 #convert to mm

    displacement *= umperpix
    nFrames -= 1 #first frame is not a length
    nFrames *= (1/50) #20 ms or 0.02 seconds

    velocity = displacement/nFrames #in mm/s
    velocity[pd.isna(velocity)] = -1

    return velocity


def metric_sum_angle(xycom):
    xcom = xycom[1]
    ycom = xycom[0]

    sum_angle = np.zeros(len(xcom))
    for i in range(len(xcom)-1):
        angle_temp = 0
        xx, yy = xcom[i], ycom[i]
        xx = np.diff(xx)
        yy = np.diff(yy)

        vectors = np.array((xx,yy)).T

        for j in range(len(vectors)-1):
            angle_temp += compute_angle(vectors[j,:], vectors[j+1,:])
        sum_angle[i] = angle_temp

    return sum_angle


def compute_angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


def compute_active_modules(seq_PFS, roi):
    seq_mean = [np.mean(s, axis=1) for s in seq_PFS]
    seq_mean = np.vstack(seq_mean)
    seq_mean_im = np.zeros((roi.shape[0], roi.shape[1], seq_mean.shape[0]))
    #seq_active_count = [np.sum(np.sum(e, axis=1)>0) for e in seq_binarized]

    for i in range(seq_mean.shape[0]):
        im = roi.astype('single')
        im[roi] = seq_mean[i,:]
        seq_mean_im[:,:,i] = im
        seq_mean_im[:,:,i] = smooth_map.lowhigh_normalize(seq_mean_im[:,:,i], mask=roi, sig_low=1.5, sig_high=12)

    contour_count = []
    for i in range(seq_mean_im.shape[2]):
        image = seq_mean_im[:,:,i]
        #level = (np.percentile(seq_mean[i,:], 99.9) + np.percentile(seq_mean[i,:], 0.1))/2
        conC = measure.find_contours(image)#, level)
        contour_count.append(len(conC))

    return contour_count


def compute_module_growth(seq_bin, roi, active_modules):
    '''
    the integer increase from first frame to the sum of all other frames
    rationale: compare first frame against spatial filtered mean image -- this image should contain all
    activity from the sequence
    '''
    active_modules_1st = np.zeros(len(seq_bin), int)
    module_growth = np.zeros(len(seq_bin), int)

    for i, seq in enumerate(seq_bin):
        im = roi.astype('single')
        im[roi] = seq[:,0]
        im = ndimage.binary_erosion(im, iterations=3)
        im = ndimage.binary_dilation(im, iterations=2)

        #level = (np.percentile(seq_mean[i,:], 99.9) + np.percentile(seq_mean[i,:], 0.1))/2
        conC = measure.find_contours(im)#, level)
        active_modules_1st[i] = len(conC)

    module_growth = active_modules - active_modules_1st

    return active_modules_1st, module_growth


def wavefront_distance(sequences, roi, umperpix=None):

    if umperpix is None:
        umperpix = 46.783625730994156

    dist_list = []
    for seq_ in sequences:
        im = roi.copy()
        im[roi] = seq_[:,0]
        com = bool_centerofmass(im)

        active = np.sum(seq_, axis=1)>0
        im[roi] = active
        coords = np.where(im)
        coords = np.array([coords[0], coords[1]])

        dist_mat = cdist(np.expand_dims(np.array(com),1).T, coords.T)
        dist_list.append(np.max(dist_mat)*umperpix)

    return dist_list


def bool_centerofmass(im):
    xx, yy = np.where(im)
    x = np.mean(xx)
    y = np.mean(yy)
    return x, y

def plot_clusters_explore():
    x,y,t = df_linear.pix_growth, df_linear.tortuousity, df_linear.pca_varex

    fig = plt.figure()
    ax = plt.axes(projection ="3d")

    ax.scatter3D(x, y, t, c=t)
    ax.set_xlabel('pix_growth', fontweight ='bold')
    ax.set_ylabel('tortuousity', fontweight ='bold')
    ax.set_zlabel('pca_varex', fontweight ='bold')


def pca_clusters_explore():
    pca = PCA(n_components=10, whiten=True)
    pca.fit(df.to_numpy())

    pca_data = pca.transform(df.to_numpy())


def angle_vector(df_sig, seq_binarized_sig, bool_180=False):
    angle_perpixel = np.zeros((seq_binarized_sig[0].shape[0], len(seq_binarized_sig)))
    for i, s in enumerate(seq_binarized_sig):
        active_pix_ = np.sum(s,axis=1)>0
        angle_perpixel[active_pix_,i] = df_sig.iloc[i,9]

    peak_angle = np.zeros(angle_perpixel.shape[0])
    for i in range(angle_perpixel.shape[0]):
        datapts = angle_perpixel[i,:]
        datapts = datapts[datapts!=0]

        bins = np.linspace(-180,180, 32)
        bin_center = (bins[0:-1] + bins[1:])/2

        if len(datapts)>5:
            hist = np.histogram(datapts, bins=bins)
            peak_angle[i] = bin_center[hist[0].argmax()]

        else:
            peak_angle[i] = -240

    return peak_angle

# %%Visualization
def sns_pairplot(df, columns = None):
    import seaborn as sns
    import matplotlib as plt
    #sns.set(style="ticks", color_codes=True)
    #columns = ['nFrames', 'active_ratio','displacement', 'distance','pix_growth', 'pca_meanLL', 'pca_varex', 'tortuousity','angle_norm']
    if columns is None:
        g = sns.pairplot(df, plot_kws=dict(s=5, alpha=0.5))
    else:
        g = sns.pairplot(df,vars=columns, plot_kws=dict(s=5, alpha=0.5))


def act_overImagingSession(df, fxxx, minutes_list = None, ax=None):
    '''
    # TODO
        make this standalone and not rely on minutes list

    this plots the number of events per minutes over an imaging session
    minutes_list is especially necessary if imaging sessions have different times
        (can be computed from pull_event_sequences)
    but minutes_list will also provide a way to compare event rates over animals easier

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    minutes_list : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''

    x = df.tser.value_counts().sort_index().index
    y = np.array(df.tser.value_counts().sort_index())

    print(x)
    print(y)

    y_mins = []
    if minutes_list is not None:
        for y_, x_ in zip(y,x):
            y_mins.append(y_/((minutes_list[1,minutes_list[0,:]==x_])[0]))
    print(y_mins)

    if ax==None:
        fig, ax = plt.subplots(1,1)
        ax.plot(x,y_mins, label=f'F{fxxx.ferret}')
        ax.set_title(f'F{fxxx.ferret} Activity')
        ax.set_xlabel('T series')
        ax.set_ylabel('Sequences/minute')
    else:
        ax.plot(x,y_mins, label=f'F{fxxx.ferret}')

    return ax


def rose_plot(data, ferret=None, ax=None, histtype='bar', metric_classifiers=None, metric_names=[], colors=None):
    '''
    data is generally trajectory angles
    '''

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax_flag = True
    else:
        ax_flag = False

    #ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)

    if metric_classifiers is None:
        h = np.histogram(data, bins=64, density=True)
        theta = np.deg2rad(h[1][1:])
        radii = h[0]

        if histtype=='bar':
            bars = ax.bar(theta, radii, width=2*np.pi/32, bottom=0.0)
        elif histtype=='step':
            steps = ax.step(theta, radii)

    else:
        for i, (value, name) in enumerate(zip(np.unique(metric_classifiers), metric_names)):
            h = np.histogram(data[metric_classifiers==value], bins=32)
            theta = np.deg2rad(h[1][1:])
            radii = h[0]
            if colors is not None:
                steps = ax.step(theta, radii, label=name, color=colors[i])
            else:
                steps = ax.step(theta, radii, label=name)

        plt.legend(loc='upper right')






    if ferret is not None:
        ax.set_title(f'F{ferret} Trajectories')
    else:
        ax.set_title('Trajectories')
    # for r,bar in zip(radii, bars):
    #     bar.set_facecolor( cm.magma(r/10.))
    #     bar.set_alpha(0.5)

    if ax_flag:
        return fig, ax
    else:
        return ax


def plot_log_regression(df, metric):
    from scipy.stats import linregress
    import matplotlib.pyplot as plt
    plt.figure()
    counts, bins, bars = plt.hist(df[metric], bins = 60)
    Regression=linregress(np.log(counts+1), np.log(bins[1:]))
    slope=Regression.slope
    plt.title(f"Slope of log-log hist: {slope}")


def plot_stationary_bars(df, ferret):
    total_counts = np.ones(df.nFrames.nunique())
    bars_stationary = df.groupby('nFrames').stationary_bool.mean()
    total_counts -= bars_stationary
    bars_indeterminate = np.zeros(df.nFrames.nunique())
    bars_indeterminate[df.nFrames.unique()<5] = total_counts[df.nFrames.unique()<5]
    total_counts -= bars_indeterminate
    bars_linear = df.groupby('nFrames').sig.mean()
    bars_linear[df.nFrames.unique()<5] = 0
    bars_nonlinear = total_counts - bars_linear

    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].bar(df.nFrames.unique(), bars_stationary, label='stationary')
    ax[0].bar(df.nFrames.unique(), np.zeros(df.nFrames.nunique()), label='--propagation--', facecolor='w', alpha=0)
    ax[0].bar(df.nFrames.unique(), bars_linear, label='linear', bottom=bars_stationary)
    ax[0].bar(df.nFrames.unique(), bars_nonlinear, label='nonlinear', bottom=(bars_stationary+bars_linear))
    ax[0].bar(df.nFrames.unique(), bars_indeterminate, label='indeterminate', bottom=(bars_stationary+bars_linear+bars_nonlinear))

    ax[0].legend()
    ax[0].set_ylabel('Proportion')

    total_counts = df.groupby('nFrames').count().tser
    bars_stationary = df.groupby('nFrames').stationary_bool.sum()
    total_counts -= bars_stationary
    bars_indeterminate = np.zeros(df.nFrames.nunique())
    bars_indeterminate[df.nFrames.unique()<5] = total_counts[df.nFrames.unique()<5]
    total_counts -= bars_indeterminate
    bars_linear = df.groupby('nFrames').sig.sum()
    bars_linear[df.nFrames.unique()<5] = 0
    bars_nonlinear = total_counts - bars_linear

    ax[1].bar(df.nFrames.unique(), bars_stationary, label='stationary')
    ax[1].bar(df.nFrames.unique(), np.zeros(df.nFrames.nunique()), label='--propagation--', facecolor='w', alpha=0)
    ax[1].bar(df.nFrames.unique(), bars_linear, label='linear', bottom=bars_stationary)
    ax[1].bar(df.nFrames.unique(), bars_nonlinear, label='nonlinear', bottom=(bars_stationary+bars_linear))
    ax[1].bar(df.nFrames.unique(), bars_indeterminate, label='indeterminate', bottom=(bars_stationary+bars_linear+bars_nonlinear))

    ax[1].legend()
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel('nFrames')

    ax[0].set_title(f'F{ferret} Sequence Distribution by Frame')


# %% a very specific function for running significance



def quick_r2_plot():
    ferrets = [261,317,335,336,337,339]
    fig, ax = plt.subplots(3,2)
    ax = ax.flatten()

    for i in range(6):
        df_unsig = dataframes[i][dataframes[i]['sig']==False]
        ax[i].hist(dataframes_sig[i]['r2'], bins=32, histtype='step', density=True)
        ax[i].hist(df_unsig['r2'], bins=32, histtype='step', density=True)
        ax[i].set_title(f'F{ferrets[i]}')
        ax[i].set_xlabel('r2')
        if i==5:
            plt.legend(['Sig', 'Not Sig'])

    plt.tight_layout()


def quick_r2_nframe_plot():
    ferrets = [261,317,335,336,337,339]
    fig, ax = plt.subplots(3,2)
    ax = ax.flatten()

    for i in range(6):
        df = dataframes[i][dataframes[i]['sig']==True]
        ax[i].hist2d(df['r2'], df['nFrames'], density=True)
        ax[i].set_title(f'F{ferrets[i]}')
        ax[i].set_xlabel('r2')
        ax[i].set_ylabel('nFrames')

    plt.tight_layout()


def quick_roseplot_angleshuffle_compare():
    ferrets = [261,317,335,336,337,339]
    fig, ax = plt.subplots(3,2, subplot_kw={'projection': 'polar'})
    ax = ax.flatten()

    for i, ft in enumerate(ferrets):
        shuffled_angles = load_dataframe(ft)['shuffle_angles']
        shuffled_angles = [s for s in shuffled_angles if s is not None]
        shuffled_angles = np.concatenate(shuffled_angles)

        df_ = load_dataframe(ft)['df']

        ax[i] = rose_plot(df_[df_.sig==True].angle, ferret=ft, ax=ax[i], histtype='step')
        ax[i] = rose_plot(shuffled_angles, ferret=ft, ax=ax[i], histtype='step')

    ax[-1].legend(['Sig', 'Shuffles'])
    plt.tight_layout()


def bunch_of_roses():
    ferrets = [261,317,335,336,337,339]

    for ft in ferrets:
        df_ = load_dataframe(ft)['df']

        fig, ax = plt.subplots(1,4, subplot_kw={'projection': 'polar'}, figsize=(13,4))
        rose_plot(df_.angle, ferret=ft, ax=ax[0])
        rose_plot(df_[df_.sig==True].angle, ferret=ft, ax=ax[1])
        rose_plot(df_[df_.sig!=True].angle, ferret=ft, ax=ax[2])

        ax[3] = rose_plot(df_[df_.sig==True].angle, ferret=ft, ax=ax[3], histtype='step')
        ax[3] = rose_plot(df_[df_.sig==False].angle, ferret=ft, ax=ax[3], histtype='step')

        ax[0].set_xlabel(f'All Events: {len(df_)}')
        ax[1].set_xlabel(f'Significant Events: {len(df_[df_.sig==True])}')
        ax[2].set_xlabel(f'Non-Sig Events: {len(df_[df_.sig!=True])}')
        ax[3].set_xlabel('Comparison')

        ax[3].legend(['Sig', 'Non-Sig'])

        date = parse_directory_structure.get_all_dates_for_ferret(ft)[0]
        fdir = day_analysis_dir.format(ferret=ft, date=date)
        fdir = os.path.join(fdir, 'trajectories')
        fig.savefig(os.path.join(fdir, f'f{ft}_roseplot_sig_compare.png'))


def rose_and_comTrajectory():
    PFS_file_identifier=None
    seq_dir_identifier='nov23'
    save_flag=True


    ferrets = [261,317,335,336,337,339]
    tsers = [(2,15), None, None, (1,13), None, None]

    for nFerret in range(6):
        #get datasets
        ferret = ferrets[nFerret]
        tser = tsers[nFerret]


        date = parse_directory_structure.get_all_dates_for_ferret(ferret)[0]
        datasets = parse_directory_structure.get_all_datasets_for_date(ferret, date)
        if tser is None:
            datasets = [d for d in datasets if d.matchDesc('stim', descriptions.stim_type.none_black)]
        else:
            datasets = datasets[tser[0]:tser[1]]


        # % load in sequences

        #dict_events = sequence_metrics.main_streamlined(datasets, seq_dir_identifier=seq_dir_identifier, minFrames=4)
        dict_events = main_streamlined(datasets, seq_dir_identifier=seq_dir_identifier, minFrames=1)
        #df = dict_events['df']

        xcom, ycom = dict_events['xycom']
        xcom_diff = [np.diff(xcom_) for xcom_ in xcom]
        ycom_diff = [np.diff(ycom_) for ycom_ in ycom]
        xcom_diff=  np.concatenate(xcom_diff)
        ycom_diff = np.concatenate(ycom_diff)
        angle_all = [np.arctan2(-x,-y) for x,y in zip(xcom_diff,ycom_diff)]

        df_ = load_dataframe(ferrets[nFerret])['df']


        fig, ax = plt.subplots(1,2, subplot_kw={'projection': 'polar'}, figsize=(7,4))
        ax[0] = rose_plot(df_.angle, ferret, ax=ax[0])
        ax[1] = rose_plot(np.rad2deg(angle_all), ferret=ferret, ax=ax[1])

        ax[0].set_xlabel(f'Significantly Linear Sequences')
        ax[1].set_xlabel(f'Center of Mass Shifts--All Active Frames')

        date = parse_directory_structure.get_all_dates_for_ferret(ferret)[0]
        fdir = day_analysis_dir.format(ferret=ferret, date=date)
        fdir = os.path.join(fdir, 'trajectories')
        fig.savefig(os.path.join(fdir, f'F{ferret}_roseplot_com_compare.png'))


# %% a plot that geoff suggested
def scatter_2frames(seq, roi, frame_start=0):
    from matplotlib.gridspec import GridSpec
    fig=plt.figure()

    gs=GridSpec(2,3) # 2 rows, 3 columns
    ax = []
    ax.append(fig.add_subplot(gs[0,0])) # First row, first column
    ax.append(fig.add_subplot(gs[1,0])) # First row, second column
    ax.append(fig.add_subplot(gs[0:,1:])) # First row, third column

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    im = roi.astype('single')
    im[roi] = seq[:, frame_start]
    ax[0].imshow(im, cmap='magma')

    im[roi] = seq[:, frame_start+1]
    ax[1].imshow(im, cmap='magma')

    ax[2].scatter(seq[:, frame_start], seq[:, frame_start+1], s=3, alpha=0.5)
    ax[2].set_xlabel('First Frame')
    ax[2].set_ylabel('Second Frame')
    ax[2].set_xlim(-0.02, 0.1)
    ax[2].set_ylim(-0.02, 0.1)

    plt.tight_layout()


# %% deprecated
# =============================================================================
# def old_main():
#     fxxx = ferret_class.create_class(ferret)
#     event_seqs = 'sf_off_2pix'
#
#
#     seq_PFS, seq_binarized, tser_list, tser_mins = activity_sequences.load_sequences(datasets, 'PFS')
#
#     date = parse_directory_structure.get_all_dates_for_ferret(ferret)[0]
#     fdir = day_analysis_dir.format(ferret=ferret, date=date)
#     if event_seqs is None:
#         fdir = os.path.join(fdir, 'Luna_analysis/event_sequences_full')
#     else:
#         fdir = os.path.join(fdir, 'Luna_analysis/event_sequences_full' + '_' + event_seqs)
#
#     try:
#         with open(os.path.join(fdir, 'xycom.json'), "r") as fp:
#             xycom = json.load(fp)
#     except FileNotFoundError:
#         xycom = event_series_centermass(seq_PFS, fxxx.roi)
#         #np.save(os.path.join(fdir, 'xycom.npy'), xycom, allow_pickle=True)
#         with open(os.path.join(fdir, 'xycom.json'), "w") as fp:
#             json.dump(xycom, fp)
#
#     len_series = [e.shape[1] for e in seq_PFS]
#     df = pd.DataFrame(len_series, columns=['nFrames'])
#     df['tser'] = tser_list
#     df['active_pix'], df['active_ratio'] = metric_active_pixels(seq_binarized)
#     df['xdisp'], df['ydisp'], df['displacement'] = metric_displacement(xycom)
#     df['distance'] = metric_distance(xycom)
#     df['tortuousity'] = df.distance/df.displacement
#     df['forwardness'] = df['displacement']/df['distance']
#     df['velocity_mm_s'] = compute_velocity(df['displacement'], df['nFrames'], fxxx.umperpix, fxxx.roi)
#     df['pix_growth'] = compute_pix_growth(seq_binarized, df['active_pix'])
#     df['sum_angle'] = metric_sum_angle(xycom)
#     df['angle_norm'] = (df.sum_angle/df.distance)
#
#
#     df_linear = df.copy()
#     df_linear = df[
#         (df['nFrames']>2) &
#         (df['tortuousity']<100)
#         ]
#
#     df_linear = df_linear.sort_values('tortuousity', ascending=True)
#
#     event_series_sf_sorted = [seq_sf[i] for i in df_linear.index]
#     event_series_bin_sorted = [seq_binarized[i] for i in df_linear.index]
#     xcom_sorted = [xycom[0][i] for i in df_linear.index]
#     ycom_sorted = [xycom[1][i] for i in df_linear.index]
#     xycom_sorted = (xcom_sorted, ycom_sorted)
#
#     if False:
#         figure = slider_plot.wrappers(fxxx, event_series_bin_sorted, xycom_sorted, df_linear)
#
#     #checking events over imaging session
#     act_overImagingSession(df_linear, fxxx, tser_mins)
#
#     return df_linear, tser_mins
#
# =============================================================================



# def metric_sig(event_series_bin_sorted, roi, recalc=True, nShuffles=2000):

# # TODO: write loading function

#     if False:
#         if recalc is not True:
#             sig_event_bool = activity_sequences.load_sig()

#     else:
#                                                          nShuffles=2000, return_shuffle_info=True)
#         sig_event_idx = np.array(sig_event_idx)
#         sig_event_bool = np.zeros(len(event_series_bin_sorted), dtype='bool')
#         sig_event_bool[sig_event_idx] = True

#     return sig_event_bool

# def metric_LR_R2(seq_binarized_list, roi):
#     from sklearn.linear_model import LinearRegression

#     lr = LinearRegression()
#     score = []
#     for sb in seq_binarized_list:
#         seq_3d, _ = transform_seq_to_3d(sb, roi)
#         lr.fit(seq_3d[:,0:2], seq_3d[:,2])
#         score.append(lr.score(seq_3d[:,0:2], seq_3d[:,2]))

#     return score


# =============================================================================
# # %%
# def main(datasets,
#         seq_dir_identifier=None,
#         sf_file_identifier=None,
#         roi_mask = None,
#         run_significance=False,
#         plot_slider=True):
#
#     seq_PFS = activity_sequences.load_sequences(datasets,
#                                             seq_type='PFS',
#                                             seq_dir_identifier=seq_dir_identifier,
#                                             sf_file_identifier=sf_file_identifier)
#
#     seq_binarized = activity_sequences.load_sequences(datasets,
#                                             seq_type='b',
#                                             seq_dir_identifier=seq_dir_identifier,
#                                             sf_file_identifier=sf_file_identifier)
#
# #If you'd like to look at spatially band-pass filtered data instead
# # =============================================================================
# #     seq_sf = activity_sequences.load_sequences(datasets,
# #                                             seq_type='sf',
# #                                             seq_dir_identifier=seq_dir_identifier,
# #                                             sf_file_identifier=sf_file_identifier)
# #
# #     seq_sfb = activity_sequences.load_sequences(datasets,
# #                                             seq_type='sf_b',
# #                                             seq_dir_identifier=seq_dir_identifier,
# #                                             sf_file_identifier=sf_file_identifier)
# #
# # =============================================================================
#
#
#     tseries_list = activity_sequences.load_tseries_list(datasets, seq_dir_identifier=seq_dir_identifier)
#     indices_list = activity_sequences.load_indices_list(datasets, seq_dir_identifier=seq_dir_identifier)
#     roi = loading.load_maksed_roi(datasets[0])
#     if False:
#         roi = roi.T
#     roi = scale.scale_quarter_bool(roi)
#
#     if roi_mask is not None:
#         roi_mask_idx = roi_masker(roi, roi_mask)
#         seq_PFS = [s[roi_mask_idx,:] for s in seq_PFS]
#         seq_binarized = [s[roi_mask_idx,:] for s in seq_binarized]
#         roi = roi_mask
#
#     xycom = event_series_centermass(seq_PFS, roi)
#
#     len_series = [e.shape[1] for e in seq_PFS]
#     df = pd.DataFrame(len_series, columns=['nFrames'])
#     df['tser'] = tseries_list
#     df['indice_start'] = [i[0] for i in indices_list]
#     df['indice_stop'] = [i[1] for i in indices_list]
#     df['xdisp'], df['ydisp'], df['displacement'] = metric_displacement(xycom)
#     df['distance'] = metric_distance(xycom)
#     df['tortuousity'] = df.distance/df.displacement
#     df['active_pix'], df['active_ratio']= metric_active_pixels(seq_binarized)
#     df['module_count'] = compute_active_modules(seq_PFS, roi)
#     df['pix_growth'], df['stationary_bool'] = compute_pix_growth(seq_binarized, df['active_pix'])
#     df['pix_growth_restricted_9pix'] = compute_restricted_pix_growth(seq_binarized, roi)
#
#
#
#     #To visualize the data
#     if plot_slider:
#         sld2 = slider_plot.wrappers(roi, seq_binarized, xycom, df, mode=3)
#
#     #I almost never return this function, but these are probably all the variables you'd need:
#     dict_events = {'df':df,
#                    'roi':roi,
#                    'xycom':xycom,
#                    'event_series':seq_PFS,
#                    "event_bin":seq_binarized}
#
#     fdir = os.path.join(datasets[0].get_path(day_analysis_dir), 'Sequences')
#     if os.path.exists(fdir)==False:
#         os.mkdir(fdir)
#     file = os.path.join(fdir, 'sequence_dataframe.pkl')
#     with open(file, 'wb') as f:
#         pickle.dump(dict_events, f)
#
#     return dict_events
#
# =============================================================================
