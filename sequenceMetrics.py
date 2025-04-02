#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:32:00 2022

@author: ckettlew
"""

import pandas as pd
import numpy as np
from skimage import measure
from scipy import ndimage
from scipy.spatial.distance import cdist
from spatialFilter import lowhigh_normalize


def event_series_centermass(event_series, roi):
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
        seq_mean_im[:,:,i] = lowhigh_normalize(seq_mean_im[:,:,i], mask=roi, sig_low=0.5, sig_high=4)

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