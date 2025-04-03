#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 12:57:21 2025

@author: loon
"""
import numpy as np
import scipy.ndimage as snd

# %% Spatial Filtering 
def spatial_filter(PFS_seq, roi, sig_low=0.5, sig_high=4.2):
    if sig_low is None:
        sig_low = 0.5
    if sig_high is None:
        sig_high = 4.2

    PFS_2d = np.zeros(
        (PFS_seq.shape[1], roi.shape[0], roi.shape[1]), dtype='float')
    for i in range(PFS_seq.shape[1]):
        PFS_2d[i, roi] = PFS_seq[:, i]

    PFS_sf = lowhigh_normalize(PFS_2d, mask=roi, sig_high=sig_high, sig_low=sig_low)

    return PFS_sf


def lowhigh_normalize(frame, mask=None, sig_high=None, sig_low=None):
	''' apply bandpass filter to frame
	specify mask and standard deviations sig_high (highpass) and sig_low (lowpass) of gaussian filters
    Citation for lowhigh_normalize:
        __author__ = 'Bettina Hein'
        __email__ = 'hein@fias.uni-frankfurt.de'
	'''
	# recursion for frames
	if len(frame.shape) > 2:
		result = np.empty_like(frame)
		for i in range(frame.shape[0]):
			result[i] = lowhigh_normalize(frame[i], mask, sig_high, sig_low)
		return result

	if mask is None:
		mask = np.ones(frame.shape, dtype=bool)
	data = np.copy(frame)
	m = np.zeros(mask.shape)
	m[mask] = 1.0
	m[~np.isfinite(frame)] = 0.
	data[np.logical_not(m)] = 0.
	m2 = np.copy(m)

	## gaussian low pass
	low_mask = snd.gaussian_filter(m2, sig_low, mode='constant', cval=0)
	low_data = 1.*snd.gaussian_filter(data, sig_low, mode='constant', cval=0)/low_mask

	## gaussian high pass
	low_data[np.logical_not(m)] = 0
	high_mask = snd.gaussian_filter(m, sig_high, mode='constant', cval=0)
	highlow_data = low_data - 1.*snd.gaussian_filter(low_data, sig_high, mode='constant', cval=0)/high_mask
	highlow_data[np.logical_not(mask)] = np.nan
	return highlow_data


def flat2img(flatdata, roi, fill=0):
    nFrames = flatdata.shape[1]
    video = np.zeros((nFrames, *roi.shape), dtype='float')
    video[:, :, :] = fill
    for n in range(nFrames):
        video[n, roi] = flatdata[:, n]

    return video