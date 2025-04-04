# Sequence Data
In this published work, the data was composed of previously processed spontaneous cortical activity in the developing visual cortex. Briefly, raw calcium imaging was z-scored (ΔF/F0), deconvolved using numerical differentiation, and segmented into temporal series of arbitrary duration.

*Stereotyped spatiotemporal dynamics of spontaneous activity in visual cortex prior to eye-opening.  
Luna Kettlewell, et al. bioRxiv 2024.06.25.600611; doi: https://doi.org/10.1101/2024.06.25.600611* 

![Event examples](/imgs/events.png)
*Three segmented events bounded on either side by inactive frames. Each event is normalized to its maximum value.*


## Sequence Class
[/SequenceLoadingClass](https://github.com/LunaKet/SequenceDataClass/blob/master/SequenceLoadingClass.py)  
One hurdle to analyzing this data was the arbitrary duration, where most data fell between 40ms-200ms. A NumPy array would not be appropriate since the durations are different. Thus, a list was used, but lists can be clunky to index and slice. This repository contains a custom class to manage this data, with slicing functionality that mimics NumPy. i.e., a series of indices can be used, or colon syntax can be used.

> seq_series = sequences.Seqs[[1,50,111,156,257,310,509]]  
> seq_firstHundred = sequences.Seqs[0:100]

Here is an interactive viewer for a subset of the sequences:  
[Interactive Sequence Viewer](https://lunaket.github.io/SequenceDataClass/index.html)

## Sequence Loading
[/SequenceLoadingFunctions](https://github.com/LunaKet/SequenceDataClass/blob/master/SequenceLoadingFunctions.py)  
The underlying data is around 1 Terabyte per animal, so care was taken to load in data in batches and to minimize data in memory. That being said, the dataset *is* big and not easily accesbile, so for now this script is only a demonstration and cannot be run.

## Sequence Metrics
[/SequenceMetrics](https://github.com/LunaKet/SequenceDataClass/blob/master/SequenceMetrics.py)  
This script calculates many intermediary and key metrics for a sequence, allowing comparison across a range of durations.

![Metrics examples](/imgs/seqMetrics.png)
*(Left) Ratio of all active pixels during entire event (RA) and the area of propagating activity (PA) extending beyond onset activation. (Right) Propagation area across all events. Red bar indicates static events (PA=0). Inset shows percentage of static and dynamic events (PA>0) for one animal. Summary of static event percentages across animals (mean=6.9%, range=4-21 %, n=6 animals).* 

## Greedy Groups
[/GreedyGroups](https://github.com/LunaKet/SequenceDataClass/blob/master/GreedyGroups.py)  
This clusters the data according to a Greedy algorithm. It is based on sequence correlations, using a shuffled control to determine the threshold for a significantly correlated pair of sequences.  
![Many-frame clusters](/imgs/greedygroupsmany.png)  
*(Left) Clustered correlation matrix of all 100 ms events in example animal with the (Center) thresholded and zoomed-in version of correlation matrix. Color-coded boxes drawn around identified clusters. (Right) The spatiotemporal clusters for animal F335. Each row is the mean projection of events in a cluster, showing the progression across five frames.*  
![Single-frame clusters](/imgs/greedygroups1.png) 
*Clustered from instantaneous, single-frame activity, showing the average group activity. Identified by finding highly repeated instances in single frames of activity. Contours are drawn to illustrate each group's active area.*



