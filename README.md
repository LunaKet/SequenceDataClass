# Sequence Data
In this project, the data was composed of previously processed spontaneous cortical activity in the developing visual cortex. Data was segmented into discrete segments, which are temporal series of arbitrary duration.

![Event examples](/imgs/events.png)
*Stereotyped spatiotemporal dynamics of spontaneous activity in visual cortex prior to eye-opening.  
Luna Kettlewell, et al. bioRxiv 2024.06.25.600611; doi: https://doi.org/10.1101/2024.06.25.600611* 

# Sequence Class
[/SequenceLoadingClass](https://github.com/LunaKet/SequenceDataClass/blob/master/SequenceLoadingClass.py)  
One hurdle to analyzing this data was the arbitrary duration, where most data fell between 40ms-200ms. A NumPy array would not be appropriate since the durations are different. Thus, a list was used, but lists can be clunky to index and slice. This repository contains a custom class to manage this data, with slicing functionality that mimics NumPy. i.e., a series of indices can be used, or colon syntax can be used.

> seq_series = sequences.Seqs[[1,50,111,156,257,310,509]]  
> seq_firstHundred = sequences.Seqs[0:100]

# Sequence Loading
[/SequenceLoadingFunctions](https://github.com/LunaKet/SequenceDataClass/blob/master/SequenceLoadingFunctions.py)  
The underlying data is around 1 Terabyte per animal, so care was taken to load in data in batches and to minimize data in memory. This script demonstrates my proficiency in efficiently working with large datasets. That being said, the dataset is big and not easily accesbile, so for now this script is only a demonstration and cannot be run.
