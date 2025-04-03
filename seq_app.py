#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:47:33 2025

@author: loon
"""

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import numpy as np

from SequenceLoadingClass import Sequences
from spatialFilter import flat2img


fdir="data/smallData.pkl"
seqs = Sequences(fdir)
seqs.removeSeqsThatDontPass(seqs.df.active_pix>500)
x,y = flat2img(seqs.Seqs[0], seqs.roi).shape[1:]
nFrames=6
seqArray = np.zeros((seqs.nSeqs, nFrames, x, y))
seqArray[:,:,:,:] = np.nan
zmin = 0
zmax = np.percentile(seqArray.flatten(), 99.9)

for i in range(len(seqArray)):
    seq_ = flat2img(seqs.Seqs[i], seqs.roi, fill=np.nan)
    if len(seq_)<nFrames:
        seqArray[i,:len(seq_), :,:] = seq_
    else:
        seqArray[i,:,:,:] = seq_[:nFrames,:,:]

labels = [i*20 for i in range(nFrames)]
fig = px.imshow(seqArray, animation_frame=0, facet_col=1, facet_col_wrap=6,
                zmin=zmin, zmax=zmax, color_continuous_scale='gray')
for i, label in enumerate(labels):
    fig.layout.annotations[i]['text'] = f'{label} ms'
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(coloraxis_showscale=False)

app = Dash()

app.layout = [
    html.Div(children='Segmented Spontaneous Activity'),
    # dcc.Slider(0,100,1, id='slider-seq', value=50),
    dcc.Graph(figure=fig, id='controls-and-graph')
    # dash_table.DataTable(data=seqs.df.to_dict('records'), page_size=10)
    # TODO! get a cleaned df to display underneath
              ]




if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
