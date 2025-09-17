# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
os.chdir('/home/demisra/QHENetworkModel/')

import numpy as np

import pickle

# +
lengths = [1000, 1000, 1000, 1000, 1000, 1000]
widths = [20, 30, 40, 50, 60]
critVal = np.pi/4
thetaRange = np.linspace(critVal, np.arctan(np.exp(0.06)), 7)
stepsPerLU = 8

iP = 0.0


# -

def LyapListPairs(WholeList, ThetaList):
    LyapRange = []
    for j in range(0, len(ThetaList)):
        LyapRange.append([ThetaList[j], -max([x for x in WholeList[j] if x<0])])

    return np.array(LyapRange)


# +
completeList = dict()

with open(f'batchLyapDataP{iP}/batchLyapDict0.pickle', 'rb') as handle:
    aggList = pickle.load(handle)

for nbatch in range(1,100000): 

    with open(f'batchLyapDataP{iP}/batchLyapDict{nbatch}.pickle', 'rb') as handle:
        batchLyapDict = pickle.load(handle)

    for width in widths: 
        
        aggList[f'{width}'] = aggList[f'{width}'] + batchLyapDict[f'{width}']

for width in widths: 
    
    completeList[f'{width}'] = LyapListPairs(aggList[f'{width}']/100000, thetaRange)

with open(f'completeLyapDataP{iP}.pickle', 'wb') as handle: 
    pickle.dump(completeList, handle, protocol=pickle.HIGHEST_PROTOCOL)

# +
import pandas as pd

iP = 0.0

with open(f"completeLyapDataP{iP}.pickle", "rb") as f:
    object = pickle.load(f)
    
widths = [20, 30, 40, 50, 60]

for width in widths: 

    df = pd.DataFrame(object[f'{width}'])
    df.to_csv(rf'lyapData_CSV/completeLyapDataM{width}P{iP}.csv')
# -


