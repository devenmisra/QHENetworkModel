# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: net
#     language: python
#     name: python3
# ---

# +
import numpy as np
import scipy as sp

def ATransferMatrixGenerator(S, T, m, diagValues): 
    Phi = np.diag(diagValues)
    A = np.array([[S, -T], [-T, S]])
    diagBlocks = []
    for i in range(m): 
        diagBlocks.append(A)
    Turn = sp.linalg.block_diag(*diagBlocks)
    
    return Turn @ Phi

def BTransferMatrixGenerator(C_s, C_o, m, diagValues): 
    Phi = np.diag(diagValues)
    A = np.array([[C_s, C_o], [C_o, C_s]])
    diagBlocks = [np.array(A[1,1])]
    for i in range(m-1): 
        diagBlocks.append(A)
    diagBlocks.append(np.array(A[0,0]))
    Turn = sp.linalg.block_diag(*diagBlocks) + sp.sparse.coo_array(([A[1,0], A[0,1]], [(0, 2*m-1), (2*m-1, 0)]))

    return Turn @ Phi


# -

def TransfMatGenerator(Theta, m, nw, phases): 
    S = 1/np.cos(Theta)
    T = np.tan(Theta)
    Cs = 1/np.sin(Theta)
    Co = np.cos(Theta)/np.sin(Theta)
    MatrixList = []
    
    for j in range(0, nw):

        AB = ATransferMatrixGenerator(S, T, m, phases[j,0]) @ BTransferMatrixGenerator(Cs, Co, m, phases[j,1])
        MatrixList.append(AB)
    
    return MatrixList


# +
def extract_block_diag_A(A,M):
    
    blocks = np.array([A[i:i+M,i:i+M] for i in range(0,len(A),M)])
    
    return blocks

def extract_block_diag_B(A,M,m):

    edge = np.array([[A[0,0], A[0,2*m-1]], [A[2*m-1, 0], A[2*m-1, 2*m-1]]])
    blocks = np.array([A[i:i+M,i:i+M] for i in range(1,len(A)-1,M)])

    return np.stack([edge, *blocks])

def RMatrixGenerator(MatrixType, ATransfMatList, BTransfMatList, m): 
    randIndex = np.random.randint(0, m, size=10)

    if MatrixType == 'A':
        M = sp.linalg.block_diag(ATransfMatList[randIndex[0]], ATransfMatList[randIndex[1]]) @ sp.linalg.block_diag(1, BTransfMatList[randIndex[2]], 1) @ sp.linalg.block_diag(ATransfMatList[randIndex[3]], ATransfMatList[randIndex[4]])

    if MatrixType == 'B': 
        M = sp.linalg.block_diag(BTransfMatList[randIndex[5]], BTransfMatList[randIndex[6]]) @ sp.linalg.block_diag(1, ATransfMatList[randIndex[7]], 1) @ sp.linalg.block_diag(BTransfMatList[randIndex[8]], BTransfMatList[randIndex[9]])
    
    R_00 = M[0,0] + (M[0,1] + M[0,2])*(M[2,0] - M[1,0]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])
    R_01 = M[0,3] + (M[0,1] + M[0,2])*(M[2,3] - M[1,3]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])
    R_10 = M[3,0] + (M[3,1] + M[3,2])*(M[2,0] - M[1,0]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])
    R_11 = M[3,3] + (M[3,1] + M[3,2])*(M[2,3] - M[1,3]) / (M[1,1] + M[1,2] - M[2,1] - M[2,2])

    RMatrix = np.array([[R_00, R_01], [R_10, R_11]])
    
    return RMatrix


# -

def TransfMatGenerator_withReplacement(Theta, m, nw, rngseed, insertProbability=0.0): 
    S = 1/np.cos(Theta)
    T = np.tan(Theta)
    Cs = 1/np.sin(Theta)
    Co = np.cos(Theta)/np.sin(Theta)
    rng = np.random.default_rng(seed = rngseed)
    phases = np.exp((np.pi/2) * 1j * rng.uniform(low=-1, high=1, size=(nw, 2, 2*m)))
    MatrixList = []
    
    for j in range(0, nw):
        A = ATransferMatrixGenerator(S, T, m, phases[j,0])
        diagBlocksA = extract_block_diag_A(A, 2)

        B = BTransferMatrixGenerator(Cs, Co, m, phases[j,1])
        diagBlocksB = extract_block_diag_B(B, 2, m)

        for i in range(m): 
            if rng.uniform(low=0, high=1, size=2*m)[i] < insertProbability:
                diagBlocksA[i] = RMatrixGenerator('A', diagBlocksA, diagBlocksB, m)

            if rng.uniform(low=0, high=1, size=2*m)[m+i] < insertProbability:
                diagBlocksB[i] = RMatrixGenerator('B', diagBlocksA, diagBlocksB, m)

        A_R = sp.linalg.block_diag(*diagBlocksA)
        B_R = sp.linalg.block_diag(*[diagBlocksB[0][0,0], *diagBlocksB[1:], diagBlocksB[0][1,1]]) +  sp.sparse.coo_array(([diagBlocksB[0][0,1], diagBlocksB[0][1,0]], [(0, 2*m-1), (2*m-1, 0)]))

        AB_R = A_R @ B_R

        MatrixList.append(AB_R)
    
    return MatrixList


# +
from scipy.sparse.linalg import splu

def LyapFinder(w, MatrixList):
    m = len(MatrixList[0])
    n = len(MatrixList)//w
    L = np.eye(m)
    LyapList = np.zeros(m)
    
    for q in range(0, n-1):
        
        if len(MatrixList[q*w : (q+1)*w]) < 2: 
            H = MatrixList[q*w : (q+1)*w][0]
        else: 
            H = np.linalg.multi_dot(MatrixList[q*w : (q+1)*w])

        B = H @ L
        LU = sp.linalg.lu(B, permute_l=True)
        L = LU[0]
        LyapList = LyapList + np.log(np.abs(np.diagonal(LU[1])))

    if len(MatrixList[(n-1)*w : n*w]) < 2:
        H = MatrixList[(n-1)*w : n*w][0]
    else:
        H = np.linalg.multi_dot(MatrixList[(n-1)*w : n*w])

    B = H @ L
    LyapList = LyapList + np.log(np.abs(np.diagonal(sp.linalg.qr(B)[1])))

    return LyapList/len(MatrixList)


# -

def ListLyap(ngen, n, m, w, ThetaList, seed):
    nmax = min([ngen, n])
    LyapRange = []
    for j in range(0, len(ThetaList)):
        MatrixList = TransfMatGenerator_withReplacement(ThetaList[j], m, ngen*w, seed)
        WholeList = LyapFinder(w, MatrixList[0:nmax])
        LyapRange.append([ThetaList[j], -max([x for x in WholeList if x<0])])

    return np.array(LyapRange)


def BatchList(ngen, n, m, w, ThetaList, seed, insertProbability):
    nmax = min([ngen, n])
    WholeList = []
    for j in range(0, len(ThetaList)):
        MatrixList = TransfMatGenerator_withReplacement(ThetaList[j], m, ngen*w, (2**seed * 3**j * 5**m), insertProbability)
        WholeList.append(LyapFinder(w, MatrixList[0:nmax]))

    return np.array(WholeList)

lengths = [1024, 1024, 1024, 1024, 1024, 1024]
widths = [20, 30, 40, 50, 60]
critVal = np.pi/4
thetaRange = np.linspace(critVal, np.arctan(np.exp(0.06)), 7)
stepsPerLU = 8

import os
import pickle
import time

os.chdir('/home/demisra/QHENetworkModel/')
os.getcwd()

iP = 0.0

def batchProcess(nbatch): 

    print(f"Processing Batch {nbatch}", flush=True)

    testList = dict()

    for length, width in zip(lengths, widths): 

        start = time.process_time()
    
        testList[f'{width}'] = BatchList(length, length, width, stepsPerLU, thetaRange, seed=nbatch, insertProbability=iP)

        with open(f'batchLyapDataP{iP}/batchLyapDict{nbatch}.pickle', 'wb') as handle:
            pickle.dump(testList, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Completed M = {width} | CPU Time: {time.process_time() - start}', flush=True)


# +
from multiprocessing import Process

def runMultiBatch(epochCount, processCount): 
    for epoch in range(epochCount):
        if __name__ == '__main__':
            processes = [Process(target=batchProcess, args=(nbatch,)) for nbatch in range(epoch * processCount, epoch * processCount + processCount)]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            print('Done', flush=True)


# +
import sys

runMultiBatch(int(sys.argv[1]), int(sys.argv[2]))