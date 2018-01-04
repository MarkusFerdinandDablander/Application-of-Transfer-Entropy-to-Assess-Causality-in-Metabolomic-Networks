#My Entropy Functions


import matplotlib.pyplot as plt
import numpy as np
import random
import math
from modules import entropy_estimators as ee
from modules import my_entropy_functions as mef
import sklearn.metrics

def mutinf_binsizefixed(x, y, bins = 12):
    x = np.array(x)
    y = np.array(y)
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = sklearn.metrics.mutual_info_score(None, None, contingency=c_xy)*math.log(math.e,2)
    return mi

def entropy(x, bins = 12):
    x = np.array(x)
    return mutinf_binsizefixed(x,x,bins)

def histedges_equalN(x, nbin):
    x = np.array(x)
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),np.arange(npt),np.sort(x))

def mutinf_binsizevar(x,y, bins = 12):
    x = np.array(x)
    y = np.array(y)
    xbins = histedges_equalN(x, bins)
    ybins = histedges_equalN(y, bins)
    c_xy = np.zeros((bins,bins), dtype=int)  
    for i in range(bins):
        for j in range(bins):         
            if i< bins-1 and j < bins-1:
                c_xy[i,j] = ((xbins[i]<=x) & (x<xbins[i+1]) & (ybins[j]<=y) & (y<ybins[j+1])).sum()
            if i == bins-1 and j < bins-1:
                c_xy[i,j] = ((xbins[i]<=x) & (x<=xbins[i+1]) & (ybins[j]<=y) & (y<ybins[j+1])).sum()
            if i< bins-1 and j == bins-1:
                c_xy[i,j] = ((xbins[i]<=x) & (x<xbins[i+1]) & (ybins[j]<=y) & (y<=ybins[j+1])).sum()
            if i == bins-1 and j == bins-1:
                c_xy[i,j] = ((xbins[i]<=x) & (x<=xbins[i+1]) & (ybins[j]<=y) & (y<=ybins[j+1])).sum()            
    mi = sklearn.metrics.mutual_info_score(None, None, contingency=c_xy)*math.log(math.e,2)
    return mi

def mutinf_binsizevar_norm_min(x, y, bins = 12):
    return mutinf_binsizevar(x, y, bins)/(np.amin([entropy(x,bins),entropy(y,bins)]))