import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import animation
from IPython.display import HTML
import random
from tssb.utils import load_time_series_segmentation_datasets, relative_change_point_distance
from sktime.annotation.clasp import ClaSPSegmentation
from sktime.annotation.clasp import find_dominant_window_sizes
from ruptures.metrics import randindex,hausdorff,precision_recall
import ruptures as rpt

list_colors = [[0.62, 0.16, 0.51],
 [1.0, 0.44, 0.84],
 [0.67, 0.64, 0.54],
 [0.28, 0.95, 0.9],
 [0.2, 0.98, 0.07],
 [0.54, 0.57, 0.81],
 [0.94, 0.41, 0.76],
 [0.52, 0.41, 0.02],
 [0.4, 0.42, 0.33],
 [0.8, 0.45, 0.16],
 [0.63, 0.87, 0.11],
 [0.61, 0.86, 0.07]]

def Fluss(X,m=500,L=500,n_regimes=18,excl_factor=5):
    mp = stumpy.stump(X, m=m)
    L = L
    cac, regime_locations = stumpy.fluss(mp[:, 1], L=L, n_regimes=n_regimes, excl_factor=5)
    regime_locations=list(regime_locations)
    regime_locations.append(65462)
    return regime_locations


def WindomSliding(X,n_regimes,width=40):
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    algo = rpt.Window(width=width, model=model).fit(X)
    regime_locations = algo.predict(n_bkps=n_regimes)
    return regime_locations

def BottomUp(X,n_regimes):
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    algo = rpt.BottomUp(model=model).fit(X)
    regime_locations = algo.predict(n_bkps=n_regimes)
    return regime_locations

def Binseg(X,n_regimes):
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar",...
    algo = rpt.Binseg(model=model).fit(X)
    regime_locations = algo.predict(n_bkps=n_regimes)
    return regime_locations

def ClaSP(X,n_regimes,period_length):
    clasp = ClaSPSegmentation(period_length=period_length, n_cps=n_regimes)
    regime_locations = clasp.fit_predict(X)
    regime_locations=list(regime_locations)
    regime_locations.sort().append(65462)
    return regime_locations

def KernelCPD (X,n_regimes,min_size):
    algo_c = rpt.KernelCPD(kernel="linear", min_size=min_size).fit(X)
    regime_locations = algo_c.predict(n_bkps=n_regimes)
    return regime_locations

def DyPr(X,n_regimes,min_size,jump):
    model = "l1"  # "l2", "rbf"
    algo = rpt.Dynp(model=model, min_size=min_size, jump=jump).fit(X)
    regime_locations = algo.predict(n_bkps=n_regimes)    
    
def plot(X,changing_points,labels_changing_points,regime_locations,label):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(X,label=label)
    fig.legend()
    for i in range(len(changing_points)):
        if i+1<len(changing_points):    
            begin = changing_points[i]
            end = changing_points[i+1]
            label = labels_changing_points[i]
            ax.axvspan(begin,end,alpha=0.2,color=load_activity_map()[label][1],ec=None)
        else: break
    for i in regime_locations:
        ax.axvline(i-1,c='k',lw = 1)



def load_activity_map():
    map = {}
    map['A'] = ['walking','b',0]
    map['B'] = ['jogging ','r',1]
    map['C'] = ['stairs ','k',2]
    map['D'] = ['sitting ','y',3]
    map['E'] = ['standing ', 'g',4]
    map['F'] = ['typing ','m',5]
    map['G'] = ['teeth ','c',6]
    map['H'] = ['soup ',(list_colors[0]),7]
    map['I'] = ['chips ',(list_colors[1]),8]
    map['J'] = ['pasta ',(list_colors[2]),9]
    map['K'] = ['drinking ',(list_colors[3]),10]
    map['L'] = ['sandwich ',(list_colors[4]),11]
    map['M'] = ['kicking ',(list_colors[5]),12]
    map['O'] = ['catch ',(list_colors[6]),13]
    map['P'] = ['dribbling',(list_colors[7]),14]
    map['Q'] = ['writing ',(list_colors[8]),15]
    map['R'] = ['clapping  ',(list_colors[9]),16]
    map['S'] = ['folding ',(list_colors[10]),17]
    return map

def Metrics(regime_locations,changing_points,time):
    AnnotationError = abs(len(changing_points)-len(regime_locations))
    Hausdorff = (hausdorff(regime_locations, changing_points))
    Precision_recall = [round(i,4) for i in precision_recall(regime_locations, 
                                                            changing_points,margin=100)]
    Randindex = round(randindex(regime_locations, changing_points),4)

    metrics = [AnnotationError,Hausdorff,Precision_recall,Randindex,time]
    return metrics

