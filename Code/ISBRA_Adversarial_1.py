#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
# import seaborn as sns
import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean

import seaborn as sns

import itertools
from itertools import product

import csv

from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit

from sklearn.decomposition import KernelPCA

import timeit
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rcParams['mathtext.fontset'] = 'cm'

## for LaTeX typefont
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

## for another LaTeX typefont
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# rc('text', usetex = True)

print("done")


# In[7]:


orig_seq = np.load("/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/gisaid_hcov_19_orig_8505_seq.npy",allow_pickle=True)
attributes = np.load("/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/gisaid_hcov_19_org_8505_attributes.npy",allow_pickle=True)


# In[8]:


# Original DNA Sequences
path = "/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/"

path_1 = "/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/"

file_name = "n2020_5x_simulated_error"
# file_name = "n2020_10x_simulated_error"
# file_name = "random_5x_simulated_error"
# file_name = "random_10x_simulated_error"

lines = []
with open(path +  file_name + '.txt') as f:
    lines = f.readlines()
asd = str(lines).replace("\n',","")
asd1 = asd.replace("\\n","")
asd2 = asd1.replace(", ","")
asd3 = asd2.replace("[","")
asd4 = asd3.split(">NC_045512.2 ''")

depth = []

for i in range(1,len(asd4)):
    asd5 = asd4[i].replace("\'","")
    depth.append(asd5)

print("Length: ",len(depth))
np.save(path_1 + file_name + ".npy",depth)

# depth_5_seq = []
# for i in range(len(protein_sequences)):
#     depth_5_seq.append(str(protein_sequences[i]))

print("Data Loaded")


# In[ ]:


# Original DNA Sequences
# path = "E:/RA/ISBRA_Adversarial_Attack/Dataset/"

# file_name = "n2020_5x_simulated_error"
file_name = "n2020_10x_simulated_error"
# file_name = "random_5x_simulated_error"
# file_name = "random_10x_simulated_error"

lines = []
with open(path +  file_name + '.txt') as f:
    lines = f.readlines()
asd = str(lines).replace("\n',","")
asd1 = asd.replace("\\n","")
asd2 = asd1.replace(", ","")
asd3 = asd2.replace("[","")
asd4 = asd3.split(">NC_045512.2 ''")

depth = []

for i in range(1,len(asd4)):
    asd5 = asd4[i].replace("\'","")
    depth.append(asd5)

print("Length: ",len(depth))
np.save(path_1 + file_name + ".npy",depth)

# depth_5_seq = []
# for i in range(len(protein_sequences)):
#     depth_5_seq.append(str(protein_sequences[i]))

print("Data Loaded")


# In[ ]:


# Original DNA Sequences
# path = "E:/RA/ISBRA_Adversarial_Attack/Dataset/"

# file_name = "n2020_5x_simulated_error"
# file_name = "n2020_10x_simulated_error"
file_name = "random_5x_simulated_error"
# file_name = "random_10x_simulated_error"

lines = []
with open(path +  file_name + '.txt') as f:
    lines = f.readlines()
asd = str(lines).replace("\n',","")
asd1 = asd.replace("\\n","")
asd2 = asd1.replace(", ","")
asd3 = asd2.replace("[","")
asd4 = asd3.split(">NC_045512.2 ''")

depth = []

for i in range(1,len(asd4)):
    asd5 = asd4[i].replace("\'","")
    depth.append(asd5)

print("Length: ",len(depth))
np.save(path_1 + file_name + ".npy",depth)

# depth_5_seq = []
# for i in range(len(protein_sequences)):
#     depth_5_seq.append(str(protein_sequences[i]))

print("Data Loaded")


# In[ ]:


# Original DNA Sequences
# path = "E:/RA/ISBRA_Adversarial_Attack/Dataset/"

# file_name = "n2020_5x_simulated_error"
# file_name = "n2020_10x_simulated_error"
# file_name = "random_5x_simulated_error"
file_name = "random_10x_simulated_error"

lines = []
with open(path +  file_name + '.txt') as f:
    lines = f.readlines()
asd = str(lines).replace("\n',","")
asd1 = asd.replace("\\n","")
asd2 = asd1.replace(", ","")
asd3 = asd2.replace("[","")
asd4 = asd3.split(">NC_045512.2 ''")

depth = []

for i in range(1,len(asd4)):
    asd5 = asd4[i].replace("\'","")
    depth.append(asd5)

print("Length: ",len(depth))
np.save(path_1 + file_name + ".npy",depth)

# depth_5_seq = []
# for i in range(len(protein_sequences)):
#     depth_5_seq.append(str(protein_sequences[i]))

print("Data Loaded")


