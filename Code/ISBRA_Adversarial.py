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


#orig_seq = np.load("/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/gisaid_hcov_19_orig_8505_seq.npy",allow_pickle=True)
#attributes = np.load("/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/gisaid_hcov_19_org_8505_attributes.npy",allow_pickle=True)


# In[8]:


# Original DNA Sequences

path_1 = "/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/"


#path = "E:/RA/ISBRA_Adversarial_Attack/Dataset/"

# name_ttmp = "n2020_5x_simulated_error"
# name_ttmp = "n2020_10x_simulated_error"
# name_ttmp = "random_5x_simulated_error"
# name_ttmp = "random_10x_simulated_error"


file_name_1 = np.load(path_1 + "n2020_5x_simulated_error" +  ".npy")
file_name_2 = np.load(path_1 + "n2020_10x_simulated_error.npy")
file_name_3 = np.load(path_1 + "random_5x_simulated_error.npy")
file_name_4 = np.load(path_1 + "random_10x_simulated_error.npy")


print("Data Loaded")


#len(file_name_1)


# In[11]:


orig_seq_1 = np.load("/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/gisaid_hcov_19_orig_8505_seq.npy",allow_pickle=True)
attributes_1 = np.load("/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/gisaid_hcov_19_org_8505_attributes.npy",allow_pickle=True)

orig_seq = orig_seq_1[0:8453]
attributes = attributes_1[0:8453]

file_name_1_final = file_name_1[0:8453]
file_name_2_final = file_name_2[0:8453]
file_name_3_final = file_name_3[0:8453]
file_name_4_final = file_name_4[0:8453]

variant_names = []

for i in range(len(attributes)):
    asd = attributes[i]
    variant_names.append(asd[1])


# In[12]:
print("Data Loaded Again")

len(file_name_1_final)


# In[13]:


idx = pd.Index(variant_names) # creates an index which allows counting the entries easily
print('Here are all of the viral species in the dataset: \n', len(idx),"entries in total")
aq = idx.value_counts()
print(aq)


# In[14]:


new_variant_names = []
new_all_attr = []
orig_seq_new = []
depth_5_seq_new = []
depth_10_seq_new = []
depth_15_seq_new = []
depth_20_seq_new = []

for i in range(len(variant_names)):
    asd = variant_names[i]
    if asd=='AY.103' or asd== 'AY.44' or asd== 'AY.100'  or asd== 'AY.3' or asd== 'AY.25' or asd== 'AY.25.1' or asd== 'AY.39'  or asd== 'AY.119' or asd== 'B.1.617.2'  or asd== 'AY.20' or asd== 'AY.26' or asd== 'AY.4' or asd== 'AY.117' or asd== 'AY.113' or asd== 'AY.118'  or asd== 'AY.43' or asd== 'AY.122'  or asd== 'BA.1' or asd== 'AY.119.2' or asd== 'AY.47' or asd== 'AY.39.1' or asd== 'AY.121' or asd== 'AY.75' or asd== 'AY.3.1'  or asd== 'AY.3.3' or asd== 'AY.107'  or asd== 'AY.34.1'  or asd== 'AY.46.6' or asd== 'AY.98.1' or asd== 'AY.13'  or asd== 'AY.116.1' or asd== 'AY.126' or asd== 'AY.114'  or asd== 'AY.125' or asd== 'AY.46.1' or asd== 'AY.34' or asd== 'AY.92'  or asd== 'AY.46.4' or asd== 'AY.127'  or asd== 'AY.98' or asd== 'AY.111':
        new_variant_names.append(asd)
        
        new_all_attr.append(attributes[i])
        
#         ac = str(list(orig_seq[i]))
        qw = str(list(orig_seq[i]))
        qw_1 = qw.replace(",","")
        qw_2 = qw_1.replace("\'","")
        qw_3 = qw_2.replace("[","")
        qw_4 = qw_3.replace("]","")
        qw_5 = qw_4.replace(" ","")

        orig_seq_new.append(qw_5)
        
        depth_5_seq_new.append(file_name_1_final[i])
        depth_10_seq_new.append(file_name_2_final[i])
        depth_15_seq_new.append(file_name_3_final[i])
        depth_20_seq_new.append(file_name_4_final[i])
        
        


# In[15]:


#len(depth_5_seq_new)


# In[16]:


print("Now saving the final Data!!")


path_new = "/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/Preprocessed/"

# name_ttmp = "n2020_5x_simulated_error"
# name_ttmp = "n2020_10x_simulated_error"
# name_ttmp = "random_5x_simulated_error"
# name_ttmp = "random_10x_simulated_error"

np.save(path_new + "n2020_5x_simulated_error" + "_" + str(len(depth_5_seq_new)) +".npy",depth_5_seq_new)
np.save(path_new + "n2020_10x_simulated_error" + "_" + str(len(depth_10_seq_new)) +".npy",depth_10_seq_new)
np.save(path_new + "random_5x_simulated_error" + "_" + str(len(depth_15_seq_new)) +".npy",depth_15_seq_new)
np.save(path_new + "random_10x_simulated_error" + "_" + str(len(depth_20_seq_new)) +".npy",depth_20_seq_new)

np.save(path_new + "variant_names_" + "_" + str(len(new_variant_names)) + ".npy",new_variant_names)
np.save(path_new + "all_attributes_" + "_" + str(len(new_all_attr)) + ".npy",new_all_attr)

np.save(path_new + "original_preprocessed_" + str(len(orig_seq_new)) + ".npy",orig_seq_new)

print("All Processing Done!!")