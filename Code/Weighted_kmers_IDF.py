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

import math

from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit

from sklearn.decomposition import KernelPCA

import timeit

from sklearn.preprocessing import LabelEncoder
from itertools import product
import timeit
import pandas as pd
from math import log

print("Packages Loading Done!!!")


# Define function to compute k-mers with IDF weighting
def build_kmers(sequence, ksize, idf_dict):
    kmer_counts = {}
    for i in range(len(sequence) - ksize + 1):
        kmer = sequence[i:i+ksize]
        if kmer not in kmer_counts:
            kmer_counts[kmer] = 0
        kmer_counts[kmer] += 1
    kmers = []
    weights = []
    for kmer, count in kmer_counts.items():
        if kmer in idf_dict:
            kmers.append(kmer)
            weights.append(count * idf_dict[kmer])
        else:
            kmers.append(kmer)
    return kmers, weights

# Define function to compute IDF weights
def compute_idf_weights(kmers_list):
    num_samples = len(kmers_list)
    idf_dict = {}
    for kmers in kmers_list:
        for kmer in set(kmers):
            if kmer not in idf_dict:
                idf_dict[kmer] = 0
            idf_dict[kmer] += 1
    for kmer, count in idf_dict.items():
        idf_dict[kmer] = log(num_samples / count)
    return idf_dict

#dataset_name_arr = ["original_preprocessed_8172","n2020_5x_simulated_error_8172","n2020_10x_simulated_error_8172","random_5x_simulated_error_8172","random_10x_simulated_error_8172"]
dataset_name_arr = ["org_red_seq_8220"]


data_path = "/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/Preprocessed/"

for e in range(0,len(dataset_name_arr)):
    dataset_name = dataset_name_arr[e]
    print("Dataset Name: ",dataset_name)
    
    seq_data_tmp = np.load(data_path + dataset_name + ".npy",allow_pickle=True)
    seq_data = []
    for q in range(len(seq_data_tmp)):
        asdf = seq_data_tmp[q]
        asdf = asdf.replace("]","")
        asdf = asdf.replace("N","-")
        asdf = asdf.replace("K","-")
        asdf = asdf.replace("W","-")
        asdf = asdf.replace("Y","-")
        asdf = asdf.replace("R","-")
        asdf = asdf.replace("M","-")
        asdf = asdf.replace("S","-")
        asdf = asdf.replace("D","-")
        asdf = asdf.replace("V","-")
        asdf = asdf.replace("H","-")
        asdf = asdf.replace("B","-")
        seq_data.append(asdf)
    
    
    print("Data Reading Done!!!")

    ############################################################################################
    # Compute IDF weights
    kmers_list = [build_kmers(smile, 3, {})[0] for smile in seq_data]
    idf_dict = compute_idf_weights(kmers_list)

    # Define characters for SMILES strings
    smiles_chars = 'ACGT-'

    # Define all possible k-mers for the given character set
    unique_seq_kmers_final_list = [''.join(c) for c in product(smiles_chars, repeat=3)]

    # Compute weighted k-mer frequency vectors using IDF weights
    start = timeit.default_timer()
    frequency_vectors = []
    for i, smile in enumerate(seq_data):
        if i%100==0:
            print("seqs: ",i,"/",len(seq_data))
        kmers, weights = build_kmers(smile, 3, idf_dict)
        frequency_vector = np.zeros(len(unique_seq_kmers_final_list))
        for kmer, weight in zip(kmers, weights):
            frequency_vector[unique_seq_kmers_final_list.index(kmer)] = weight
        frequency_vectors.append(frequency_vector)
    end = timeit.default_timer()
    print("Time taken to compute weighted k-mer frequency vectors:", end-start, "seconds")
    ############################################################################################


    np.save("/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/Vectors/Weighted_kmers_IDF_" + dataset_name + ".npy",frequency_vectors)

print("All processing done!!")