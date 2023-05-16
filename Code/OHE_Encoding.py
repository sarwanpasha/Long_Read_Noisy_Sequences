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

print("Packages Loading Done!!!")


#dataset_name_arr = ["n2020_5x_simulated_error_8172","n2020_10x_simulated_error_8172","random_5x_simulated_error_8172","random_10x_simulated_error_8172"]
dataset_name_arr = ["original_preprocessed_8172"]


data_path = "/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/Preprocessed/"

for e in range(0,len(dataset_name_arr)):
    dataset_name = dataset_name_arr[e]
    
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

    max_sequence_length = 0

    for i in range(len(seq_data)):
        if len(seq_data[i])>=max_sequence_length:
            max_sequence_length = len(seq_data[i])

    # Getting the unique values
    res = list(set(i for j in seq_data for i in j))

    # printing result
    print ("Unique values : ", str(res))
    print ("Unique values Length : ", len(res))
    print ("Max Sequence Length : ", max_sequence_length)


    # np.array(res).to_categorical() 

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(res)
    integer_encoded,max(integer_encoded)

    final_data = []
    for ind in range(len(seq_data)):
        #print(dataset_name," => ",ind,"/",len(seq_data))
        asd = list(str(seq_data[ind]))
        for i in range(len(res)):
    #         asd = np.where(asd == res[i], integer_encoded[i], asd)
            asd = list(map(lambda x: x.replace(res[i], str(integer_encoded[i])), asd))
        final_data.append(asd)

    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list


    one_hot_data = []
    ohe_vector_length = len(res)*max_sequence_length
    for i in range(len(final_data)):
        if i%1000==0:
            print(dataset_name, " => ",i,"/",len(final_data))
        row_wise = (final_data[i])

    #     row_wise = []
    #     for w in range(len(seq_1)):
    #         if seq_1[w]=='A' or seq_1[w]=='C' or seq_1[w]=='G' or seq_1[w]=='T':
    #             row_wise.append(seq_1[w])
    #         else:
    #             row_wise.append("-")

        row_vector = []
        for j in range(len(row_wise)):
            temp_vector = [0]* len(res)
            temp_vector[int(row_wise[j])] = 1
            row_vector.append(list(temp_vector))
        row_vec = flatten_list(row_vector)
        if(len(row_vec)<ohe_vector_length):
            for k in range(len(row_vec),ohe_vector_length):
                row_vec.append(0)
        one_hot_data.append(row_vec)
    ##############################################################################################################

    np.save("/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/Vectors/OHE_" + dataset_name + ".npy",one_hot_data)

print("All processing done!!")