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

#dataset_name_arr = ["original_preprocessed_8172","n2020_5x_simulated_error_8172","n2020_10x_simulated_error_8172","random_5x_simulated_error_8172","random_10x_simulated_error_8172","depth_5_red_seq_new_8220","depth_10_red_seq_new_8220"]
#dataset_name_arr = ["depth_5_red_seq_new_8220","depth_10_red_seq_new_8220"]
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
    # seq="ATGCGATATCGTAGGCGTCGATGGAGAGCTAGATCGATCGATCTAAATCCCGATCGATTCCGAGCGCGATCAAAGCGCGATAGGCTAGCTAAAGCTAGCAA"

    start = timeit.default_timer()

    unique_seq_kmers_final_list = [''.join(c) for c in product('ACGT-', repeat=3)]  
    Kmer = 3

    final_feature_vector = []
    for ind_loop in range(len(seq_data)):
        if ind_loop%100==0:
            print("Index :",ind_loop,"/",len(seq_data))
        seq = seq_data[ind_loop]
        ################ Generate k-mers (Start) #########################
        L = len(seq)
        
        k_mers_final = []
        for i in range(0, L-Kmer+1):
            sub_f=seq[i:i+Kmer]
            k_mers_final.append(sub_f)
        #     print(sub_f)

        ################ Generate k-mers (end) #########################


        ################ Generate PWM (Start) #########################
        # To create a new txt file for writing "EI_nine_pwm.txt"
        # Initialize the PWM with four rows and nine columns [i.e., 4 lists of zeros]
        # a = [0]*9
        # c = [0]*9
        # g = [0]*9
        # t = [0]*9

        a_val = [0]*Kmer
        c_val = [0]*Kmer
        g_val = [0]*Kmer
        t_val = [0]*Kmer


        # input_file = open("E:/RA/Position Weight Matrix/Code/EI_nine.txt","r")   
        count_lines = 0 # Initialize the total number of sequences to 0
        # Read line by line, stripping the end of line character and
        # updating the PWM with the frequencies of each base at the 9 positions
        for ii in range(len(k_mers_final)):
            line = k_mers_final[ii]
            count_lines += 1 # Keep counting the sequences
            for i in range(len(line)):
                if line[i] == 'A':
                    a_val[i] = a_val[i]+1
                elif line[i] == 'C':
                    c_val[i] = c_val[i]+1
                elif line[i] == 'G':
                    g_val[i] = g_val[i]+1
                elif line[i] == 'T':
                    t_val[i] = t_val[i]+1



        LaPlace_pseudocount = 0.1
        equal_prob_nucleotide = 0.04
        

        for i in range(len(k_mers_final[0])):

            a_val[i] = round(math.log((a_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/0.06557377049180328,2),3)
            c_val[i] = round(math.log((c_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/0.03278688524590164,2),3)
            g_val[i] = round(math.log((g_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/0.06557377049180328,2),3)
            t_val[i] = round(math.log((t_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/0.06557377049180328,2),3)

        ################ Generate PWM (End) #########################

        ################ Assign Individual k-mers Score (Start) #########################
        each_k_mer_score = []
        listofzeros = [0] * len(unique_seq_kmers_final_list)
        for ii in range(len(k_mers_final)):
            line = k_mers_final[ii]
            score = 0
            for i in range(len(line)):
                if line[i] == 'A':
                    score += a_val[i]
                elif line[i] == 'C':
                    score += c_val[i]
                elif line[i] == 'G':
                    score += g_val[i]
                elif line[i] == 'T':
                    score += t_val[i]
                    
            # Write each input sequence followed by its score into the file
            # "EI_nine_output.txt"
            final_score_tmp = round(score, 3)
            each_k_mer_score.append(final_score_tmp)
            
            ###################### assign weughted k-mers frequency score ###############
            kmer_val_check = str(line)
            aa_lst_1 = kmer_val_check.replace(",","")
            aa_lst_2 = aa_lst_1.replace("[","")
            aa_lst_3 = aa_lst_2.replace("\"","")
            aa_lst_4 = aa_lst_3.replace("]","")
            aa_lst_5 = aa_lst_4.replace("'","")
            aa_lst_6 = aa_lst_5.replace(" ","")
        
            if "B" in aa_lst_6 or "J" in aa_lst_6 or "X" in aa_lst_6 or "Z" in aa_lst_6:
                aa_1 = aa_lst_6.replace("B","-")
                aa_2 = aa_1.replace("J","-")
                aa_3 = aa_2.replace("X","-")
                aa_4 = aa_3.replace("Z","-")
                ind_tmp = unique_seq_kmers_final_list.index(aa_4)
            else:
                ind_tmp = unique_seq_kmers_final_list.index(aa_lst_6)
            listofzeros[ind_tmp] = listofzeros[ind_tmp] + (1 * final_score_tmp)
            
    #     final_feature_vector.append(each_k_mer_score)
        final_feature_vector.append(listofzeros)
        ################ Assign Individual k-mers Score (end) #########################
        
    stop = timeit.default_timer()
    print("kmers PSSMFreq2Vec Time : ", stop - start)
    ############################################################################################


    np.save("/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/Vectors/PWM_kmers_" + dataset_name + ".npy",final_feature_vector)

print("All processing done!!")