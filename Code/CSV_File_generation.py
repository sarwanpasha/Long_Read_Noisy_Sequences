import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot

import csv

print("Packages Loaded!!")

#location_id_val = 1559
#location_id_val = 480
#location_id_val = 18728
#location_id_val = 1041
#location_id_val = 944
#location_id_val = ["OHE_n2020_5x_simulated_error_8172","OHE_n2020_10x_simulated_error_8172","OHE_random_5x_simulated_error_8172","OHE_random_10x_simulated_error_8172"]
location_id_val = ["original_preprocessed_8172","n2020_5x_simulated_error_8172","n2020_10x_simulated_error_8172","random_5x_simulated_error_8172","random_10x_simulated_error_8172"]


for location_id_ind in location_id_val:
    data = np.load("/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/Preprocessed/" + str(location_id_ind) + ".npy")
    
    print("Vector Length: ",len(data[0]))
    
    write_path_11 = "/alina-data2/Sarwan/ISBRA_Adversarial_Attack/Dataset/Vectors/" + location_id_ind + ".csv"


    with open(write_path_11, 'w', newline='') as file:
        writer = csv.writer(file)
        for q in range(0,len(data)):
            aa_1 = data[q]
            writer.writerow(aa_1)
    
    
    ##########################################################


print("All processing done!!")

