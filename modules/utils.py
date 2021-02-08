import pandas as pd
import numpy as np
import tensorflow.keras.backend as K

def f_load_data(prep_option):
    if prep_option == 1:
        print("Normal Scaling of numeric variables (prep_option = 1)")
        data_path = "Data/Scaled_data.csv"
    if prep_option == 2:
        print("Binning (following Rudin) and one hot encoding (prep_option = 2)")
        data_path = "Data/Bin_Encoded_data_v2.csv"
    if prep_option == 3:
        print("Binning and applying WOE, calculating WOE on Rudin's bins (prep_option = 3)")
        data_path = "Data/WOE_data.csv"
    if prep_option == 4:
        print("Binning and applying WOE, following Rudin (prep_option = 4)")
        data_path = "Data/WOE_Rud_data.csv"

    ori_path = "Data/heloc_dataset_v1.csv"

    CLASS = 'RiskPerformance' 

    data = pd.read_csv(ori_path)
    X1 = pd.read_csv(data_path)
    y = pd.read_csv("Data/y_data.csv")

    print('Target: Bad (y=1)')
    class_names = sorted(y[CLASS].unique(),  reverse=True)
    print(y[CLASS].value_counts())
    y_onehot = pd.get_dummies(y[CLASS])[['Bad']]
    print(np.array(np.unique(y_onehot, return_counts=True)).T)

    print('X shape:',X1.shape)
    
    return data, X1, y, y_onehot


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))