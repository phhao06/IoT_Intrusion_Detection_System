from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer 
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector


def load_data(filepath):
    try:
        return pd.read_csv(filepath, delimiter=',')
    except:
        print("File Not Found")
        return

def processing_data(df):
    x_col = df.columns.drop("id")
    x = df[x_col]
    onehotencoder = OneHotEncoder() 
    #print(temp)
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough') 
    #x = np.array(columnTransformer.fit_transform(x), dtype=object)
    x =columnTransformer.fit_transform(x)
    temp_df = pd.DataFrame(x.todense())

def split_label_trainset(dataset):
    #split label column
    l =len(dataset.columns) - 1
    label_list = dataset.iloc[:,-1]
    data_list = dataset.iloc[:,1:l]
    return data_list, label_list

def train_test_splitor(data, train_per):
    total_rows = data.shape[0]
    random_id_list= np.random.permutation(total_rows)
    train_idx = random_id_list[0:int(train_per*total_rows)]
    test_idx = random_id_list[int(train_per*total_rows):-1]
    train_data = data.iloc[train_idx,:]
    test_data = data.iloc[test_idx,:]
    return train_data, test_data

def main():
    filepath= "../Dataset/UNSW/UNSW_NB15_training-set.csv"
    testpath = "../Dataset/UNSW/UNSW_NB15_testing-set.csv"

    dataset = load_data(filepath)
    testset = load_data(testpath)

    le = LabelEncoder() 
  
    df =    dataset.sample(frac=0.08, replace=False)
    test_df = testset.sample(frac=0.001, replace=False)
    
    print(df.shape)
    df['proto']= le.fit_transform(df['proto']) 
    df['service']= le.fit_transform(df['service'])
    df['state']= le.fit_transform(df['state']) 
    df['attack_cat']= le.fit_transform(df['attack_cat']) 

    test_df['proto']= le.fit_transform(test_df['proto']) 
    test_df['service']= le.fit_transform(test_df['service'])
    test_df['state']= le.fit_transform(test_df['state']) 
    test_df['attack_cat']= le.fit_transform(test_df['attack_cat']) 

    #split data, label of dataset
    train_data, train_label = split_label_trainset(df)
    test_data, test_label = split_label_trainset(test_df)

    print("Shape of Train Data: {}".format(train_data.shape))
    print("Shape of Test Data: {}".format(test_data.shape))

    enc = OneHotEncoder(handle_unknown='ignore')
    enc_trans = enc.fit(train_data)
    enc_train_data = enc_trans.transform(train_data).toarray()
    # print(enc)
    # print(enc_train_data)
    enc_test_data = enc_trans.transform(test_data).toarray()
    #print(enc_test_data)
    

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(enc_train_data,train_label)
    pred = clf.predict(enc_test_data)
    #enc_test_data = enc_trans.transform(test_data)
    #print(enc_test_data)
    # enc_train_df = pd.DataFrame.sparse.from_spmatrix(enc_data)
    # enc_test_df = pd.DataFrame.sparse.from_spmatrix(enc_test_data)
    accurancy = accuracy_score(test_label,pred)
    print(accurancy)

##############
    # print("data shape: {}".format(enc_train_df.shape))
    # print("label shape: {}".format(train_label.shape))

if __name__ == "__main__":
    main()