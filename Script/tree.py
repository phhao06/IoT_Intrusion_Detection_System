from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def load_data(filepath):
    try:
        return pd.read_csv(filepath, delimiter=',')
    except:
        print("File Not Found")
        return

def split_label_trainset(dataset):
    #split label column
    l =len(dataset.columns) - 1
    label_list = dataset.iloc[:,-1]
    data_list = dataset.iloc[:,1:l-1]
    return data_list, label_list

def change_label_to_num(input):
    encoder = OrdinalEncoder()
    input = input.values.reshape(1,-1)
    encoder.fit(input)
    input_encoded = encoder.transform(input)
    return input_encoded


def train_test_splitor(data, per):
    total_rows = data.shape[0]
    random_id_list= np.random.permutation(total_rows)
    train_idx = random_id_list[0:int(per*total_rows)]
    test_idx = random_id_list[int(per*total_rows):-1]
    train_data = data.iloc[train_idx,:]
    test_data = data.iloc[test_idx,:]
    return train_data, test_data

def transform_data(col):
    le = preprocessing.LabelEncoder()
    trans = preprocessing.LabelEncoder()
    attribute = list(set(col))
    le.fit(attribute)
    return le.transform(col)


#Encode Numeric Cols
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()
    if sd is None:
        sd = df[name].std()
    df[name] = (df[name] - mean)/sd
    
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def main():
    data_file_path = '../Dataset/UNSW/UNSW_NB15_training-set.csv'
    test_file_path = '../Dataset/UNSW/UNSW_NB15_testing-set.csv'
    dataset = load_data(data_file_path)
    #testset = load_data(test_file_path)
    

    print("Read {} rows from datafile".format(len(dataset)))
    #print("Read {} rows from testfile".format(len(testset)))
    

    

    headers = dataset.columns.values.tolist()
    
    # encode_text_dummy(data,"proto")
    # encode_text_dummy(data,"state")
    # encode_text_dummy(data,"service")
    
    for h in headers:
        if h == "proto" or h =="service" or h == "state":
            encode_text_dummy(dataset, h)
        else:
            encode_numeric_zscore(dataset, h)
    
    trainset, testset = train_test_splitor(dataset, per=0.7)

    data, label = split_label_trainset(trainset)
    data_for_test, true_label = split_label_trainset(testset)

    #print(data_for_test[0:5])

    #print(data)
    #data["attack_cat"] = transform_data(data["attack_cat"])


    # testset["proto"] = transform_data(testset["proto"])
    # testset["service"] = transform_data(testset["service"])
    # testset["state"] = transform_data(testset["state"])
    # encode_text_dummy(data_for_test,"proto")
    # encode_text_dummy(data_for_test,"state")
    # encode_text_dummy(data_for_test,"service")

    #data_for_test["attack_cat"] = transform_data(data_for_test["attack_cat"])


    

    clf = DecisionTreeClassifier()
    clf.fit(data, label)
    prediction = clf.predict(data_for_test[0:5])
    accurancy = accuracy_score(true_label, prediction)
    #conf_matrix =confusion_matrix(true_label, prediction)
    #print(conf_matrix )
    # print("=================================")
    #print(accurancy)
    
if __name__ == "__main__":
    main()