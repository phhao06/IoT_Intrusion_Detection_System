from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import csv


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
    data_list = dataset.iloc[:,:l]
    return data_list, label_list

def change_text_to_num(input):
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

def main():
    filepath = './kddcup.data_10_percent'
    test_file = './kddcup.testdata.unlabeled'
    #test_file = './test.data'
    dataset = load_data(filepath)
    #testset = load_data(test_file)
    train_data, test_data = train_test_splitor(dataset,0.7)
    convert_data = change_text_to_num(train_data.iloc[:100,2])
    print(convert_data)
    # data, label = split_label_trainset(train_data)

    # test_data, true_label = split_label_trainset(test_data)
    
    # label_float = change_text_to_num(label)
    # clf = DecisionTreeClassifier()
    # clf.fit(data,label)
    # #print(data.shape[0])
    # pred = clf.predict(test_data)
    # accurancy = accuracy_score(true_label, pred)
    # print(accurancy)
    
    

if __name__ == "__main__":
    main()