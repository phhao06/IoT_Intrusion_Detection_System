import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
    return "[{}]".format(",".join(result))
        
def analyze(df):
    print()
    cols = df.columns.values
    total = float(len(df))

    #print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
        else:
            print("** {}:{}".format(col,expand_categories(df[col])))
            expand_categories(df[col])

def split_label_trainset(dataset):
    #split label column
    l =len(dataset.columns) - 1
    label_list = dataset.iloc[:,-1]
    data_list = dataset.iloc[:,1:l]
    return data_list, label_list


def swap_col(df):
    le = LabelEncoder() 
    col_proto = df["proto"]
    col_service = df["service"]
    col_state = df["state"]
    #col_att_cat = df["attack_cat"]
    temp = df.drop(columns=["proto","service","state"],axis=1,inplace=False)
    swap_df = pd.concat([col_proto,col_service,col_state,temp], axis=1,sort=False)
    # swap_df["proto"] = le.fit_transform(swap_df["proto"])
    # swap_df["service"] = le.fit_transform(swap_df["service"])
    # swap_df["state"] = le.fit_transform(swap_df["state"])
    #
    return swap_df

def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd

def main():
    
    dataset = pd.read_csv("../Dataset/UNSW/UNSW_NB15_training-set.csv")

    testset = pd.read_csv("../Dataset/UNSW/UNSW_NB15_testing-set.csv")


    dataset.drop(columns="attack_cat", axis=1, inplace=True)
    testset.drop(columns="attack_cat", axis=1, inplace=True)

    sample = dataset.sample(frac=0.1, replace=False)
    test_sample = dataset.sample(frac=0.001, replace=False)
    print(len(sample.columns))
    train_data,train_label = split_label_trainset(sample)
    test_data,test_label = split_label_trainset(test_sample)
    
    swap_train_data = swap_col(train_data)
    swap_test_data = swap_col(test_data)
    
    scale_col = swap_train_data.columns.drop(["proto","service","state","is_sm_ips_ports","is_ftp_login"])
    for col in scale_col:
        encode_numeric_zscore(swap_train_data,col)
        encode_numeric_zscore(swap_test_data,col)
    # df["proto"] = le.fit_transform(df["proto"])
    # df["service"] = le.fit_transform(df["service"])
    # df["state"] = le.fit_transform(df["state"])
    # df["attack_cat"] = le.fit_transform(df["attack_cat"])
    # enc_fit_data = pd.concat([col_proto,col_service,col_state,col_att_cat], axis=1,sort=False)
    # columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0,1,2,3])], remainder='passthrough') 
    # ct_df = columnTransformer.fit_transform(df)
    #print(swap_train_data)
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0,1,2])], remainder='passthrough') 
    ct = columnTransformer.fit(swap_train_data)

    #print(swap_test_data)
    ct_train_data = ct.transform(swap_train_data).toarray()
    ct_test_data = ct.transform(swap_test_data).toarray()
    # print(ct_train_data.shape)
    # print(ct_train_data[5].shape)
    #print(ct_train_data.toarray())
    #clf = SVC(gamma='auto')
    
    #clf = DecisionTreeClassifier()
    clf = RandomForestClassifier(random_state=0)
    clf.fit(ct_train_data, train_label)
    pred = clf.predict(ct_test_data)
    accurancy = accuracy_score(test_label,pred)
    plot_confusion_matrix(clf, ct_test_data, test_label)
    print(accurancy)
    print(confusion_matrix(test_label, pred))
    plt.show()

if __name__ == "__main__":
    main();