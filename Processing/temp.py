import os
import subprocess
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import joblib
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import csv
# dataset = pd.read_csv("../Dataset/UNSW/UNSW_NB15_training-set.csv")

# argus_fields = [
#     "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload",
#     "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat", "smean", "dmean"
# ]
# argus_unsw = dataset[argus_fields]

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
    data_list = dataset.iloc[:,1:l]
    return data_list, label_list

def drop_nan(df):
    features = df.columns.drop("label")
    for f in features:
        df.loc[df[f] == '-', f] = np.nan
    df.dropna(inplace=True)


def encode_state_label(df):
    le = preprocessing.LabelEncoder()
    df["state"] = le.fit_transform(df["state"])

#RSP
def encode_state(s):
    le = preprocessing.LabelEncoder()
    state_list = ['ECO', 'ECR', 'CON', 'INT', 'TST', 'MAS', 'no', 'PAR', 'RST', 'ACC', 'URN', 'FIN', 'URH', 'REQ', 'CLO', 'TXD',]
    try:
        lenc = le.fit(state_list)
        s[4] = lenc.transform([s[4]])[0]
    except ValueError as e:
        state_list.append(str(s[4]))
        lenc = le.fit(state_list)
        s[4] = lenc.transform([s[4]])[0]

def clear_hex_value(s,index):
    cols = s[index].items()
    for index, value in cols:
        if isinstance(value, int):
            pass
        else:
            r = int(str(value),0)
            df = df.replace(to_replace=value, value=r)

def get_number_proto():
    file_proto_num = "../proto_num.csv"
    #proto_num = load_data(file_proto_num)
    f = csv.reader(open(file_proto_num, 'r'))
    pro_num = dict()
    for row in f:
        if row[1] != "":
            k = row[1].lower()
            v = row[0]
            pro_num[k] = v
        else:
            continue
    del pro_num['keyword']
    return pro_num


argus_fields = [
    "sport", "dport","dur", "proto", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat", "smeansz", "dmeansz",'stime', 'ltime'
]



"""from http://blog.kagesenshi.org/2008/02/teeing-python-subprocesspopen-output.html
"""

statement = "ra -S 192.168.1.11:561 -u -nn -s sport, dport,dur, proto, state, spkts, dpkts, sbytes, dbytes, sttl, dttl, sload, dload, sloss, dloss, swin, dwin, stcpb, dtcpb, tcprtt, synack, ackdat, smeansz, dmeansz,stime, ltime"
cmd = statement.split(" ")
print(cmd)
# p = subprocess.Popen(["ping","www.facebook.com","-n","10"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# stdout = []
# while True:
#     line = p.stdout.readline()
#     if not isinstance(line, (str)):
#         line = line.decode('utf-8')
#         print(type(pd.Series(line)))
#         print(pd.Series(line))
#     if (line == '' and p.poll() != None):
#         break

