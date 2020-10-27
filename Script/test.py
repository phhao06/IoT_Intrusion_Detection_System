from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
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
    #col_proto = df["proto"]
    #col_service = df["service"]
    #state = df["state"]
    #temp = df.drop(columns=["proto","state"],axis=1,inplace=False)
    #swap_df = pd.concat([col_proto,col_state,temp], axis=1,sort=False)
    df["state"] = le.fit_transform(df["state"])
    #swap_df["state"] = le.fit_transform(swap_df["state"])
    #return swap_df

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


argus_fields_2 = [
    "sport", "dport","dur", "proto", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat", "smeansz", "dmeansz",'stime', 'ltime'
]


filepath2 = "../Testset/attack_2.csv"


# test = [
#     'Sport', 'Dur', 'Proto', 'State', 'SrcPkts', 'DstPkts', 'SrcBytes',
#        'DstBytes', 'sTtl', 'dTtl', 'SrcLoad', 'DstLoad', 'SrcLoss', 'DstLoss',
#        'SIntPkt', 'DIntPkt', 'SrcJitter', 'DstJitter', 'SrcWin', 'DstWin',
#        'SrcTCPBase', 'DstTCPBase', 'TcpRtt', 'SynAck', 'AckDat', 'sMeanPktSz',
#        'dMeanPktSz', 'StartTime', 'LastTime'
# ]

argus_fields = [
    "sport", "dport","dur", "proto", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sintpkt", "dintpkt", "sjit", "djit", "swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat", "smeansz", "dmeansz","stime", "ltime"
]

print(len(argus_fields_2))

#sport, dport,dur, proto, state, spkts, dpkts, sbytes, dbytes, sttl, dttl, sload, dload, sloss, dloss, swin, dwin, stcpb, dtcpb, tcprtt, synack, ackdat, smeansz, dmeansz,stime, ltime


proto_num = get_number_proto()
# clf = joblib.load("../Model/model_rf.sav")
# predictions = []
# with open(filepath2) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         line_count = 0
#         for row in csv_reader:
#             if line_count == 0:
#                 #print("column header")
#                 line_count += 1
#             else:
#                 #print(row)
#                 sample = pd.Series(row)
#                 # #en_sample = encode_label(sample)
                
#                 encode_state(sample)
#                 sample = pd.to_numeric(sample, errors='coerce').fillna(np.random.randint(100), downcast='infer')
#                 #sample = sample.fillna(np.random.randint(100), inplace=True)
#                 #print(len(sample))
#                 prediction = clf.predict(sample.values.reshape(1,-1))
#                 predictions.append(prediction[0])
#                 line_count += 1
                
#         print(f'Processed {line_count} lines.')
# print(predictions)
testset = load_data(filepath2)
testset.columns = argus_fields_2
# 
testset = testset.fillna(0)
scols = testset['sport'].items()
for index, value in scols:
    if isinstance(value, int):
        pass
    else:
        r = float.fromhex(str(value))
        testset = testset.replace(to_replace=value, value=r)

dcols = testset['dport'].items()
for index, value in dcols:
    if isinstance(value, int):
        pass
    else:
        r = float.fromhex(str(value))
        testset = testset.replace(to_replace=value, value=r)

#test_df = testset[argus_fields_2]
encode_state_label(testset)
# print(en_df)
clf = joblib.load("../Model/model_rf_ids.sav")
predictions = clf.predict(testset)
print(predictions)
sum = 0
for x in predictions:
    if x == 1:
        sum+=1
print("Accurancy: {}".format(sum/len(predictions)))
