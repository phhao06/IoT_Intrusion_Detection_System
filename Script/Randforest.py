import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import joblib
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import re
import csv

def load_data(filepath):
    try:
        return pd.read_csv(filepath, delimiter=',',low_memory=False)
    except:
        print("File Not Found")
        return
def split_label_trainset(dataset):
    #split label column
    l =len(dataset.columns) - 1
    label_list = dataset.iloc[:,-1]
    data_list = dataset.iloc[:,0:l]
    return data_list, label_list

#split dataset to trainset and testset
def train_test_splitor(data, per):
    total_rows = data.shape[0]
    random_id_list= np.random.permutation(total_rows)
    train_idx = random_id_list[0:int(per*total_rows)]
    test_idx = random_id_list[int(per*total_rows):-1]
    train_data = data.iloc[train_idx,:]
    test_data = data.iloc[test_idx,:]
    return train_data, test_data

def drop_nan(df):
    features = df.columns.drop("label")
    for f in features:
        df.loc[df[f] == '-', f] = np.nan
    df.dropna(inplace=True)


def clear_hex_value(df,col):
    cols = df[col].items()
    for index, value in cols:
        if isinstance(value, int):
            pass
        else:
            r = int(str(value),0)
            df = df.replace(to_replace=value, value=r)

def encode_label(df):
    le = preprocessing.LabelEncoder()
    #col_proto = df["proto"]
    #col_service = df["service"]
    # col_state = df["state"]
    # temp = df.drop(columns=["state"],axis=1,inplace=False)
    # swap_df = pd.concat([col_state,temp], axis=1,sort=False)
    #swap_df["proto"] = le.fit_transform(swap_df["proto"])
    df["state"] = le.fit_transform(df["state"])
    #return swap_df

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




features = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 
'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']

argus_fields = [
    "sport", "dport","dur", "proto", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sintpkt", "dintpkt", "sjit", "djit", "swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat", "smeansz", "dmeansz",'stime', 'ltime',"label"
]



filepath = "../Dataset/UNSW/UNSW-train.csv"
filepath1 = "../Dataset/UNSW/UNSW-NB15_train_2.csv"
#dt = load_data(filepath)

proto_num = get_number_proto()
dt = load_data(filepath)

dt.columns = argus_fields
# print(len(set(dt["proto"])))
# s = set(dt["proto"])
# print(s)
# print(proto_num)
data = dt.replace({"proto": proto_num})
encode_label(data)
data = data.fillna(np.random.randint(100))
#print(set(en_data["proto"]))
# dt.loc[dt.proto == "tcp", "proto"] = 6
# dt.loc[dt.proto == "udp", "proto"] = 17
# dataset, testset = train_test_splitor(en_data, 0.8)


argus_fields_2 = [
    "sport", "dport","dur", "proto", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat", "smeansz", "dmeansz",'stime', 'ltime',"label"
]
dataset =data[argus_fields_2]
# df = dataset[argus_fields_2]
# test_df = testset[argus_fields_2]
# # en_df = encode_label(df)
# # en_test_df = encode_label(test_df)
x_train ,y_train = split_label_trainset(dataset)
# x_test, y_test = split_label_trainset(testset)


print('The scikit-learn version is {}.'.format(sklearn.__version__))

clf = RandomForestClassifier()
clf.fit(x_train, y_train)
# predictions = clf.predict(x_test)
# accurancy = accuracy_score(y_test,predictions)
# conf = confusion_matrix(y_test, predictions)
# # # #save the model to disk
filename = '../Model/ids_rf_3.9.sav'
# # filename = '../Model/dectree_model.sav'
joblib.dump(clf, filename)