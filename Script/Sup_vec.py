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
    data_list = dataset.iloc[:,1:l]
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
    col_proto = df["proto"]
    #col_service = df["service"]
    col_state = df["state"]
    temp = df.drop(columns=["proto","state"],axis=1,inplace=False)
    swap_df = pd.concat([col_proto,col_state,temp], axis=1,sort=False)
    swap_df["proto"] = le.fit_transform(swap_df["proto"])
    swap_df["state"] = le.fit_transform(swap_df["state"])
    return swap_df


features = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 
'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']



filepath = "../Dataset/UNSW/UNSW-train.csv"

# dt.columns = features
#testset.columns = features
dt = load_data(filepath)

#dt.dropna(axis=1,inplace=True)
# testset.dropna(axis=1,inplace=True)
# drop_nan(dt)

dataset, testset = train_test_splitor(dt, 0.7)

# clear_hex_value(dataset,'sport')

# clear_hex_value(testset,'dport')

argus_fields = [
    "sport", "dsport","dur", "proto", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sintpkt", "dintpkt", "sjit", "djit", "swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat", "smeansz", "dmeansz",'stime', 'ltime',"label"
]

# df = dataset[argus_fields]
# test_df = testset[argus_fields]



#clear_hex_value(df,'sport')
# clear_hex_value(test_df,'sport')

# clear_hex_value(df,'dsport')
# clear_hex_value(test_df,'dsport')


en_df = encode_label(dataset)
en_test_df = encode_label(testset)

#encode all data
#en_df = encode_label(dt)


x_train ,y_train = split_label_trainset(en_df)
x_test, y_test = split_label_trainset(en_test_df)

print('The scikit-learn version is {}.'.format(sklearn.__version__))

clf = make_pipeline(preprocessing.StandardScaler(), SVC(gamma='auto'))
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
accurancy = accuracy_score(y_test,predictions)
conf = confusion_matrix(y_test, predictions)
print(conf)
print(accurancy)
#save the model to disk
# filename = '../Model/svm_model.sav'
# joblib.dump(clf, filename)