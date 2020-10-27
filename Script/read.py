import csv
import pandas as pd
import numpy as np
# headers = []
# with open('../Dataset/UNSW/NUSW-NB15_features.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for line in csv_reader:
#         headers.append(line[1])

# headers.pop(0)
# features = [x.lower() for x in headers]

# data = pd.read_csv("../Dataset/UNSW/UNSW-NB15_1.csv",delimiter=',')
# data.columns = features

# data.dropna(inplace=True,axis=1)
# print(data["service"])
# print(data["dsport"])
def load_data(filepath):
    try:
        return pd.read_csv(filepath, delimiter=',',low_memory=False)
    except:
        print("File Not Found")
        return


def drop_nan(df):
    features = df.columns.drop(["label"])
    for f in features:
        df.loc[df[f] == '-', f] = np.nan
    df.dropna(axis=0, inplace=True)

features = ['saddr', 'sport', 'daddr', 'dport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 
'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']
argus_fields = [
    "sport", "dport","dur", "proto", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sintpkt", "dintpkt", "sjit", "djit", "swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat", "smeansz", "dmeansz",'stime', 'ltime',"label"
]


filepath_1 = "../Dataset/UNSW/UNSW-NB15_train_1.csv"
#filepath_2 = "../Dataset/UNSW/UNSW-NB15_train_2.csv"
filepath_3 = "../Dataset/UNSW/UNSW-NB15_train_3.csv"
filepath_4 = "../Dataset/UNSW/UNSW-NB15_train_4.csv"


dt1 = load_data(filepath_1)
#dt2 = load_data(filepath_2)
dt3 = load_data(filepath_3)
dt4 = load_data(filepath_4)

dt1.columns = features
#dt2.columns = features
dt3.columns = features
dt4.columns = features

df1 = dt1[argus_fields]
#df2 = dt2[argus_fields]
df3 = dt3[argus_fields]
df4 = dt4[argus_fields]

print(df1.shape)
#print(df2.shape)
print(df3.shape)
print(df4.shape)
# print(len(dt1.loc[dt1['service'] == '-', "service"]))
# print(len(dt2.loc[dt2['service'] == '-', "service"]))
# print(len(dt3.loc[dt3['service'] == '-', "service"]))
# print(len(dt4.loc[dt4['service'] == '-', "service"]))
drop_nan(df1)
#drop_nan(df2)
drop_nan(df3)
drop_nan(df4)
# dt1.dropna(inplace=True)
# dt2.dropna(inplace=True)
# dt3.dropna(inplace=True)
# dt4.dropna(inplace=True)
final_df = pd.concat([df1,df3,df4], ignore_index=True, axis=0)
final_df.to_csv("../Dataset/UNSW/UNSW-train_134.csv",sep=",", index=False)