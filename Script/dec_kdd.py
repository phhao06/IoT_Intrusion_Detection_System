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
import seaborn as sns


def load_data(filepath):
    try:
        return pd.read_csv(filepath, delimiter=',')
    except:
        print("File Not Found")
        return

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

#split dataset to trainset and testset
def train_test_splitor(data, per):
    total_rows = data.shape[0]
    random_id_list= np.random.permutation(total_rows)
    train_idx = random_id_list[0:int(per*total_rows)]
    test_idx = random_id_list[int(per*total_rows):-1]
    train_data = data.iloc[train_idx,:]
    test_data = data.iloc[test_idx,:]
    return train_data, test_data

def split_label_trainset(dataset):
    #split label column
    l =len(dataset.columns) - 1
    label_list = dataset.iloc[:,-1]
    data_list = dataset.iloc[:,1:l-1]
    return data_list, label_list

def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


def swap_col(df):
    le = LabelEncoder() 
    col_proto = df["proto"]
    col_service = df["service"]
    col_flag = df["flag"]
    col_land = df["land"]
    col_login = df["logged_in"]
    col_ishost_login = df["is_host_login"]
    col_isguest_login = df["is_guest_login"]
    temp = df.drop(columns=["proto","service","flag","land","logged_in","is_host_login","is_guest_login"],axis=1,inplace=False)
    swap_df = pd.concat([col_proto,col_service,col_flag,col_land,col_login,col_ishost_login,col_isguest_login,temp], axis=1,sort=False)
    swap_df["proto"] = le.fit_transform(swap_df["proto"])
    swap_df["service"] = le.fit_transform(swap_df["service"])
    swap_df["flag"] = le.fit_transform(swap_df["flag"])
    swap_df["land"] = le.fit_transform(swap_df["land"])
    swap_df["logged_in"] = le.fit_transform(swap_df["logged_in"])
    swap_df["is_host_login"] = le.fit_transform(swap_df["is_host_login"])
    swap_df["is_guest_login"] = le.fit_transform(swap_df["is_guest_login"])
    #swap_df["attack_cat"] = le.fit_transform(swap_df["attack_cat"])
    return swap_df

def change_label(s):
    labels = s.values
    l = len(labels)
    for i in range(l):
        if labels[i] =='normal':
            labels[i]=0
        else:
            labels[i]=1
    return labels
    
def main():
    filepath = '../Dataset/KDDcup99/kddcup.data'#_10_percent'
    #file_test_path = '../Dataset/KDDcup99/corrected'


    df = load_data(filepath)
    df.columns = ['duration', 'proto','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label']
    labels = df['label']
    dataset = df.sample(frac=0.4, replace=False)
    dataset.dropna(inplace=True,axis=1)
    # testset = load_data(file_test_path)
    trainset, testset = train_test_splitor(dataset, per=0.7)
    

    # trainset.columns = ['duration', 'proto','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    # 'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    # 'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label']

    # testset.columns = [
    # 'duration', 'proto','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    # 'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    # 'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
    # ]

    
    # train_df = trainset.sample(frac=0.1,replace=False)
    # test_df = testset.sample(frac=0.1, replace=False)
    
    train_data,train_label = split_label_trainset(trainset)
    test_data,test_label = split_label_trainset(testset)

    # ch_train_label = change_label(train_label)
    # ch_test_label = change_label(test_label)
    #print(ch_train_label['unknown'])
    swap_train_data = swap_col(train_data)
    swap_test_data = swap_col(test_data)

    scale_col = swap_train_data.columns.drop(["proto","service","flag","land","logged_in","is_host_login","is_guest_login"])
    for col in scale_col:
        encode_numeric_zscore(swap_train_data,col)
        encode_numeric_zscore(swap_test_data,col)
    

    swap_train_data.dropna(inplace=True,axis=1)
    swap_test_data.dropna(inplace=True,axis=1)
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0,1,2])], remainder='passthrough') 
    ct = columnTransformer.fit(swap_train_data)
    ct_train_data = ct.transform(swap_train_data)#.toarray()
    ct_test_data = ct.transform(swap_test_data)#.toarray()

    #clf = SVC(gamma='auto')
    clf = DecisionTreeClassifier()
    #clf = RandomForestClassifier(random_state=0)
    clf.fit(ct_train_data, train_label)
    pred = clf.predict(ct_test_data)
    accurancy = accuracy_score(test_label,pred)
    fig, ax = plt.subplots(figsize=(50, 40))
    disp = plot_confusion_matrix(clf, ct_test_data, test_label, xticks_rotation='vertical',cmap=plt.cm.Blues,ax=ax)
    disp.ax_.set_title("Confusion Matrix")
    
    #sns.heatmap(conf,annot=True, fmt=".1f")
    print(accurancy)
    #print(confusion_matrix(test_label, pred))
    
    plt.show()

if __name__ == "__main__":
    main()