import os
import subprocess
import logging
import csv
import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
from Netflow import Netflow

def main():
    clf = joblib.load("../Model/model_rf_ids.sav")
    
    #statement = "ra -S 192.168.1.11:561 -u -nn -s sport, dport,dur, proto, state, spkts, dpkts, sbytes, dbytes, sttl, dttl, sload, dload, sloss, dloss, swin, dwin, stcpb, dtcpb, tcprtt, synack, ackdat, smeansz, dmeansz,stime, ltime"
    stm = input("cmd > ")
    cmd = stm.split(" ")
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = []
    while True:
        line = p.stdout.readline()
        if not isinstance(line, (str)):
            line = line.decode('utf-8')
            temp_list = line.split(",")
            temp_sr = pd.Series(temp_list)
            temp_sr.index = argus_fields
            nf =Netflow(temp_sr)
            nf.encode_state()
            nf.clear_hex_value()
            nf.to_num()
            nf.reshape()
            predictions = clf.predict(nf.flow)
            print(predictions)
        if (line == '' and p.poll() != None):
            break

if __name__ == "__main__":
    main()