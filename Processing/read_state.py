import pandas as pd
file = "../Dataset/UNSW/UNSW-train.csv"
df = pd.read_csv(file,delimiter=",")
print(list(set(df["state"])))