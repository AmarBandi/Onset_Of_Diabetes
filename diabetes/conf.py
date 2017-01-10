import os
import csv

DIRPATH = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.dirname(DIRPATH) + "/data"

CSVPATH = DATAPATH + "/diabetes.csv"
# DBPATH = DATAPATH + "/diabetic.csv"
# NDBPATH = DATAPATH + "/not_diabetic.csv"

#SPLIT_PROPORTION = 0.25

def read_csv(path=CSVPATH):
    data = []
    with open(CSVPATH, "rb") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
           data.append(row)
    return data
