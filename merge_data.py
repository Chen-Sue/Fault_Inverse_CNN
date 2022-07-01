import sys, os
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
import pickle
import re
import time

t = time.time()

print("\n  Begin merge_data.py ... \n")

# fileLocation = r"C:\chengshu\ShiYaolin\Program\data_create\10w"  # location可修改
file_location = r"/mnt/c/chengshu/ShiYaolin/Program/data_create/10w"
fileFrom = ".csv"

def numberFile():
    global file_location
    number = 0
    file_glob = os.path.join(file_location, r"/disp_csv")
    for _ in os.listdir(file_glob):
        number += 1
    return number

numbers = numberFile()
print("\n  number of files: ", numbers)

class MergeData:

    def __init__(self):
        pass

    def pkl_dump_x(self, x):
        inputdata = open(file_location + r"/inputdata.pkl", "wb")
        pickle.dump(x, inputdata)
        inputdata.close()         

    def pkl_dump_y(self, y):
        outputdata = open(file_location + r"/outputdata.pkl", "wb")
        pickle.dump(y, outputdata)        
        outputdata.close()

    def mergeInputFile(self): 
        global fileLocation 
        global fileFrom
        x = pd.DataFrame() 
        t0 = time.time()
        for i in range(1, numbers+1):
            inputData = pd.read_csv(fileLocation + r"/disp_csv" + r"/%d"%i + fileFrom, \
                                   delimiter=",", \
                                   usecols=[0,1,2,3,4],\
                                   header=None)
            x = pd.concat([x, inputData])
            if i % 1000 == 0:
                print("  mergeInputFile() has been executed at {0} times! It takes {1:.2f} minutes. ".format(i, (time.time() - t0)/60))
        x.set_axis(["ux", "uy", "x", "y", "z"], axis="columns", inplace=True)
        x = x.replace(r"E+", " * 10 ** ")
        print("\n  x[:5]: \n", x[:5])        
        self.pkl_dump_x(x)
        print("\n  mergeInputFile() has been executed! ")

    def mergeOutputFile(self): 
        global fileLocation 
        global fileFrom
        y = pd.DataFrame(dtype="float")  
        t0 = time.time() 
        for i in range(1, numbers+1): 
            outputData = pd.read_csv(fileLocation + r"/inp_csv" + r"/%d"%i + fileFrom, \
                                     delimiter=",", \
                                     usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], \
                                     header=None)
            y = pd.concat([y, outputData])
            if i % 1000 == 0:
                print("  mergeOutputFile() has been executed at {0} times! It takes {1:.2f} minutes. ".format(i, (time.time() - t0)/60))
        y.set_axis(["slip", "xs", "ys", "depth", "length", "width", "strike", "dip", "rake"], axis="columns", inplace=True)
        print("  y.loc[:, depth][:5]: ", y.loc[:, "depth"][:5])
        t0 = time.time()
        for i in range(len(y)):
            for j in range(1, y.columns.size-3):
                y.iloc[i, j] = float(y.iloc[i, j].replace("d+03", ""))*1000
            if i % 1000 == 0:
                print("string_to_numeric has been executed at {0} times! It takes {1:.2f} minutes. ".format(i, (time.time() - t0)/60))
        print("\n  y.loc[:, depth][:5]: ", y.loc[:, "depth"][:5])
        print("\n  y[:5]: ", y[:5])
        print("\n  y.dtypes: ", y.dtypes)        
        self.pkl_dump_y(y)
        print("\n  mergeOutputFile() has been executed! ")

    def pkl_load_x(self):
        
        inputdata = open(file_location + r"/inputdata.pkl", "rb")
        x = pickle.load(inputdata)
        inputdata.close()
        print("\n  pkl_load_x() has been executed! ")
        return x

    def pkl_load_y(self):
        outputdata = open(file_location + r"/outputdata.pkl", "rb")
        y = pickle.load(outputdata)
        outputdata.close()
        print("\n  pkl_load_y() has been executed! ")
        return y

    # 缺失数据

    # 加上噪音


def main():
    m = MergeData()  
    _ = m.mergeInputFile()  
    _ = m.mergeOutputFile()
    x = m.pkl_load_x()
    y = m.pkl_load_y()  
    # print("\n  x: \n{}".format(x), "\n  and y: \n{}".format(y))
    

if __name__ == "__main__":
    main()

print("\n  It takes {:.2f} minutes. ".format((time.time() - t)/60))
print("\n  End ...\n")