"""
Transform *.disp files to *.csv as NN"S input files
"""

import sys, os
sys.path.append(os.pardir)
import pandas as pd
import time

print("\n  Begin to execute disp_csv.py ... \n")

t = time.time()

# file_location = r"C:\chengshu\ShiYaolin\Program\data_create\10w"
file_location = r"/mnt/c/chengshu/ShiYaolin/Program/data_create/10w"
fileFrom = ".csv"
numbers = numberFile()
print("\n  number of files: \n", numbers)

def numberFile():
    global file_location
    number = 0
    file_glob = os.path.join(file_location, "/inp")
    for _ in os.listdir(file_glob):
        number += 1
    return number

def disp_csv():
    for k in range(1, numbers+1):
        data = pd.read_csv(file_location + r"/disp" + r"/%d"%k + fileFrom)
        data1 = data.loc[2:,:]
        a1 = pd.Series(data1["# Displacements calculated with edcmp"].values, \
            index=data1["# Displacements calculated with edcmp"])
        a2 = list(a1)
        x_data = []
        for i in range(len(a2)):
            x_data.append(a2[i].split(" "))
        for i in range(len(x_data)):
            for _ in range(x_data[i].count("")):
                x_data[i].remove("")
        c =pd.DataFrame(x_data)
        c.to_csv(file_location + r"/disp_csv" + r"/%d"%k + fileFrom, header=0, index=0)
        if k % 5000 == 0:
            print("  disp_csv() has been executed at {0} times! It takes {1:.2f} minutes. ".format(k, (time.time() - t)/60))        
        if k == numberFile():
            print("\n  disp_csv() has been executed! \n")

if __name__ == "__main__":
    disp_csv()

print("\n It takes {:.2f} minutes. ".format((time.time() - t)/60))
print("\n End ... \n")