"""
Modify *.inp files to *.csv as NN"S output files
"""

import sys, os
sys.path.append(os.pardir)
import time
import re
import csv
import pandas as pd

print("\n  Begin to execute inp_mean_csv.py ... \n")

t = time.time()

# file_location = r"C:\chengshu\ShiYaolin\Program\data_create\10w_mean"
file_location = r"/mnt/c/chengshu/ShiYaolin/Program/data_create/10w_mean"
fileFrom = ".csv"
numbers = numberFile()
print("\n  number of files: \n", numbers)

def numberFile():
    global file_location
    number = 0
    file_glob = os.path.join(file_location, "inp_mean")
    for _ in os.listdir(file_glob):
        number += 1
    return number

def inp_mean_csv():
    global file_location
    for k in range(1, numbers+1):
        with open(file_location + r"/inp_mean" + r"/%d"%k + fileFrom, "r", encoding="utf-8") as f, \
                open(file_location + r"/inp_mean_csv" + r"/%d"%k + fileFrom, mode="w+",encoding="utf-8") as f_new:     
            f_context = f.read()
            # Slip   [0.02, 1.50)m         \d\.\d{2}\s
            # xs     [-400.00, 400.00)km   .\d{0,3}\.\d{2}.\+\b03\s
            # ys     [-400.00, 400.00)km   .\d{0,3}\.\d{2}.\+\b03\s
            # zs     [0.00, 20.00)km       \d{1,2}\.\d{2}.\+\b03\s   
            # length [6.00, 60.00)km       \d{1,2}\.\d{2}.\+\b03\s
            # width  [4.00, 20.00)km       \d{1,2}\.\d{2}.\+\b03\s
            # strike [0.00, 360.00)        \d{1,3}\.\d{2}\s
            # dip    [0.00, 90.00)         \d{1,2}\.\d{2}\s
            # rake   [0.00, 360.00)        \d{1,3}\.\d{2}
            result = re.findall(r"\d\.\d{2}\s.\d{0,3}\.\d{2}.\+\b03\s.\d{0,3}\.\d{2}.\+\b03\s\d{1,2}\.\d{2}.\+\b03\s\d{1,2}\.\d{2}.\+\b03\s\d{1,2}\.\d{2}.\+\b03\s.\d{0,3}\.\d{2}\s\d{1,2}\.\d{2}\s.\d{0,3}\.\d{2}", f_context)
            for item in result:
                f_new.write(item)
        if k % 5000 == 0:
            print("  inp_mean_csv() has been executed at {} times. ".format(k))
        if k == numberFile():
            print("\n  inp_mean_csv() has been executed! \n")
            f.close()
            f_new.close()

def inp_mean_csv1():
    global file_location
    for k in range(1, numbers+1):
        data= pd.read_csv(file_location + r"/inp_mean_csv" + r"/%d"%k + fileFrom)
        a1 = pd.Series(data.columns.values, index=data.columns)
        a2 = list(a1)
        x_data = []
        for i in range(len(a2)):
            x_data.append(a2[i].split(" "))
        for j in range(len(x_data)):
            for _ in range(x_data[j].count("")):
                x_data[j].remove("")
        c = pd.DataFrame(x_data)
        c.to_csv(file_location + r"/inp_mean_csv" + r"/%d"%k + fileFrom, header=0, index=0)
        if k % 5000 == 0:
            print("  inp_mean_csv1() has been executed at {} times. ".format(k))          
        if k == numberFile():
            print("\n  inp_mean_csv1() has been executed! \n")

if __name__ == "__main__":
    inp_mean_csv()
    inp_mean_csv1()


print("\n  It takes {:.2f} minutes. ".format((time.time() - t)/60))
print("\n  End ... \n")