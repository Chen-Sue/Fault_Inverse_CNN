"""
Modify *.inp files to *.csv as NN"S output files
"""

import sys, os
sys.path.append(os.pardir)
import time
import re
import csv
import pandas as pd

print("\n  Begin to execute inp_csv.py... \n")

t = time.time()
# file_location = r"C:\chengshu\ShiYaolin\Program\data_create\10w"
file_location = r"/mnt/c/chengshu/ShiYaolin/Program/data_create/10w"
fileFrom0 = ".inp"
fileFrom1 = ".csv"
print("\n  number of files: \n", numberFile())

def numberFile():
    global file_location
    count = 0
    file_glob = os.path.join(file_location, "/inp")
    for _ in os.listdir(file_glob):
        count += 1
    return count

def inp_csv():
    global file_location
    for k in range(1, numberFile()+1):  
        with open(file_location + r"/inp" + r"/%d"%k + fileFrom0, "r", encoding="utf-8") as f, \
                open(file_location + r"/inp_csv" + r"/%d"%k + fileFrom0, mode="w+",encoding="utf-8") as f_new:     
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
            print("  inp_csv() has been executed at {} times. ".format(k))
        if k == numberFile():
            print("\n  inp_csv() has been executed! \n")
            f.close()
            f_new.close()

def inp_csv1():
    global file_location
    for k in range(1, count+1):  
        data= pd.read_csv(file_location + r"/inp_csv" + r"/%d"%k + fileFrom1)
        a1 = pd.Series(data.columns.values, index=data.columns)
        a2 = list(a1)
        x_data = []
        for i in range(len(a2)):
            x_data.append(a2[i].split(" "))
        for j in range(len(x_data)):
            for _ in range(x_data[j].count("")):
                x_data[j].remove("")
        c = pd.DataFrame(x_data)
        c.to_csv(file_location + r"/inp_csv" + r"/%d"%k + fileFrom1, header=0, index=0)
        if k % 5000 == 0:
            print("  inp_csv1() has been executed at {} times. ".format(k))          
        if k == numberFile():
            print("\n  inp_csv1() has been executed! \n")

if __name__ == "__main__":
    inp_csv()
    inp_csv1()


print("\n  It takes {:.2f} minutes. ".format((time.time() - t)/60))
print("\n  End ... \n")