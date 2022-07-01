"""
Modify fault parameters to generate *.inp files
"""

import os, sys
import time
import numpy as np
import re

print("\n  Begin to execute inp.py... \n")

t = time.time()

# file_location = r"C:\chengshu\ShiYaolin\Program\data_create"
file_location = r"/mnt/c/chengshu/ShiYaolin/Program/data_create"
fileFrom = ".inp"
numbers = numberFile()
print("\n  number of files: \n", numbers)

def numberFile():
    global file_location
    number = 0
    file_glob = os.path.join(file_location, r"/10w/inp")
    for _ in os.listdir(file_glob):
        number += 1
    return number

def main():
    for i in range(1, numbers+1):
        with open(file_location + r"/edcmpdata_bam/edcmp_bam.inp", mode="r", encoding="utf-8") as fin, \
                open(file_location + r"/10w/inp_mean" + r"/%d"%i + fileFrom, mode="w+", encoding="utf-8") as fout:
            for line in fin:     
                if "izmhs" in line:  
                    # modify  file_name
                    line = line.replace("izmhs", str("%d" %i))
                if "1.00 5.00d+03 -25.00d+03 1.00d+03 40.00d+03 10.00d+03 90.00 90.00 0.00" in line:  
                    # Slip [0.02, 1.50)m
                    slip = (1.50 - 0.02) * np.random.random() + 0.02
                    # xs [-400.00, 400.00)km
                    xs = (400 - (-400)) * np.random.random() + (-400)
                    # ys [-400.00, 400.00)km
                    ys = (400 - (-400)) * np.random.random() + (-400)
                    # zs [0, 20.00)km
                    zs = (20.00 - 0.00) * np.random.random() + 0.00
                    # length [6.00, 60.00)km
                    length = (60.00 - 6.00) * np.random.random() + 6.00
                    # width [4.00, 20.00)km
                    width = (20.00 - 4.00) * np.random.random() + 4.00
                    # strike（走向） [0, 360)
                    strike = 0 + np.random.random()*360
                    # dip（倾向） [0, 90)
                    dip = np.random.random()*90
                    # rake（滑动角） [0, 360)
                    rake = 0 + np.random.random()*360                    
                    line = line.replace("1.00 5.00d+03 -25.00d+03 1.00d+03 40.00d+03 10.00d+03 90.00 90.00 0.00", \
                                        "{:.2f} {:.2f}d+03 {:.2f}d+03 {:.2f}d+03 {:.2f}d+03 {:.2f}d+03 {:.2f} {:.2f} {:.2f}".\
                                            format(slip, xs, ys, zs, length, width, strike, dip, rake))
                fout.write(line)
        if i % 5000 == 0:
            print("  inp() has been executed at {} times. ".format(i))

        fin.close()
        fout.close()
 
if __name__ == "__main__":
    main()


print("\n  It takes {:.2f} minutes. ".format((time.time() - t)/60))
print("\n  End ... \n")