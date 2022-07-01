# cd /mnt/c/chengshu/ShiYaolin/Program/Code
# cd C:\chengshu\ShiYaolin\Program\data_create\10w_mean

import os
import time
import numpy as np
import pandas as pd 
import pickle
from six.moves import cPickle 


print("\n  Begin to execute merge_input_data.py... \n")

#=================================================================================
# Environment
#============
t = time.time()
seed = 22
#---------------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # 忽略tensorflow警告信息
#---------------------------------------------------------------------------------


#=================================================================================
# Loading data
#=============  
file_location = r"/mnt/c/chengshu/ShiYaolin/Program/data_create/10w_mean"
input_data_path1 = file_location + r"/inputdata1.pkl"
input_data_path2 = file_location + r"/inputdata2.pkl"
input_data_path3 = file_location + r"/inputdata3.pkl"
input_data_path4 = file_location + r"/inputdata4.pkl"
input_data_path5 = file_location + r"/inputdata5.pkl"
with open(input_data_path1, "rb") as input_pickle_file:
	input_data1 = cPickle.load(input_pickle_file)
with open(input_data_path2, "rb") as input_pickle_file:
	input_data2 = cPickle.load(input_pickle_file)
with open(input_data_path3, "rb") as input_pickle_file:
	input_data3 = cPickle.load(input_pickle_file)
with open(input_data_path4, "rb") as input_pickle_file:
	input_data4 = cPickle.load(input_pickle_file)
with open(input_data_path5, "rb") as input_pickle_file:
	input_data5 = cPickle.load(input_pickle_file)
#---------------------------------------------------------------------------------
print("  ******")
print("  initial shape: input_data1={0}, input_data2={1}, input_data3={2}, input_data4={3}, input_data5={4}".\
	format(input_data1.shape, input_data2.shape, input_data3.shape, input_data4.shape, input_data5.shape))
print(  "******")
#---------------------------------------------------------------------------------


#=================================================================================
# Merge input data
#================= 
input_data = pd.concat([input_data1, input_data2, input_data3, input_data4, input_data5])
#---------------------------------------------------------------------------------
inputdata_pkl = open(file_location + r"/inputdata_mean.pkl", "wb")
pickle.dump(input_data, inputdata_pkl, protocol = 4)
inputdata_pkl.close()
#---------------------------------------------------------------------------------


print("\n  It takes {:.2f} minutes. ".format((time.time() - t)/60))
print("\n  End ...\n")