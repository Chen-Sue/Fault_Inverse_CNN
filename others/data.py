import sys, os
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import time
import pickle
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split

from merge_data import MergeData

# randnum = random.randint(0, 100)
randnum = 0

rate = 0.25
input_node = 246
output_node = 7

class Data:

    def __init__(self, x , y):
        self.x = x
        self.y = y       
      
    # 在类的内部定义访问数据的函数
    def print_xy(self):
        print('x:{}'.format(self.x), ', and y:{}'.format(self.y))

    # 私有变量（private），只有内部可以访问，代码更加健壮
    def print_private_xy(self):
        print('x:{}'.format(self.__x), ', and y:{}'.format(self.__y))

    # 外部代码获取x, y
    def get_xy(self):
        return self.__x, self.__y

    # 允许外部代码修改x, y
    def set_x(self, x, y):
        self.__x = x
        self.__y = y

    # 重新定义x y 的形状
    def reshape_xy(self, x, y):
        self.x = np.array(x).reshape(len(y) , input_node)  
        self.y = np.array(y).reshape(len(y) , output_node)
        return self.x, self.y

    # 打乱数据集对
    def shuffle_xy(self, x, y):
        np.random.seed(randnum)
        self.x = np.array(shuffle(list(x)))
        np.random.seed(randnum)
        self.y = np.array(shuffle(list(y)))
        return self.x, self.y

    # 分割数据
    def split_xy(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=rate, random_state=randnum, shuffle=False)
        return x_train, x_test, y_train, y_test

    # 标准化
    def standerScaler(self, x_train, x_test, y_train, y_test):
        scaler_x = StandardScaler().fit(x_train) 
        x_train_scaler = scaler_x.transform(x_train) 
        x_test_scaler = scaler_x.transform(x_test)  
        scaler_y = StandardScaler().fit(y_train)
        y_train_scaler = scaler_y.transform(y_train.reshape(-1, 7)) 
        y_test_scaler = scaler_y.transform(y_test.reshape(-1, 7))  
        return x_train_scaler, x_test_scaler, y_train_scaler, y_test_scaler
    
    # 区间缩放
    def minMaxScaler(self, x_train, x_test, y_train, y_test):
        scaler_x = MinMaxScaler().fit(x_train) 
        x_train_scaler = scaler_x.transform(x_train) 
        x_test_scaler = scaler_x.transform(x_test)  
        scaler_y = MinMaxScaler().fit(y_train) 
        y_train_scaler = scaler_y.transform(y_train.reshape(-1, 7)) 
        y_test_scaler = scaler_y.transform(y_test.reshape(-1, 7))  
        return x_train_scaler, x_test_scaler, y_train_scaler, y_test_scaler

    # 归一化
    def normalizeScaler(self, x_train, x_test, y_train, y_test):
        scaler_x = Normalizer().fit(x_train) 
        x_train_scaler = scaler_x.transform(x_train) 
        x_test_scaler = scaler_x.transform(x_test)  
        scaler_y = Normalizer().fit(y_train) 
        y_train_scaler = scaler_y.transform(y_train.reshape(-1, 7)) 
        y_test_scaler = scaler_y.transform(y_test.reshape(-1, 7))  
        return x_train_scaler, x_test_scaler, y_train_scaler, y_test_scaler

    # 利用零均值和单位方差的缩放方法进行预处理
    def inverse_standerScaler_x(self, x):
        scaler = StandardScaler()
        scaler.fit(x)
        return scaler.inverse_transform(x)   

def main():    
    t1 = time.time()
    m = MergeData()
    m.numberFile()
    x = m.pkl_load_x()
    y = m.pkl_load_y()  
    print('\n x: \n{}'.format(x), '\n \n , and y: \n{}'.format(y))
    print('\n data.py has been executed!!! It takes {:.3f} minutes.'.format((time.time() - t1)/60))

if __name__ == '__main__':
    main()

