# cd /mnt/c/chengshu/ShiYaolin/Program/Code
# cd C:\chengshu\ShiYaolin\Program\Code\bayesian

import  os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # 忽略tensorflow警告信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 只使用第一块GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # 忽略tensorflow警告信息
os.environ["PATH"] += os.pathsep + r"C:\Users\cs\Anaconda3\Library\bin\graphviz"
import time
import tensorflow as tf
assert tf.__version__.startswith("2.") # 判断tf版本是否以"2.0"开头
tf.keras.backend.clear_session()  # 销毁当前的TF图并创建一个新图。有避免旧模型/图层混乱
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import mpl_toolkits.axisartist.axislines as axislines
from scipy.stats import norm
from six.moves import cPickle 
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from tensorflow import keras
from tensorflow.keras import layers, optimizers, metrics, Sequential, initializers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, LeakyReLU, GlobalMaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model


# In[1]:

seed = 22
tf.random.set_seed(seed)
np.random.seed(seed)

#=================================================================================
# Loading data
# ============  
file_location = r"C:\chengshu\ShiYaolin\Program\data_create\10w"
# file_location = r"/mnt/c/chengshu/ShiYaolin/Program/data_create/10w"
#---------------------------------------------------------------------------------
input_data_path = file_location + r"/inputdata.pkl"
with open(input_data_path, "rb") as input_pickle_file:
	input_data = cPickle.load(input_pickle_file)
#---------------------------------------------------------------------------------
output_data_path = file_location + r"/outputdata.pkl"
with open(output_data_path, "rb") as output_pickle_file:
	output_data = cPickle.load(output_pickle_file)
input_pickle_file.close()
output_pickle_file.close()
#---------------------------------------------------------------------------------
print("******")
print("  Initial shape: input_data={0}, output_data={1}".format(input_data.shape, output_data.shape))
#---------------------------------------------------------------------------------

#=================================================================================
# Reshaping data
# ==============
input_data = np.array(input_data, dtype=np.float32).reshape(-1, 5)
# input_data = np.array(input_data, dtype=np.float32).reshape(-1, 51*51*5)
output_data = np.array(output_data, dtype=np.float32).reshape(-1, 9)
#---------------------------------------------------------------------------------

#=================================================================================
# Normalized data
# ===============
scaler_x = MaxAbsScaler().fit(input_data)   # MaxAbsScaler
input_data = scaler_x.transform(input_data).reshape(-1, 82*5)
#---------------------------------------------------------------------------------
scaler_y = MaxAbsScaler().fit(output_data)  # MinMaxScaler
output_data = scaler_y.transform(output_data).reshape(-1, 9)
#---------------------------------------------------------------------------------
print("******")
print("  Normalized shape: input_data={0}, output_data={1}".format(input_data.shape, output_data.shape))
#---------------------------------------------------------------------------------

#=================================================================================
# Filtering data
# ==============
# for i in range(len(input_data)):
# 	sum = 0
# 	for j in range(2601*2):
# 		if abs(input_data[i, j]) < 2 * 10 ** -3:
# 			sum += 1
# 		if sum >= 5:
# 			input_data_new = np.delete(input_data, i, axis=0)
# 			output_data_new = np.delete(output_data, i, axis=0)
# 			continue		
# print(len(input_data_new))
#---------------------------------------------------------------------------------

#=================================================================================
# Spliting data
# =============
# num1, num2, num3 = int(100000 * 0.6), int(100000 * 0.8), int(100000 * 1.0) 
# x_train, x_validation, x_test = input_data[:num1], input_data[num1:num2], input_data[num2:num3]
# y_train, y_validation, y_test = output_data[:num1], output_data[num1:num2], output_data[num2:num3]
num1 = int(100000 * 0.8)
num2 = int(100000)
x_train, x_test = input_data[:num1], input_data[num1:num2]
y_train, y_test = output_data[:num1], output_data[num1:num2]
print("******")
print("  Spliting shape: x_train={0}, y_train={1}".format(x_train.shape, y_train.shape))
# print("  Spliting shape: x_validation={0}, y_validation={1}".format(x_validation.shape, y_validation.shape))
print("  Spliting shape: x_test={0}, y_test={1}".format(x_test.shape, y_test.shape))
#---------------------------------------------------------------------------------
x_train = x_train.reshape((-1, 82*5))
# x_validation = x_validation.reshape((-1, 51, 51, 3))
x_test = x_test.reshape((-1, 82*5))
y_train = y_train.reshape((-1, 9))
# y_validation = y_validation.reshape((-1, 9))
y_test = y_test.reshape((-1, 9))
print("******")
print("  x_train.shape={0}, y_train.shape={1}".format(x_train.shape, y_train.shape))
print("  x_test.shape={0}, y_test.shape={1}".format(x_test.shape, y_test.shape))
print("  x_train[0]={0}".format(x_train[0]))
print("  y_train[0]={0}".format(y_train[0]))
#---------------------------------------------------------------------------------

#=================================================================================
# Dataset
# =======
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)  
    y = tf.cast(y, dtype=tf.float32) 
    return x, y
batch_size = 64 * 4
#---------------------------------------------------------------------------------
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(preprocess).batch(batch_size)
#---------------------------------------------------------------------------------
# validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
# validation_dataset = validation_dataset.map(preprocess).batch(batch_size)
#---------------------------------------------------------------------------------
test_dataset =  tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(preprocess).batch(batch_size)
#---------------------------------------------------------------------------------
train_sample = next(iter(train_dataset))
# validation_sample = next(iter(validation_dataset))
test_sample = next(iter(test_dataset))
print("******")
print("  Batch: \n  train_sample[0].shape={0}, train_sample[1].shape={1}".format(train_sample[0].shape, train_sample[1].shape))
#---------------------------------------------------------------------------------


# In[3]:

def main():
    # cd C:\chengshu\ShiYaolin\Program\Code\bayesian
    # python bayesian.py
    input_node = 82*5
    output_node = 9    
    batch_size = 64 * 4
    epochs = 1000
    initial_learning_rate = 1e-6
    print("  initial_learning_rate = ", initial_learning_rate)
    learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=False)
    import tensorflow_probability as tfp
    # build model and optimizer
    model = Sequential()
    model.add(tfp.layers.DenseReparameterization(512, activation=tf.nn.relu))
    # model.add(BatchNormalization(axis=1))
    # model.add(LeakyReLU())
    model.add(tfp.layers.DenseReparameterization(256, activation=tf.nn.relu))
    # model.add(BatchNormalization(axis=1))
    # model.add(LeakyReLU())
    model.add(tfp.layers.DenseReparameterization(128, activation=tf.nn.relu))
    # model.add(BatchNormalization(axis=1))
    # model.add(LeakyReLU())
    model.add(tfp.layers.DenseReparameterization(64, activation=tf.nn.relu))
    # model.add(BatchNormalization(axis=1))
    # model.add(LeakyReLU())
    model.add(tfp.layers.DenseReparameterization(output_node))
    # model.add(layers.Activation("tanh"))
    #---------------------------------------------------------------------------------
    model.build(input_shape=[None, x_train.shape[1]])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    print("Number of variables in the model :", len(model.variables))
    model.summary()
    print("******")
    print("  layer nums:", len(model.layers))
    # train
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), verbose=2)
    # model.save("./models/self_cnn_model.h5")
    plot_model(model=model, to_file="./pictures/model_net.png", show_shapes=True)
    # History of the loss values and metric values during training
    file = open("./history.pkl", "wb")
    history_dict = history.history
    print("******")
    print("  history_dict.keys(): {}".format(history_dict.keys()))
    print("  history dict: \n{}".format(history_dict))
    pickle.dump(history_dict, file)
    file.close()
    # evaluate on test set
    test_loss, test_mae, test_rmse = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
    print("******")
    print("  Evaluate on test data: ")
    print("  test_mse={0:.4f}%, test_mae={1:.4f}%, test_rmse={2:.4f}%".\
	    format(test_loss*100, test_mae*100, test_rmse*100))
    # Value of Slip Difference
    y_test_pre = model.predict(x_test)
    targets, predictions = scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(y_test_pre)
    diff_slip = (targets[:, 0] - predictions[:, 0])
    print("******")
    print("  diff_slip={}".format(diff_slip))
    #---------------------------------------------------------------------------------
    # Picture
    fig = plt.figure(figsize=(20.0, 6.0))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    xticks = range(1, epochs+1, 1)
    #---------------------------------------------------------------------------------
    # subpicture1
    ax1 = axislines.Subplot(fig, 311)
    fig.add_subplot(ax1)
    ax1.axis[:].major_ticks.set_tick_out(True)
    n, bins, patches = ax1.hist(diff_slip, bins="auto", density=True, cumulative=False, facecolor="yellowgreen", alpha=0.5, label="histogram")
    y = norm.pdf(bins, diff_slip.mean(), diff_slip.std()) # 生成正态分布函数
    ax1.plot(bins, y, "b--", label="Gauss Distribution")
    ax1.set_title("Histogram of Slip Difference distribution: $\mu = {0:.2f}$, $\sigma={1:.2f}$".format(diff_slip.mean(), diff_slip.std()))
    ax1.set_xlabel("Value of Slip Difference(m)")
    ax1.set_ylabel("Percent")
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=len(diff_slip)))
    ax1.legend(loc="upper left", shadow=True)
    #---------------------------------------------------------------------------------
    # subpicture2
    ax2 = axislines.Subplot(fig, 312)
    fig.add_subplot(ax2)
    ax2.axis[:].major_ticks.set_tick_out(True)
    ax2.set_xticks(xticks)
    # ax2.set_yticks(yticks)
    ax2.axis["bottom"].label.set_text("Epochs")
    ax2.axis["left"].label.set_text("Erroe") 
    ax2.set_title("mse (Mean square error)")
    rmse = history_dict["rmse"]
    val_rmse = history_dict["val_rmse"]
    ax2.plot(xticks, rmse, "--b*", label="Training")
    ax2.plot(xticks, val_rmse, "--r*", label="Validation")
    ax2.legend(loc="best", shadow=True)
    #---------------------------------------------------------------------------------
    # subpicture3
    ax3 = axislines.Subplot(fig, 313)
    fig.add_subplot(ax3)
    ax3.axis[:].major_ticks.set_tick_out(True)
    ax3.set_xticks(xticks)
    # ax3.set_yticks(yticks)
    ax3.axis["bottom"].label.set_text("Epochs")
    ax3.axis["left"].label.set_text("error") 
    ax3.set_title("mae (Mean absolute error)")
    acc = history_dict["mae"]
    val_acc = history_dict["val_mae"]
    ax3.plot(xticks, acc, "--b*", label="Training")
    ax3.plot(xticks, val_acc, "--r*", label="Validation")
    ax3.legend(loc="best", shadow=True)
    #---------------------------------------------------------------------------------
    plt.savefig("./pictures/self_cnn.png")
    fig.show()
    #---------------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n******")
    print("  Begin to execute main_mean.py... \n")
    t = time.time()
    main()
    print("\nIt takes {:.2f} minutes. ".format((time.time() - t)/60))
    print("\nEnd ... \n")