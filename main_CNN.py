# cd /mnt/c/chengshu/ShiYaolin/Program/Code
# cd C:\chengshu\ShiYaolin\Program\Code
# python main_CNN.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # 忽略tensorflow警告信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 只使用第一块GPU
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
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, LeakyReLU, ReLU, GlobalMaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from matplotlib.pyplot import MultipleLocator 

print("\n******")
print("  Begin to execute main_mean.py... \n")
t = time.time()

# In[1]:

seed = 22
tf.random.set_seed(seed)
np.random.seed(seed)

#=================================================================================
# Parameters
# ==========
input_node = 51*51  # input_node = input("input node : ")
output_node = 9
batch_size = 64 * 4
epochs = 1000
initial_learning_rate = 1e-3
print("  initial_learning_rate = ", initial_learning_rate)
learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=False)										
#---------------------------------------------------------------------------------

#=================================================================================
# Loading data
# ============  
file_location = r"C:\chengshu\ShiYaolin\Program\data_create\10w_mean"
# file_location = r"/mnt/c/chengshu/ShiYaolin/Program/data_create/10w_mean"
#---------------------------------------------------------------------------------
input_data_path = file_location + r"/inputdata_mean.pkl"
with open(input_data_path, "rb") as input_pickle_file:
	input_data = cPickle.load(input_pickle_file)
#---------------------------------------------------------------------------------
output_data_path = file_location + r"/outputdata_mean.pkl"
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
input_data = np.array(input_data, dtype=np.float32)[:, 2:5].reshape(-1, 51*51*3)
# input_data = np.array(input_data, dtype=np.float32).reshape(-1, 51*51*5)
output_data = np.array(output_data, dtype=np.float32).reshape(-1, 9)
#---------------------------------------------------------------------------------


#=================================================================================
# Normalized data
# ===============
scaler_x = MaxAbsScaler().fit(input_data)   # MaxAbsScaler
input_data = scaler_x.transform(input_data).reshape(-1, 51*51*3)
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

x_train = x_train.reshape((-1, 51, 51, 3))
# x_validation = x_validation.reshape((-1, 51, 51, 3))
x_test = x_test.reshape((-1, 51, 51, 3))
y_train = y_train.reshape((-1, 9))
# y_validation = y_validation.reshape((-1, 9))
y_test = y_test.reshape((-1, 9))
print("******")
print("  x_train.shape={0}, y_train.shape={1}".format(x_train.shape, y_train.shape))
print("  x_test.shape={0}, y_test.shape={1}".format(x_test.shape, y_test.shape))
# print("  x_train[0]={0}".format(x_train[0]))
# print("  y_train[0]={0}".format(y_train[0]))
#---------------------------------------------------------------------------------

#=================================================================================
# Dataset
# =======
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)  
    y = tf.cast(y, dtype=tf.float32) 
    return x, y
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

weight_decay = 1e-4

#=================================================================================
# Building model
#===============
model = Sequential()
#---------------------------------------------------------------------------------
model.add(Conv2D(filters=16, 
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_size=(5, 5), 
                    strides=(1, 1), 
                    kernel_initializer=initializers.he_normal(), 
                    bias_initializer=tf.zeros_initializer()))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
#---------------------------------------------------------------------------------
model.add(Conv2D(filters=32, 
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_size=(5, 5), 
                    strides=(1, 1), 
                    kernel_initializer=initializers.he_normal(), 
                    bias_initializer=tf.zeros_initializer()))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
#---------------------------------------------------------------------------------
model.add(Conv2D(filters=64, 
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_size=(5, 5), 
                    strides=(1, 1), 
                    kernel_initializer=initializers.he_normal(), 
                    bias_initializer=tf.zeros_initializer()))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
#---------------------------------------------------------------------------------
model.add(Conv2D(filters=128, 
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_size=(5, 5), 
                    strides=(1, 1), 
                    kernel_initializer=initializers.he_normal(), 
                    bias_initializer=tf.zeros_initializer()))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
#---------------------------------------------------------------------------------
model.add(Conv2D(filters=256, 
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_size=(5, 5), 
                    strides=(1, 1), 
                    kernel_initializer=initializers.he_normal(), 
                    bias_initializer=tf.zeros_initializer()))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
#---------------------------------------------------------------------------------
# model.add(Flatten())
model.add(GlobalMaxPool2D())
#---------------------------------------------------------------------------------
model.add(Dense(256, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer(), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(LeakyReLU())
#---------------------------------------------------------------------------------
model.add(Dense(128, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer(), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(LeakyReLU())
#---------------------------------------------------------------------------------
# model.add(Dense(output_node, activation=tf.nn.tanh, kernel_initializer=initializers.glorot_normal(), bias_initializer=tf.zeros_initializer()))
model.add(Dense(output_node, kernel_initializer=initializers.glorot_normal(), bias_initializer=tf.zeros_initializer()))
#---------------------------------------------------------------------------------
x_shape  = x_train.shape
model.build(input_shape=[None, x_shape[1], x_shape[2], x_shape[3]])
#---------------------------------------------------------------------------------
model.summary()
print("******")
print("  layer nums:", len(model.layers))
#---------------------------------------------------------------------------------


#=================================================================================
# Checkpoint
# ==========
# checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="auto", save_freq=1)
# reducelr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="auto", cooldown=0, min_lr=0)
# callback_lists = [checkpoint, reducelr]
#---------------------------------------------------------------------------------


#=================================================================================
# Optimizer
# =========
optimizer = optimizers.Adam(learning_rate=learning_rate)
#---------------------------------------------------------------------------------


#=================================================================================
# Compile
# =======
model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name='rmse')])
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
log_dir = os.path.join("log")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
error_dir = os.path.join("error")
if not os.path.exists(error_dir):
    os.mkdir(error_dir)
model_dir = os.path.join("model")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
weight_dir = os.path.join("weight")
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)
#---------------------------------------------------------------------------------
# output_model_file = os.path.join("./model/the_best_model.h5")
output_model_file = os.path.join("./model/model_{epoch:05d}-{loss:.9f}-{mae:.9f}-{rmse:.9f}-{val_loss:.9f}-{val_mae:.9f}-{val_rmse:.9f}.h5")
#---------------------------------------------------------------------------------
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir),
				tf.keras.callbacks.CSVLogger(os.path.join(log_dir,"logs.log"),separator=","),
				tf.keras.callbacks.ModelCheckpoint(output_model_file, monitor="val_loss", verbose=1, save_best_only = True, save_weights_only=False),
				tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=epochs, verbose=True)]  # val_loss loss
#---------------------------------------------------------------------------------


#=================================================================================
# Save weights
# ============
# model.save_weights("./weights/model")
# model.load_weights("./weights/model")
# model.save_weights("./model.h5")
# model.load_weights("./model.h5")
#---------------------------------------------------------------------------------


#=================================================================================
# Save networls
# ============
# import json
# import pprint
# json_str = model.to_json()  # 序列化成json
# pprint.pprint(json.loads(json_str))
# fresh_model = tf.keras.models.model_from_json(json_str)
# yaml_str = model.to_yaml()  # 保持为yaml格式  #需要提前安装pyyaml
# print(yaml_str)
# fresh_model = tf.keras.models.model_from_yaml(yaml_str)


#=================================================================================
# Train on training set
# =====================
# history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(x_validation, y_validation)) #, callbacks=callback_lists)
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(x_test, y_test)) #, callbacks=callback_lists)
# history = model.fit(train_dataset, epochs=epochs,	validation_data=validation_dataset, verbose=2)
#---------------------------------------------------------------------------------


#=================================================================================
# Save all model
# ==============
model.save("./models/self_cnn_model.h5")
# model = tf.keras.models.load_model("all_model.h5")
#---------------------------------------------------------------------------------

#=================================================================================
# Save model"s picture
# ====================
plot_model(model=model, to_file="./pictures/model_net.png", show_shapes=True)
#---------------------------------------------------------------------------------


#=================================================================================
# History of the loss values and metric values during training
# ============================================================
file = open("./history.pkl", "wb")
history_dict = history.history
print("******")
print("  history_dict.keys(): {}".format(history_dict.keys()))
# print("  history dict: \n{}".format(history_dict))
pickle.dump(history_dict, file)
file.close()
#---------------------------------------------------------------------------------


#=================================================================================
# Evaluate on test set
# ====================
test_loss, test_mae, test_rmse = model.evaluate(x_test, y_test, verbose=2)
print("Evaluate on test data: ")
print("  test_mse={0:.4f}%, test_mae={1:.4f}%, test_rmse={2:.4f}%".\
	format(test_loss*100, test_mae*100, test_rmse*100))
#---------------------------------------------------------------------------------


#=================================================================================
# Value of Slip Difference
# ========================
y_test_pre = model.predict(x_test)
targets, predictions = scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(y_test_pre)
#---------------------------------------------------------------------------------
diff_slip = (targets[:, 0] - predictions[:, 0])
print("******")
print("  diff_slip={}".format(diff_slip))
#---------------------------------------------------------------------------------


#=================================================================================
# Picture
# =======
fig = plt.figure(figsize=(25.0, 18.0))
plt.tick_params(axis='both', which='major', labelsize=13)
fig.subplots_adjust(hspace=0.4, wspace=0.4)
xticks = range(1, epochs+1, 1)
# yticks = np.arange(0, 0.33, 0.03)
#---------------------------------------------------------------------------------
# subpicture1
ax1 = axislines.Subplot(fig, 311)
fig.add_subplot(ax1)
ax1.axis[:].major_ticks.set_tick_out(True)
n, bins, patches = ax1.hist(diff_slip, bins="auto", density=True, cumulative=False, facecolor="yellowgreen", alpha=0.5, label="histogram")
y = norm.pdf(bins, diff_slip.mean(), diff_slip.std()) # 生成正态分布函数
ax1.plot(bins, y, "b--", label="Gauss Distribution")
ax1.set_title("Histogram of Slip Difference distribution: $\mu = {0:.2f}$, $\sigma={1:.2f}$".format(diff_slip.mean(), diff_slip.std()))
ax1.set_xlabel("Value of Slip Difference(m)", fontsize=20)
ax1.set_ylabel("Percent", fontsize=20)
ax1.yaxis.set_major_formatter(PercentFormatter(xmax=len(diff_slip)))
# # ax2=plt.gca() #ax为两条坐标轴的实例
# ax1.xaxis.set_major_locator(MultipleLocator(epochs/10)) #把x轴的主刻度设置为epochs/10的倍数
# ax1.yaxis.set_major_locator(MultipleLocator(0.05)) #把y轴的主刻度设置为0.05的倍数
ax1.legend(loc="upper left", shadow=True, fontsize=15)
#---------------------------------------------------------------------------------
# subpicture2
ax2 = axislines.Subplot(fig, 312)
fig.add_subplot(ax2)
ax2.axis[:].major_ticks.set_tick_out(True)
ax2.set_xticks(xticks)
# ax2.set_yticks(yticks)
ax2.axis["bottom"].label.set_text("Epochs")
ax2.axis["left"].label.set_text("error") 
ax2.set_title("RMSE", fontsize=18)
rmse = history_dict["rmse"]
val_rmse = history_dict["val_rmse"]
ax2.plot(xticks, rmse, "b", label="Training")
ax2.plot(xticks, val_rmse, "r", label="Validation")
# ax2.xaxis.set_major_locator(MultipleLocator(epochs/10)) #把x轴的主刻度设置为epochs/10的倍数
ax2.legend(loc="upper right", shadow=True, fontsize=15)
#---------------------------------------------------------------------------------
# subpicture3
ax3 = axislines.Subplot(fig, 313)
fig.add_subplot(ax3)
ax3.axis[:].major_ticks.set_tick_out(True)
ax3.set_xticks(xticks)
# ax3.set_yticks(yticks)
ax3.axis["bottom"].label.set_text("Epochs")
ax3.axis["left"].label.set_text("error") 
ax3.set_title("MAE", fontsize=18)
acc = history_dict["mae"]
val_acc = history_dict["val_mae"]
ax3.plot(xticks, acc, "b", label="Training")
ax3.plot(xticks, val_acc, "r", label="Validation")
# ax3.xaxis.set_major_locator(MultipleLocator(epochs/10)) #把x轴的主刻度设置为epochs/10的倍数
ax3.legend(loc="upper right", shadow=True, fontsize=15)
#---------------------------------------------------------------------------------
plt.show()
plt.savefig("./pictures/self_cnn.png")
fig.show()
#---------------------------------------------------------------------------------


#=================================================================================
# Prediction on new data
# ======================
# x_new_location = r"/mnt/c/chengshu/ShiYaolin/Program"
# # x_new_location = r"C:\chengshu\ShiYaolin\Program"
# x_new = pd.read_csv(x_new_location + r"/Code/x_new.csv", \
# 					delimiter=",", \
# 					usecols=[0, 1, 2, 3, 4], \
# 					header=None)
# x_new = np.array(x_new, dtype="float32").reshape(1, 2601, 5)
# x_new[:, 4] = -x_new[:, 4]
# x_new = x_new.reshape(-1, 5)[:, 2:5]
# x_new_scaler = scaler_x.transform(x_new)
# #---------------------------------------------------------------------------------
# x_new = x_new_scaler.reshape(-1, 51, 51, 3)
# #---------------------------------------------------------------------------------
# y_new = model.predict(x_new)
# y_new = scaler_y.inverse_transform(y_new)
# print("******")
# print("Prediction on new data: ")
# print("  y_new={}".format(y_new))
# print("******")
# #---------------------------------------------------------------------------------
# magnitude = y_new[:, 0] * y_new[:, 2] * y_new[:, 3] * 30 * (10 ** 9)  
# print("\n", "magnitude", magnitude, "\n")
# if magnitude>0:
# 	magnitude = (2.0 / 3.0) * math.log(magnitude[0], 10)  - 6.033
# 	print("\n", "predict x_new : {:.4f}".format(magnitude), "\n")


#---------------------------------------------------------------------------------
print("\nIt takes {:.2f} minutes. ".format((time.time() - t)/60))
print("\nEnd ... \n")
#---------------------------------------------------------------------------------