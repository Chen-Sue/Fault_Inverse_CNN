# cd C:\chengshu\ShiYaolin\Program\Code
# python main_BP.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # 忽略tensorflow警告信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 只使用第一块GPU
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from tensorflow import keras
from tensorflow.keras import layers, optimizers, metrics, Sequential, initializers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, LeakyReLU, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2



# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#=================================================================================
# Environment
# ===========
t = time.time()
seed = 22
tf.random.set_seed(seed)   # 对全局随机数生成种子的设置
np.random.seed(seed)  # 使用相同的参数，每次生成的随机数都相同
#---------------------------------------------------------------------------------

#=================================================================================
# Parameters
# ==========
input_node = 82*5 # input_node = input("input node : ") # 82*3=246
output_node = 51*51*5
batch_size = 64
epochs = 10
initial_learning_rate = 1e-3
learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate,
													decay_steps=10000,
													decay_rate=0.96,
													staircase=False)										
#---------------------------------------------------------------------------------


#=================================================================================
# Loading data
# ============   
file_location = r"C:\chengshu\ShiYaolin\Program\data_create\10w"
input_data_path = file_location + r"/inputdata.pkl"
file_location = r"C:\chengshu\ShiYaolin\Program\data_create\10w_mean"
output_data_path = file_location + r"/inputdata_mean.pkl" 
# output_data_path = file_location + r"/outputdata.pkl" 
with open(input_data_path, "rb") as input_pickle_file:
	input_data = cPickle.load(input_pickle_file)
with open(output_data_path, "rb") as output_pickle_file:
	output_data = cPickle.load(output_pickle_file)
print("******")
print("initial shape: \n  input_data.shape={0}, output_data.shape={1}".\
	format(input_data.shape, output_data.shape))
print("******")
#---------------------------------------------------------------------------------


#=================================================================================
# Reshaping data
# ==============
input_data = np.array(input_data, dtype=np.float32).reshape(-1, 5)[:, 2:]
# input_data = np.array(input_data, dtype=np.float32).reshape(-1, 5)
output_data = np.array(output_data, dtype=np.float32).reshape(-1, 51*51*5)
#---------------------------------------------------------------------------------


#=================================================================================
# Normalized data
# ===============
scaler_x = MaxAbsScaler().fit(input_data)   # MinMaxScaler
input_data = scaler_x.transform(input_data).reshape(-1, 82 * 3)
# input_data = scaler_x.transform(input_data).reshape(-1, 82 * 5)
#---------------------------------------------------------------------------------
scaler_y = MaxAbsScaler().fit(output_data)  # MaxAbsScaler
output_data = scaler_y.transform(output_data).reshape(-1, 51*51*5)
#---------------------------------------------------------------------------------
print("******")
print("reshape: \n  input_data.shape={0}, output_data.shape={1}".\
	format(input_data.shape, output_data.shape))
print("******")
#---------------------------------------------------------------------------------


#=================================================================================
# Filtering data
# ==============
# for i in range(len(input_data)):
# 	sum = 0
# 	for j in range(82*2):
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
num1, num2, num3 = int(100000 * 0.6), int(100000 * 0.8), int(100000 * 1.0) 
x_train, x_validation, x_test = input_data[:num1], input_data[num1:num2], input_data[num2:num3]
y_train, y_validation, y_test = output_data[:num1], output_data[num1:num2], output_data[num2:num3]
print("******")
print("shape: \n  x_train={0}, y_train{1}, \n  x_validation={2}, y_validation={3}, \n  x_test={4}, y_test={5}"\
    .format(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape, x_test.shape, y_test.shape))
print("******")
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
validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
validation_dataset = validation_dataset.map(preprocess).batch(batch_size)
#---------------------------------------------------------------------------------
test_dataset =  tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(preprocess).batch(batch_size)
#---------------------------------------------------------------------------------
train_sample = next(iter(train_dataset))
validation_sample = next(iter(validation_dataset))
test_sample = next(iter(test_dataset))
print("******")
print("batch: \n  {0}, {1}".format(train_sample[0].shape, train_sample[1].shape))
print("******")
#---------------------------------------------------------------------------------


#=================================================================================
# Building model
#===============
# model = Sequential([
# 	# layers.Dropout(0.1), 
# 	# layers.BatchNormalization(),
# 	layers.Dense(units=1024, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()),
#     layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()),
#     layers.Dense(units=256, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()),
# 	layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()),
# 	layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), 	bias_initializer=tf.zeros_initializer()),
# 	layers.Dense(units=32, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), 	bias_initializer=tf.zeros_initializer()),	
# 	layers.Dense(units=output_node, activation=tf.nn.tanh, kernel_initializer=initializers.glorot_normal(), bias_initializer=tf.zeros_initializer())
# ])

model = Sequential()

model.add(Dense(1024, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()))

model.add(Dense(512, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()))
# model.add(BatchNormalization(axis=1))
# model.add(LeakyReLU())

model.add(Dense(256, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()))
# model.add(BatchNormalization(axis=1))
# model.add(LeakyReLU())

model.add(Dense(128, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()))
# model.add(BatchNormalization(axis=1))
# model.add(LeakyReLU())

model.add(Dense(64, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer()))
# model.add(BatchNormalization(axis=1))
# model.add(LeakyReLU())

model.add(Dense(output_node, kernel_initializer=initializers.glorot_normal(), bias_initializer=tf.zeros_initializer()))
model.add(Activation("tanh"))

#---------------------------------------------------------------------------------
model.build(input_shape=[None, x_train.shape[1]])
#---------------------------------------------------------------------------------
print("******************************************************************")
print("model.summary(): \n{} ".format(model.summary()))
print("  layer nums:", len(model.layers))
print("******************************************************************")
#---------------------------------------------------------------------------------


#=================================================================================
# Checkpoint
# ==========
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", \
	verbose=2, save_best_only=True, mode="auto", save_freq=1)
reducelr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, \
	patience=5, verbose=2, mode="auto", cooldown=0, min_lr=0)
callback_lists = [checkpoint, reducelr]
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
# Train on training set
# =====================
print("******************************************************************")
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, \
	verbose=2, validation_data=(x_validation, y_validation)) #, callbacks=callback_lists)
# history = model.fit(train_dataset, epochs=epochs, \
# 	validation_data=validation_dataset, verbose=2)
print("******************************************************************")
#---------------------------------------------------------------------------------


#=================================================================================
# history of the loss values and metric values during training
# ======================
file = open("./history.pkl", "wb")
history_dict = history.history
print("******")
print("history_dict.keys(): {}".format(history_dict.keys()))
print("history dict: \n{}".format(history_dict))
print("******")
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
# Prediction on new data
# ======================
# x_new_location = r"/mnt/c/chengshu/ShiYaolin/Program"
# x_new = pd.read_csv(x_new_location + r"/Code/x_new.csv", \
# 					delimiter=",", \
# 					usecols=[0, 1, 2, 3, 4], \
# 					header=None)
# x_new = np.array(x_new, dtype="float32")[:, 2:].reshape(1, 82, 3)
# x_new[:, 2] = -x_new[:, 2]
# # x_new[:, 4] = -x_new[:, 4]
# x_new = x_new.reshape(-1, 3)
# # x_new = x_new.reshape(-1, 5)
# x_new_scaler = scaler_x.transform(x_new)
# #---------------------------------------------------------------------------------
# x_new = x_new.reshape(-1, 82*3)
# # x_new = x_new.reshape(-1, 82*5)
# #---------------------------------------------------------------------------------
# y_new = model.predict(x_new)
# print("  y_new={}".format(y_new))
# y_new[:, 0] = 0.02 + (1.5 - 0.02)/2 * (y_new[:, 0]+1)
# y_new[:, 1] = 400 + (400 - (-400))/2 * (y_new[:, 1]+1)
# y_new[:, 2] = 400 + (400 - (-400))/2 * (y_new[:, 2]+1)
# y_new[:, 3] = 0 + (20 - 0)/2 * (y_new[:, 3]+1)
# y_new[:, 4] = 6 + (60 - 6)/2 * (y_new[:, 4]+1)
# y_new[:, 5] = 4 + (20 - 4)/2 * (y_new[:, 5]+1)
# y_new[:, 6] = 0 + (360 - 0)/2 * (y_new[:, 6]+1)
# y_new[:, 7] = 0 + (90 - 0)/2 * (y_new[:, 7]+1)
# y_new[:, 8] = 0 + (360 - 0)/2 * (y_new[:, 8]+1)

# # y_new = scaler_y.inverse_transform(y_new)
# print("******")
# print("Prediction on new data: ")
# print("  y_new={}".format(y_new))
# print("******")
# #---------------------------------------------------------------------------------


#=================================================================================
# Test data
# ======================
y_test_pre = model.predict(x_test)
targets, predictions = scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(y_test_pre)
#---------------------------------------------------------------------------------
diff_slip = (targets[:, 0] - predictions[:, 0])
print("******")
print("  diff_slip={}".format(diff_slip))
#---------------------------------------------------------------------------------

from matplotlib.ticker import PercentFormatter
from tensorflow.keras.utils import plot_model

#=================================================================================
# Picture
# =======
fig = plt.figure(figsize=(20.0, 10.0))
xticks = range(1, epochs+1, 1)
# yticks = np.arange(0, 0.33, 0.03)
#---------------------------------------------------------------------------------
# subpicture2
ax2 = axislines.Subplot(fig, 211)
fig.add_subplot(ax2)
ax2.axis[:].major_ticks.set_tick_out(True)
ax2.set_xticks(xticks)
# ax2.set_yticks(yticks)
ax2.axis["bottom"].label.set_text("Epochs")
ax2.axis["left"].label.set_text("Erroe") 
ax2.set_title("RMSE")
rmse = history_dict["rmse"]
val_rmse = history_dict["val_rmse"]
ax2.plot(xticks, rmse, "b", label="Training")
ax2.plot(xticks, val_rmse, "r", label="Validation")
ax2.legend(loc="upper right", shadow=True)
#---------------------------------------------------------------------------------
# subpicture3
ax3 = axislines.Subplot(fig, 212)
fig.add_subplot(ax3)
ax3.axis[:].major_ticks.set_tick_out(True)
ax3.set_xticks(xticks)
# ax3.set_yticks(yticks)
ax3.axis["bottom"].label.set_text("Epochs")
ax3.axis["left"].label.set_text("error") 
ax3.set_title("MAE")
mae = history_dict["mae"]
val_mae = history_dict["val_mae"]
ax3.plot(xticks, mae, "b", label="Training")
ax3.plot(xticks, val_mae, "r", label="Validation")
ax3.legend(loc="upper right", shadow=True)
#---------------------------------------------------------------------------------
plt.savefig("./pictures/self_BP.png")
fig.show()
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
print("Code has been executed!!! It takes {:.2f} minutes.".format((time.time() - t)/60))
#---------------------------------------------------------------------------------