import os
import time
import tensorflow as tf
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from matplotlib.ticker import PercentFormatter

print("\n  Begin to execute main_mean.py... \n")

#=================================================================================
# Environment
# ===========
t = time.time()
seed = 22
tf.random.set_seed(seed)   # 对全局随机数生成种子的设置
np.random.seed(seed)  # 使用相同的参数，每次生成的随机数都相同
#---------------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # 忽略tensorflow警告信息
assert tf.__version__.startswith("2.") # 判断tf版本是否以"2.0"开头
tf.keras.backend.clear_session()  # 销毁当前的TF图并创建一个新图。有避免旧模型/图层混乱
#---------------------------------------------------------------------------------


#=================================================================================
# Parameters
# ==========
input_node = 51*51 # input_node = input("input node : ")
output_node = 9
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
# file_location = r"C:\chengshu\ShiYaolin\Program\data_create\10w_mean"
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
output_data_path = file_location + r"/outputdata_mean.pkl"
with open(output_data_path, "rb") as output_pickle_file:
	output_data = cPickle.load(output_pickle_file)
print("******")
print("initial shape: input_data1={0}, input_data2={1}, input_data3={2}, input_data4={3}, input_data5={4}".\
	format(input_data1.shape, input_data2.shape, input_data3.shape, input_data4.shape, input_data5.shape))
print("initial shape: output_data={0}".	format(output_data.shape))
print("******")
#---------------------------------------------------------------------------------
input_data = pd.concat([input_data1, input_data2, input_data3, input_data4, input_data5])


#=================================================================================
# Reshaping data
# ==============
input_data = np.array(input_data, dtype=np.float32)[:, 2:5].reshape(-1, 51*51*3)
input_data = input_data[:10000, :]
output_data = np.array(output_data, dtype=np.float32).reshape(-1, 9)
#---------------------------------------------------------------------------------
output_data = output_data[:10000, :]

#=================================================================================
# Normalized data
# ===============
scaler_x = MinMaxScaler().fit(input_data)   # MaxAbsScaler
input_data = scaler_x.transform(input_data).reshape(-1, 51*51*3)
#---------------------------------------------------------------------------------
scaler_y = MinMaxScaler().fit(output_data)  # MinMaxScaler
output_data = scaler_y.transform(output_data).reshape(-1, 9)
#---------------------------------------------------------------------------------

print("******")
print("reshape shape: input_data={0}, output_data={1}".format(input_data.shape, output_data.shape))
print("******")
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
num1, num2, num3 = int(10000 * 0.6), int(10000 * 0.8), int(10000 * 1.0) 
# num1, num2, num3 = int(100000 * 0.6), int(100000 * 0.8), int(100000 * 1.0) 
x_train, x_validation, x_test = input_data[:num1], input_data[num1:num2], input_data[num2:num3]
y_train, y_validation, y_test = output_data[:num1], output_data[num1:num2], output_data[num2:num3]
print("******")
print("shape: \n  x_train={0}, y_train{1}, \n  x_validation={2}, y_validation={3}, \n  x_test={4}, y_test={5}"\
    .format(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape, x_test.shape, y_test.shape))
print("******")
#---------------------------------------------------------------------------------

x_train = x_train.reshape((-1, 51, 51, 3))
x_validation = x_validation.reshape((-1, 51, 51, 3))
x_test = x_test.reshape((-1, 51, 51, 3))
y_train = y_train.reshape((-1, 9))
y_validation = y_validation.reshape((-1, 9))
y_test = y_test.reshape((-1, 9))
print("******")
print("x_train.shape={0}, y_train.shape={1}".format(x_train.shape, y_train.shape))
print("x_test.shape={0}, y_test.shape={1}".format(x_test.shape, y_test.shape))
print("x_train[0]={0}".format(x_train[0]))
print("y_train[0]={0}".format(y_train[0]))
print("******")


#=================================================================================
# Dataset
# =======
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)  
    y = tf.cast(y, dtype=tf.float32) 
    return x, y
#---------------------------------------------------------------------------------
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.map(preprocess).batch(batch_size)
# #---------------------------------------------------------------------------------
# validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
# validation_dataset = validation_dataset.map(preprocess).batch(batch_size)
# #---------------------------------------------------------------------------------
# test_dataset =  tf.data.Dataset.from_tensor_slices((x_test, y_test))
# test_dataset = test_dataset.map(preprocess).batch(batch_size)
# #---------------------------------------------------------------------------------
# train_sample = next(iter(train_dataset))
# validation_sample = next(iter(validation_dataset))
# test_sample = next(iter(test_dataset))
# print("******")
# print("batch: \n  {0}, {1}".format(train_sample[0].shape, train_sample[1].shape))
# print("******")
#---------------------------------------------------------------------------------

x_shape  = x_train.shape
img_shape=(x_shape[1], x_shape[2], x_shape[3])

freeze_layers = -5

def model(mode='fc'):
    if mode == 'fc':
        # FC层设定为含有128个参数的隐藏层
        base_model = VGG16(input_shape=img_shape, include_top=False, pooling='none', weights='imagenet')
        x = base_model.output
        x = Flatten()(x)
        x = Dense(128, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer())(x)
        prediction = Dense(output_node, activation='sigmoid')(x)
    elif mode == 'avg':
        # GAP层通过指定pooling='avg'来设定
        base_model = VGG16(input_shape=img_shape, include_top=False, pooling='avg', weights='imagenet')
        x = base_model.output
        x = Dense(128, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer())(x)
        prediction = Dense(output_node, activation='sigmoid')(x)
    else:
        # GMP层通过指定pooling='max'来设定
        base_model = VGG16(input_shape=img_shape, include_top=False, pooling='max', weights='imagenet')
        x = base_model.output
        x = Dense(128, activation=tf.nn.relu, kernel_initializer=initializers.he_normal(), bias_initializer=tf.zeros_initializer())(x)
        prediction = Dense(output_node, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=prediction)
    
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    # block5_pool_features = model.predict(x)

    model.summary()
    print("layer nums:", len(model.layers))

    # 除了FC层，靠近FC层的一部分卷积层可参与参数训练，
    # 一般来说，模型结构已经标明一个卷积块包含的层数，freeze_layers为17，表示最后一个卷积块和FC层要参与参数训练
    for layer in model.layers[:freeze_layers]:
        layer.trainable = False
    for layer in model.layers[freeze_layers:]:
        layer.trainable = True
    for layer in model.layers:
        print("layer.trainable:", layer.trainable)

    model.summary()

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_validation, y_validation), verbose=2)
    
    test_mse, test_mae = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
    print("  test_mse={0:.4f}%, test_mae={1:.4f}%".format(test_mse*100, test_mae*100))

    y_test_pre = model.predict(x_test)

    global diff_slip
    diff_slip = (y_test_pre[:, 0] - y_test[:, 0])
    print("diff_slip={}".format(diff_slip))

    return history



print("GAP层通过指定pooling='avg'来设定")
avg_history = model('avg')
avg_history_dict = avg_history.history

print("GMP层通过指定pooling='max'来设定")
max_history = model('max')
max_history_dict = max_history.history

print("FC层设定为含有128个参数的隐藏层")
fc_history = model('fc')
fc_history_dict = fc_history.history

#=================================================================================
# Picture
# =======
fig = plt.figure(figsize=(20.0, 6.0))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
xticks = range(1, epochs+1, 1)
yticks = np.arange(0, 0.33, 0.03)
#---------------------------------------------------------------------------------
# sub_picture1
ax1 = axislines.Subplot(fig, 131)
fig.add_subplot(ax1)
ax1.axis[:].major_ticks.set_tick_out(True)
n, bins, patches = ax1.hist(diff_slip, bins="auto", density=True, cumulative=False, facecolor="yellowgreen", alpha=0.5, label="histogram")
y = norm.pdf(bins, diff_slip.mean(), diff_slip.std()) #生成正态分布函数
ax1.plot(bins, y, "b--", label="Gauss Distribution")
ax1.set_title("Histogram of Slip Difference distribution: $\mu = {0:.2f}$, $\sigma={1:.2f}$".format(diff_slip.mean(), diff_slip.std()))
ax1.set_xlabel("Value of Slip Difference(m)")
ax1.set_ylabel("Percent")
ax1.legend(loc="upper left", shadow=True)
ax1.yaxis.set_major_formatter(PercentFormatter(xmax=len(diff_slip)))

#---------------------------------------------------------------------------------
# sub_picture2
ax2 = axislines.Subplot(fig, 132)
fig.add_subplot(ax2)
ax2.axis[:].major_ticks.set_tick_out(True)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.axis["bottom"].label.set_text("Epochs")
ax2.axis["left"].label.set_text("Error") 
ax2.set_title("mse (Mean square error)")

avg_loss_values = avg_history_dict["loss"]
avg_val_loss_values = avg_history_dict["val_loss"]
ax2.plot(xticks, avg_loss_values, "--b*", label="Training")
ax2.plot(xticks, avg_val_loss_values, "--r*", label="Validation")

max_loss_values = max_history_dict["loss"]
max_val_loss_values = max_history_dict["val_loss"]
ax2.plot(xticks, max_loss_values, "-.b*", label="Training")
ax2.plot(xticks, max_val_loss_values, "-.r*", label="Validation")

fc_loss_values = fc_history_dict["loss"]
fc_val_loss_values = fc_history_dict["val_loss"]
ax2.plot(xticks, fc_loss_values, "-b*", label="Training")
ax2.plot(xticks, fc_val_loss_values, "-r*", label="Validation")

ax2.legend(['FC', 'AVG', 'MAX'], loc='upper right')
ax2.grid(True)
#---------------------------------------------------------------------------------
# sub_picture3
ax3 = axislines.Subplot(fig, 133)
fig.add_subplot(ax3)
ax3.axis[:].major_ticks.set_tick_out(True)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.axis["bottom"].label.set_text("Epochs")
ax3.axis["left"].label.set_text("Error") 
ax3.set_title("mae (Mean absolute error)")

avg_acc = avg_history_dict["mae"]
avg_val_acc = avg_history_dict["val_mae"]
ax3.plot(xticks, avg_acc, "--b*", label="Training")
ax3.plot(xticks, avg_val_acc, "--r*", label="Validation")

max_acc = max_history_dict["mae"]
max_val_acc = max_history_dict["val_mae"]
ax3.plot(xticks, max_acc, "-.b*", label="Training")
ax3.plot(xticks, max_val_acc, "-.r*", label="Validation")

fc_acc = fc_history_dict["mae"]
fc_val_acc = fc_history_dict["val_mae"]
ax3.plot(xticks, fc_acc, "-b*", label="Training")
ax3.plot(xticks, fc_val_acc, "-r*", label="Validation")

ax3.legend(['FC', 'AVG', 'MAX'], loc='upper right')
ax3.grid(True)

#---------------------------------------------------------------------------------
plt.savefig("VGG16.png")
fig.show()
#---------------------------------------------------------------------------------




#---------------------------------------------------------------------------------
print("\n  It takes {:.2f} minutes. ".format((time.time() - t)/60))
print("\n  End ... \n")
#---------------------------------------------------------------------------------