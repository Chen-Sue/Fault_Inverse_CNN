import time
import tensorflow as tf
import pickle

from merge_data import MergeData
from data import Data

model_names = ['loss_AE', 'loss_MAE', 'loss_relative_l1', 'loss_relative_l2', \
                'loss_MSE', 'loss_RMSE', 'loss_r2', 'loss_compute_weighted', \
                'loss_hinge', 'loss_huber','loss_log','loss_mean_pairwise_squared_error']

class Loss:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_names = model_names


    # Absolute Error: 绝对误差(越小, 拟合效果越好)
    def loss_AE(self): 
        with tf.variable_scope('absolute_difference', reuse=tf.AUTO_REUSE):
            ae_loss = tf.losses.absolute_difference(self.y_true, self.y_pred)   
        return ae_loss  

    # Mean Absolute Error: 平均绝对误差(值越小, 拟合效果越好)
    def loss_MAE(self):
        with tf.variable_scope('mean_absolute_error', reuse=tf.AUTO_REUSE):
            mae_loss = tf.reduce_mean(tf.losses.absolute_difference(self.y_true, self.y_pred))
        return mae_loss 
    
    # relative_l1: 平均相对误差(值越小, 拟合效果越好)
    """def loss_relative_l1(self):
        with tf.variable_scope('relative_difference_l1', reuse=tf.AUTO_REUSE):
            l1_loss = tf.reduce_mean(tf.div(tf.abs(tf.subtract(self.y_true, self.y_pred)), tf.abs(self.y_true)))
        return l1_loss"""

    # relative_l2: 平均相对误差(值越小, 拟合效果越好)
    def loss_relative_l2(self):
        with tf.variable_scope('relative_difference_l2', reuse=tf.AUTO_REUSE):
            l2_loss = tf.reduce_mean(tf.div(tf.nn.l2_normalize(tf.subtract(self.y_true, self.y_pred)), tf.nn.l2_normalize(self.y_true)))
        return l2_loss
 
    # Mean Square Error: 均方误差
    def loss_MSE(self):
        with tf.variable_scope('mean_square_error', reuse=tf.AUTO_REUSE):
            mae_loss = tf.losses.mean_squared_error(self.y_true, self.y_pred) 
        return mae_loss

    # Root Mean Square Error: 均方根误差
    def loss_RMSE(self):
        with tf.variable_scope('root_mean_absolute_error', reuse=tf.AUTO_REUSE):
            root_mean_absolute_error_loss = tf.sqrt(tf.losses.mean_squared_error(self.y_true, self.y_pred))
        return root_mean_absolute_error_loss

    # R2决定系数（拟合优度），值取值范围是[0,1]，越小则说明效果越差
    def loss_r2(self):
        with tf.variable_scope('r2', reuse=tf.AUTO_REUSE):
            r2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(self.y_true, self.y_pred))))
        return r2_loss

    def loss_compute_weighted(self):
        with tf.variable_scope('compute_weighted_loss', reuse=tf.AUTO_REUSE):
            compute_weighted_loss = tf.losses.compute_weighted_loss(self.y_true, self.y_pred)
        return compute_weighted_loss

    def loss_hinge(self):
        with tf.variable_scope('hinge', reuse=tf.AUTO_REUSE):
            hinge_loss = tf.losses.hinge_loss(self.y_true, self.y_pred)
        return hinge_loss

    def loss_huber(self):
        with tf.variable_scope('huber', reuse=tf.AUTO_REUSE):
           huber_loss = tf.losses.huber_loss(self.y_true, self.y_pred)
        return huber_loss

    def loss_log(self):
        with tf.variable_scope('log_loss', reuse=tf.AUTO_REUSE):
            log_loss_loss = tf.losses.log_loss(self.y_true, self.y_pred)
        return log_loss_loss

    def loss_mean_pairwise_squared_error(self):
        with tf.variable_scope('mean_pairwise_squared_error', reuse=tf.AUTO_REUSE):
            mean_pairwise_squared_error_loss = tf.losses.mean_pairwise_squared_error(self.y_true, self.y_pred)
        return mean_pairwise_squared_error_loss

           
    # 模型效果指标评估
    def performance(self):

        tmp_list=[] 

        """for i in range(len(model_names)):  
            tmp_score = self.model_names[i]()  
            tmp_list.append(tmp_score)  """

        tmp_score = self.loss_AE()
        tmp_list.append(tmp_score)
        
        tmp_score = self.loss_MAE()
        tmp_list.append(tmp_score)

        """tmp_score = self.loss_relative_l1()
        tmp_list.append(tmp_score)"""

        tmp_score = self.loss_relative_l2()
        tmp_list.append(tmp_score)

        tmp_score = self.loss_MSE()
        tmp_list.append(tmp_score)

        tmp_score = self.loss_RMSE()
        tmp_list.append(tmp_score)

        tmp_score = self.loss_r2()
        tmp_list.append(tmp_score)

        tmp_score = self.loss_compute_weighted()
        tmp_list.append(tmp_score)

        tmp_score = self.loss_hinge()
        tmp_list.append(tmp_score)

        tmp_score = self.loss_huber()
        tmp_list.append(tmp_score)

        tmp_score = self.loss_log()
        tmp_list.append(tmp_score)

        tmp_score = self.loss_mean_pairwise_squared_error()
        tmp_list.append(tmp_score)        

        """print(['loss_AE', 'loss_MAE', 'loss_relative_l1', 'loss_relative_l2', \
                'loss_MSE', 'loss_RMSE', 'loss_r2', 'loss_compute_weighted', \
                'loss_hinge', 'loss_huber','loss_log','loss_mean_pairwise_squared_error'])"""

        print('\n', 'loss_AE', 'loss_MAE', 'loss_relative_l2', \
                'loss_MSE', 'loss_RMSE', 'loss_r2', 'loss_compute_weighted', \
                'loss_hinge', 'loss_huber','loss_log','loss_mean_pairwise_squared_error')
        print('\n tmp_list:{:.3f}'.format(tmp_list))

        return tmp_list
        

    # Adjusted_R2
    # def Adjusted_R2(self):
    #     R_squared =  Loss.r2(self, self.y_true, self.y_pred)
    #     return 1-((1-R_squared(self.y_true, self.y_pred))*(len(self.y_true.eval())-1))/(len(self.y_true.eval())-1-1)

    #  Pseudo-Huber with delta = 0.25   L = delta^2 * (sqrt(1 + ((pred - actual)/delta)^2) - 1)
    # def loss_PH(self):
    #     with tf.name_scope('loss_PH1'):
    #         loss = tf.multiply(tf.square(self.delta), tf.sqrt(1. + tf.square((self.y_true-self.y_pred)/self.delta)) - 1.) 
    #     return loss

    # SMAPE 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error） 需要自己实现
    # def smape(self):
    #     return 2.0 * np.mean(np.abs(self.y_true, self.y_pred) / (np.abs(self.y_pred) + np.abs(self.y_true))) * 100

    # explained_variance_score：解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。


def main():
    
    t1 = time.time()
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 2.2, 3.3, 4.4]
    loss = Loss(y_true, y_pred)
    losses = loss.loss_MSE()
    tf.Session().run(tf.initialize_all_variables())

    print('\n loss: {:.3f}'.format(tf.Session().run(losses)))
    # loss.performance()
    print('\n loss_function.py has been executed!!! It takes {:.3f} minutes.'.format((time.time() - t1)/60))

if __name__ == '__main__':
    main()