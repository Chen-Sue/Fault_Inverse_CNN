import numpy as np
import tensorflow as tf
import time

from loss_function import Loss

learning_rate = 0.0001

class Optimizer:

    def __init__(self, learning_rate, loss):
        self.learning_rate = learning_rate
        self.loss = loss

    # SGD
    def sgd(self):
        with tf.variable_scope('SGD', reuse=tf.AUTO_REUSE):
            sgd_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        return sgd_optimizer

    # Momentum    
    def momentum(self):
        with tf.variable_scope('Momentum', reuse=tf.AUTO_REUSE):
            momentum_optimizer = tf.train.MomentumOptimizer(self.learning_rate, 
                                                            momentum=0.9, 
                                                            name='Momentum').minimize(self.loss)
        return momentum_optimizer

     # Nesterov's Accelerated Gradient
    def nesterov(self):
        with tf.variable_scope('Nesterov', reuse=tf.AUTO_REUSE):
             Nesterov_optimizer = tf.train.MomentumOptimizer(self.learning_rate, 
                                                                momentum=0.9, 
                                                                use_nesterov=False, 
                                                                name='Nesterov').minimize(self.loss)
        return Nesterov_optimizer

    # AdaGrad
    def adaGrad(self):
        with tf.variable_scope('AdaGrad', reuse=tf.AUTO_REUSE):
            adaGrad_optimizer = tf.train.AdagradOptimizer(self.learning_rate, 
                                                            name='AdaGrad').minimize(self.loss)
        return adaGrad_optimizer

    # ProximalAdagrad
    def proximalAdagrad(self):
        with tf.variable_scope('ProximalAdagrad', reuse=tf.AUTO_REUSE):
            proximalAdagrad_optimizer = tf.train.ProximalAdagradOptimizer(self.learning_rate, 
                                                                            name='ProximalAdagrad').minimize(self.loss)
        return proximalAdagrad_optimizer

    # ProximalGradientDescent
    def proximalGradientDescent(self):
        with tf.variable_scope('ProximalGradientDescent', reuse=tf.AUTO_REUSE):
            proximalGradientDescent_optimizer = tf.train.ProximalGradientDescentOptimizer(self.learning_rate, 
                                                                                            name='ProximalGradientDescent'
                                                                                            ).minimize(self.loss)
        return proximalGradientDescent_optimizer

    # Adadelta
    def adadelta(self):
        with tf.variable_scope('Adadelta', reuse=tf.AUTO_REUSE):
            adadelta_optimizer = tf.train.AdadeltaOptimizer(self.learning_rate, 
                                                            name='Adadelta').minimize(self.loss)
        return adadelta_optimizer

    # RMSprop
    def RMSprop(self):
        with tf.variable_scope('RMSprop', reuse=tf.AUTO_REUSE):
            RMSprop_optimizer = tf.train.RMSPropOptimizer(self.learning_rate, 
                                                            name='RMSProp').minimize(self.loss)
        return RMSprop_optimizer

    # Adam
    def adam(self):
        with tf.variable_scope('Adam', reuse=tf.AUTO_REUSE):
            adam_optimizer = tf.train.AdamOptimizer(self.learning_rate, 
                                                    name='Adam').minimize(self.loss)
        return adam_optimizer

    # 模型效果指标评估
    def performance(self):

        tmp_list=[] 

        tmp_score = self.sgd()
        tmp_list.append(tmp_score)

        tmp_score = self.momentum()
        tmp_list.append(tmp_score)

        tmp_score = self.nesterov()
        tmp_list.append(tmp_score)

        tmp_score = self.adaGrad()
        tmp_list.append(tmp_score)

        tmp_score = self.proximalAdagrad()
        tmp_list.append(tmp_score)

        tmp_score = self.proximalGradientDescent()
        tmp_list.append(tmp_score)

        tmp_score = self.adadelta()
        tmp_list.append(tmp_score)

        tmp_score = self.RMSprop()
        tmp_list.append(tmp_score)

        tmp_score = self.adam()
        tmp_list.append(tmp_score)

        print('\n', 'sgd', 'momentum', 'nesterov', \
                'adaGrad', 'proximalAdagrad', 'proximalGradientDescent', \
                'adadelta', 'RMSprop', 'adam')
        print('tmp_list: {:.3f}'.format(tmp_list))

        return tmp_list


def main():
    t1 = time.time()
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 2.2, 3.3, 4.4]

    """loss = Loss(y_true, y_pred)
    losses = loss.loss_MSE()"""
    # loss.performance()

    losses = tf.losses.mean_squared_error(y_true, y_pred)

    """tf.Session().run(tf.initialize_all_variables())
    print('\n', tf.Session().run(losses))"""

    """optimizer = Optimizer(learning_rate, losses)
    optimizer.sgd()"""
    # optimizer.performance()

    _ = tf.train.AdamOptimizer(learning_rate, name='Adam').minimize(losses)

    print('\n optimizer.py has been executed!!! It takes {:.3f} minutes.'.format((time.time() - t1)/60))

if __name__ == '__main__':
    main()

