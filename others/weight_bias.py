import tensorflow as tf

momentum = 0.98
keep_prob = 0.5

def get_weight_variable(shape, name=None, type='xavier_uniform', regularize=True, **kwargs):   
    initialise_from_constant = False
    if type == 'xavier_uniform':
        initial = tf.contrib.layers.xavier_initializer(uniform=True, seed=0, dtype=tf.float32)
    elif type == 'xavier_normal':
        initial = tf.contrib.layers.xavier_initializer(uniform=False, seed=0, dtype=tf.float32)
    elif type == 'he_normal':
        initial = tf.contrib.layers.variance_scaling_initializer(uniform=False, seed=0, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'he_uniform':
        initial = tf.contrib.layers.variance_scaling_initializer(uniform=True, seed=0, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'caffe_uniform':
        initial = tf.contrib.layers.variance_scaling_initializer(uniform=True, seed=0, factor=1.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'simple':
        stddev = kwargs.get('stddev', 0.02)
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'bilinear':
        weights = tf.contrib.layers._bilinear_upsample_weights(shape)
        initial = tf.constant(weights, shape=shape, dtype=tf.float32)
        initialise_from_constant = True
    else:
        raise ValueError('Unknown initialisation requested: %s' % type)
    if name is None:  # This keeps to option open to use unnamed Variables
        weight = tf.Variable(shape, initial)
    else:
        if initialise_from_constant:
            weight = tf.get_variable(name, initializer=initial)
        else:
            weight = tf.get_variable(name, shape, initializer=initial)
    if regularize:
        tf.add_to_collection('weight_variables', weight)
    return weight 


def get_bias_variable(shape, name=None, type=None, regularize=True, **kwargs):
    if type == '0':
        initial = tf.constant_initializer(0)
    if name is None:  # This keeps to option open to use unnamed Variables
        bias = tf.Variable(shape, initial)
    else:
        bias = tf.get_variable(shape, name, initializer=initial)
    if regularize:
        tf.add_to_collection('bias_variables', bias)
    return bias


