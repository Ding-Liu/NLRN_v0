import tensorflow as tf


def build_model(model_input, state_num, is_training=True):
    x = tf.layers.batch_normalization(model_input, training=is_training)
    x = tf.layers.conv2d(x, 128, 3, padding='same', activation=None, name='conv1')
    y = x
    with tf.variable_scope("rnn"):
        for i in range(state_num):
            if i == 0:
                x = residual_block(x, y, 128, is_training, name='RB1', reuse=False)
            else:
                x = residual_block(x, y, 128, is_training, name='RB1', reuse=True)

    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 1, 3, padding='same', activation=None, name='conv_end')

    return x


def residual_block(x, y, filter_num, is_training, name, reuse):
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    x = non_local_block(x, 64, 128, name='non_local', reuse=reuse)

    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, filter_num, 3, padding='same', activation=None, name=name + '_a', reuse=reuse)

    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filter_num, 3, padding='same', activation=None, name=name + '_b', reuse=reuse)

    x = tf.add(x, y)
    return x


def non_local_block(x, filter_num, output_filter_num, name, reuse=False):
    x_theta = tf.layers.conv2d(x, filter_num, 1, padding='same', activation=None, name=name + '_theta', reuse=reuse)
    x_phi = tf.layers.conv2d(x, filter_num, 1, padding='same', activation=None, name=name + '_phi', reuse=reuse)
    x_g = tf.layers.conv2d(x, output_filter_num, 1, padding='same', activation=None, name=name + '_g', reuse=reuse, kernel_initializer=tf.zeros_initializer())

    x_theta_reshaped = tf.reshape(x_theta, [tf.shape(x_theta)[0], tf.shape(x_theta)[1] * tf.shape(x_theta)[2],
                                            tf.shape(x_theta)[3]])
    x_phi_reshaped = tf.reshape(x_phi,
                                [tf.shape(x_phi)[0], tf.shape(x_phi)[1] * tf.shape(x_phi)[2], tf.shape(x_phi)[3]])
    x_phi_permuted = tf.transpose(x_phi_reshaped, perm=[0, 2, 1])
    x_mul1 = tf.matmul(x_theta_reshaped, x_phi_permuted)
    x_mul1_softmax = tf.nn.softmax(x_mul1, axis=-1)  # normalization for embedded Gaussian

    x_g_reshaped = tf.reshape(x_g, [tf.shape(x_g)[0], tf.shape(x_g)[1] * tf.shape(x_g)[2], tf.shape(x_g)[3]])
    x_mul2 = tf.matmul(x_mul1_softmax, x_g_reshaped)
    x_mul2_reshaped = tf.reshape(x_mul2, [tf.shape(x_mul2)[0], tf.shape(x_phi)[1], tf.shape(x_phi)[2], output_filter_num])

    return tf.add(x, x_mul2_reshaped)
