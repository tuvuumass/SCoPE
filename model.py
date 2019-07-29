import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def token_embedder(x, opts, prefix ='', reuse=None):
    with tf.variable_scope(prefix+'embed', reuse=reuse):
        if opts.from_scratch:
            weights_initializer = tf.random_uniform_initializer(-0.001, 0.001)
        else:
            assert (np.shape(np.array(opts.embed)) == (opts.vocab_size, opts.embed_dim))
            weights_initializer = tf.constant_initializer(opts.embed)
        W = tf.get_variable('W', [opts.vocab_size, opts.embed_dim],
                            initializer=weights_initializer, trainable=opts.unfrozen)
    W_normalized = tf.nn.l2_normalize(W, 1)
    x_embed = tf.nn.embedding_lookup(W_normalized, x)
    return x_embed, W_normalized


def BOW(x, embed):
    x_bow = []
    for i in range(len(x)):
        x_bow.append(np.array([embed[t] for t in x[i] if t != 0]).mean(axis=0))
    return x_bow


def BOW_encoder(x, embed):
    p = x[:, 0]
    s = x[:, 1]
    p = BOW(p, embed)
    s = BOW(s, embed)
    return np.concatenate((p, s), axis=1)


def MLP(h, opts, keep_prob=1.0, prefix='', num_outputs=1, reuse=None):
    h = tf.squeeze(h)
    biases_initializer = tf.constant_initializer(0.001, dtype=tf.float32)
    h = layers.fully_connected(tf.nn.dropout(h, keep_prob=keep_prob),
                               num_outputs=opts.mlp_hidden_dim,
                               biases_initializer=biases_initializer,
                               activation_fn=tf.nn.relu,
                               scope=prefix + 'mlp1',
                               reuse=reuse)
    logits = layers.linear(tf.nn.dropout(h, keep_prob=keep_prob),
                           num_outputs=num_outputs,
                           biases_initializer=biases_initializer,
                           scope=prefix + 'mlp2', reuse=reuse)
    return logits


def regularization(x, opts, train, reuse= None, prefix=''):
    if 'x_' not in prefix:
        if opts.batch_norm:
            x = layers.batch_norm(x, decay=0.9, center=True, scale=True, is_training=train, scope=prefix + '_bn',
                                  reuse=reuse)
        x = tf.nn.relu(x)
    x = x if not opts.cnn_layer_dropout else layers.dropout(x, keep_prob=opts.dropout_keep_prob, scope=prefix + '_dropout')
    return x


def CNN(x, opts, prefix='', reuse=None, num_outputs=None, train=True):
    if hasattr(opts, 'multiplier'):
        multiplier = opts.multiplier
    else:
        multiplier = 2

    if opts.from_scratch:
        weights_initializer_W1 = tf.constant_initializer(0.001, dtype=tf.float32)
        biases_initializer_b1 = None if opts.batch_norm else tf.constant_initializer(0.001, dtype=tf.float32)
        weights_initializer_W2 = tf.constant_initializer(0.001, dtype=tf.float32)
        biases_initializer_b2 = None if opts.batch_norm else tf.constant_initializer(0.001, dtype=tf.float32)
        weights_initializer_W3 = tf.constant_initializer(0.001, dtype=tf.float32)
        biases_initializer_b3 = None if opts.batch_norm else tf.constant_initializer(0.001, dtype=tf.float32)
    else:
        weights_initializer_W1 = tf.constant_initializer(opts.W1)
        biases_initializer_b1 = tf.constant_initializer(opts.b1)
        weights_initializer_W2 = tf.constant_initializer(opts.W2)
        biases_initializer_b2 = tf.constant_initializer(opts.b2)
        weights_initializer_W3 = tf.constant_initializer(opts.W3)
        biases_initializer_b3 = tf.constant_initializer(opts.b3)

    x = regularization(x, opts, prefix=prefix + 'x_regularized', train=train, reuse=reuse)
    h = layers.conv2d(x,
                      num_outputs=opts.filter_size,
                      kernel_size=[opts.filter_shape, opts.embed_dim],
                      stride=[opts.stride[0], 1],
                      weights_initializer=weights_initializer_W1,
                      biases_initializer=biases_initializer_b1,
                      activation_fn=None,
                      padding='VALID',
                      scope=prefix + 'H1_3',
                      reuse=reuse,
                      trainable=opts.unfrozen)
    h = regularization(h, opts, prefix=prefix + 'H1_regularized', train=train, reuse=reuse)
    h = layers.conv2d(h,
                      num_outputs=opts.filter_size*multiplier,
                      kernel_size=[opts.filter_shape, 1],
                      stride=[opts.stride[1],1],
                      weights_initializer=weights_initializer_W2,
                      biases_initializer=biases_initializer_b2,
                      activation_fn=None,
                      padding='VALID',
                      scope=prefix + 'H2_3',
                      reuse=reuse,
                      trainable=opts.unfrozen)
    h = regularization(h, opts, prefix=prefix + 'H2_regularized', train=train, reuse=reuse)
    h = layers.conv2d(h,
                      num_outputs=opts.n_gan,
                      kernel_size=[opts.cnn_output_sizes[-1], 1],
                      weights_initializer=weights_initializer_W3,
                      biases_initializer=biases_initializer_b3,
                      activation_fn=tf.nn.tanh,
                      padding='VALID',
                      scope=prefix + 'H3_3',
                      reuse=reuse,
                      trainable=opts.unfrozen)
    return h


