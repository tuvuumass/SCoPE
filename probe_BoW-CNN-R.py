import os
import cPickle
import fire
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from model import *
from utils import *
from sklearn.metrics import accuracy_score


GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

logging.set_verbosity(logging.INFO)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


class Options(object):
    def __init__(self):
        self.max_seq_len = 252
        self.embed = None
        self.embed_dim = 300
        self.mlp_hidden_dim = 300

        self.optimizer = 'Adam'
        self.learning_rate = 2e-4
        self.clip_gradients = None
        self.batch_size = 32
        self.num_epochs = 100
        self.dropout_keep_prob = 0.5

        self.log_path = None
        self.save_path = None
        self.model_archive_path = None
        self.print_freq = None
        self.valid_freq = None

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


def classifier(x, y, dropout_keep_prob, opts):
    logits = MLP(x, opts, keep_prob=dropout_keep_prob, prefix='classify')
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

    prob = tf.nn.sigmoid(logits)
    correct_prediction = tf.equal(tf.round(prob), tf.round(y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', loss)
    summaries = [
        "loss"
    ]
    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        optimizer=opts.optimizer,
        learning_rate=opts.learning_rate,
        summaries=summaries
    )
    return loss, train_op, prob, accuracy


def train_model(opts, x_train, y_train, x_val, y_val, x_test, y_test):

    with tf.device('/gpu:{}'.format(str(GPU_ID))):
        input_x = tf.placeholder(tf.float32, [opts.batch_size, 2 * opts.embed_dim], name='input_x')
        input_y = tf.placeholder(tf.float32, shape=[opts.batch_size, 1], name='input_y')
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        loss, train_op, prob, acc = classifier(input_x, input_y, dropout_keep_prob, opts)
        merged = tf.summary.merge_all()

    train_step = 0
    max_val_acc = 0.0
    max_test_acc = 0.0
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(opts.log_path, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(opts.log_path, 'test'), sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch in range(opts.num_epochs):
            tf.logging.info('Epoch {}\n'.format(epoch))
            train_batches = get_minibatches(len(x_train), opts.batch_size, shuffle=True)
            for _, train_batch in train_batches:
                train_step += 1
                x_train_batch = [x_train[i] for i in train_batch]
                y_train_batch = [y_train[i] for i in train_batch]
                y_train_batch = np.array(y_train_batch).reshape((-1,1))

                _, train_loss, _ = sess.run([train_op, loss, acc],
                                                         feed_dict={input_x: x_train_batch,
                                                                    input_y: y_train_batch,
                                                                    dropout_keep_prob: opts.dropout_keep_prob})
                summary = sess.run(merged, feed_dict={input_x: x_train_batch,
                                                      input_y: y_train_batch,
                                                      dropout_keep_prob: opts.dropout_keep_prob})
                train_writer.add_summary(summary, train_step)

                if train_step % opts.print_freq == 0:
                    tf.logging.info('Train: step {}, loss {}'.format(train_step, train_loss))
                if train_step % opts.valid_freq == 0:
                    val_batches = get_minibatches(len(x_val), opts.batch_size, shuffle=False)
                    val_probs = []
                    for _, val_batch in val_batches:
                        x_val_batch = [x_val[i] for i in val_batch]
                        y_val_batch = [y_val[i] for i in val_batch]
                        y_val_batch = np.array(y_val_batch).reshape((-1, 1))

                        val_probs_batch = sess.run(prob, feed_dict={input_x: x_val_batch,
                                                                    dropout_keep_prob: 1.0})
                        for p in val_probs_batch:
                            val_probs.append(p)

                    val_true = []
                    val_pred = []
                    for i in range(len(val_probs)):
                        if val_probs[i] > 0.5:
                            val_pred.append(1)
                        else:
                            val_pred.append(0)
                        val_true.append(y_val[i])

                    val_acc = accuracy_score(val_true, val_pred)
                    tf.logging.info('Val: accuracy {}'.format(val_acc))
                    summary = sess.run(merged, feed_dict={input_x: x_val_batch,
                                                          input_y: y_val_batch,
                                                          dropout_keep_prob:1.0})
                    test_writer.add_summary(summary, train_step)

                    if val_acc >= max_val_acc:
                        max_val_acc = val_acc
                        test_batches = get_minibatches(len(x_test), opts.batch_size, shuffle=False)
                        test_probs = []
                        for _, test_batch in test_batches:
                            x_test_batch = [x_test[i] for i in test_batch]

                            test_probs_batch = sess.run(prob, feed_dict={input_x: x_test_batch,
                                                                         dropout_keep_prob: 1.0})
                            for p in test_probs_batch:
                                test_probs.append(p)

                        test_true = []
                        test_pred = []
                        for i in range(len(test_probs)):
                            if test_probs[i] > 0.5:
                                test_pred.append(1)
                            else:
                                test_pred.append(0)
                            test_true.append(y_test[i])

                        test_acc = accuracy_score(test_true, test_pred)
                        tf.logging.info('Test: accuracy {}'.format(test_acc))
                        max_test_acc = test_acc
            tf.logging.info('Epoch {} : max val accuracy {}'.format(epoch, max_val_acc))
            tf.logging.info('Epoch {} : max test accuracy {}'.format(epoch, max_test_acc))
            saver.save(sess, os.path.join(opts.save_path, 'model'), global_step=epoch)


def load_model_archive(opts):
    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            checkpoint_file = tf.train.latest_checkpoint(opts.model_archive_path)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            # restore
            opts.embed = sess.run(graph.get_tensor_by_name('embed/W:0'))


def main(data_path, model_archive_path, log_path, save_path,
         embed_dim, learning_rate, batch_size, num_epochs, dropout_keep_prob,
         print_freq=100, valid_freq=1000):

    # loading data
    data = cPickle.load(open(data_path, 'rb'))
    x_train, x_val, x_test = data[3], data[4], data[5]
    y_train, y_val, y_test = data[6], data[7], data[8]
    index2token = data[10]

    # randomly shuffle data
    x_train, y_train = shuffle_data(x_train, y_train)
    x_val, y_val = shuffle_data(x_val, y_val)
    x_test, y_test = shuffle_data(x_test, y_test)

    opts = Options()
    opts.vocab_size = len(index2token)
    opts.model_archive_path = model_archive_path
    opts.log_path = log_path
    opts.save_path = save_path
    opts.embed_dim = embed_dim
    opts.learning_rate = learning_rate
    opts.batch_size = batch_size
    opts.num_epochs = num_epochs
    opts.dropout_keep_prob = dropout_keep_prob
    opts.print_freq = print_freq
    opts.valid_freq = valid_freq

    # load model archive
    load_model_archive(opts)
    opts.embed = l2_norm(opts.embed)
    x_train = BOW_encoder(x_train, opts.embed)
    x_val = BOW_encoder(x_val, opts.embed)
    x_test = BOW_encoder(x_test, opts.embed)
    train_model(opts, x_train, y_train, x_val, y_val, x_test, y_test)

    
if __name__ == '__main__': fire.Fire(main)

   
