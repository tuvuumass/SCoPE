import numpy as np
from sklearn import preprocessing


def shuffle_data(x, y=None):
    np.random.seed(42)
    x = np.array(x)
    shuffle_indices = np.random.permutation(np.arange(len(x)))
    x_shuffled = x[shuffle_indices]
    if y is not None:
        y = np.array(y)
        y_shuffled = y[shuffle_indices]
        return x_shuffled, y_shuffled
    else:
        return x_shuffled


def get_data(x, y, portion):
    x = np.array(x)
    y = np.array(y)

    n_samples = int(float(portion * len(x)) / 100.0)
    sample_indices = np.random.choice(len(x), n_samples, replace=False)
    return x[sample_indices], y[sample_indices]


def l2_norm(x):
    return preprocessing.normalize(x, norm='l2')


def get_minibatches(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = np.arange(n, dtype='int32')

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # if (minibatch_start != n):
    #    # Make a minibatch out of what is left
    #    minibatches.append(idx_list[minibatch_start:])
    return zip(range(len(minibatches)), minibatches)


def prepare_data_for_CNN(seqs, opts):
    max_seq_len = opts.max_seq_len
    filter_shape = opts.filter_shape
    seq_lens = [len(s) for s in seqs]

    if max_seq_len != None:
        seqs, seq_lens = zip(*([(s, l) if l <= max_seq_len else (s[:max_seq_len], max_seq_len)
                                for s, l in zip(seqs, seq_lens)]))
        if len(seq_lens) < 1:
            return None, None

    seqs_padded = []
    for s in seqs:
        s_padded = [0] * (filter_shape-1)
        s_padded += [idx for idx in s]
        s_padded += [0] * (max_seq_len + 2 * (filter_shape-1) - len(s_padded))
        seqs_padded.append(s_padded)
    return np.array(seqs_padded, dtype='int32')


