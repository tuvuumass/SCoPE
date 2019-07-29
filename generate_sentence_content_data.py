# -*- coding: utf-8 -*-
import fire
import cPickle
import numpy as np
import nltk


def get_sentence_content_data(data, labels, token2index, index2token, setting='rand'):
    x_sc = []
    y_sc = []
    for i in range(len(data)):
        para = nltk.sent_tokenize(' '.join([index2token[idx] for idx in data[i]
                                            if idx != 0]).decode('utf-8'))
        if len(para) >= 2:
            pos_sent = para[np.random.randint(len(para))]
            pos_sent_split = pos_sent.strip().split(' ')
            if(len(pos_sent_split) >= 2):
                pos_idxs = [token2index[tok.encode('utf-8')] for tok in pos_sent_split]
                pos_idxs.append(0)
                positive_example = np.asarray(pos_idxs, dtype=np.int32)
                j = np.random.randint(len(data))
                para_tmp = nltk.sent_tokenize(' '.join([index2token[idx] for idx in data[j]
                                                        if idx != 0]).decode('utf-8'))
                if setting == 'in':
                    while (j == i) or (len(para_tmp) < 2) or (labels[j] != labels[i]):
                        j = np.random.randint(len(data))
                        para_tmp = nltk.sent_tokenize(' '.join([index2token[idx] for idx in data[j]
                                                                if idx != 0]).decode('utf-8'))
                    assert j != i
                    assert labels[j] == labels[i]
                elif setting == 'out':
                    while (j == i) or (len(para_tmp) < 2) or (labels[j] == labels[i]):
                        j = np.random.randint(len(data))
                        para_tmp = nltk.sent_tokenize(' '.join([index2token[idx] for idx in data[j]
                                                                if idx != 0]).decode('utf-8'))
                    assert j != i
                    assert labels[j] != labels[i]
                else:   # random
                    while (j == i) or len(para_tmp) < 2:
                        j = np.random.randint(len(data))
                        para_tmp = nltk.sent_tokenize(' '.join([index2token[idx] for idx in data[j]
                                                                if idx != 0]).decode('utf-8'))
                neg_sent = para_tmp[np.random.randint(len(para_tmp))]
                neg_sent_split = pos_sent.strip().split(' ')

                if (neg_sent in para) or (len(neg_sent_split) < 2):
                    continue
                else:
                    neg_idxs = [token2index[tok.encode('utf-8')] for tok in neg_sent_split]
                    neg_idxs.append(0)
                    negative_example = np.asarray(neg_idxs, dtype=np.int32)
                    x_sc.append([np.asarray(data[i], dtype=np.int32), positive_example])
                    y_sc.append(1)
                    x_sc.append([np.asarray(data[i], dtype=np.int32), negative_example])
                    y_sc.append(0)
    assert (len(x_sc) == len(y_sc))
    return x_sc, y_sc


def get_sentence_content_data_all(data, labels, token2index, index2token, setting='rand'):
    x_sc = []
    y_sc = []
    for i in range(len(data)):
        para = nltk.sent_tokenize(' '.join([index2token[idx] for idx in data[i]
                                            if idx != 0]).decode('utf-8'))
        if len(para) >= 2:
            for pos_sent in para:
                pos_sent_split = pos_sent.strip().split(' ')
                if(len(pos_sent_split) >= 2):
                    pos_idxs = [token2index[tok.encode('utf-8')] for tok in pos_sent_split]
                    pos_idxs.append(0)
                    positive_example = np.asarray(pos_idxs, dtype=np.int32)
                    j = np.random.randint(len(data))
                    para_tmp = nltk.sent_tokenize(' '.join([index2token[idx] for idx in data[j]
                                                            if idx != 0]).decode('utf-8'))

                    if setting == 'in':
                        while (j == i) or (len(para_tmp) < 2) or (labels[j] != labels[i]):
                            j = np.random.randint(len(data))
                            para_tmp = nltk.sent_tokenize(' '.join([index2token[idx] for idx in data[j]
                                                                    if idx != 0]).decode('utf-8'))
                        assert j != i
                        assert labels[j] == labels[i]
                    elif setting == 'out':
                        while (j == i) or (len(para_tmp) < 2) or (labels[j] == labels[i]):
                            j = np.random.randint(len(data))
                            para_tmp = nltk.sent_tokenize(' '.join([index2token[idx] for idx in data[j]
                                                                    if idx != 0]).decode('utf-8'))
                        assert j != i
                        assert labels[j] != labels[i]
                    else:
                        while (j == i) or len(para_tmp) < 2:
                            j = np.random.randint(len(data))
                            para_tmp = nltk.sent_tokenize(' '.join([index2token[idx] for idx in data[j]
                                                                    if idx != 0]).decode('utf-8'))
                    neg_sent = para_tmp[np.random.randint(len(para_tmp))]
                    neg_sent_split = pos_sent.strip().split(' ')

                    if (neg_sent in para) or (len(neg_sent_split) < 2):
                        continue
                    else:
                        neg_idxs = [token2index[tok.encode('utf-8')] for tok in neg_sent_split]
                        neg_idxs.append(0)
                        negative_example = np.asarray(neg_idxs, dtype=np.int32)
                        x_sc.append([np.asarray(data[i], dtype=np.int32), positive_example])
                        y_sc.append(1)
                        x_sc.append([np.asarray(data[i], dtype=np.int32), negative_example])
                        y_sc.append(0)
    assert (len(x_sc) == len(y_sc))
    return x_sc, y_sc


def main(dataname, data_path, use_all_sentences=True, setting='rand'):
    data = cPickle.load(open(data_path + '.p', 'rb'))

    x_train, x_val, x_test = data[0], data[1], data[2]
    y_train, y_val, y_test = data[3], data[4], data[5]
    token2index, index2token = data[6], data[7]

    print('Generating sentence content data ...')
    if use_all_sentences:
        x_train_sc, y_train_sc = get_sentence_content_data_all(x_train, y_train,
                                                           token2index, index2token, setting=setting)
        x_val_sc, y_val_sc = get_sentence_content_data_all(x_val, y_val,
                                                       token2index, index2token, setting=setting)
        x_test_sc, y_test_sc = get_sentence_content_data_all(x_test, y_test,
                                                         token2index, index2token, setting=setting)
    else:
        x_train_sc, y_train_sc = get_sentence_content_data(x_train, y_train,
                                                               token2index, index2token, setting=setting)
        x_val_sc, y_val_sc = get_sentence_content_data(x_val, y_val,
                                                           token2index, index2token, setting=setting)
        x_test_sc, y_test_sc = get_sentence_content_data(x_test, y_test,
                                                             token2index, index2token, setting=setting)
    print('Saving sentence content data ...')
    cPickle.dump((x_train_sc, x_val_sc, x_test_sc,
                  y_train_sc, y_val_sc, y_test_sc,
                  token2index, index2token),
                 open(dataname + '_sentence_content_' + setting + '.p', 'wb'), protocol=2)


if __name__ == '__main__': fire.Fire(main)

    
