# Imports
import numpy as np
import os
# import sys
# sys.path.append(r'home/xiajie/pyutils/PycharmProjects/deeplearning20210221/UCIbaseline/utils')
# from utils.utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import numpy as np
import os

def read_data(data_path, split="train"):
    """ Read data """

    # Fixed params
    n_class = 6
    n_steps = 128

    # Paths
    path_ = os.path.join(data_path, split)
    path_signals = os.path.join(path_, "Inertial_Signals")

    # Read labels and one-hot encode
    label_path = os.path.join(path_, "y_" + split + ".txt")
    labels = pd.read_csv(label_path, header=None)

    # Read time-series data
    channel_files = os.listdir(path_signals)
    channel_files.sort()
    n_channels = len(channel_files)
    posix = len(split) + 5  # 10

    # Initiate array
    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))
    i_ch = 0
    for fil_ch in channel_files:
        # print(fil_ch)  # body_acc_x_train.txt
        channel_name = fil_ch[:-posix]
        # print(channel_name)  # body_acc_x
        dat_ = pd.read_csv(os.path.join(path_signals, fil_ch), delim_whitespace=True, header=None)
        # print(dat_)  # [7352 rows x 128 columns]
        # X[:, :, i_ch] = dat_.as_matrix()
        X[:, :, i_ch] = dat_.iloc[:, :].values
        # print(X)

        # Record names
        list_of_channels.append(channel_name)

        # iterate
        i_ch += 1

    # Return
    return X, labels[0].values, list_of_channels


def standardize(train, test):
    """ Standardize data """

    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
    X_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]

    return X_train, X_test


def one_hot(labels, n_class=2):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels - 1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y


def get_batches(X, y, batch_size=100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]

import h5py
# 读取数据
# len6000data_DAE.py
# 加载hdpy成np的形式
# def load_dataset(path):
#
#     h5_file = h5py.File(path, 'r')
#     eegdatas = h5_file['fre']
#     labels = h5_file['labels']
#
#     return eegdatas, labels

# 加载hdpy成np的形式
def load_h5py_to_np(path):
    h5_file = h5py.File(path, 'r')
    print('打印一下h5py中有哪些关键字', h5_file.keys())
    permutation = np.random.permutation(len(h5_file['labels']))   # 打乱
    shuffled_image = h5_file['fre'][:][permutation, :]
    shuffled_label = h5_file['labels'][:][permutation]
    print('经过打乱之后数据集中的标签顺序是:\n', shuffled_label, len(h5_file['labels']))
    return shuffled_image, shuffled_label

def load_dataset(path):
    dataa, labels = load_h5py_to_np(path)  # shuffle过

    # dataset = h5py.File('datasets/train_signs.h5', "r")

    train_set_x_orig = dataa[:27024]  # your train set features  # 5405
    train_set_y_orig = labels[:27024]  # your train set labels  # 23220

    # test_dataset = h5py.File('datasets/test_signs.h5', "r")
    # test_set_x_orig = np.array(dataset["image"][2000:])  # your test set features
    # test_set_y_orig = np.array(dataset["labels"][2000:])  # your test set labels
    test_set_x_orig = dataa[27024:]  # your test set features  # 23220
    test_set_y_orig = labels[27024:]  # your test set labels  # 23220

    classes = np.array([0, 1])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset('20210519_Dataset_200x10.h5')


# X_train, labels_train, list_ch_train = read_data(data_path="./data/", split="train")  # train
# X_test, labels_test, list_ch_test = read_data(data_path="./data/", split="test")  # test
#
# assert list_ch_train == list_ch_test, "Mistmatch in channels!"  # 条件为 false 触发异常

# train_set_x_orig = np.reshape(train_set_x_orig, train_set_x_orig.shape+(1,))
# # print(train_set_x_orig.shape)
# test_set_x_orig = np.reshape(train_set_x_orig, train_set_x_orig.shape+(1,))
# # print(test_set_x_orig.shape)


# Standardize
print(train_set_x_orig.shape)
print(test_set_x_orig.shape)
X_train, X_test = standardize(train_set_x_orig, test_set_x_orig)
print(X_train)
print(X_test)

# 依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样。
# print(X_train)
# print(train_set_y_orig)
X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, train_set_y_orig[0], stratify=train_set_y_orig[0], random_state=123)

y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(test_set_y_orig[0])

# print(y_tr)
# print(len(y_vld))  # 1838
# print(len(X_tr))  # 5514
# print(len(X_tr[0]))  # 128
# print(X_tr[0][0])  # [x,x,x,x...,x] 9个
# print(X_tr[0][0][0])  # 一个float
# print(X_vld)

# Hyperparameters
# Imports
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# print(tf.__version__)    #查看tensorflow版本

lstm_size = 30         # 3 times the amount of channels
lstm_layers = 2        # Number of layers
batch_size = 600       # Batch size
seq_len = 200          # Number of steps
learning_rate = 0.001  # Learning rate (default is 0.001)
epochs = 50  # 1000

# Fixed
n_classes = 2
n_channels = 10

# Construct the graph
# Placeholders
graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name='inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name='labels')
    keep_prob_ = tf.placeholder(tf.float32, name='keep')  # 类似于dropout
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

# Construct inputs to LSTM
with graph.as_default():
    # Construct the LSTM inputs and LSTM cells
    # print(inputs_)  # Tensor("inputs:0", shape=(?, 128, 9), dtype=float32)
    lstm_in = tf.transpose(inputs_, [1, 0, 2])  # (N, seq_len, channels) reshape into (seq_len, N, channels)
    # print(lstm_in)  # Tensor("transpose:0", shape=(128, ?, 9), dtype=float32)
    lstm_in = tf.reshape(lstm_in, [-1, n_channels])  # Now (seq_len*N, n_channels)
    # print(lstm_in)  # Tensor("Reshape:0", shape=(?, 9), dtype=float32)
    # To cells
    lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?  # units：输出的维度大小，改变inputs的最后一维

    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, seq_len, 0)  # 返回128个[5514，9]的分量

    # Add LSTM layers
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)  # lstm的个数
    initial_state = cell.zero_state(batch_size, tf.float32)

# Define forward pass, cost function and optimizer:
with graph.as_default():
    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32, initial_state=initial_state)

    # We only need the last output tensor to pass into a classifier
    logits = tf.layers.dense(outputs[-1], n_classes, name='logits')

    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    # optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost) # No grad clipping

    # Grad clipping
    train_op = tf.train.AdamOptimizer(learning_rate_)

    gradients = train_op.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    optimizer = train_op.apply_gradients(capped_gradients)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# #Train the network
if os.path.exists('checkpoints') == False:
    print('!mkdir checkpoints')

validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()


with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1

    for e in range(epochs):
        # Initialize
        state = sess.run(initial_state)

        # Loop over batches
        for x, y in get_batches(X_tr, y_tr, batch_size):

            # Feed dictionary
            feed = {inputs_: x, labels_: y, keep_prob_: 0.5,
                    initial_state: state, learning_rate_: learning_rate}

            loss, _, state, acc = sess.run([cost, optimizer, final_state, accuracy], feed_dict=feed)  # session执行
            train_acc.append(acc)
            train_loss.append(loss)

            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))

            # Compute validation loss at every 25 iterations
            if (iteration % 25 == 0):

                # Initiate for validation set
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))

                val_acc_ = []
                val_loss_ = []
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1.0, initial_state: val_state}

                    # Loss
                    loss_v, state_v, acc_v = sess.run([cost, final_state, accuracy], feed_dict=feed)

                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)

                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))

            # Iterate
            iteration += 1

    saver.save(sess, "checkpoints/coca-lstm.ckpt")

# Plot training and test loss
t = np.arange(iteration-1)

plt.figure(figsize=(6, 6))
plt.plot(t, np.array(train_loss), 'r-', t[t % 25 == 0], np.array(validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# Plot Accuracies
plt.figure(figsize=(6, 6))

plt.plot(t, np.array(train_acc), 'r-', t[t % 25 == 0], validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


test_acc = []

with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))

    # 画图
    init = tf.initialize_all_variables()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/", sess.graph)  # 目录结构尽量简单，复杂了容易出现找不到文件，原因不清楚
    sess.run(init)
    print("tensorboard --logdir=./logs  --host=127.0.0.1")

    for x_t, y_t in get_batches(X_test, y_test, batch_size):
        feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1,
                initial_state: test_state}

        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))