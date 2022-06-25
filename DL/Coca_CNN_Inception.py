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
    posix = len(split) + 5

    # Initiate array
    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))
    i_ch = 0
    for fil_ch in channel_files:
        channel_name = fil_ch[:-posix]
        dat_ = pd.read_csv(os.path.join(path_signals, fil_ch), delim_whitespace=True, header=None)
        # X[:, :, i_ch] = dat_.as_matrix()
        X[:, :, i_ch] = dat_.iloc[:, :].values

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


# X_train, labels_train, list_ch_train = read_data(data_path="./data/", split="train")  # train
# X_test, labels_test, list_ch_test = read_data(data_path="./data/", split="test")  # test
#
# assert list_ch_train == list_ch_test, "Mistmatch in channels!"
#
# # Standardize
# X_train, X_test = standardize(X_train, X_test)
#
# X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, stratify=labels_train, random_state=123)
#
# y_tr = one_hot(lab_tr)
# y_vld = one_hot(lab_vld)
# y_test = one_hot(labels_test)





import h5py
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

    train_set_x_orig = dataa[:27024]  # your train set features  # 23220  # 3512
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

batch_size = 600       # Batch size
seq_len = 200          # Number of steps
learning_rate = 0.001
epochs = 50

n_classes = 2
n_channels = 10

graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

with graph.as_default():
    # (batch, 128, 9) --> (batch, 64, 18)
    # (batch, 200, 10) --> (batch, 100, 20)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=20, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    # (batch, 64, 18) --> (batch, 32, 18)
    # (batch, 100, 20) --> (batch, 50, 20)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=20, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    # (batch, 32, 18) --> (batch, 16, 36)
    # (batch, 50, 20) --> (batch, 25, 40)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=40, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

    # (batch, 16, 36) --> (batch, 8, 36)
    # (batch, 25, 40) --> (batch, 5, 40)
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=40, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=5, padding='same')

with graph.as_default():
    # convolution: input to output of inception (size=1)
    # (batch, 8, 36) --> (batch, 8, 36)
    # (batch, 5, 40) --> (batch, 5, 40)
    conv1_11 = tf.layers.conv1d(inputs=max_pool_4, filters=40, kernel_size=1, strides=1,
                                padding='same', activation=tf.nn.relu)

    # convolution: input to middle layer of inception (size=1)
    # (batch, 8, 36) --> (batch, 8, 18)
    # (batch, 5, 40) --> (batch, 5, 20)
    conv1_21 = tf.layers.conv1d(inputs=max_pool_4, filters=20, kernel_size=1, strides=1,
                                padding='same', activation=tf.nn.relu)

    # convolution: input to middle layer of inception (size=1)
    # (batch, 8, 36) --> (batch, 8, 18)
    # (batch, 5, 20) --> (batch, 5, 20)
    conv1_31 = tf.layers.conv1d(inputs=max_pool_4, filters=20, kernel_size=1, strides=1,
                                padding='same', activation=tf.nn.relu)

    # average pool: input to middle layer of inception
    # (batch, 8, 36) --> (batch, 8, 36)
    # (batch, 5, 20) --> (batch, 5, 20)
    avg_pool_41 = tf.layers.average_pooling1d(inputs=max_pool_4, pool_size=2, strides=1, padding='same')

    ## Middle layer of inception

    # convolution: middle to out layer of inception (size=2)
    # (batch, 8, 18) --> (batch, 8, 36)
    # (batch, 5, 20) --> (batch, 5, 40)
    conv2_22 = tf.layers.conv1d(inputs=conv1_21, filters=40, kernel_size=2, strides=1,
                                padding='same', activation=tf.nn.relu)

    # convolution: middle to out layer of inception (size=4)
    # (batch, 8, 18) --> (batch, 8, 36)
    # (batch, 5, 20) --> (batch, 5, 40)
    conv4_32 = tf.layers.conv1d(inputs=conv1_31, filters=40, kernel_size=4, strides=1,
                                padding='same', activation=tf.nn.relu)

    # convolution: middle to out layer of inception (size=1)
    # (batch, 8, 36) --> (batch, 8, 36)
    # (batch, 5, 20) --> (batch, 5, 40)
    conv1_42 = tf.layers.conv1d(inputs=avg_pool_41, filters=40, kernel_size=1, strides=1,
                                padding='same', activation=tf.nn.relu)

    ## Out layer: Concatenate filters
    # (batch, 5, 4*40)
    inception_out = tf.concat([conv1_11, conv2_22, conv4_32, conv1_42], axis=2)

with graph.as_default():
    # Flatten and add dropout
    flat = tf.reshape(inception_out, (-1, 5 * 160))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

    # Predictions
    logits = tf.layers.dense(flat, n_classes)

    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# if (os.path.exists('checkpoints-cnn') == False):
#     !mkdir checkpoints-cnn

validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1

    # Loop over epochs
    for e in range(epochs):

        # Loop over batches
        for x, y in get_batches(X_tr, y_tr, batch_size):

            # Feed dictionary
            feed = {inputs_: x, labels_: y, keep_prob_: 0.5, learning_rate_: learning_rate}

            # Loss
            loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict=feed)
            train_acc.append(acc)
            train_loss.append(loss)

            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))

            # Compute validation loss at every 10 iterations
            if (iteration % 10 == 0):
                val_acc_ = []
                val_loss_ = []

                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1.0}

                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict=feed)
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

    saver.save(sess, "checkpoints-cnn/har.ckpt")

# Plot training and test loss
t = np.arange(iteration-1)

plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# Plot Accuracies
plt.figure(figsize = (6,6))

plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

test_acc = []

with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))

    # 画图
    init = tf.initialize_all_variables()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logss/", sess.graph)  # 目录结构尽量简单，复杂了容易出现找不到文件，原因不清楚
    sess.run(init)
    print("tensorboard --logdir=./logss  --host=127.0.0.1")

    for x_t, y_t in get_batches(X_test, y_test, batch_size):
        feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1}

        batch_acc = sess.run(accuracy, feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

