import cPickle as pkl
import os
import time
from datetime import timedelta
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import *

from wrapped_loadCriteo import loadLRF


class LR_f_criteo():
    def __init__(self, path, learning_rate=0.1, epochs=10000):
        self.graph = tf.Graph()

        self._path = path
        self._save_path, self._logs_path = None, None
        self.cross_entropy, self.train_step, self.prediction = None, None, None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.features = 5867
        self.classes = 2

        with self.graph.as_default():
            self._define_sparse_inputs()
            self._build_sparse_graph()
            self.saver = tf.train.Saver()
            self.global_initializer = tf.global_variables_initializer()
            self.local_initializer = tf.local_variables_initializer()
        self._initialize_session()

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = '%s/checkpoint' % self._path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = '%s/logs' % self._path
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return self._logs_path

    def _define_inputs(self):
        self.X = tf.placeholder(tf.float32, [None, self.features])
        self.Y = tf.placeholder(tf.float32, [None, self.classes])

    def _define_sparse_inputs(self):
        self.X = tf.sparse_placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32, [None, self.classes])

    def _initialize_session(self, set_logs=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.global_initializer)
        self.sess.run(self.local_initializer)
        if set_logs:
            logswriter = tf.summary.FileWriter
            self.summary_writer = logswriter(self.logs_path, graph=self.graph)

    def _build_graph(self):
        W = tf.Variable(tf.random_normal([self.features, self.classes]))
        B = tf.Variable(tf.random_normal([self.classes]))

        pY = tf.sigmoid(tf.matmul(self.X, W) + B)
        pY = tf.nn.softmax(pY)

        cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pY, labels=self.Y))
        self.cross_entropy = cost_func
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(cost_func)
        self.train_step = opt
        self.prediction = pY

    def _build_sparse_graph(self):
        W = tf.Variable(tf.random_normal([self.features, self.classes]))
        B = tf.Variable(tf.random_normal([self.classes]))
        X = tf.sparse_tensor_to_dense(self.X)

        pY = tf.sigmoid(tf.matmul(X, W) + B)
        cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pY, labels=self.Y))
        pY = tf.nn.softmax(pY)
        self.cross_entropy = cost_func
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(cost_func)
        self.train_step = opt
        self.prediction = pY
        self.W = W

    def train_one_epoch(self):
        total_time = 0
        total_loss = []
        pred = []
        label = []
        trainfile = open('data/train_usr.yzx.txt', 'rb')
        while True:
            train_X, train_Y = loadLRF(500, 20, 12, trainfile)
            feed_dict = {
                self.X: train_X,
                self.Y: train_Y
            }
            fetches = [self.train_step, self.cross_entropy, self.prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, prediction = result
            total_loss.append(loss)
            pred += prediction.tolist()
            label += train_Y
            if len(train_Y) < 500:
                break
        pred = np.array(pred)
        auc = roc_auc_score(np.argmax(label, 1), pred[:, 1])
        print("AUC = " + "{:.4f}".format(auc))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        print "Load_time = " + str(timedelta(seconds=total_time))
        _pY = np.argmax(pred, 1)
        Y = np.argmax(label, 1)
        precision = precision_score(Y, _pY)
        recall = recall_score(Y, _pY)
        F1 = f1_score(Y, _pY)
        accuracy = accuracy_score(Y, _pY)
        return mean_loss, auc, accuracy, precision, recall, F1

    def train_all_epochs(self):
        total_start_time = time.time()
        losses = []
        for epoch in range(self.epochs):
            print '-' * 30, 'Train epoch: %d' % epoch, '-' * 30
            start_time = time.time()

            print("Training...")
            result = self.train_one_epoch()
            loss = result[0]
            losses.append(loss)
            self.test(epoch)
            self.log(epoch, result, 'train')
            time_per_epoch = time.time() - start_time
            seconds_left = int((self.epochs - epoch) * time_per_epoch)
            print ('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))
            ))
            if epoch > 3:
                if losses[-1] >= losses[-2] and losses[-2] >= losses[-3]:
                    break
            self.save_model()

        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))

    def log(self, epoch, result, prefix):
        s = prefix + '\t' + str(epoch)
        for i in result:
            s += ('\t' + str(i))
        fout = open("%s/%s" % (self.logs_path, str(self.learning_rate)), 'a')
        fout.write(s + '\n')

    def test(self, epoch):
        total_loss = []
        pred = []
        label = []
        file = open('data/test_usr.yzx.txt', 'rb')
        while True:
            train_X, train_Y = loadLRF(500, 20, 12, file)
            feed_dict = {
                self.X: train_X,
                self.Y: train_Y
            }
            fetches = [self.train_step, self.cross_entropy, self.prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, prediction = result
            total_loss.append(loss)
            pred += prediction.tolist()
            label += train_Y
            if len(train_Y) < 500:
                break
        pred = np.array(pred)
        auc = roc_auc_score(np.argmax(label, 1), pred[:, 1])
        loglikelyhood = -log_loss(np.argmax(label, 1), pred[:, 1])
        print("AUC = " + "{:.4f}".format(auc))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        _pY = np.argmax(pred, 1)
        Y = np.argmax(label, 1)
        precision = precision_score(Y, _pY)
        recall = recall_score(Y, _pY)
        F1 = f1_score(Y, _pY)
        accuracy = accuracy_score(Y, _pY)
        result = mean_loss, auc, loglikelyhood, accuracy, precision, recall, F1
        self.log(epoch, result, 'test')

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)

    def attr(self):
        file = open('data/test_usr.yzx.txt', 'rb')
        channelfile = open('index2channel.pkl', 'rb')
        channel_set = pkl.load(channelfile)
        outfile = open('/newNAS/Workspaces/AdsGroup/fyc/attribute_criteo/lr_f.txt', 'w')
        train_X, train_Y = loadLRF(1, 20, 12, file)
        feed_dict = {
            self.X: train_X,
            self.Y: train_Y
        }
        W = self.sess.run(self.W, feed_dict)
        for key in channel_set:
            outfile.write(channel_set[key] + '\t' + str(W[int(key)][1]) + '\n')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        learning_rate = float(sys.argv[1])
    elif len(sys.argv) == 1:
        learning_rate = 1e-4
    else:
        print 'usage: python LR.py [learning rate]'
        exit(1)
    model = LR_f_criteo("./Model/LR", learning_rate=learning_rate)
    model.train_all_epochs()
    model.test(0)
