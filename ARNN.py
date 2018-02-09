from __future__ import print_function

import cPickle as pkl
import os
import time
from datetime import timedelta
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import *
import sys

from model_config import config
from wrapped_loadCriteo import loadrnnattention


class RnnWithattention(object):
    def __init__(self, path, train_dataset, test_dataset, config):
        self.graph = tf.Graph()

        self._path = path
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self._save_path, self._logs_path = None, None
        self.config = config
        self.batches_step = 0
        self.cross_entropy, self.train_step, self.prediction = None, None, None
        self.data_shape = config.data_shape
        self.label_shape = config.label_shape
        self.n_classes = config.n_classes
        self.attention = None
        with self.graph.as_default():
            self._define_inputs()
            self._build_graph()
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
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
        shape = [None]
        shape.extend(self.data_shape)
        label_shape = [None]
        self.input = tf.placeholder(
            tf.float32,
            shape=[None, self.config.seq_max_len, 12],
            name='input'
        )
        self.labels = tf.placeholder(
            tf.float32,
            shape=label_shape,
            name='labels'
        )
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate'
        )
        self.seqlen = tf.placeholder(
            tf.int32,
            shape=[None],
            name='seqlen'
        )
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    def _initialize_session(self, set_logs=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.sess.run(self.initializer)
        if set_logs:
            logswriter = tf.summary.FileWriter
            self.summary_writer = logswriter(self.logs_path, graph=self.graph)

    def _build_graph(self):
        x = self.input
        batchsize = tf.shape(x)[0]
        embedding_matrix = tf.Variable(
            tf.random_normal([self.config.max_features, self.config.embedding_output], stddev=0.1))
        x1, x2 = tf.split(x, [2, 10], 2)
        x2 = tf.to_int32(x2)
        x2 = tf.nn.embedding_lookup(embedding_matrix, x2)
        x2 = tf.reshape(x2, [-1, self.config.seq_max_len, 10 * self.config.embedding_output])
        x = tf.concat((x1, x2), axis=2)
        weights = {
            'out': tf.Variable(tf.random_normal([self.config.n_hidden])),
            'attention_h': tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden])),
            'attention_x': tf.Variable(tf.random_normal([self.config.n_input, self.config.n_hidden])),
            'v_a': tf.Variable(tf.random_normal([self.config.n_hidden]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal(shape=[self.config.batch_size], dtype=tf.float32))
        }

        index = tf.range(0, self.config.batch_size) * self.config.seq_max_len + (self.seqlen - 1)
        x_last = tf.gather(tf.reshape(x, [-1, self.config.n_input]), index)
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.config.n_input])
        x = tf.split(axis=0, num_or_size_splits=self.config.seq_max_len, value=x)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob,
                                                  output_keep_prob=self.keep_prob)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=self.seqlen)

        # attention
        e = []
        Ux = tf.matmul(x_last, weights['attention_x'])
        for output in outputs:
            e_ = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(output, weights['attention_h']) + Ux), weights['v_a']),
                               reduction_indices=1)
            e.append(e_)
        e = tf.stack(e)
        a = tf.nn.softmax(e, dim=0)
        a = tf.split(a, self.config.seq_max_len, 0)
        c = tf.zeros([self.config.batch_size, self.config.n_hidden])
        for i in range(self.config.seq_max_len):
            c = c + tf.multiply(outputs[i], tf.transpose(a[i]))
        cvr = tf.reduce_sum(tf.multiply(c, weights['out']), axis=1) + biases['out']
        cvr = tf.nn.dropout(cvr, keep_prob=self.keep_prob)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=cvr))
        cvr = tf.nn.sigmoid(cvr)
        for v in tf.trainable_variables():
            loss += self.config.miu * tf.nn.l2_loss(v)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        gvs, v = zip(*optimizer.compute_gradients(loss))
        gvs, _ = tf.clip_by_global_norm(gvs, 5.0)
        gvs = zip(gvs, v)

        self.cross_entropy = loss
        self.train_step = optimizer.apply_gradients(gvs)
        self.prediction = cvr
        self.attention = a

    def train_one_epoch(self, batch_size, learning_rate):
        total_loss = []
        cvr_pred = []
        cvr_label = []
        infile = open(self.train_dataset, 'rb')
        while True:
            batch = loadrnnattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            train_data, train_compaign_data, train_label, train_seqlen = batch
            if len(train_label) != batch_size:
                break
            feed_dict = {
                self.input: train_data,
                self.labels: train_label,
                self.seqlen: train_seqlen,
                self.learning_rate: learning_rate,
                self.is_training: True,
                self.keep_prob: 0.5
            }

            result = self.sess.run([self.train_step, self.cross_entropy, self.prediction], feed_dict=feed_dict)
            _, loss, cvr = result
            total_loss.append(loss)
            cvr_pred += cvr.tolist()
            cvr_label += train_label
        auc_cov = roc_auc_score(cvr_label, cvr_pred)
        print("conversion_AUC = " + "{:.4f}".format(auc_cov))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        return mean_loss, auc_cov

    def train_all_epochs(self, start_epoch=1):
        n_epoches = self.config.n_epochs
        learning_rate = self.config.learning_rate
        batch_size = self.config.batch_size

        total_start_time = time.time()
        for epoch in range(start_epoch, n_epoches + 1):
            print('\n', '-' * 30, 'Train epoch: %d' % epoch, '-' * 30, '\n')
            start_time = time.time()

            print("Training...")
            result = self.train_one_epoch(batch_size, learning_rate)
            self.log(epoch, result, prefix='train')
            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epoches - epoch) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))
            ))

        self.save_model()

        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))

    def train_until_cov(self):
        learning_rate = self.config.learning_rate
        batch_size = self.config.batch_size

        total_start_time = time.time()
        epoch = 1
        losses = []
        n_epochs = self.config.n_epochs

        while True:
            print('-' * 30, 'Train epoch: %d' % epoch, '-' * 30)
            start_time = time.time()

            print("Training...")
            # if epoch > 50 or (epoch > 3 and clk_losses[-1] < clk_losses[-2] < clk_losses[-3]):
            #     fetch = [self.train_step, self.clk_loss, self.cov_loss, self.cross_entropy, self.click_prediction,
            #              self.conversion_prediction]
            result = self.train_one_epoch(batch_size, learning_rate)
            self.log(epoch, result, prefix='train')

            self.test(epoch)
            loss = result[0]
            time_per_epoch = time.time() - start_time
            losses.append(loss)
            if epoch > 10 and losses[-1] > losses[-2] > losses[-3]:
                break
            print('Time per epoch: %s' % (
                str(timedelta(seconds=time_per_epoch))
            ))
            epoch += 1
            self.save_model()

        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def test(self, epoch):
        batch_size = self.config.batch_size
        total_loss = []
        cvr_pred = []
        cvr_label = []
        infile = open(self.test_dataset, 'rb')
        while True:
            batch = loadrnnattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            test_data, test_compaign_data, test_label, test_seqlen = batch
            if len(test_label) != batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.keep_prob: 1
            }
            fetches = [self.cross_entropy, self.prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            loss, cvr = result
            total_loss.append(loss)
            cvr_pred += cvr.tolist()
            cvr_label += test_label
        auc_cov = roc_auc_score(cvr_label, cvr_pred)
        log = log_loss(cvr_label, cvr_pred)
        print("loglikelyhood = " + "{:.4f}".format(log))
        print("conversion_AUC = " + "{:.4f}".format(auc_cov))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        self.log(epoch, [mean_loss, auc_cov], 'test')
        return auc_cov

    def log(self, epoch, result, prefix):
        s = prefix + '\t' + str(epoch)
        for i in result:
            s += ('\t' + str(i))
        fout = open("%s/%s_%s_%s_%s" % (
            self.logs_path, str(self.config.learning_rate), str(self.config.batch_size), str(self.config.n_hidden),
            str(self.config.miu)),
                    'a')
        fout.write(s + '\n')

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)

    def attr(self):
        filec = open('index2channel.pkl', 'rb')
        Channel_Set = pkl.load(filec)
        Channel_value = {}
        Channel_time = {}
        infile = open(self.test_dataset, 'rb')
        outfile = open('/newNAS/Workspaces/AdsGroup/fyc/attribute_criteo_s1/rnn_withattention.txt', 'w')
        while True:
            batch = loadrnnattention(self.config.batch_size, self.config.seq_max_len, self.config.feature_number,
                                     infile)
            test_data, test_compaign_data, test_label, test_seqlen = batch
            if len(test_label) != self.config.batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.keep_prob: 1
            }
            fetches = self.attention
            attention = self.sess.run(fetches, feed_dict=feed_dict)
            for i in range(self.config.batch_size):
                if test_label[i] != 0:
                    for j in range(test_seqlen[i]):
                        # if click_label[i][j] == 1:
                        index = Channel_Set[str(test_data[i][j][2])]
                        v = attention[j][0][i]
                        if Channel_value.has_key(index):
                            Channel_value[index] += v
                            Channel_time[index] += 1
                        else:
                            Channel_value[index] = v
                            Channel_time[index] = 1
        # infile = open(self.train_dataset, 'rb')
        # while True:
        #     batch = loadrnnattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
        #     test_data, test_compaign_data, test_label, test_seqlen = batch
        #     if len(test_label) != batch_size:
        #         break
        #     feed_dict = {
        #         self.input: test_data,
        #         self.labels: test_label,
        #         self.seqlen: test_seqlen,
        #         self.is_training: False,
        #         self.keep_prob: 1
        #     }
        #     fetches = self.attention
        #     attention = self.sess.run(fetches, feed_dict=feed_dict)
        #     for i in range(batch_size):
        #         for j in range(test_seqlen[i]):
        #             # if click_label[i][j] == 1:
        #             index = Channel_Set[str(test_data[i][j][2])]
        #             v = attention[j][0][i]
        #             if Channel_value.has_key(index):
        #                 Channel_value[index] += v
        #                 Channel_time[index] += 1
        #             else:
        #                 Channel_value[index] = v
        #                 Channel_time[index] = 1

        for key in Channel_value:
            outfile.write(key + '\t' + str(Channel_value[key] / Channel_time[key]) + '\n')

    def vertical_attr(self, lenth):
        batch_size = self.config.batch_size
        value = [0.] * lenth
        infile = open(self.test_dataset, 'rb')
        while True:
            batch = loadrnnattention(self.config.batch_size, self.config.seq_max_len, self.config.feature_number,
                                     infile)
            test_data, test_compaign_data, test_label, test_seqlen = batch
            if len(test_label) != self.config.batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.keep_prob: 1
            }
            fetches = self.attention
            attention = self.sess.run(fetches, feed_dict=feed_dict)
            for i in range(batch_size):
                if test_seqlen[i] == lenth and test_label[i] == 1:
                    for j in range(test_seqlen[i]):
                        # if click_label[i][j] == 1:
                        v = attention[j][0][i]
                        value[j] += v

        print(value)


if __name__ == '__main__':
    traindata = 'data/train_usr.yzx.txt'
    testdata = 'data/test_usr.yzx.txt'
    if len(sys.argv) != 4 and len(sys.argv) != 1:
        print('usage: python ARNN.py [learning rate] [batch size] [mu]')
        exit(1)
    elif len(sys.argv) == 4:
        learning_rate = float(sys.argv[1])
        batch_size = float(sys.argv[2])
        mu = float(sys.argv[3])
    else:
        learning_rate = 0.001
        batch_size = 256
        mu = 1e-6

    C = config(max_features=5897, learning_rate=learning_rate, batch_size=batch_size, feature_number=12,
               seq_max_len=20, n_input=2,
               embedding_output=256, n_hidden=512, n_classes=2, n_epochs=50, isseq=True, miu=mu)
    path = './Model/ARNN'
    model = RnnWithattention(path, traindata, testdata, C)
    model.train_until_cov()
    model.test(0)
