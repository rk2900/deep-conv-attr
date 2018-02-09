from __future__ import print_function

import cPickle as pkl
import os
import time
from datetime import timedelta
import random
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import *

from model_config import config
from wrapped_loadCriteo import loaddualattention


class DualAttention(object):
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
            shape=[None, self.config.seq_max_len, 12]
        )
        self.click_label = tf.placeholder(
            tf.float32,
            shape=[None, self.config.seq_max_len, self.config.n_classes]
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
        # Define Variables
        W = tf.Variable(tf.random_normal([self.config.n_classes, self.config.n_hidden], stddev=0.1), name='W')
        U = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='U')
        C = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='C')
        U_a = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='U_a')
        W_a = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='W_a')
        v_a = tf.Variable(tf.random_normal([self.config.n_hidden], stddev=0.1), name='v_a')
        W_z = tf.Variable(tf.random_normal([self.config.n_classes, self.config.n_hidden], stddev=0.1), name='W_z')
        U_z = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='U_z')
        C_z = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='C_z')
        W_r = tf.Variable(tf.random_normal([self.config.n_classes, self.config.n_hidden], stddev=0.1), name='W_r')
        U_r = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='U_r')
        C_r = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='C_r')
        W_s = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='W_s')
        W_o = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_classes], stddev=0.1), name='W_o')
        b_o = tf.Variable(tf.random_normal([self.config.n_classes], stddev=0.1), name='b_o')
        W_h = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='W_h')
        U_s = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='U_s')
        W_x1 = tf.Variable(tf.random_normal([self.config.n_input, self.config.n_hidden], stddev=0.1), name='W_x1')
        W_x2 = tf.Variable(tf.random_normal([self.config.n_input, self.config.n_hidden], stddev=0.1), name='W_x2')
        v_a2 = tf.Variable(tf.random_normal([self.config.n_hidden], stddev=0.1), name='v_a2')
        v_a3 = tf.Variable(tf.random_normal([self.config.n_hidden], stddev=0.1), name='v_a3')
        W_c = tf.Variable(tf.random_normal([self.config.n_hidden], stddev=0.1), name='W_c')
        b_c = tf.Variable(tf.random_normal([self.config.batch_size]))

        index = tf.range(0, batchsize) * self.config.seq_max_len + (self.seqlen - 1)
        x_last = tf.gather(params=tf.reshape(x, [-1, self.config.n_input]), indices=index)
        x = tf.transpose(x, [1, 0, 2])
        # x = tf.reshape(x,[-1, self.config.n_input])
        # x = tf.split(axis=0, num_or_size_splits=self.config.seq_max_len, value=x)
        y = tf.transpose(self.click_label, [1, 0, 2])
        y = tf.reshape(y, [-1, self.config.n_classes])
        y = tf.split(value=y, num_or_size_splits=self.config.seq_max_len, axis=0)
        # encoder
        gru_cell = tf.contrib.rnn.GRUCell(self.config.n_hidden)
        gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=self.keep_prob,
                                                 output_keep_prob=self.keep_prob)
        states_h, last_h = tf.nn.dynamic_rnn(gru_cell, x, self.seqlen, dtype=tf.float32, time_major=True)
        states_h = tf.reshape(states_h, [-1, self.config.n_hidden])
        states_h = tf.split(states_h, self.config.seq_max_len, 0)

        Uhs = []
        for state_h in states_h:
            Uh = tf.matmul(state_h, U_a)
            Uhs.append(Uh)

        # decoder
        state_s = tf.tanh(tf.matmul(states_h[-1], W_s))
        # s0 =  tanh(Ws * h_last)
        states_s = [state_s]
        outputs = []
        output = tf.zeros(shape=[batchsize, self.config.n_classes])
        for i in range(self.config.seq_max_len):
            # e = []
            # for Uh in Uhs:
            # 	e_ = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(states_s[i], W_a) + Uh), v_a), reduction_indices=1)
            # 	e.append(e_)
            # e = tf.stack(e)
            # # (seq_max_len, batch_size)
            # a1 = tf.nn.softmax(e, dim=0)
            # a1 = tf.split(a1, self.config.seq_max_len, 0)
            # c = tf.zeros([batchsize, self.config.n_hidden])
            # for j in range(self.config.seq_max_len):
            # 	c = c + tf.multiply(states_h[j], tf.transpose(a1[j]))
            c = states_h[-1]
            if self.is_training == True:
                last_output = y[i]
            else:
                last_output = tf.nn.softmax(output)
            r = tf.sigmoid(tf.matmul(last_output, W_r) + tf.matmul(states_s[i], U_r) + tf.matmul(c, C_r))
            z = tf.sigmoid(tf.matmul(last_output, W_z) + tf.matmul(states_s[i], U_z) + tf.matmul(c, C_z))
            s_hat = tf.tanh(tf.matmul(last_output, W) + tf.matmul(tf.multiply(r, states_s[i]), U) + tf.matmul(c, C))
            state_s = tf.multiply(tf.subtract(1.0, z), states_s[i]) + tf.multiply(z, s_hat)
            states_s.append(state_s)
            state_s = tf.nn.dropout(state_s, self.keep_prob)
            output = tf.matmul(state_s, W_o) + b_o
            outputs.append(output)

        e2 = []
        e3 = []
        Ux = tf.matmul(x_last, W_x1)
        Ux2 = tf.matmul(x_last, W_x2)
        for i in range(self.config.seq_max_len):
            e2_ = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(states_h[i], W_h) + Ux), v_a2), reduction_indices=1)
            e2.append(e2_)
            e3_ = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(states_s[i], U_s) + Ux2), v_a3), reduction_indices=1)
            e3.append(e3_)
        e2 = tf.stack(e2)
        e3 = tf.stack(e3)
        a2 = tf.nn.softmax(e2, dim=0)
        a3 = tf.nn.softmax(e3, dim=0)
        a2 = tf.split(a2, self.config.seq_max_len, 0)
        a3 = tf.split(a3, self.config.seq_max_len, 0)
        c2 = tf.zeros([batchsize, self.config.n_hidden])
        c3 = tf.zeros([batchsize, self.config.n_hidden])
        for i in range(self.config.seq_max_len):
            c2 = c2 + tf.multiply(states_h[i], tf.transpose(a2[i]))
            c3 = c3 + tf.multiply(states_s[i], tf.transpose(a3[i]))
        C = c2 + c3
        cvr = tf.reduce_sum(tf.multiply(C, W_c), axis=1) + b_c
        cvr = tf.nn.dropout(cvr, keep_prob=self.keep_prob)
        conversion_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=cvr))
        cvr = tf.nn.sigmoid(cvr)
        mask = tf.sequence_mask(self.seqlen, self.config.seq_max_len)
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        # (batchsize, max_seq_len, n_classes)
        loss_click = tf.nn.softmax_cross_entropy_with_logits(labels=self.click_label, logits=outputs)
        loss_click = tf.boolean_mask(loss_click, mask)
        loss_click = tf.reduce_mean(loss_click)
        click_pred = tf.nn.softmax(outputs)
        loss = loss_click + conversion_loss
        for v in tf.trainable_variables():
            loss += self.config.miu * tf.nn.l2_loss(v)
            loss_click += self.config.miu * tf.nn.l2_loss(v)
            conversion_loss += self.config.miu * tf.nn.l2_loss(v)

        global_step = tf.Variable(0, trainable=False)
        start_learningrate = self.config.learning_rate
        cov_learning_rate = tf.train.exponential_decay(start_learningrate, global_step, 50000, 0.96)
        clk_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        optimizer = tf.train.AdamOptimizer(learning_rate=cov_learning_rate)
        gvs_clk, v_clk = zip(*clk_optimizer.compute_gradients(loss_click))
        gvs_clk, _ = tf.clip_by_global_norm(gvs_clk, 5.0)
        gvs_clk = zip(gvs_clk, v_clk)
        gvs_cov, v_cov = zip(*optimizer.compute_gradients(conversion_loss))
        gvs_cov, _ = tf.clip_by_global_norm(gvs_cov, 5.0)
        gvs_cov = zip(gvs_cov, v_cov)
        gvs, v = zip(*optimizer.compute_gradients(loss))
        gvs, _ = tf.clip_by_global_norm(gvs, 5.0)
        gvs = zip(gvs, v)

        self.clk_train_step = clk_optimizer.apply_gradients(gvs_clk)
        self.cov_train_step = optimizer.apply_gradients(gvs_cov, global_step=global_step)
        self.train_step = optimizer.apply_gradients(gvs, global_step=global_step)
        self.click_prediction = click_pred
        self.conversion_prediction = cvr
        self.cross_entropy = loss
        self.clk_loss = loss_click
        self.cov_loss = conversion_loss
        self.click_attention = a3
        self.attention = a2

    # self.impression_attention = a1

    def train_one_epoch(self, batch_size, learning_rate, fetches):
        total_loss = []
        total_clk_loss = []
        total_cov_loss = []
        clk_pred = []
        clk_label = []
        cvr_pred = []
        cvr_label = []
        infile = open(self.train_dataset, 'rb')
        while True:
            batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            train_data, train_compaign_data, click_label, train_label, train_seqlen = batch
            if len(train_label) != batch_size:
                break
            feed_dict = {
                self.input: train_data,
                self.click_label: click_label,
                self.labels: train_label,
                self.seqlen: train_seqlen,
                self.learning_rate: learning_rate,
                self.is_training: True,
                self.keep_prob: 0.5
            }
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, clk_loss, cov_loss, loss, clk, cvr = result
            total_loss.append(loss)
            total_clk_loss.append(clk_loss)
            total_cov_loss.append(cov_loss)
            clk = np.reshape(clk, (-1, 2)).tolist()
            click_label = np.reshape(click_label, (-1, 2)).tolist()
            clk_pred += clk
            clk_label += click_label
            cvr_pred += cvr.tolist()
            cvr_label += train_label
        clk_pred = np.array(clk_pred)
        auc_clk = roc_auc_score(np.argmax(clk_label, 1), clk_pred[:, 1])
        auc_cov = roc_auc_score(cvr_label, cvr_pred)
        print("click_AUC = " + "{:.4f}".format(auc_clk))
        print("conversion_AUC = " + "{:.4f}".format(auc_cov))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        mean_clk_loss = np.mean(total_clk_loss)
        mean_cov_loss = np.mean(total_cov_loss)
        print("Clk Loss = " + "{:.4f}".format(mean_clk_loss))
        print("Cov_Loss = " + "{:.4f}".format(mean_cov_loss))
        return mean_loss, mean_cov_loss, mean_clk_loss, auc_clk, auc_cov

    def train_all_epochs(self, start_epoch=1):
        n_epoches = self.config.n_epochs
        learning_rate = self.config.learning_rate
        batch_size = self.config.batch_size

        total_start_time = time.time()
        for epoch in range(start_epoch, n_epoches + 1):
            print('\n', '-' * 30, 'Train epoch: %d' % epoch, '-' * 30, '\n')
            start_time = time.time()

            print("Training...")
            result = self.train_one_epoch(batch_size, learning_rate,
                                          [self.clk_train_step, self.clk_loss, self.cov_loss, self.cross_entropy,
                                           self.click_prediction,
                                           self.conversion_prediction])
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
        clk_losses = []
        n_epochs = self.config.n_epochs
        fetch = [self.clk_train_step, self.clk_loss, self.cov_loss, self.cross_entropy, self.click_prediction,
                 self.conversion_prediction]
        flag = 0
        while True:
            print('-' * 30, 'Train epoch: %d' % epoch, '-' * 30)
            start_time = time.time()

            print("Training...")
            if flag == 0 and (epoch > 10 or (epoch > 3 and clk_losses[-1] < clk_losses[-2] < clk_losses[-3])):
                flag = epoch
                fetch = [self.train_step, self.clk_loss, self.cov_loss, self.cross_entropy, self.click_prediction,
                         self.conversion_prediction]
            result = self.train_one_epoch(batch_size, learning_rate, fetch)
            self.log(epoch, result, prefix='train')

            loss = self.test(epoch)
            time_per_epoch = time.time() - start_time
            losses.append(loss[0])
            clk_losses.append(loss[1])
            if flag != 0 and (epoch > flag + 3 and losses[-1] < losses[-2] < losses[-3]):
                self.save_model()
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
        clk_pred = []
        clk_label = []
        cvr_pred = []
        total_clk_loss = []
        total_cov_loss = []
        cvr_label = []
        infile = open(self.test_dataset, 'rb')
        while True:
            batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
            if len(test_label) != batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.click_label: click_label,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.keep_prob: 1
            }
            fetches = [self.clk_loss, self.cov_loss, self.cross_entropy, self.click_prediction,
                       self.conversion_prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            clk_loss, cov_loss, loss, clk, cvr = result
            total_loss.append(loss)
            total_clk_loss.append(clk_loss)
            total_cov_loss.append(cov_loss)
            clk = np.reshape(clk, (-1, 2)).tolist()
            click_label = np.reshape(click_label, (-1, 2)).tolist()
            clk_pred += clk
            clk_label += click_label
            cvr_pred += cvr.tolist()
            cvr_label += test_label
        clk_pred = np.array(clk_pred)
        auc_clk = roc_auc_score(np.argmax(clk_label, 1), clk_pred[:, 1])
        auc_cov = roc_auc_score(cvr_label, cvr_pred)
        loglikelyhood = -log_loss(cvr_label, cvr_pred)
        print("click_AUC = " + "{:.4f}".format(auc_clk))
        print("conversion_AUC = " + "{:.4f}".format(auc_cov))
        print("loglikelyhood = " + "{:.4f}".format(loglikelyhood))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        mean_clk_loss = np.mean(total_clk_loss)
        mean_cov_loss = np.mean(total_cov_loss)
        print("Clk Loss = " + "{:.4f}".format(mean_clk_loss))
        print("Cov_Loss = " + "{:.4f}".format(mean_cov_loss))
        self.log(epoch, [mean_loss, mean_cov_loss, mean_clk_loss, auc_clk, auc_cov], 'test')
        return auc_cov, auc_clk

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
        batch_size = self.config.batch_size
        infile = open(self.test_dataset, 'rb')
        outfile = open('/newNAS/Workspaces/AdsGroup/fyc/attribute_criteo_DARNN/dualattention_withclick.txt', 'w')
        while True:
            batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
            if len(test_label) != batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.click_label: click_label,
                self.keep_prob: 1
            }
            fetches = self.attention
            attention = self.sess.run(fetches, feed_dict=feed_dict)
            click_label = np.array(click_label)
            click_label = click_label[:, :, 1]
            for i in range(batch_size):
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
        #     batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
        #     test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
        #     if len(test_label) != batch_size:
        #         break
        #     feed_dict = {
        #         self.input: test_data,
        #         self.labels: test_label,
        #         self.seqlen: test_seqlen,
        #         self.is_training: False,
        #         self.click_label: click_label,
        #         self.keep_prob: 1
        #     }
        #     fetches = self.attention
        #     attention = self.sess.run(fetches, feed_dict=feed_dict)
        #     click_label = np.array(click_label)
        #     click_label = click_label[:, :, 1]
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
            outfile.write(key + '\t' + str(Channel_value[key]) + '\n')

    # for (i, j, k) in zip(Channel_List, Channel_value, Channel_time):
    #     if k != 0:
    #         print(i + '\t' + str(j / k))
    #     else:
    #         print(i + '\t0')
    def attr_click(self):
        filec = open('index2channel.pkl', 'rb')
        Channel_Set = pkl.load(filec)
        Channel_value = {}
        Channel_time = {}
        batch_size = self.config.batch_size
        infile = open(self.test_dataset, 'rb')
        outfile = open('/newNAS/Workspaces/AdsGroup/fyc/attribute_criteo_DARNN/dualattention_withclick_click.txt', 'w')
        while True:
            batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
            if len(test_label) != batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.click_label: click_label,
                self.keep_prob: 1
            }
            fetches = self.click_attention
            attention = self.sess.run(fetches, feed_dict=feed_dict)
            click_label = np.array(click_label)
            click_label = click_label[:, :, 1]
            for i in range(batch_size):
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
        #     batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
        #     test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
        #     if len(test_label) != batch_size:
        #         break
        #     feed_dict = {
        #         self.input: test_data,
        #         self.labels: test_label,
        #         self.seqlen: test_seqlen,
        #         self.is_training: False,
        #         self.click_label: click_label,
        #         self.keep_prob: 1
        #     }
        #     fetches = self.attention
        #     attention = self.sess.run(fetches, feed_dict=feed_dict)
        #     click_label = np.array(click_label)
        #     click_label = click_label[:, :, 1]
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
            outfile.write(key + '\t' + str(Channel_value[key]) + '\n')

    # for (i, j, k) in zip(Channel_List, Channel_value, Channel_time):
    #     if k != 0:
    #         print(i + '\t' + str(j / k))
    #     else:
    #         print(i + '\t0')
    def vertical_attr(self, lenth):
        batch_size = self.config.batch_size
        infile = open(self.test_dataset, 'rb')
        value1 = [0.] * lenth
        value2 = [0.] * lenth
        while True:
            batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
            if len(test_label) != batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.click_label: click_label,
                self.keep_prob: 1
            }
            fetches = [self.click_attention, self.attention]
            a1, a2 = self.sess.run(fetches, feed_dict=feed_dict)

            for i in range(batch_size):
                if test_seqlen[i] == lenth and test_label[i] == 1:
                    for j in range(test_seqlen[i]):
                        # if click_label[i][j] == 1:
                        v1 = a1[j][0][i]
                        v2 = a2[j][0][i]
                        value1[j] += v1 * 0.2 + v2 * 0.8
                        value2[j] += v2 * 0.9 + v2 * 0.1

        print(value1)
        print(value2)


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
    path = './Model/DARNN'
    model = DualAttention(path, traindata, testdata, C)
    model.train_until_cov()
    model.test(0)
