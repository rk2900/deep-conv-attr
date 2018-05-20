import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
import tqdm
from sklearn.metrics import *
import cPickle as pkl

import sys
import loadCriteo

'''
 data:([t-ti,[xi]]...,last_time)
'''


class AMTA():
    def __init__(self, path, max_seq_len=20, embedding_size=64, feature_number=11,
                 learning_rate=0.001, epochs=10000, miu=1e-3, batchsize=32):
        self.graph = tf.Graph()

        self._path = path
        self._save_path, self._logs_path = None, None
        self.cross_entropy, self.train_step, self.prediction = None, None, None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.Channel_Set, self.Channel_List = ChannelSet, ChannelList
        self.features = num_of_channel
        self.feature_number = feature_number
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.classes = 2
        self.n_input = num_feat
        self.miu = miu
        self.batchsize = batchsize

        with self.graph.as_default():
            self._define_inputs()
            self._build_graph()
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

    def _initialize_session(self, set_logs=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.global_initializer)
        self.sess.run(self.local_initializer)
        if set_logs:
            logswriter = tf.summary.FileWriter
            self.summary_writer = logswriter(self.logs_path, graph=self.graph)

    def _define_inputs(self):
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, self.max_seq_len, self.feature_number])
        self.time = tf.placeholder(dtype=tf.float32, shape=[None, self.max_seq_len])
        self.channel = tf.placeholder(dtype=tf.int32, shape=[None, self.max_seq_len])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None])
        self.seqlen = tf.placeholder(dtype=tf.int32, shape=[None])

    def _build_graph(self):
        index = tf.range(0, self.batchsize) * self.max_seq_len + (self.seqlen - 1)
        x_last = tf.gather(params=tf.reshape(self.x, [-1, self.feature_number]), indices=index)
        x = tf.transpose(self.x, [1, 0, 2])

        y = self.y
        time = self.time
        time = tf.transpose(time, [1, 0])
        seqlen = self.seqlen

        w_c = tf.Variable(tf.random_normal([self.n_input], stddev=0.001))
        w_e = tf.Variable(tf.random_normal([self.n_input], stddev=0.001))
        w_d = tf.Variable(tf.random_normal([self.n_input], stddev=0.001))

        mask = tf.sequence_mask(seqlen, self.max_seq_len, dtype=tf.float32)
        mask = tf.transpose(mask, [1, 0])

        alpha = tf.exp(tf.reduce_sum(tf.gather(w_e, x), axis=2))
        gamma = tf.exp(tf.reduce_sum(tf.gather(w_d, x), axis=2))
        lam = tf.multiply(gamma, tf.exp(-tf.multiply(gamma, time)))
        Gamma = tf.subtract(tf.ones(shape=tf.shape(gamma)), tf.exp(-tf.multiply(gamma, time)))

        S = tf.exp(-tf.reduce_sum(tf.multiply(tf.multiply(alpha, Gamma), mask), axis=0))
        h = tf.reduce_sum(tf.multiply(tf.multiply(alpha, lam), mask), axis=0)
        p = tf.divide(tf.ones(shape=tf.shape(y)),
                      tf.add(tf.ones(shape=tf.shape(y)), tf.exp(-tf.reduce_sum(tf.gather(w_c, x_last), axis=1))))

        ny = tf.subtract(tf.ones(shape=tf.shape(y)), y)
        initial = tf.constant(1e-5, shape=[self.batchsize])
        biases = tf.Variable(initial)
        attribution = tf.multiply(tf.multiply(alpha, lam), mask)
        attribution = attribution / h
        self.attribution = attribution
        # self.loss_0 = tf.reduce_mean(tf.log(tf.multiply(y, h) + biases))
        # self.loss_1 = tf.reduce_mean(tf.log(tf.add(ny, tf.multiply(p, tf.subtract(S, ny))) + biases))

        self.loss_0 = tf.reduce_mean(tf.multiply(y, tf.log(p))) + tf.reduce_mean(
            tf.multiply(y, tf.log(h))) + tf.reduce_mean(tf.multiply(y, tf.log(S)))
        self.loss_1 = tf.reduce_mean(
            tf.multiply(ny, tf.log(1 - tf.multiply(p, tf.subtract(tf.ones(shape=tf.shape(y)), S)))))
        self.loss_2 = self.miu * (tf.nn.l2_loss(w_c) + tf.nn.l2_loss(w_e) + tf.nn.l2_loss(w_d))
        loss = -tf.add(self.loss_0, self.loss_1) + self.loss_2
        self.cross_entropy = loss
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        # self.prediction = p * (1 - S)
        # self.prediction = p * S * h
        self.prediction = p * (1 - S + S * h)
        self.h = h

    def train_one_epoch(self):
        num_examples = traindata_size
        print num_examples
        total_loss = []
        total_loss_0 = []
        total_loss_1 = []
        total_loss_2 = []
        pred = []
        label = []
        f_train = open(train_path)
        pbar = tqdm.tqdm(total=traindata_size // self.batchsize)
        for i in range(num_examples // self.batchsize):
            # print("batch {}:".format(i))
            pbar.update(1)
            data, compaign_data, labels, seqlen, time = loadCriteo.loadCriteoBatch_AMTA(batchsize=self.batchsize,
                                                                                        max_input=self.feature_number + 2,
                                                                                        max_seq_len=self.max_seq_len,
                                                                                        fin=f_train)
            feed_dict = {
                self.x: data,
                self.y: labels,
                self.time: time,
                self.seqlen: seqlen,
                # self.channel: compaign_data
            }
            # print(data)
            # print(labels)
            # print(time)
            # print(seqlen)
            fetches = [self.train_step, self.cross_entropy, self.prediction, self.h, self.loss_0, self.loss_1,
                       self.loss_2]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, prediction, h, loss_0, loss_1, loss_2 = result
            total_loss.append(loss)
            total_loss_0.append(loss_0)
            total_loss_1.append(loss_1)
            total_loss_2.append(loss_2)
            pred += prediction.tolist()
            label += labels
        # print(loss)
        print("training finished")
        mean_loss = np.mean(total_loss)
        mean_loss_0 = np.mean(total_loss_0)
        mean_loss_1 = np.mean(total_loss_1)
        mean_loss_2 = np.mean(total_loss_2)
        auc = roc_auc_score(label, pred)
        print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(mean_loss, mean_loss_0, mean_loss_1, mean_loss_2, auc))
        f_train.close()
        f_log.write(
            "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(mean_loss, mean_loss_0, mean_loss_1, mean_loss_2, auc))
        return mean_loss, auc, mean_loss_0, mean_loss_1, mean_loss_2

    def train_all_epoch(self):
        total_start_time = time.time()
        losses = []
        print("Loss\tLoss_0\tLoss_1\tLoss_2\tAUC")
        f_log.write("Loss\tLoss_0\tLoss_1\tLoss_2\tAUC\n")
        for epoch in range(self.epochs):
            print '-' * 30, 'Train epoch: %d' % epoch, '-' * 30
            start_time = time.time()
            f_log.write("Train epoch {}".format(epoch))
            print("Training...")
            result = self.train_one_epoch()
            loss = result[1]
            losses.append(loss)
            self.log(epoch, result, 'train')
            self.test(epoch)
            time_per_epoch = time.time() - start_time
            seconds_left = int((self.epochs - epoch) * time_per_epoch)
            print ('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))
            ))
            # if epoch > 3:
            #    if losses[-1] < losses[-2] < losses[-3]:
            #        break

            self.save_model()

        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))

    def test(self, epoch=0):
        num_examples = testdata_size
        print num_examples
        total_loss = []
        total_loss_0 = []
        total_loss_1 = []
        total_loss_2 = []
        pred = []
        label = []
        f_test = open(test_path)
        for i in range(num_examples // self.batchsize):
            data, compaign_data, labels, seqlen, time = loadCriteo.loadCriteoBatch_AMTA(batchsize=self.batchsize,
                                                                                        max_input=self.feature_number + 2,
                                                                                        max_seq_len=self.max_seq_len,
                                                                                        fin=f_test)
            feed_dict = {
                self.x: data,
                self.y: labels,
                self.time: time,
                self.seqlen: seqlen,
                # self.channel: compaign_data
            }
            fetches = [self.cross_entropy, self.prediction, self.h, self.loss_0, self.loss_1, self.loss_2]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            loss, prediction, h, loss_0, loss_1, loss_2 = result
            total_loss.append(loss)
            total_loss_0.append(loss_0)
            total_loss_1.append(loss_1)
            total_loss_2.append(loss_2)
            pred += prediction.tolist()
            label += labels

        print("testing {}-th epoch finished".format(epoch))
        mean_loss = np.mean(total_loss)
        mean_loss_0 = np.mean(total_loss_0)
        mean_loss_1 = np.mean(total_loss_1)
        mean_loss_2 = np.mean(total_loss_2)
        auc = roc_auc_score(label, pred)
        print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(mean_loss, mean_loss_0, mean_loss_1, mean_loss_2, auc))
        f_log.write(
            "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(mean_loss, mean_loss_0, mean_loss_1, mean_loss_2, auc))
        self.log(epoch, [auc], 'test')
        f_test.close()

    # self.log(epoch, [mean_loss, auc], 'test')

    def log(self, epoch, result, prefix):
        s = prefix + '\t' + str(epoch)
        for i in result:
            s += ('\t' + str(i))
        fout = open("%s/%s" % (self.logs_path, str(self.learning_rate)), 'a')
        fout.write(s + '\n')

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)

    def attr(self):
        num_examples = testdata_size

        f_test = open(test_path)
        outfile = open('./attribute_criteo_s1/AMTA.txt', 'w')
        filec = open('index2channel.pkl', 'rb')
        Channel = pkl.load(filec)
        Channel_value = {}
        Channel_time = {}
        for i in range(num_examples // self.batchsize):
            data, compaign_data, labels, seqlen, time = loadCriteo.loadCriteoBatch_AMTA(batchsize=self.batchsize,
                                                                                        max_input=self.feature_number + 2,
                                                                                        max_seq_len=self.max_seq_len,
                                                                                        fin=f_test)
            feed_dict = {
                self.x: data,
                self.y: labels,
                self.time: time,
                self.seqlen: seqlen,
                # self.channel: compaign_data
            }
            fetches = [self.attribution]
            attribution = self.sess.run(fetches, feed_dict)[0]
            for i in range(self.batchsize):
                if labels[i] != 0:
                    for j in range(seqlen[i]):
                        # if click_label[i][j] == 1:
                        index = Channel[str(data[i][j][0])]
                        v = attribution[j][i]
                        if index in Channel_value:
                            Channel_value[index] += v
                            Channel_time[index] += 1
                        else:
                            Channel_value[index] = v
                            Channel_time[index] = 1

        for key in Channel_value:
            outfile.write(key + '\t' + str(Channel_value[key] / Channel_time[key]) + '\n')

    def vertical_attr(self, lenth):
        attr = [0.] * lenth
        num_examples = testdata_size

        for i in range(num_examples // self.batchsize):
            data, compaign_data, labels, seqlen, time = loadCriteo.loadCriteoBatch_AMTA(batchsize=self.batchsize,
                                                                                        max_input=self.feature_number + 2,
                                                                                        max_seq_len=self.max_seq_len,
                                                                                        fin=f_test)
            feed_dict = {
                self.x: data,
                self.y: labels,
                self.time: time,
                self.seqlen: seqlen,
                # self.channel: compaign_data
            }
            fetches = [self.attribution]
            attribution = self.sess.run(fetches, feed_dict)[0]
            for i in range(len(labels)):
                if seqlen[i] == lenth and labels[i] == 1:
                    for j in range(seqlen[i]):
                        # if click_label[i][j] == 1:

                        v = attribution[j][i]
                        attr[j] += v

        print attr


f_log = open("log_AMTA.txt", 'w')
train_path = '../data/train_usr.yzx.txt'
test_path = '../data/test_usr.yzx.txt'
traindata_size = loadCriteo.count(train_path)
testdata_size = loadCriteo.count(test_path)
num_feat = 5867
# non-click: 5867, click: 5868
num_feat += 2
ChannelList = [1367, 2481, 3193, 2879, 3166, 3364, 1222, 4274, 1967, 4516, 1984, 4219, 3740, 3634, 3934, 4670, 5405,
               5820, 4340, 5444, 3298, 4186, 482, 1718, 3648, 5302, 3712, 1107, 1804, 4830, 4956, 3905, 5394, 2084,
               4168, 3007, 2418, 865, 3149, 1509, 1155, 2545, 631, 3897, 2318, 4458, 2371, 1392, 4213, 1457, 4228, 3652,
               725, 3030, 5498, 729, 5848, 2478, 5214, 1353, 4549, 1179, 2354, 4772, 3875, 4941, 396, 912, 2312, 4943,
               592, 3730, 2945, 1478, 4050, 3230, 4807, 531, 775, 3058, 3039, 4418, 4057, 3205, 4518, 2986, 1352, 5489,
               3391, 779, 2129, 1218, 3806, 428, 4944, 2571, 4942, 911, 2915, 3184, 3805, 3403, 2238, 5707, 5397, 3607,
               2393, 4290, 2259, 149, 1178, 5335, 3239, 1555, 2956, 30, 2357, 1069, 1933, 3982, 5698, 1458, 74, 681,
               5784, 3408, 811, 1437, 569, 1278, 414, 4437, 5772, 2055, 1782, 4039, 2682, 1197, 570, 830, 216, 4730,
               5602, 2940, 274, 2132, 1823, 1082, 3146, 5052, 4631, 3052, 1389, 4175, 1480, 3361, 1519, 5097, 3789,
               1920, 929, 1516, 1930, 4659, 4628, 3542, 632, 4616, 3006, 1164, 4624, 5459, 4358, 5426, 1078, 2480, 1789,
               2964, 4940, 5846, 3701, 317, 348, 4914, 1094, 1095, 2567, 1070, 998, 744, 1499, 4548, 5028, 4315, 4933,
               1801, 3074, 1263, 5629, 1705, 1379, 568, 4748, 3776, 2338, 381, 3536, 922, 2562, 3495, 3044, 1282, 4325,
               5095, 2372, 1730, 4412, 3480, 2473, 5582, 4986, 4341, 4100, 3553, 65, 1513, 5067, 3933, 2538, 5002, 3078,
               3432, 5285, 1860, 4792, 1728, 154, 4179, 5264, 1815, 2446, 3231, 3263, 473, 1813, 1812, 2799, 4880, 3377,
               1058, 1198, 763, 3085, 2957, 498, 58, 1697, 1161, 789, 1539, 2569, 2918, 3506, 451, 1376, 214, 791, 5609,
               5372, 4074, 2221, 2757, 2518, 2585, 4409, 2739, 1114, 4629, 1611, 1927, 4579, 1980, 4420, 4916, 2651,
               5094, 2778, 2997, 3062, 1425, 5273, 4717, 2040, 2688, 833, 37, 4502, 413, 5376, 4553, 1564, 1279, 3876,
               3683, 3964, 2590, 4124, 3656, 4954, 5494, 3088, 4759, 5115, 4541, 3040, 5610, 5263, 5721, 1622, 1744,
               5786, 1199, 5866, 908, 2661, 1176, 171, 3987, 4263, 1767, 817, 4615, 3604, 545, 629, 3518, 3449, 1415,
               1368, 783, 5215, 1620, 4747, 5564, 1163, 2563, 1650, 1008, 4625, 3654, 3623, 4367, 176, 5389, 1840, 2524,
               4331, 1380, 3995, 5102, 4061, 2306, 1150, 4396, 2060, 3822, 5005, 1616, 4874, 2844, 2665, 5753, 1589,
               4086, 3086, 5197, 4915, 3589, 5361, 4291, 4990, 838, 1009, 3277, 3672, 4898, 1435, 5745, 29, 3796, 2574,
               3600, 16, 950, 2943, 1811, 4676, 766, 3953, 1028, 2952, 1116, 5828, 3537, 2148, 2747, 3643, 1349, 4065,
               144, 3075, 255, 3453, 5733, 5750, 3622, 718, 3308, 5161, 2434, 4598, 1873, 4715, 3041, 5341, 5467, 3008,
               1501, 4330, 2097, 4095, 4605, 2185, 3659, 517, 5601, 2436, 1237, 3992, 5237, 2039, 4492, 1510, 3680,
               3269, 4738, 3534, 2098, 4913, 4063, 1946, 4332, 1160, 36, 917, 3836, 244, 4128, 4339, 5571, 3079, 5157,
               1416, 541, 4720, 1998, 1308, 5035, 342, 1866, 5004, 5287, 5659, 2058, 3464, 2600, 1057, 288, 859, 2601,
               4910, 5732, 4356, 4982, 2656, 5027, 240, 269, 697, 2756, 3952, 1266, 299, 5357, 4617, 1344, 2896, 2346,
               1129, 232, 797, 4694, 2897, 491, 1162, 2345, 2951, 5596, 5166, 4852, 3029, 5377, 5591, 5253, 2890, 1968,
               2492, 5809, 4547, 2959, 4145, 3512, 1394, 4395, 1085, 3760, 2902, 2475, 2057, 2018, 4423, 2536, 4630,
               361, 539, 5219, 389, 1877, 654, 4683, 2373, 1583, 3870, 1790, 4422, 3397, 78, 4509, 5754, 3050, 2910,
               955, 1628, 2041, 4703, 4897, 5503, 2135, 5274, 2884, 5106, 3276, 565, 4808, 2308, 127, 1159, 5627, 626,
               2410, 4491, 1590, 5003, 1226, 2655, 1934, 156, 1267, 5849, 1990, 5026, 1765, 5590, 4669, 5045, 4578,
               4785, 1969, 4411, 5000, 1957, 2007, 2593, 3358, 1904, 3027, 5324, 3028, 1931, 2662, 5785, 2012, 1826,
               1446, 3690, 1312, 4268, 2732, 2872, 2063, 4416, 2042, 4094, 496, 0, 2035, 3864, 1821, 5831, 415, 1956,
               1851, 1217, 1511, 5014, 4822, 3983, 4812, 5863, 3963, 4857, 2315, 3842, 1853, 5155, 5666, 5262, 3476,
               1636, 4342, 4867, 2672, 1317, 2220, 3753, 3678, 2474, 1210, 5173, 1372, 128, 3681, 3333, 5469, 1084,
               5616, 4653, 5589, 4674, 1588, 2984, 4902, 2476, 2720, 3777, 4546, 921, 2078, 5843, 792, 5170]
ChannelSet = {0: 614, 4100: 222, 2055: 133, 2057: 528, 2058: 473, 2060: 364, 2063: 609, 16: 392, 4124: 306, 29: 388,
              30: 115, 4128: 457, 2084: 33, 37: 295, 2097: 429, 2098: 447, 58: 255, 65: 224, 4168: 34, 3753: 644,
              74: 122, 78: 546, 4175: 153, 2129: 90, 4179: 237, 2132: 145, 2135: 557, 4186: 21, 2148: 404, 4213: 48,
              4219: 11, 2410: 569, 127: 565, 128: 650, 3776: 203, 4228: 50, 2185: 432, 144: 409, 149: 109, 154: 236,
              156: 576, 4263: 328, 171: 326, 4268: 606, 2221: 270, 176: 352, 4274: 7, 2078: 667, 2238: 102, 4290: 107,
              4291: 379, 2259: 108, 214: 265, 216: 140, 36: 453, 4315: 193, 4325: 212, 232: 499, 4330: 428, 4331: 356,
              4332: 451, 240: 486, 4339: 458, 4340: 18, 4341: 221, 4342: 639, 3534: 446, 255: 411, 2306: 361, 4356: 482,
              4358: 172, 2312: 68, 2315: 631, 269: 487, 2318: 44, 4367: 351, 4578: 585, 274: 144, 5166: 508, 288: 477,
              2338: 204, 4145: 520, 2345: 505, 2346: 497, 299: 492, 4396: 363, 3464: 474, 2354: 62, 2357: 116,
              4409: 274, 4411: 588, 4412: 216, 317: 181, 4416: 610, 4418: 81, 2371: 46, 4420: 282, 2373: 540, 4422: 544,
              4423: 530, 4492: 441, 4437: 131, 342: 468, 2393: 106, 348: 182, 361: 533, 4458: 45, 5863: 628, 2418: 36,
              2672: 641, 381: 205, 2434: 419, 2436: 436, 389: 536, 4491: 570, 396: 66, 2446: 240, 4502: 296, 413: 297,
              414: 130, 415: 619, 4516: 9, 4518: 84, 2473: 218, 2474: 646, 2475: 527, 428: 93, 2478: 57, 4509: 547,
              2480: 175, 2481: 1, 2492: 516, 4541: 313, 4546: 665, 451: 263, 4548: 191, 4549: 60, 4553: 299, 5197: 375,
              2518: 272, 473: 243, 2524: 355, 482: 22, 4579: 280, 2536: 531, 2538: 228, 491: 503, 496: 613, 2545: 41,
              498: 254, 4598: 420, 4605: 431, 2562: 208, 2563: 345, 517: 434, 2567: 186, 4616: 167, 2569: 260, 2571: 95,
              2574: 390, 4624: 170, 4625: 348, 531: 77, 4628: 164, 4629: 277, 4630: 532, 4631: 150, 2585: 273, 539: 534,
              541: 463, 2590: 305, 545: 333, 2600: 475, 2601: 479, 1458: 121, 4659: 163, 5214: 58, 568: 201, 569: 128,
              570: 138, 4669: 583, 4670: 15, 4674: 658, 4676: 396, 4683: 539, 565: 562, 592: 70, 4694: 501, 2651: 284,
              4703: 554, 2656: 484, 2661: 324, 2662: 599, 2665: 370, 4715: 422, 4717: 291, 4720: 464, 626: 568,
              629: 334, 631: 42, 632: 166, 2682: 136, 2476: 662, 2688: 293, 4738: 445, 4747: 342, 4748: 202, 654: 538,
              5000: 589, 5571: 459, 4759: 311, 2720: 663, 4772: 63, 681: 123, 2732: 607, 4785: 586, 2739: 275,
              4792: 234, 697: 488, 2747: 405, 2896: 496, 2756: 489, 2757: 271, 4807: 76, 4808: 563, 4812: 627, 718: 416,
              725: 52, 4822: 625, 729: 55, 2778: 286, 4830: 29, 4730: 141, 744: 189, 2799: 246, 4852: 509, 4857: 630,
              763: 251, 766: 397, 4867: 640, 5590: 582, 775: 78, 4874: 368, 779: 89, 783: 339, 4880: 247, 789: 258,
              791: 266, 792: 669, 2844: 369, 797: 500, 5253: 513, 4897: 555, 4898: 385, 2959: 519, 4902: 661, 4913: 448,
              811: 126, 4910: 480, 817: 330, 4914: 183, 4915: 376, 4916: 283, 2872: 608, 830: 139, 2879: 3, 833: 294,
              2884: 559, 4933: 194, 838: 381, 5601: 435, 2890: 514, 4940: 178, 4941: 65, 4942: 96, 4943: 69, 4944: 94,
              2897: 502, 2902: 526, 4954: 308, 859: 478, 4956: 30, 2910: 550, 865: 37, 2915: 98, 2918: 261, 4982: 483,
              1513: 225, 4986: 220, 2940: 143, 4990: 380, 2943: 394, 2945: 72, 2951: 506, 2952: 400, 5002: 229,
              5003: 572, 908: 323, 2957: 253, 911: 97, 912: 67, 2964: 177, 917: 454, 5014: 624, 921: 666, 922: 207,
              5274: 558, 929: 160, 5026: 580, 5027: 485, 5028: 192, 2984: 660, 2986: 85, 5035: 467, 2997: 287, 950: 393,
              955: 551, 5052: 149, 3006: 168, 3007: 35, 3008: 426, 5067: 226, 3027: 595, 3028: 597, 3029: 510, 3030: 53,
              3039: 80, 3040: 314, 3041: 423, 3044: 210, 5094: 285, 5095: 213, 5097: 157, 3050: 549, 5287: 471,
              3052: 151, 5102: 359, 1008: 347, 1009: 382, 3058: 79, 3062: 288, 5115: 312, 3074: 196, 3075: 410,
              1028: 399, 3078: 230, 3079: 460, 2220: 643, 3085: 252, 3086: 374, 3088: 310, 1057: 476, 1058: 249,
              5155: 634, 5157: 461, 5161: 418, 4615: 331, 1069: 117, 1070: 187, 5170: 670, 5173: 648, 1078: 174,
              4617: 494, 1082: 147, 1084: 654, 1085: 524, 1094: 184, 1095: 185, 3146: 148, 3149: 38, 1107: 27,
              1114: 276, 1116: 401, 3166: 4, 5215: 340, 5219: 535, 1129: 498, 3184: 99, 5237: 439, 3193: 2, 1150: 362,
              1155: 40, 3205: 83, 1159: 566, 1160: 452, 1161: 257, 1162: 504, 1163: 344, 1164: 169, 5262: 636,
              5263: 316, 5264: 238, 1176: 325, 5273: 290, 1178: 110, 1179: 61, 3230: 75, 3231: 241, 1904: 594,
              5285: 232, 3239: 112, 1197: 137, 1198: 250, 1199: 321, 5302: 25, 1210: 647, 3263: 242, 1217: 622,
              1218: 91, 3269: 444, 1222: 6, 2593: 592, 5324: 596, 1226: 573, 3276: 561, 3277: 383, 1237: 437, 5335: 111,
              5341: 424, 3298: 20, 3308: 417, 5357: 493, 1263: 197, 5361: 378, 1266: 491, 1267: 577, 5372: 268,
              1278: 129, 1279: 301, 5376: 298, 5377: 511, 1282: 211, 3333: 652, 5389: 353, 4653: 656, 5394: 32,
              5397: 104, 1308: 466, 5405: 16, 3358: 593, 1312: 605, 3361: 155, 3364: 5, 1317: 642, 3377: 248, 5426: 173,
              3391: 88, 1344: 495, 3659: 433, 5444: 19, 1349: 407, 1352: 86, 2956: 114, 3403: 101, 5005: 366, 3408: 125,
              5459: 171, 1934: 575, 1367: 0, 1368: 338, 5467: 425, 1372: 649, 5469: 653, 1376: 264, 1379: 200,
              1380: 357, 3432: 231, 1389: 152, 4547: 518, 1392: 47, 5489: 87, 1394: 522, 5494: 309, 3449: 336, 5498: 54,
              3453: 412, 5503: 556, 1415: 337, 1416: 462, 1425: 289, 3476: 637, 3480: 217, 1435: 386, 1437: 127,
              1446: 603, 3495: 209, 1457: 49, 3506: 262, 3512: 521, 244: 456, 5564: 343, 3518: 335, 1611: 278, 1478: 73,
              1480: 154, 5582: 219, 3536: 206, 3537: 403, 5589: 657, 3542: 165, 5591: 512, 1499: 190, 5596: 507,
              1501: 427, 3553: 223, 5602: 142, 1509: 39, 1510: 442, 1511: 623, 5609: 267, 5610: 315, 1516: 161,
              1519: 156, 5616: 655, 5627: 567, 5629: 198, 1539: 259, 3589: 377, 3600: 391, 1555: 113, 3604: 332,
              3607: 105, 2308: 564, 5659: 472, 1564: 300, 5666: 635, 3622: 415, 3623: 350, 1583: 541, 3634: 13,
              1588: 659, 1589: 372, 1590: 571, 3643: 406, 5045: 584, 3648: 24, 5698: 120, 3652: 51, 3654: 349,
              3656: 307, 5707: 103, 1616: 367, 1620: 341, 1622: 318, 3672: 384, 5721: 317, 1628: 552, 3678: 645,
              3680: 443, 3681: 651, 3683: 303, 5732: 481, 5733: 413, 3690: 604, 5745: 387, 1650: 346, 3701: 180,
              5750: 414, 5753: 371, 5754: 548, 3712: 26, 5772: 132, 3730: 71, 5784: 124, 5785: 600, 5786: 320, 3740: 12,
              1697: 256, 1705: 199, 3760: 525, 5809: 517, 1718: 23, 5820: 17, 1728: 235, 3777: 664, 1730: 215,
              5828: 402, 5831: 618, 3789: 158, 1744: 319, 5843: 668, 3796: 389, 5846: 179, 5848: 56, 5849: 578,
              3805: 100, 3806: 92, 1765: 581, 1767: 329, 5866: 322, 3822: 365, 1782: 134, 3836: 455, 1789: 176,
              1790: 543, 3842: 632, 4395: 523, 1801: 195, 1804: 28, 1811: 395, 1812: 245, 1813: 244, 1815: 239,
              3864: 616, 1821: 617, 3870: 542, 1823: 146, 1826: 602, 3875: 64, 3876: 302, 1840: 354, 5004: 470,
              3897: 43, 1851: 621, 1853: 633, 3905: 31, 1860: 233, 1866: 469, 1873: 421, 1877: 537, 3933: 227, 3934: 14,
              998: 188, 3952: 490, 3953: 398, 2655: 574, 3963: 629, 3964: 304, 1920: 159, 1927: 279, 1930: 162,
              1931: 598, 1933: 118, 3982: 119, 3983: 626, 3987: 327, 3992: 438, 2372: 214, 1946: 450, 3995: 358,
              3397: 545, 1956: 620, 1957: 590, 5106: 560, 1967: 8, 1968: 515, 1969: 587, 1353: 59, 1980: 281, 1984: 10,
              1990: 579, 4039: 135, 1998: 465, 4050: 74, 2007: 591, 4057: 82, 2012: 601, 4061: 360, 4063: 449,
              4065: 408, 2018: 529, 4074: 269, 1636: 638, 2035: 615, 4086: 373, 2039: 440, 2040: 292, 2041: 553,
              2042: 611, 4094: 612, 4095: 430}
num_of_channel = len(ChannelList)
if len(sys.argv) == 2:
    learning_rate = float(sys.argv[1])
    batchsize = int(sys.argv[2])
elif len(sys.argv) == 1:
    learning_rate = 1e-4
    batchsize = 1000
else:
    print 'usage: python AMTA.py [learning rate] [batch size]'
    exit(1)
f_train = open(train_path)
f_test = open(test_path)
AMTAModel = AMTA('../Model/AMTA', batchsize=1000, learning_rate=learning_rate)
AMTAModel.train_all_epoch()
AMTAModel.test()
