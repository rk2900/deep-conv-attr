import cPickle as pkl
import time

import numpy as np
import loadCriteo
from sklearn.metrics import *

'''
	data Format:
	on list of user behaviors,
		one element in list -> one user(u) sequence {'action':[...,[a_i(channel),t_i(current_time)],...], 'isconversion': X_u, 'conversionTime': T_u}
'''


def optmize(data, num_of_channel, beta, omega):
    # Preparation

    for item in data:
        item['p'] = np.zeros(len(item['action']))

    # update_P
    for item in data:
        X_u = item['isconversion']
        if (X_u == 0):
            continue
        T_u = item['conversionTime']
        l = len(item['action'])
        sum_p = 0.
        for i in range(l):
            a_i = item['action'][i][0]
            t_i = item['action'][i][1]
            item['p'][i] = beta[a_i] * omega[a_i] * np.exp(-omega[a_i] * (T_u - t_i))
            sum_p += item['p'][i]
        for i in range(l):
            item['p'][i] = item['p'][i] / sum_p

    # update_Beta
    fz = np.zeros(num_of_channel)
    fmb = np.zeros(num_of_channel)
    for item in data:
        X_u = item['isconversion']
        T_u = item['conversionTime']
        l = len(item['action'])
        for i in range(l):
            a_i = item['action'][i][0]
            t_i = item['action'][i][1]
            fmb[a_i] += 1 - np.exp(-omega[a_i] * (T_u - t_i))
            if (X_u == 1):
                fz[a_i] += item['p'][i]

    # #update_P
    # for item in data:
    # 	X_u = item['isconversion']
    # 	if (X_u == 0):
    # 		continue;
    # 	T_u = item['conversionTime']
    # 	l = len(item['action'])
    # 	sum_p = 0.
    # 	for i in range(l):
    # 		a_i = item['action'][i][0]
    # 		t_i = item['action'][i][1]
    # 		item['p'][i] = beta[a_i]*omega[a_i]*np.exp(-omega[a_i]*(T_u - t_i))
    # 		sum_p += item['p'][i]
    # 	for i in range(l):
    # 		item['p'][i] = item['p'][i] / sum_p

    # update_ogema
    fz = np.zeros(num_of_channel)
    fm = np.zeros(num_of_channel)
    for item in data:
        X_u = item['isconversion']
        T_u = item['conversionTime']
        l = len(item['action'])
        for i in range(l):
            a_i = item['action'][i][0]
            t_i = item['action'][i][1]
            fm[a_i] += item['p'][i] * (T_u - t_i) + beta[a_i] * (T_u - t_i) * np.exp(-omega[a_i] * (T_u - t_i))
            if (X_u == 1):
                fz[a_i] += item['p'][i]

    for k in range(num_of_channel):
        if (fm[k] > 0.):
            if fm[k] < 1e-10:
                fm[k] = 1e-5
            omega[k] = fz[k] / fm[k]

    for k in range(num_of_channel):
        if (fmb[k] > 0.):
            beta[k] = fz[k] / fmb[k]
    return beta, omega


def attr(beta, omega, num_of_channel, data):
    for item in data:
        item['p'] = np.zeros(len(item['action']), dtype=np.float64)

    for item in data:
        X_u = item['isconversion']
        if (X_u == 0):
            continue
        T_u = item['conversionTime']
        l = len(item['action'])
        sum_p = 0.
        for i in range(l):
            a_i = item['action'][i][0]
            t_i = item['action'][i][1]
            item['p'][i] = beta[a_i] * omega[a_i] * np.exp(-omega[a_i] * (T_u - t_i))
            sum_p += item['p'][i]
        for i in range(l):
            item['p'][i] = item['p'][i] / sum_p
    global ChannelList
    ChannelSet = pkl.load(open('index2channel.pkl', 'rb'))
    channel_value = {}
    channel_time = {}
    for item in data:
        X_u = item['isconversion']
        if (X_u == 0):
            continue
        l = len(item['action'])
        for i in range(l):
            a_i = item['action'][i][0]
            channel = ChannelList[a_i]
            if channel_value.has_key(channel):
                channel_value[channel] += item['p'][i]
                channel_time[channel] += 1
            else:
                channel_value[channel] = item['p'][i]
                channel_time[channel] = 1
    outfile = open('/newNAS/Workspaces/AdsGroup/fyc/attribute_criteo_s1/survival.txt', 'w')
    for channel in channel_value:
        outfile.write(ChannelSet[str(channel)] + '\t' + str(channel_value[channel] / channel_time[channel]) + '\n')


def vertical_attr(beta, omega, num_of_channel, data, lenth):
    for item in data:
        item['p'] = np.zeros(len(item['action']), dtype=np.float64)

    for item in data:
        X_u = item['isconversion']
        if (X_u == 0):
            continue
        T_u = item['conversionTime']
        l = len(item['action'])
        sum_p = 0.
        for i in range(l):
            a_i = item['action'][i][0]
            t_i = item['action'][i][1]
            item['p'][i] = beta[a_i] * omega[a_i] * np.exp(-omega[a_i] * (T_u - t_i))
            sum_p += item['p'][i]
        for i in range(l):
            item['p'][i] = item['p'][i] / sum_p
    v = [0.] * lenth
    for item in data:
        X_u = item['isconversion']
        if (X_u == 0):
            continue
        l = len(item['action'])
        if l != lenth:
            continue
        for i in range(l):
            v[i] += item['p'][i]
    print v


def test(beta, omega, num_of_channel, test_data):
    y = []
    pred = []
    for item in test_data:
        y.append(item['isconversion'])
        ans = 1.
        T_u = item['conversionTime']
        Diff_Set = {}
        for tmp in item['action']:
            a_i = tmp[0]
            if not Diff_Set.has_key(a_i):
                Diff_Set[a_i] = 1
            t_i = tmp[1]
            # print T_u - t_i
            pred_now = np.exp(-beta[a_i] * (1 - np.exp(-omega[a_i] * (T_u - t_i + 0.1))))
            ans *= pred_now
        pred.append((1. - ans) * (0.95 ** len(Diff_Set)))

    auc = roc_auc_score(y, pred)
    loglikelyhood = -log_loss(y, pred)
    print "Testing Auc= " + "{:.6f}".format(auc)
    print "Testing loglikelyhood " + "{:.6f}".format(loglikelyhood)
    return auc


num_of_epoches = 20

train_path = '../data/train_usr.yzx.txt'
test_path = '../data/test_usr.yzx.txt'
traindata_size = loadCriteo.count(train_path)
testdata_size = loadCriteo.count(test_path)

ChannelSet = {}
ChannelList = []


def loadCriteoData(datasize, fin):
    total_data = []
    for i in range(datasize):
        tmpseq = {}
        try:
            (tmp_seq_len, tmp_label) = [int(_) for _ in fin.readline().split()]
            tmpseq['isconversion'] = tmp_label
            tmp_action = []
            for _ in range(tmp_seq_len):
                tmpline = fin.readline().split()
                tmp_campaign, tmp_time = int(tmpline[2]), float(tmpline[0]) * 31.
                global ChannelSet, ChannelList
                if tmp_campaign not in ChannelSet:
                    ChannelSet[tmp_campaign] = len(ChannelList)
                    ChannelList.append(tmp_campaign)
                tmp_action.append((ChannelSet[tmp_campaign], tmp_time))
                if _ == tmp_seq_len - 1:
                    tmpseq['conversionTime'] = tmp_time
            tmpseq['action'] = tmp_action
        except:
            continue
        total_data.append(tmpseq)
    print(len(total_data))
    return total_data


def main():
    start_time = time.time()
    with open(train_path) as f_train:
        train_data = loadCriteoData(traindata_size, f_train)
    with open(test_path) as f_test:
        test_data = loadCriteoData(testdata_size, f_test)
    finish_time = time.time()
    print("load dataset finished, {} seconds took".format(finish_time - start_time))
    num_of_channel = len(ChannelList)
    beta = np.random.uniform(0, 1, num_of_channel)
    omega = np.random.uniform(0, 1, num_of_channel)
    auc = []
    for epoch in range(num_of_epoches):
        beta, omega = optmize(train_data, num_of_channel, beta, omega)
        print 'Test on Epoch %d' % epoch
        auc.append(test(beta, omega, num_of_channel, test_data))
        if epoch > 10 and auc[-1] < auc[-2] < auc[-3]:
            break


if __name__ == "__main__":
    main()
