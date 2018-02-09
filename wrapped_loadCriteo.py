import numpy as np

from loadCriteo import loadCriteoBatch


def loadLRF(batchsize, max_seq_len, max_input, fin):
    total_data, _, total_seqlen, total_label = loadCriteoBatch(batchsize, max_seq_len, max_input, fin)
    batchsize = len(total_data)
    value = []
    indice = []
    shape = [batchsize, 5867]
    for i in range(batchsize):
        ant = {}
        if total_seqlen[i] > 20:
            total_seqlen[i] = 20
        s = 1. / total_seqlen[i]

        for j in range(total_seqlen[i]):
            if not ant.has_key(0):
                ant[0] = total_data[i][j][0] / total_seqlen[i]
            else:
                ant[0] += total_data[i][j][0] / total_seqlen[i]

            if not ant.has_key(1):
                ant[1] = total_data[i][j][1] / total_seqlen[i]
            else:
                ant[1] += total_data[i][j][1] / total_seqlen[i]
            for k in total_data[i][j][2:]:
                if not ant.has_key(k):
                    ant[k] = s
                else:
                    ant[k] += s
        ant = sorted(ant.items(), key=lambda item: item[0])
        for pair in ant:
            k, v = pair
            indice.append([i, k])
            value.append(v / float(s))
    data = (indice, value, shape)
    for i in range(len(total_label)):
        if total_label[i] == 1:
            total_label[i] = [0, 1]
        else:
            total_label[i] = [1, 0]
    return data, total_label


def loadSparse(batchsize, max_seq_len, max_input, fin):
    total_data, _, total_seqlen, total_label = loadCriteoBatch(batchsize, max_seq_len, max_input, fin)
    batchsize = len(total_data)
    value = []
    indice = []
    shape = [batchsize, max_seq_len, 5869]
    for i in range(batchsize):
        if total_seqlen[i] > 20:
            total_seqlen[i] = 20
        for j in range(total_seqlen[i]):
            indice.append([i, j, 0])
            value.append(total_data[i][j][0])
            indice.append([i, j, 1])
            value.append(total_data[i][j][1])
            for index in total_data[i][j][2:]:
                indice.append([i, j, index + 2])
                value.append(1.)
    indice.sort()
    data = (indice, value, shape)
    for i in range(len(total_label)):
        if total_label[i] == 1:
            total_label[i] = [0, 1]
        else:
            total_label[i] = [1, 0]
    return data, None, total_label, total_seqlen


def loadLR(batchsize, max_seq_len, max_input, fin):
    total_data, _, total_seqlen, total_label = loadCriteoBatch(batchsize, max_seq_len, max_input, fin)
    batchsize = len(total_data)
    value = []
    indice = []
    shape = [batchsize, 5867]
    for i in range(batchsize):
        if total_seqlen[i] > 20:
            total_seqlen[i] = 20
        j = total_seqlen[i]
        indice.append([i, 0])
        value.append(total_data[i][j - 1][0])
        indice.append([i, 1])
        value.append(total_data[i][j - 1][1])
        for index in total_data[i][j - 1][2:]:
            indice.append([i, index])
            value.append(1.)
    indice = sorted(indice)
    data = (indice, value, shape)
    for i in range(len(total_label)):
        if total_label[i] == 1:
            total_label[i] = [0, 1]
        else:
            total_label[i] = [1, 0]
    return data, total_label


def loaddualattention(batchsize, max_seq_len, max_input, fin):
    total_data, _, total_seqlen, total_label = loadCriteoBatch(batchsize, max_seq_len, max_input, fin)
    batchsize = len(total_data)
    click_label = []
    for i in range(batchsize):
        l = []
        if total_seqlen[i] > max_seq_len:
            total_seqlen[i] = max_seq_len
        for j in range(total_seqlen[i]):
            if total_data[i][j][1] == 1:
                l.append([0, 1])
            else:
                l.append([1, 0])
        click_label.append(l + [[0, 0]] * (max_seq_len - total_seqlen[i]))
    # total_data = np.array(total_data)
    # total_data = np.concatenate((total_data[:, :, 0:1], total_data[:, :, 2:]), axis=2)
    return total_data, None, click_label, total_label, total_seqlen


def loadrnnattention(batchsize, max_seq_len, max_input, fin):
    total_data, _, total_seqlen, total_label = loadCriteoBatch(batchsize, max_seq_len, max_input, fin)
    batchsize = len(total_data)
    for i in range(batchsize):
        if total_seqlen[i] > max_seq_len:
            total_seqlen[i] = max_seq_len
    return total_data, None, total_label, total_seqlen


def savedense(fin):
    fout = open('criteo_train.pkl', 'wb')
    data, _, seqlen, label = loadCriteoBatch(1, 20, 10, fin)
    ant = []
    for i in range(seqlen[0]):
        ans = [0.0] * 5869
        ans[0] = data[0][i][0]
        ans[1] = data[0][i][1]
        for j in range(2, 13):
            ans[data[0][i][j]] = 1.
        ant.append(ans)
