from loadCriteo import loadCriteoBatch

import cPickle as pkl
from sklearn.metrics import*
from tqdm import tqdm

class SP:
    def __init__(self, path, trainpath, testpath, channelpath):
        self.pob = {}
        self.path = path
        self.trainpath = trainpath
        self.testpath = testpath
        self.channelpath = channelpath

    def savemodel(self):
        outfile = open(self.path, 'wb')
        pkl.dump(self.pob, outfile)
        print('saved')

    def loadmodel(self):
        self.pob = pkl.load(open(self.path, 'rb'))
        print('loaded')

    def train(self):
        infile = open(self.trainpath, 'r')
        filec = open(self.channelpath, 'rb')
        Channel_Set = pkl.load(filec)
        Channel_value = {}
        Channel_time = {}
        batch_size = 10
        while True:
            batch = loadCriteoBatch(batch_size, 20, 12, infile)
            data, _, seqlen, label = batch
            for i in range(len(label)):
                for j in range(seqlen[i]):
                    index = Channel_Set[str(data[i][j][2])]
                    v = label[i]
                    if index in Channel_value:
                        Channel_value[index] += v
                        Channel_time[index] += 1.
                    else:
                        Channel_value[index] = v
                        Channel_time[index] = 1.
            if len(label) != batch_size:
                break
        for key in Channel_value:
            self.pob[key] = Channel_value[key]/Channel_time[key]
        self.savemodel()

    def vali(self, set):
        if(set == 'train'):
            infile = open(self.trainpath, 'r')
        else:
            infile = open(self.testpath, 'r')
        filec = open(self.channelpath, 'rb')
        Channel_Set = pkl.load(filec)
        batch_size = 10
        labels = []
        preds = []
        while True:
            batch = loadCriteoBatch(batch_size, 20, 12, infile)
            data, _, seqlen, label = batch

            for i in range(len(label)):
                labels.append(label[i])
                pred = 1
                for j in range(seqlen[i]):
                    index = Channel_Set[str(data[i][j][2])]
                    pred *= (1-self.pob[index])
                pred = 1 - pred
                preds.append(pred)
            if len(label) != batch_size:
                break
        #print(labels)
        print('finished')
        #print(preds)
        #print(len(labels) == len(preds))
        auc = roc_auc_score(labels, preds)
        logloss = log_loss(labels, preds)
        print("AUC = " + "{:.4f}".format(auc))
        print("logloss = " + "{:.4f}".format(logloss))

    def attr(self):
        outfile = open('./attribute_criteo/SP.txt', 'w')
        for key in self.pob:
            outfile.write(key + '\t' + str(self.pob[key]) + '\n')



if __name__ == '__main__':
    trainpath = '../data/train_usr.yzx.txt'
    testpath = '../data/test_usr.yzx.txt'
    channelpath = '../data/index2channel.pkl'
    path = '../Model/SP.pkl'
    model = SP(path, trainpath, testpath, channelpath)
    model.train()
    #model.loadmodel()
    model.vali('train')
    model.vali('test')
    #model.attr()
