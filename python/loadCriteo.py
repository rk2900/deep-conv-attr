def loadCriteoBatch(batchsize, max_seq_len, max_input, fin):
    total_data = []
    total_seqlen = []
    total_label = []
    for i in range(batchsize):
        tmpseq = []
        try:
            (seq_len, label) = [int(_) for _ in fin.readline().split()]
            # print(seq_len,label)
            for _ in range(seq_len):
                tmpline = fin.readline().split()
                new_tmpline = [float(tmpline[0])] + [int(t) for t in tmpline[1:max_input]]
                tmpseq.append(new_tmpline)
            tmpseq = tmpseq[-1 * max_seq_len:]
            # padding zero
            if seq_len < max_seq_len:
                for _ in range(seq_len, max_seq_len):
                    tmpseq.append([0] * max_input)
            if seq_len > max_seq_len:
                seq_len = max_seq_len
        except:
            continue
        total_data.append(tmpseq)
        total_seqlen.append(seq_len)
        total_label.append(label)
    # print(len(total_data),len(total_seqlen),len(total_label))
    return total_data, None, total_seqlen, total_label


def loadCriteoBatch_AMTA(batchsize, max_seq_len, max_input, fin):
    total_data = []
    total_seqlen = []
    total_label = []
    total_time = []
    for i in range(batchsize):
        tmpseq = []
        tmptime = []
        try:
            (seq_len, label) = [int(_) for _ in fin.readline().split()]
            for _ in range(seq_len):
                tmpline = fin.readline().split()
                newtmpline = [int(t) for t in tmpline[2:max_input]]
                if int(tmpline[1]) == 1:
                    newtmpline.append(5868)
                else:
                    newtmpline.append(5867)
                tmpseq.append(newtmpline)
                tmptime.append(float(tmpline[0]))
            tmpseq = tmpseq[-1 * max_seq_len:]
            tmptime = tmptime[-1 * max_seq_len:]
            last_time = tmptime[-1]
            new_tmptime = [last_time - curr_time for curr_time in tmptime]
            # padding zero
            if seq_len < max_seq_len:
                for _ in range(seq_len, max_seq_len):
                    tmpseq.append([0] * (max_input - 2))
                    new_tmptime.append(-1)
            if seq_len > max_seq_len:
                seq_len = max_seq_len
        except:
            continue
        total_data.append(tmpseq)
        total_seqlen.append(seq_len)
        total_label.append(label)
        total_time.append(new_tmptime)
    return total_data, None, total_label, total_seqlen, total_time

def count(path):
    c = 0
    f = open(path,'r')
    for line in f.readlines():
        if len(line.split()) == 2:
            c += 1
    return c
maxdim = 12
maxseqlen = 20
batchsize = 1

