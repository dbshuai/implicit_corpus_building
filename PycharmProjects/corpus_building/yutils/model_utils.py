# coding:utf-8
from torch.utils.data import dataloader
import torch.utils.data as data
import numpy as np
import pickle
import random


def shuffle(lol, seed=1234567890):
    """
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def read_file2list(filename):
    with open(filename, 'r') as f:
        contents = [line.strip() for line in f]
    print("The file has lines: ", len(contents))
    return contents

def dict2pickle(your_dict, desfile):
    with open(desfile, 'wb') as f:
        pickle.dump(your_dict, f,protocol= 4)


def pickle2dict(picklefile):
    with open(picklefile, 'rb') as f:
        your_dict = pickle.load(f)
    return your_dict

def test_prf(pred, labels, num_class, name='test'):
    """
    4. log and return prf scores
    :return:
    """
    total = len(labels)
    pred_right = [0]*num_class
    pred_all = [0]*num_class
    gold = [0]*num_class
    for i in range(total):
        # print(i, pred[i])
        pred_all[pred[i]] += 1
        if pred[i] == labels[i]:
            pred_right[pred[i]] += 1
        gold[labels[i]] += 1

    print("         ****** {} data ******     ".format(name))
    print("  Prediction:", pred_all, " Right:", pred_right, " Gold:", gold)
    ''' -- for all labels -- '''
    print("  -- Neu|Neg|Pos --")
    accuracy = 1.0 * sum(pred_right) / total
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                                     formation=False,
                                     metric_type="macro")
    _, _, micro_f1 = cal_prf(pred_all, pred_right, gold,
                                     formation=False,
                                     metric_type="micro")
    print("    Accuracy on %s is %d/%d = %f" %
          (name, sum(pred_right), total, accuracy))
    print("    Precision: {}\n    Recall   : {}\n    F1 score : {}".format(p, r, f1))
    print("    Macro F1 score on {} (Neu|Neg|Pos) is {}".format(name, macro_f1))
    print("    Micro F1 score on {} (Neu|Neg|Pos) is {}".format(name, micro_f1))
    print("         *********************     \n")

    return accuracy

def cal_prf(pred, right, gold, formation=True, metric_type=""):
    """
    :param pred: predicted labels
    :param right: predicting right labels
    :param gold: gold labels
    :param formation: whether format the float to 6 digits
    :param metric_type:
    :return: prf for each label
    """
    ''' Pred: [0, 2905, 0]  Right: [0, 2083, 0]  Gold: [370, 2083, 452] '''
    num_class = len(pred)
    precision = [0.0] * num_class
    recall = [0.0] * num_class
    f1_score = [0.0] * num_class

    for i in range(num_class):
        ''' cal precision for each class: right / predict '''
        precision[i] = 0 if pred[i] == 0 else 1.0 * right[i] / pred[i]

        ''' cal recall for each class: right / gold '''
        recall[i] = 0 if gold[i] == 0 else 1.0 * right[i] / gold[i]

        ''' cal recall for each class: 2 pr / (p+r) '''
        f1_score[i] = 0 if precision[i] == 0 or recall[i] == 0 \
            else 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        if formation:
            precision[i] = precision[i].__format__(".6f")
            recall[i] = recall[i].__format__(".6f")
            f1_score[i] = f1_score[i].__format__(".6f")

    ''' PRF for each label or PRF for all labels '''
    if metric_type == "macro":
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        f1_score = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0
    elif metric_type == "micro":
        precision = 1.0 * sum(right) / sum(pred) if sum(pred) > 0 else 0
        recall = 1.0 * sum(right) / sum(gold) if sum(recall) > 0 else 0
        f1_score = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def get_padding(sentences, max_len):
    """
    :param sentences: raw sentence --> index_padded sentence
                    [2, 3, 4], 5 --> [2, 3, 4, 0, 0]
    :param max_len: number of steps to unroll for a LSTM
    :return: sentence of max_len size with zero paddings
    """
    seq_len = np.zeros((0,))
    padded = np.zeros((0, max_len))
    for sentence in sentences:
        num_words = len(sentence)
        num_pad = max_len - num_words
        ''' Answer 60=45+15'''
        if max_len == 60 and num_words > 60:
            sentence = sentence[:45] + sentence[num_words-15:]
            sentence = np.asarray(sentence, dtype=np.int64).reshape(1, -1)
        else:
            sentence = np.asarray(sentence[:max_len], dtype=np.int64).reshape(1, -1)
        if num_pad > 0:
            zero_paddings = np.zeros((1, num_pad), dtype=np.int64)
            sentence = np.concatenate((sentence, zero_paddings), axis=1)
        else:
            num_words = max_len

        padded = np.concatenate((padded, sentence), axis=0)
        seq_len = np.concatenate((seq_len, [num_words]))
    return padded.astype(np.int64), seq_len.astype(np.int64)


def get_mask_matrix(seq_lengths, max_len):
    """
    [5, 2, 4,... 7], 10 -->
            [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             ...,
             [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
            ]
    :param seq_lengths:
    :param max_len:
    :return:
    """
    mask_matrix = np.ones((0, max_len))
    for seq_len in seq_lengths:
        num_mask = max_len - seq_len
        mask = np.ones((1, seq_len), dtype=np.int64)
        if num_mask > 0:
            zero_paddings = np.zeros((1, num_mask), dtype=np.int64)
            mask = np.concatenate((mask, zero_paddings), axis=1)
        mask_matrix = np.concatenate((mask_matrix, mask), axis=0)
    return mask_matrix

class Dataset(data.Dataset):

    def __init__(self,sentences_index,labels,to_pad = True,max_len = 40):
        """
        :param sentences_index: all sentences' index like [[1,3,2,4],[...],...]
        :param labels: all sentences' labels like [0,1,2,...]
        :param to_pad: padding sentences to max_len
        :param max_len:for padding
        """
        self.features = sentences_index
        self.labels = labels
        self.to_pad = to_pad
        self.pad_max_len = max_len

        if to_pad:
            self.seq_lens = None
            self.mask_matrix = None
            if max_len:
                self._padding()
                self._mask()
            else:
                print("Need more information about padding max_length")

    def _padding(self):
        self.features, self.seq_lens = get_padding(self.features, max_len=self.pad_max_len)

    def _mask(self):
        self.mask_matrix = get_mask_matrix(self.seq_lens, max_len=self.pad_max_len)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        if self.to_pad:
            seq_len = self.seq_lens[index]
            mask = self.mask_matrix[index]
            return [feature,label,seq_len,mask]
        else:
            return [feature,label]

    def __len__(self):
        return len(self.features)




