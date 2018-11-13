import numpy as np
import os
import swang_yutils

np.random.seed(123456)


#################
# read embeddings
#################


def read_emb_idx(filename):
    """
    1.read embeddings files to
        "embeddings": numpy matrix, each row is a vector with corresponding index
        "word2idx": word2idx[word] = idx in the "embeddings" matrix
        "idx2word": the reverse dict of "word2idx"
    2. add padding and unk to 3 dictionaries
    :param filename:
        file format: word<space>emb, '\n' (line[0], line[1:-1], line[-1])
    :return:
        vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}
    """
    with open(filename, 'r') as f:
        embeddings = []
        word2idx = dict()

        word2idx["_padding"] = 0  # PyTorch Embedding lookup need padding to be zero
        word2idx["_unk"] = 1

        for i,line in enumerate(f):
            if i>1:
                line = line.strip()
                one = line.split(' ')
                word = one[0]
                emb = [float(i) for i in one[1:]]
                embeddings.append(emb)
                word2idx[word] = len(word2idx)

        ''' Add padding and unknown word to embeddings and word2idx'''
        emb_dim = len(embeddings[0])
        embeddings.insert(0, np.zeros(emb_dim))  # _padding
        embeddings.insert(1, np.random.random(emb_dim))  # _unk

        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = embeddings.reshape(len(embeddings), emb_dim)

        idx2word = dict((word2idx[word], word) for word in word2idx)
        vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}

        print("Finish loading embedding %s * * * * * * * * * * * *" % filename)
        return vocab


#############################################################
""" Raw data --> pickle
output file style looks like this:
    {"training":{
        "xIndexes":[]
        "yLabels":[]
            }
     "validation": ...
     "test": ...
     "word2idx":{"_padding":0,"_unk":1, "1st":2, "hello":3, ...}
     "embedding":[ [word0], [word1], [word2], ...]
    }
"""
#################
# evaluation
#################


def sentence_to_index(word2idx, sentences):
    """
    Transform sentence into lists of word index
    :param word2idx:
        word2idx = {word:idx, ...}
    :param sentences:
        list of sentences which are list of word
    :return:
    """
    print("-------------begin making sentence xIndexes-------------")
    sentences_indexes = []
    for sentence in sentences:
        s_index = []
        for word in sentence:
            word = word
            if word == "\n":
                continue
            if word in word2idx:
                s_index.append(word2idx[word])
            else:
                s_index.append(word2idx["_unk"])
                print("  --", word, "--  ")

        if len(s_index) == 0:
            print(len(sentence), "+++++++++++++++++++++++++++++++++empty sentence")
            s_index.append(word2idx["_unk"])
        sentences_indexes.append(s_index)
    assert len(sentences_indexes) == len(sentences)
    print("-------------finish making sentence xIndexes-------------")
    return sentences_indexes


def make_datasets(word2idx, raw_data):
    """
    :param word2idx:
        word2idx = {word:idx, ...}
    :param raw_data:
        raw_data = {"training": (inputs, labels),
                    "validation",
                    "test"}
    :return:
    """
    datasets = dict()

    for i in ["training", "validation", "test"]:
        sentences, labels = raw_data[i]
        xIndexes = sentence_to_index(word2idx, sentences)
        yLabels = [int(label) for label in labels]
        yLabels = np.asarray(yLabels, dtype=np.int64).reshape(len(labels))
        # yAspects = [int(aspect) for aspect in aspects]
        # yAspects = np.asarray(yAspects,dtype=np.int64).reshape(len(labels))
        datasets[i] = {"xIndexes": xIndexes,
                       "yLabels": yLabels}

    return datasets

#############################################################


def processing():
    output_dir = "data/pkl"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # read raw text
    fns = "data/all_data"
    sentences = swang_yutils.read_file2list(fns + ".sentences")
    labels = swang_yutils.read_file2list(fns + ".labels")
    sentences=sentences[:500]
    labels = labels[:500]
    # aspects = swang_yutils.read_file2list(fns + ".aspects")
    data=[sentences, labels]
    swang_yutils.shuffle(data, seed=123456)
    assert len(data[0]) == len(data[1])

    # split the dataset: train, test
    train_num = int(len(data[0]) * 0.8)
    valid_num = int(len(data[0]) * 0.9)
    train = [d[:train_num] for d in data]
    valid = [d[train_num:valid_num] for d in data]
    test = [d[valid_num:] for d in data]

    assert len(train[0]) == len(train[1])
    assert len(valid[0]) == len(valid[1])
    assert len(test[0])  == len(test[1])

    raw_data = {"training": train,
                "validation": valid,
                "test": test}

    # read the embedding files
    emb_dir = "/users4/gzluo/dataset/embedding/"
    emb_file = emb_dir+"Tencent_AILab_ChineseEmbedding.txt"
    vocab = read_emb_idx(emb_file)
    word2idx, embeddings,idx2word = vocab["word2idx"], vocab["embeddings"],vocab["idx2word"]

    # transform sentence to word index
    datasets = make_datasets(word2idx, raw_data)

    # output the transformed files
    swang_yutils.dict2pickle(datasets, output_dir + "/features_glove.pkl")
    swang_yutils.dict2pickle(word2idx, output_dir + "/word2idx_glove.pkl")
    swang_yutils.dict2pickle(embeddings, output_dir + "/embeddings_glove.pkl")
    swang_yutils.dict2pickle(idx2word, output_dir + "/idx2word_glove.pkl")



if __name__ == "__main__":
    processing()

