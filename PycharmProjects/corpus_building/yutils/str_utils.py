import os
import sys
sys.path.append("../../../")
sys.path.append("../../")
from pyltp import Segmentor,Postagger,Parser

MODELDIR = "/Users/wangshuai/PycharmProjects/ltp_data/"

def segment_str(sentences_list,lexicon=None):
    """
    :param sentences_list:
    :param lexicon:
    :return:
    """
    segmentor = Segmentor()
    SEGDIR = os.path.join(MODELDIR,"cws.model")
    print("loading models from "+ SEGDIR + "...")
    if lexicon == None:
        segmentor.load(SEGDIR)
    else:
       segmentor.load_with_lexicon(SEGDIR,lexicon)
    seg_list = []
    for sentence in sentences_list:
        words = list(segmentor.segment(sentence))
        seg_list.append(words)
    segmentor.release()
    print("finished..")
    return seg_list

def postag_str(seg_list):
    """
    :param seg_list:
    :return:
    """
    postagger = Postagger()
    POSDIR = os.path.join(MODELDIR,"pos.model")
    print("loading models from " + POSDIR + "...")
    postagger.load(POSDIR)
    pos_list = []
    for words in seg_list:
        postags = list(postagger.postag(words))
        pos_list.append(postags)
    postagger.release()
    print("finished..")
    return pos_list

def parse_str(seg_list,pos_list):
    """
    :param seg_list:
    :param pos_list:
    :return:
    """
    parser =Parser()
    PARDIR = os.path.join(MODELDIR,"parser.model")
    print("loading models from " + PARDIR + "...")
    parser.load(PARDIR)
    par_list = []
    for (words,poses) in zip(seg_list,pos_list):
        par = list(parser.parse(words,poses))
        parse = []
        for p in par:
            parse.append([p.head,p.relation])
        par_list.append(parse)
    parser.release()
    print("finished..")
    return par_list


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False