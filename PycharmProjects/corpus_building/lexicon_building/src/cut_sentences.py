import argparse
import os
import sys
sys.path.append("../../")
import pickle
import xlrd
import xlwt
import re
from yutils.str_utils import is_number

parse = argparse.ArgumentParser(description="Instructions for processing")
parse.add_argument("--rp",type=str,default="../../data/original_data/phone/",
                   help="path of original data")
parse.add_argument("--sl",type=str,default="../../data/sentiment_lexicon/sqh_情感词典最终版.xlsx",
                   help="sentiment lexicon")
parse.add_argument("--sp",type=str,default="../../data/corpus/",
                   help="where to save")
parse.add_argument("--st",type=str,default="txt",
                   help="save type")
parse.add_argument("--min",type=int,default=5,
                   help="min length of implicit sentence")
args = parse.parse_args()
##########################################################################################
#read original data
##########################################################################################
def read_original(path):

    """
    :param directory:
    :return:
    """
    sentences = []
    polarities = []
    for file in os.listdir(path):
        file = os.path.join(path,file)
        with open(file,"r") as f:
            lines = f.readlines()
            print("reading data from "+file)
            for line in lines:
                if len(line)>2:
                    line = line.strip().lower()
                    info = line.split("   ")
                    if len(info) == 2:
                        polarity = info[0]
                        sentence = info[1]
                        if polarity == "差评":
                            polarity = 1
                        else:
                            polarity = 0
                        sentence = clause(sentence)
                        if len(sentence)>=3:
                            sentences.append(sentence)
                            polarities.append(polarity)
    print(path,"finished!",len(sentences))
    return sentences,polarities
##########################################################################################
#clause
##########################################################################################
def clause(sentence):
    pattern = r'？|,|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）|hellip'
    shorts_filter = []
    shorts = re.split(pattern,sentence)
    for short in shorts:
        if len(short)>0:
            shorts_filter.append(short)
    return shorts_filter
##########################################################################################
#sentiment lexicon
##########################################################################################
def read_sentiment_lexicon(sl):
    book = xlrd.open_workbook(sl)
    sheet_pos = book.sheet_by_index(5)
    sheet_neg = book.sheet_by_index(6)
    words_pos = []
    words_neg = []
    rows_pos = sheet_pos.nrows
    rows_neg = sheet_neg.nrows
    for i in range(1,rows_pos):
        word = sheet_pos.cell_value(i,2)
        if word != "":
            words_pos.append(word)
    for i in range(1,rows_neg):
        word = sheet_neg.cell_value(i,2)
        if word != "":
            words_neg.append(word)
    return words_pos,words_neg
##########################################################################################
#implicit sentiment sentences
##########################################################################################
def identify_implicit_sentiment(min,sentences,polariyies,words_pos,words_neg):
    implicit_sentences = []
    implicit_sentiments = []
    for sentence,polarity in zip(sentences,polariyies):
        implicit_index = []
        new_sentence = []
        cur = 0
        length = len(sentence)
        if polarity == 0:
            while cur<length:
                if not is_number(sentence[cur]) and len(sentence[cur])>=min:
                    token = 0
                    for word in words_pos:
                        if word in sentence[cur]:
                            token = 1
                            break
                    implicit_index.append(token)
                    new_sentence.append(sentence[cur])
                cur += 1
        if polarity == 1:
            while cur<length:
                if not is_number(sentence[cur]) and len(sentence[cur])>=min:
                    token = 0
                    for word in words_neg:
                        if word in sentence[cur]:
                            token = 1
                            break
                    implicit_index.append(token)
                    new_sentence.append(sentence[cur])
                cur += 1
        cur = 0
        while cur<len(implicit_index)-2:
            if implicit_index[cur] == 1 and implicit_index[cur+1] == 0 and implicit_index[cur+2] == 1:
                if new_sentence[cur+1] not in implicit_sentences:
                    implicit_sentences.append(new_sentence[cur+1])
                    implicit_sentiments.append(polarity)
            cur += 1
    assert len(implicit_sentiments) == len(implicit_sentences)
    return implicit_sentences,implicit_sentiments

def save_implicit_sentences(sp,st,sentences,polarities):
    if st == "txt":
        sentences_save_path = os.path.join(sp,"implicit.sentence")
        labels_save_path = os.path.join(sp,"implicit.label")
        file_sentences = open(sentences_save_path,"w")
        file_labels = open(labels_save_path,"w")
        for sentence,polarity in zip(sentences,polarities):
            file_sentences.write(sentence+"\n")
            file_labels.write(str(polarity)+"\n")
    if st == "xls":
        save_path = os.path.join(sp,"implicit.xls")
        book = xlwt.Workbook()
        sheet = book.add_sheet("implicit",cell_overwrite_ok=True)
        i = 0
        sheet.write(i,0,"polarity")
        sheet.write(i,1,"sentence")
        sheet.write(i,2,"true")
        i = i+1
        for sentence,polarity in  zip(sentences,polarities):
            sheet.write(i,0,polarity)
            sheet.write(i,1,sentence)
            i += 1
        book.save(save_path)

def main(args):
    sentences,polarities = read_original(args.rp)
    words_pos,words_neg = read_sentiment_lexicon(args.sl)
    sentences,polarities = identify_implicit_sentiment(args.min,sentences,polarities,words_pos,words_neg)
    save_implicit_sentences(args.sp,args.st,sentences,polarities)

if __name__ == "__main__":
    main(args)