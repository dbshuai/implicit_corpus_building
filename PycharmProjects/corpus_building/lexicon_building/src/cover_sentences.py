import argparse
import os
import sys
sys.path.append("../../")
import pickle
import xlrd
import xlwt
import re
from yutils.str_utils import is_number,segment_str

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
    bad_num = 0
    good_num = 0
    sentences = []
    polarities = []
    for file in os.listdir(path):
        if file != "华为荣耀8.txt":
            file = os.path.join(path,file)
            with open(file,"r") as f:
                lines = f.readlines()
                print("reading data from "+file)
                for line in lines:
                    if len(line)>5:
                        line = line.strip().lower()
                        info = line.split("   ")
                        if len(info) == 2:
                            polarity = info[0]
                            sentence = info[1]
                            if polarity == "差评":
                                polarity = 1
                                sentences.append(sentence)
                                polarities.append(polarity)
                                bad_num += 1
                            if polarity == "好评":
                                polarity = 0
                                sentences.append(sentence)
                                polarities.append(polarity)
                                good_num += 1
        else:
            file = os.path.join(path, file)
            with open(file, "r") as f:
                lines = f.readlines()
                print("reading data from !!!" + file)
                for line in lines:
                    if bad_num>=good_num:
                        if len(line) > 5:
                            line = line.strip().lower()
                            info = line.split("   ")
                            if len(info) == 2:
                                polarity = info[0]
                                sentence = info[1]
                                if polarity == "差评":
                                    polarity = 1
                                    sentences.append(sentence)
                                    polarities.append(polarity)
                                    # bad_num += 1
                                if polarity == "好评":
                                    polarity = 0
                                    sentences.append(sentence)
                                    polarities.append(polarity)
                                    good_num += 1
    print(path,"finished!",len(sentences))
    return sentences,polarities
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
#cover sentiment words
##########################################################################################
def cover_sentiment_words(sentences,words_pos,words_neg):
    """
    :param sentences:
    :param words_pos:
    :param words_neg:
    :return:
    """
    marks = ["？",",",".",";","'","?",":","~","!","，","。","、","；","！","…","hellip"]
    implicit_sentences = []
    seg_sentences = segment_str(sentences)
    for i,words in enumerate(seg_sentences):
        new_words = []
        mark_list = []
        for j,word in enumerate(words):
            if word in marks:
                mark_list.append(j)
        if len(mark_list)>0:
            if mark_list[-1] != len(words) - 1:
                mark_list.append(len(words)-1)
            cur = 0
            for k in range(len(mark_list)):
                flag = True
                for word in words[cur:mark_list[k]]:
                    if word in words_neg or word in words_pos:
                        flag = False
                        break
                if flag == True:
                    new_words.extend(words[cur:mark_list[k]])
                else:
                    new_words.append("_")
                new_words.append(words[mark_list[k]])
                cur = mark_list[k]+1
        sentence = "".join(new_words)
        # print(sentence)
        implicit_sentences.append(sentence)
    return implicit_sentences
def save_implicit_sentences(sp,st,sentences,polarities):
    if st == "txt":
        sentences_save_path = os.path.join(sp,"cover_implicit.sentence")
        labels_save_path = os.path.join(sp,"cover_implicit.label")
        file_sentences = open(sentences_save_path,"w")
        file_labels = open(labels_save_path,"w")
        for sentence,polarity in zip(sentences,polarities):
            if len(sentence)>5:
                file_sentences.write(sentence+"\n")
                file_labels.write(str(polarity)+"\n")
    if st == "xls":
        save_path = os.path.join(sp,"cover_implicit.xls")
        book = xlwt.Workbook()
        sheet = book.add_sheet("cover_implicit",cell_overwrite_ok=True)
        i = 0
        sheet.write(i,0,"polarity")
        sheet.write(i,1,"sentence")
        sheet.write(i,2,"true")
        i = i+1
        for sentence,polarity in  zip(sentences,polarities):
            if len(sentence)>5:
                sheet.write(i,0,polarity)
                sheet.write(i,1,sentence)
                i += 1
        book.save(save_path)

def main(args):
    sentences,polarities = read_original(args.rp)
    words_pos,words_neg = read_sentiment_lexicon(args.sl)
    sentences = cover_sentiment_words(sentences,words_pos,words_neg)
    save_implicit_sentences(args.sp,args.st,sentences,polarities)

if __name__ == "__main__":
    main(args)