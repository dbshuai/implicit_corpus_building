import xlrd
import pickle
import argparse
import cmath
import sys
import os

sys.path.append("../../")
from yutils.str_utils import segment_str,postag_str,parse_str

def vvyxhqc_xlsx_read(xlsx_file):
    """
    :param xlsx_file:
    :return:
    """
    read_file = xlrd.open_workbook(xlsx_file)
    sheet = read_file.sheet_by_index(0)
    cols = sheet.col_values(0)[1:]
    norm_contents = {}
    polarities = []
    sentences = []
    proportion = [0.0,0.0]
    for col in cols:
        if col != "\n":
            col_info = col.split("\n")
            if len(col_info) == 5:
                col_sentence = col_info[0].lower()
                col_aspect = col_info[1][len("<aspect>"):len("</aspect>")].lower()
                col_polarity_expression = col_info[2][len("<polarity-expression>"):len("</polarity-expression>")].lower()
                col_polarity = col_info[3][len("<polarity>"):len("</polarity>")].lower()
                if col_polarity == "p":
                    proportion[0] += 1
                else:
                    proportion[1] += 1
                col_other_polarity_word = col[4][len("<other-polarity-word>"):len("</other-polarity-word>")].lower()
                if col_sentence not in norm_contents:
                    norm_contents[col_sentence] = []
                elif len(norm_contents[col_sentence])>0:
                    continue
                norm_contents[col_sentence].append(col_aspect)
                norm_contents[col_sentence].append(col_polarity_expression)
                norm_contents[col_sentence].append(col_polarity)
                norm_contents[col_sentence].append(col_other_polarity_word)
                polarities.append(col_polarity)
                sentences.append(col_sentence)
    print("there are " + str(len(sentences)) +" sentences to process..")
    return norm_contents,sentences,polarities,proportion

def phone_dir_read(directory):
    """
    :param directory:
    :return:
    """
    sentences = []
    polarities = []
    proportion = [0.0,0.0]
    for file in os.listdir(directory):
        file = os.path.join(directory,file)
        with open(file,"r") as f:
            lines = f.readlines()
            print("reading data from "+file)
            for line in lines:
                line = line.strip().lower()
                info = line.split("   ")
                if len(info) == 2:
                    polarity = info[0]
                    sentence = info[1]
                    if polarity == "差评":
                        polarity = "n"
                        proportion[0] += 1
                    else:
                        polarity = "p"
                        proportion[1] += 1
                    sentences.append(sentence)
                    polarities.append(polarity)
            print(file,"finished!",len(lines))
    return sentences,polarities,proportion
"""
考虑语料不均匀问题
    p = proportion[0]/(proportion[0] + proportion[1])
    n = proportion[1]/(proportion[0] + proportion[1])
"""
def get_feature(sentences):
    """
    :param sentences:[sentence1,sentence2,..]
    :return: [[w1,w2,..],..],[[pos1,pos2,..],..],[[parse1,parse2,..],..]
    """
    words_list = segment_str(sentences,lexicon=args.lexicon)
    pos_list = postag_str(words_list)
    parse_list = parse_str(words_list,pos_list)
    # print(words_list[4])
    # print(parse_list[4])
    return words_list,pos_list,parse_list

def determining_candidate(words_list,pos_list,parse_list,polarities):
    positive_noun = []
    negative_noun = []
    statistic_for_words = {}
    for (words,poses,parses,polarity) in zip(words_list,pos_list,parse_list,polarities):
        for (word,pos) in zip(words,poses):
            if pos == "n":
                if word not in statistic_for_words:
                    statistic_for_words[word] = [0.0,0.0]
                if polarity == "p":
                    statistic_for_words[word][0] += 1
                if polarity == "n":
                    statistic_for_words[word][1] += 1
    for k,v in statistic_for_words.items():
        n = v[0] + v[1]
        if n > 10:
            if v[0]>v[1]:
                p = v[0]/n
                p0 = args.p0+0.5
                up = p-p0
                down = cmath.sqrt(p0*(1-p0)/n)
                Z = up/down
                Z = Z.real
                if Z >= -1.64:
                    positive_noun.append(k)
            else:
                p = v[1]/n
                p0 = args.p0
                up = p - p0
                down = cmath.sqrt(p0*(1 - p0)/n)
                Z = up/down
                Z = Z.real
                if Z >= -1.64:
                    negative_noun.append(k)
    return positive_noun,negative_noun

def att_to_end(id,parses):
    """
    找到定中结构的尾
    :param id:
    :param parses:
    :return:
    """
    dep_id = parses[id][0]-1
    dep_typ = parses[id][1]
    if dep_typ != "ATT":
        return id
    else:
        return att_to_end(dep_id,parses)

def pruning(positive_noun,negative_noun,words_list,pos_list,parse_list):
    """
    对候选词表进行裁剪
    :param positive_noun:
    :param negative_noun:
    :param words_list:
    :param pos_list:
    :param parse_list:
    :return:
    """
    after_pruning_positive = []
    after_pruning_negative = []
    del_words = []
    for (words,poses,parses) in zip(words_list,pos_list,parse_list):
        for i,word in enumerate(words):
            if word in positive_noun or word in negative_noun:
                id = att_to_end(i,parses)
                if parses[id][1] == "SBV":
                    SBV_word = words[parses[id][0]-1]
                    if poses[parses[id][0]-1] == "a":
                        print("形容词是 "+ SBV_word+" 删除 "+word)
                        # print(words)
                        # print(poses)
                        # print(parses)
                        del_words.append(word)
                    elif poses[parses[id][0]-1] == "v" and parses[parses[id][0]-1][1] == "ATT":
                        adj_id = parses[parses[id][0]-1][0]-1
                        adj_word = words[parses[parses[id][0]-1][0]-1]
                        if poses[adj_id] == "a":
                            print("形容词是 " + adj_word + " 删除 " + word)
                            # print(words)
                            # print(poses)
                            # print(parses)
                            del_words.append(words)
    for noun in positive_noun:
        if noun not in del_words:
            after_pruning_positive.append(noun)
    for noun in negative_noun:
        if noun not in del_words:
            after_pruning_negative.append(noun)
    return after_pruning_positive,after_pruning_negative

def file_write(positive,negative,type="pickle"):
    if type == "pickle":
        with open("../../data/pickle_data/positive.pkl","wb") as f:
            pickle.dump(positive,f)
        with open("../../data/pickle_data/negative.pkl", "wb") as f:
            pickle.dump(negative,f)
    if type == "txt":
        with open("../../data/pickle_data/positive.txt","w") as f:
            for noun in positive:
                f.write(noun+"\n")
        with open("../../data/pickle_data/negative.txt", "w") as f:
            for noun in negative:
                f.write(noun+"\n")

def main():
    if args.rf == "../../data/original_data/vvyxhqc.xlsx":
        _,sentences,polarities,proportion = vvyxhqc_xlsx_read(args.rf)
    if args.rf == "../../data/original_data/phone/":
        sentences, polarities,proportion = phone_dir_read(args.rf)
    words_list,pos_list,parse_list = get_feature(sentences)
    positive_noun,negative_noun = determining_candidate(words_list,pos_list,parse_list,polarities)
    positive_noun,negative_noun = pruning(positive_noun,negative_noun,words_list,pos_list,parse_list)
    file_write(positive_noun,negative_noun,args.type)


if __name__ == "__main__":
    """
       使用已经标注好极性的语料抽取表意名词
       规则：
       1、候选确定：判断每个名词特征的上下文情感，若一个名词特征更多地处于积极情感，则该名词特征是积极的，反之亦然。统计测试保留可靠的词。
       得到候选积极词表以及候选消极词表
       2、裁剪规则：如果一个名词在句子中即被积极形容词修饰又被消极形容词修饰，则不可能是表意名词。
    """
    parse = argparse.ArgumentParser(description="args for corpus building")
    parse.add_argument("--rf",type=str,default="../../data/original_data/vvyxhqc.xlsx",
                       help="from where to read original corpus")
    parse.add_argument("--p0",type=float,default=0.8,
                       help="hypothesized value")
    parse.add_argument("--cf",type=float,default=0.95,
                       help="statistical confidence level")
    parse.add_argument("--type",type=str,default="txt",
                       help="how to save the lexicon")
    parse.add_argument("--lexicon",type=str,default="../../data/seg_lexicon/special_vvyxhqc.txt",
                       help="the lexicon for segment")
    args = parse.parse_args()
    main()
