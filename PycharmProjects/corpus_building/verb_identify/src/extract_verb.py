from yutils.str_utils import  segment_str,parse_str,postag_str
def read_file(path):
    sentences = []
    polaritys = []
    with open(path,"r") as f:
        lines = f.readlines()
        for line in lines:
                line = line.split("   ")
                if len(line)==2:
                    sentence = line[1].strip()
                    polarity = line[0]
                    if len(sentence)>0 and len(polarity)>0:
                        sentences.append(sentence)
                        polaritys.append(polarity)
    return sentences,polaritys
def action(sentence_list):
    actions = []
    indexes = []
    seg_list = segment_str(sentence_list)
    pos_list = postag_str(seg_list)
    parse_list = parse_str(seg_list,pos_list)
    assert len(seg_list)==len(pos_list)==len(parse_list)
    for words,postags,parses in zip(seg_list,pos_list,parse_list):
        # print(words)
        # print(postags)
        # print(parses)
        # exit(0)
        cur = 0



    return actions,indexes
if __name__ ==  "__main__":
    path = "../../data/original_data/phone/phone_2.txt"
    sentences,polaritys = read_file(path)
    chunk_list,index_list = action(sentences)