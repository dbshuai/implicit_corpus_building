import sys
sys.path.append("../")
sys.path.append("../../")
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import yutils.model_utils as model_utils
import os
import time
from model.blstm import BLSTM
def get_implicite_expression(plabels,features,attw,labels):
    expressions = {}
    idx_to_word = model_utils.pickle2dict("../../data/pickle_data/idx2word_glove.pkl")
    for plabel,feature,w,label in zip(plabels,features,attw,labels):
        if plabel == label:
            expression = ""
            last = -1
            for i,weight in enumerate(w):
                if weight>0.15:
                    word = idx_to_word[feature[i]]
                    if "_" not in word and "unk" not in word:
                        if i == last + 1:
                            expression += word
                            last = i
                        else:
                            expression += " " + word
                            last = i
            if len(expression)>0 and expression not in expressions:
               expressions[expression] = label
    return expressions
def save_res(expressions):
    with open("implicit_lexicon_10_p.txt","w") as f:
        for k,v in expressions.items():
            content = k + "   " + str(v) + "\n"
            f.write(content)
def var_batch(args,features,labels,seq_lens,masks):
    if args.cuda:
        features = features.cuda()
        seq_lens = seq_lens.cuda()
        masks = masks.cuda()
        labels = labels.cuda()
    return features,labels,seq_lens,masks

def position_select(args,model,dataloader):
    for step, (features, labels, seq_lens, masks) in enumerate(dataloader):
        features, labels, seq_lens, masks = var_batch(args, features, labels, seq_lens, masks)
        pred,attw = model(features, seq_lens, masks)
        # print(pred.size())
        # print(attw.size())
        # exit(0)
        _,plabels = torch.max(pred,dim=1)
        if args.cuda == True:
            plabels = plabels.cpu().numpy().tolist()
            attw = attw.cpu().detach().numpy().tolist()
            features = features.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
        else:
            plabels = plabels.numpy().tolist()
            attw = attw.detach().numpy().tolist()
            features = features.numpy().tolist()
            labels = labels.numpy().tolist()
    return plabels,features,attw,labels

def main(args):
    if args.cuda:
        model = torch.load(args.model, map_location='gpu0')
    else:
        model = torch.load(args.model,map_location='cpu')
    in_dir = "../../data/pickle_data/"
    dataset = model_utils.pickle2dict(in_dir + "features_glove.pkl")
    if torch.cuda.is_available():
        if not args.cuda:
            print("Waring: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
            model.cuda()
    else:
        args.cuda = False
    pre_set = {"xIndexes":[],"yLabels":[]}
    for name,dic in dataset.items():
        pre_set["xIndexes"].extend(dic["xIndexes"])
        pre_set["yLabels"].extend(dic["yLabels"])
    batch = len(pre_set["xIndexes"])
    pre_set = model_utils.Dataset(pre_set["xIndexes"],
                              pre_set["yLabels"],
                              to_pad=True,
                              max_len=args.sen_max_len)
    set_loader = DataLoader(dataset=pre_set,batch_size=batch,shuffle=False)
    plabels,features,attW,labels = position_select(args,model,set_loader)
    expressions = get_implicite_expression(plabels,features,attW,labels)
    save_res(expressions)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments for lexicon building!")
    parser.add_argument("--model",type=str,default="../save_model/model.pt")
    parser.add_argument("--cuda",type=bool,default=False)
    parser.add_argument("--sen_max_len", type=int, default=60)
    args = parser.parse_args()
    main(args)