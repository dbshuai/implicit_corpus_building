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
# def collate_fn(batch):
#     sen = []
#     label = []
#     seq_lens = []
#     mask = []
#     for key in batch:
#         sen.append(key[0])
#         label.append(key[1])
#         seq_lens.append(key[2])
#         mask.append(key[3])
#     return sen, label, seq_lens, mask

def var_batch(args,features,labels,seq_lens,masks):
    # print(features.type())
    # features = features.long()
    # seq_lens = seq_lens.long()
    # masks = masks.long()
    # print(features.type())
    # seq_lens.requires_grad = True
    # masks.requires_grad = True
    #features.requires_grad = True
    # features = features.float()
    if args.cuda:
        features = features.cuda()
        seq_lens = seq_lens.cuda()
        masks = masks.cuda()
        labels = labels.cuda()
    return features,labels,seq_lens,masks

def train_epoch(args,model,optimizer,criterion,dataset_loader):
    model.train()
    total_loss = 0
    # for param in model.parameters():
    #     print(param.requires_grad)
    for step, (features, labels, seq_lens, masks) in enumerate(dataset_loader):
        features,labels,seq_lens,masks = var_batch(args,features,labels,seq_lens,masks)
        model.zero_grad()
        # features.requires_grad = False
        pred,att_w = model(features,seq_lens,masks)
        loss = criterion(pred.view(len(labels),-1),labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss

def test(args, model, dataset_loader, name="test"):
    model.eval()
    accuracy = None
    for step, (features, labels, seq_lens, masks) in enumerate(dataset_loader):
        tic = time.time()
        features, labels, seq_lens, masks = var_batch(args, features, labels, seq_lens, masks)
        pred = model(features, seq_lens, masks)
        tit = time.time() - tic
        print("  Predicting {:d} examples using {:5.4f} seconds".format(len(features), tit))
        _, pred = torch.max(pred[0], dim=1)
        if args.cuda:
            pred = pred.view(-1).cpu().numpy()
        else:
            pred = pred.view(-1).numpy()
        labels = labels.cpu().numpy()
        ''' log and return prf scores '''
        accuracy = model_utils.test_prf(pred, labels,num_class=2,name=name)
    return accuracy

def main(args):
    # define location to save the model
    if args.save == "__":
        # LSTM_100_40_8
        args.save = "../save_model/%s_%d_%d_%d" % \
                    (args.model,args.nhid, args.sen_max_len, args.batch_size)
    ''' make sure the folder to save models exist '''
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    in_dir = "../../data/pickle_data/"
    dataset = model_utils.pickle2dict(in_dir + "features_glove.pkl")
    embeddings = model_utils.pickle2dict(in_dir + "embeddings_glove.pkl")
    emb_np = np.asarray(embeddings, dtype=np.float32)  # from_numpy
    emb = torch.from_numpy(emb_np)
    model = BLSTM(embeddings=emb,
                  input_dim=args.embsize,
                  hidden_dim=args.nhid,
                  num_layers=args.nlayers,
                  output_dim=2,
                  max_len=args.sen_max_len,
                  dropout=args.dropout,)
    if torch.cuda.is_available():
        if not args.cuda:
            print("Waring: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
            model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    training_set = dataset["training"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]
    training_batch = args.batch_size
    validation_batch = len(validation_set["xIndexes"])
    test_batch = len(test_set["xIndexes"])
    training_set = model_utils.Dataset(training_set["xIndexes"],
                                       training_set["yLabels"],
                                       to_pad=True,
                                       max_len=args.sen_max_len)
    validation_set=model_utils.Dataset(validation_set["xIndexes"],
                                        validation_set["yLabels"],
                                        to_pad=True,
                                        max_len=args.sen_max_len)
    test_set = model_utils.Dataset(test_set["xIndexes"],
                                    test_set["yLabels"],
                                    to_pad=True,
                                    max_len=args.sen_max_len)
    training_set_loader = DataLoader(dataset=training_set,
                                     batch_size=training_batch,
                                     shuffle=True,)

    validation_set_loader = DataLoader(dataset=validation_set,
                                       batch_size=validation_batch,
                                       shuffle=False)
    test_set_loader = DataLoader(dataset=test_set,
                                    batch_size=test_batch,
                                    shuffle=False)
    best_acc_test, best_acc_valid = -np.inf, -np.inf
    tic = time.time()
    print("-----------------------------",args.epochs, len(training_set), args.batch_size)
    for i in range(args.epochs):
        print("--------------\nEpoch %d begins!"%(i))
        loss = train_epoch(args, model, optimizer, criterion, training_set_loader)
        print("loss = ",loss)
        print("  using %.5f seconds" % (time.time() - tic))
        tic = time.time()
        print("\n  Begin to predict the results on Validation")
        acc_score = test(args, model, validation_set_loader, name="validation")
        print("  ----Old best acc score on validation is %f" % best_acc_valid)
        if acc_score > best_acc_valid:
            print("  ----New acc score on validation is %f" % acc_score)
            best_acc_valid = acc_score
            with open(args.save + "/model.pt", 'wb') as to_save:
                torch.save(model, to_save)
            acc_test = test(args, model, test_set_loader, name="test")
            print("  ----Old best acc score on test is %f" % best_acc_test)
            if acc_test > best_acc_test:
                best_acc_test = acc_test
                print("  ----New acc score on test is %f" % acc_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch for 30suo project")

    ''' load data and save model'''
    parser.add_argument("--save", type=str, default="../save_model/",
                        help="path to save the model")

    ''' model parameters '''
    parser.add_argument("--model", type=str, default="__",
                        help="name of model")
    parser.add_argument("--embsize", type=int, default=100,
                        help="size of word embeddings")
    parser.add_argument("--emb", type=str, default="glove",
                        help="type of word embeddings")
    parser.add_argument("--nhid", type=int, default=50,
                        help="size of RNN hidden layer")
    parser.add_argument("--nlayers", type=int, default=1,
                        help="number of layers of LSTM")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epoch")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--dropout", type=float, default=0.75,
                        help="dropout rate")
    parser.add_argument("--seed", type=int, default=123456,
                        help="random seed for reproduction")
    parser.add_argument("--cuda", action="store_true",
                        help="use CUDA")

    parser.add_argument("--sen_max_len", type=int, default=60,
                        help="max time step of tweet sequence")
    # ''' test purpose'''
    # parser.add_argument("--is_test", action="store_true",
    #                     help="flag for training model or only test")

    my_args = parser.parse_args()
    torch.manual_seed(my_args.seed)
    np.random.seed(my_args.seed)
    main(my_args)
