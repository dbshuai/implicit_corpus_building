#!/usr/bin/env python
# coding:utf8
import sys
sys.path.append("./")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from model.attentionLayer import  AttentionLayer

torch.manual_seed(123456)


class BLSTM(nn.Module):
    """
        Implementation of BLSTM Concatenation for sentiment classification task
    """

    def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
        super(BLSTM, self).__init__()

        self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
                                embedding_dim=embeddings.size(1),
                                padding_idx=0)
        self.emb.weight = nn.Parameter(embeddings)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # sen encoder
        self.sen_len = max_len
        self.sen_rnn = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)
        self.attention = AttentionLayer(self.hidden_dim * 2)

        self.output = nn.Linear(2 * self.hidden_dim, output_dim)

    def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
        """

        :param sen_batch: (batch, sen_length), tensor for sentence sequence
        :param sen_lengths:
        :param sen_mask_matrix:
        :return:
        """

        ''' 
        Embedding Layer | Padding | Sequence_length 40
        '''
        sen_lengths_sort,sort = torch.sort(sen_lengths,dim=0,descending = True)
        _,unsort = torch.sort(sort,dim=0)
        batch_size = len(sen_batch)
        sen_batch = self.emb(sen_batch)
        sen_batch = torch.index_select(sen_batch,dim=0,index=sort)
        sen_batch = utils.rnn.pack_padded_sequence(sen_batch,sen_lengths_sort,batch_first=True)
        ''' Bi-LSTM Computation '''
        sen_outs,(hn,cn) = self.sen_rnn(sen_batch)
        # sen_outs = utils.rnn.pad_packed_sequence(sen_outs,batch_first=True) #batch_size,max_len,hid_dim*2
        # sen_outs = torch.index_select(sen_outs,dim=0,index=unsort)
        # sen_rnn = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, sen_len, 2*hid)

        ''' 
        Fetch the truly hidden layer of both sides
        '''
        sen_outs = utils.rnn.pad_packed_sequence(sen_outs,batch_first=True)
        sen_outs = torch.index_select(sen_outs[0],dim=0,index=unsort) #batch_size,sen_len,hidden_dim*2
        sen_outs = torch.tanh(sen_outs)
        outs,att_w = self.attention(sen_outs)
        res = self.output(outs)
        out_prob = F.softmax(res,dim=1)
        return out_prob,att_w
