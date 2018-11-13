import torch.nn as nn
import torch
class AttentionLayer(nn.Module):
    def __init__(self,layer_size):
        super(AttentionLayer,self).__init__()
        self.att = nn.Sequential(
            nn.Linear(layer_size,layer_size,bias=False),
            nn.Tanh(),
            nn.Linear(layer_size,1)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input):
        wM = self.att(input).squeeze(2) #batch,max_len,1 - > batch,max_len
        att_w = self.softmax(wM)
        a = att_w.unsqueeze(2).expand_as(input)
        r = input*a
        r = torch.sum(r,1)
        return r,att_w