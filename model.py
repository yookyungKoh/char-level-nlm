# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:50:39 2018

@author: Yookyung Koh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import *

class Model(nn.Module):   
    
    # CNN
  
    def __init__(self, args):
        super(Model, self).__init__()
                
        # parameters
        self.kernel_w = args.kernel_w
        self.num_feature = args.num_feature
        self.embed_dim = args.embed_dim
        self.bias = args.bias
        self.max_word_len = args.max_word_len
        self.vocab_size = args.vocab_size
        
        # Embedding
        self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim).cuda()
        
        # CNN
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_feature[idx], (self.kernel_w[idx], self.embed_dim)) for idx in range(6)])
            
        # Highway
        self.num_layer = args.num_highway_layer
        self.bias = args.bias
        
        size = sum(args.num_feature)    
        self.nonlinear = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
        self.gate.bias.data.fill_(self.bias)
        self.linear_hw = nn.Linear(size, size)
    
        # RNN
    
        self.num_lstm_layer = args.num_lstm_layer
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.embed_dim
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        
        _, _, data = word_embedding('train.txt')
        self.word_size = len(data) # word dictionary size
        self.lstm = nn.LSTM(size, self.hidden_dim, self.num_lstm_layer, dropout=args.dropout)
        self.linear_rnn = nn.Linear(self.hidden_dim, self.word_size)
        self.dropout = nn.Dropout(args.dropout)
        
        self.init_weights()

    def init_weights(self):
        self.embeddings.weight.data.uniform_(-0.05, 0.05)
        self.linear_hw.weight.data.uniform_(-0.05, 0.05)
        self.linear_rnn.weight.data.uniform_(-0.05, 0.05)
        self.nonlinear.weight.data.uniform_(-0.05, 0.05)
        self.gate.weight.data.uniform_(-0.05, 0.05)

    def forward(self, inputs, h):
        
        inputs = inputs.contiguous().view(-1, self.max_word_len) # [(20x35) x 21]
        x = self.embeddings(Variable(inputs).cuda()) # [(20x35) x 21 x 15]
        
        x = x.unsqueeze(1) # [(20x35) x 1 x 21 x 15]
        
        layers = []
        for i, c in enumerate(self.convs):
            out = [F.tanh(c(x).squeeze())]
            out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]        
            layers.append(out)
            
        cnnout = [torch.cat(i, 1) for i in layers]
        cnnout = torch.cat(cnnout, 1)

        # Highway
        """ 
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        g: non-linearity (ReLU), t: transform gate, (1 - t): carry gate
        """
        
        gate = F.sigmoid(self.gate(cnnout))
        nonlinear = F.relu(self.nonlinear(cnnout))
        linear = self.linear_hw(cnnout)

#        for layer in range(self.num_layer):
#            gate = F.sigmoid(self.gate[layer](x))
#            nonlinear = F.relu(self.nonlinear[layer](x))
#            linear = self.linear[layer](x)
        highwayout = gate * nonlinear + (1 - gate) * linear
    
    # LSTM
        x = highwayout.view([self.seq_len, self.batch_size, -1]) # [35 x 20 x 525]
        
        # Forward propagate RNN
        out, h = self.lstm(x, h)
        out = out.view(self.batch_size, self.seq_len, self.hidden_dim)
        # out size: [20 x 35 x 300]

        # Decode hidden state of last time step
        out = self.dropout(out)
        logit = self.linear_rnn(out) #[20 x 35 x word_vocab_size]
#        out = F.softmax(out, dim=0)

        return logit, h

       
