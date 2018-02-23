import os
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from dataset import *
from model import *

def main():
    
    parser = argparse.ArgumentParser(description='Character-Aware NLM')
    
    parser.add_argument('--data', type=str, default='./data', help='location of data')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='save model dir')
    parser.add_argument('--valid_dir', type=str, default='./valid_dir', help='valid model dir')
    parser.add_argument('--test_dir', type=str, default='./test_dir', help='test result dir')
    parser.add_argument('--filename', type=str, default='train.txt', help='dataset name')
    
    # CNN
    parser.add_argument('--vocab_size', type=int, default=50, help='character vocab size')
    parser.add_argument('--max_word_len', type=int, default=21, help='maximum word length in dataset')
    parser.add_argument('--embed_dim', type=int, default=15, help='dimensionality of character embedding')
    parser.add_argument('--kernel_w', type=list, default=[1,2,3,4,5,6], help='kernel width')
    parser.add_argument('--num_feature', type=list, default=[25,50,75,100,125,150], help='number of features')
    
    # Highway
    parser.add_argument('--num_highway_layer', type=int, default=1, help='number of layers in Highway network')
    parser.add_argument('--bias', type=float, default=-2.0, help='bias')
    
    # RNN_LSTM
    parser.add_argument('--num_lstm_layer', type=int, default=2, help='number of layers in RNN')
    parser.add_argument('--hidden_dim', type=int, default=300, help='hidden dimension in RNN')
    
    # Optimization
    parser.add_argument('--seq_len', type=int, default=35, help='time steps for backpropagation')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--epoch', type=int, default=25, help='number of training epoch')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--constraint', type=float, default=5.0, help='L2 norm constraint')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    if not os.path.exists(args.valid_dir):
        os.makedirs(args.valid_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    train(args)
    #test(args)

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

def detach(states):
    return [state.detach() for state in states]

def train(args):
    # for train
    data_tr, trX, trY = get_batch('train.txt', args)
    train_iter = trX.size(1) // args.seq_len
    _, word_index, vocab_tr = word_embedding('train.txt', args)

    model = Model(args, len(vocab_tr)).cuda()
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    # Training
    start_process = time.time()
    for epoch in tqdm(range(1, args.epoch+1)):
        
        avg_loss = 0
        states = (Variable(torch.zeros(args.num_lstm_layer, args.batch_size, args.hidden_dim)).cuda(),
                Variable(torch.zeros(args.num_lstm_layer, args.batch_size, args.hidden_dim)).cuda())

        for i in range(train_iter):
            inputs = trX[:, i:i+args.seq_len, :]
            targets = Variable(trY[:, (i+1):(i+1)+args.seq_len].contiguous().view(-1)).cuda()
            model.zero_grad()
            states = detach(states)
            train_out, states = model(inputs, states)
            loss = loss_function(train_out.view(-1,len(vocab_tr)), targets)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.constraint)
            optimizer.step()
            
            avg_loss += loss.data[0]/train_iter
            
        PPL = np.exp(avg_loss)

        print("epoch: {}, loss: {}, PPL: {}".format(epoch, avg_loss, PPL))
    torch.save(model, '%s/checkpoint.pth' %(args.checkpoint))
        
#        print('==========Validation==========')
#        _, index_val, vocab_val = word_embedding('valid.txt', args)
#        data_val, valX, valY = get_batch('valid.txt', args)
#        valid_iter = valX.size(1)//args.seq_len
#        with open('%s/checkpoint.pth'%(args.checkpoint), 'rb') as f:
#            model = torch.load(f)
#        model.eval()
#        val_loss = 0
#        ppl_valid = []
        
#        states = (Variable(torch.zeros(args.num_lstm_layer, args.batch_size, args.hidden_dim)).cuda(),
#                Variable(torch.zeros(args.num_lstm_layer, args.batch_size, args.hidden_dim)).cuda())
#
#        for i in range(valid_iter):
#            val_input = valX[:, i:i+args.seq_len, :]
#            val_target = Variable(valY[:, i+1:(i+1)+args.seq_len]).contiguous().view(-1).cuda()
#            
#            valid_out, states = model(val_input, states)
#            val_loss += loss_function(valid_out.view(-1, len(vocab_val)), val_target).data[0]
#        
#        val_loss /= train_iter
#        ppl_ = np.exp(val_loss)
#        ppl_valid.append(ppl_)
#        
#        print('val_loss: {}, val_PPL: {}'.format(val_loss, ppl_))
#         
#        
#    torch.save(model, '%s/model.pth'%(args.checkpoint))
#
#    end_process = time.time()
#    print("Train completed in: {}".format(int(end_process - start_process), "sec"))    

def test(args):       
    # for test
    print('==========Testing==========')
    with open('%s/model.pth'%(args.checkpoint), 'rb') as f:
        model = torch.load(f)
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    test_loss = 0
    
    _, index_te, vocab_te = word_embedding('test.txt', args)
    data_te, teX, teY= get_batch('test.txt', args)
    test_iter = teX.size(1)//args.seq_len

    states = (Variable(torch.zeros(args.num_lstm_layer, args.batch_size, args.hidden_dim)).cuda(),
            Variable(torch.zeros(args.num_lstm_layer, args.batch_size, args.hidden_dim)).cuda())
    
    for i in tqdm(range(test_iter)):
        
        test_input = teX[:, i:i+args.seq_len, :]
        test_target = Variable(teY[:, i+1:(i+1)+args.s1eq_len]).contiguous().view(-1).cuda()
        
        test_out, states = model(test_input, states)
        test_loss += loss_function(test_out.view(-1, len(vocab_te)), test_target).data[0]
        states = detach(states)
        ppl = np.exp(test_loss / test_iter)
        
        print('TEST LOSS: {}, PPL: {}'.format(test_loss, ppl))

if __name__ == '__main__':
    main()
    
