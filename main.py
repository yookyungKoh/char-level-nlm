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
    #parser.add_argument('--filename', type=str, default='train.txt', help='dataset name')
    
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
    train(args)

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

def detach(states):
    return [state.detach() for state in states]

def train(args):
    
    model = Model(args)
    if args.cuda:
        model.cuda()

    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    # for train
    data_tr, trX, trY = get_batch('train.txt', args)
    train_iter = trX.size(1) // args.seq_len
    _, word_index, vocab_tr = word_embedding('train.txt')

    # Training
    start_process = time.time()
    for epoch in tqdm(range(1, args.epoch+1)):
        avg_loss = 0
        states = (Variable(torch.zeros(args.num_lstm_layer, args.batch_size, args.hidden_dim)).cuda(),
                Variable(torch.zeros(args.num_lstm_layer, args.batch_size, args.hidden_dim)).cuda())

        for i in range(train_iter):
            
            # Step 1. Prepare inputs --> batches of trX, trY
            inputs = trX[:, i:i+args.seq_len, :]
            targets = Variable(trY[:, (i+1):(i+1)+args.seq_len].contiguous().view(-1)).cuda()

            # Step 2. Before passing in a new instance, zero_grad
            model.zero_grad()
            
            # Step 3. Run the forward pass
            states = detach(states)
            train_out, states = model(inputs, states)
            
            # Step 4. Compute loss function (with Variable)
            loss = loss_function(train_out.view(-1,len(vocab_tr)), targets)
            
            # Step 5. Backpropagate and update the gradient
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.constraint)
            optimizer.step()
            
            avg_loss += loss.data[0]
            
        avg_loss /= train_iter
        PPL = np.exp(avg_loss)

        print("epoch: {}, loss: {}, PPL: {}, lr: {}".format(epoch, avg_loss, PPL, args.lr))
        torch.save(model.state_dict(), '%s/checkpoint.pth' %(args.checkpoint))
 
    end_process = time.time()
    print("Train completed in: {}".format(int(end_process - start_process), "sec"))    
    
    torch.save(model.state_dict(), '%s/checkpoint.pth' % (args.checkpoint))

def valid(args):
    # for validation
    model.eval()
    valid_loss = 0
    ppl_valid = []
    
    for epoch in range(1, args.epoch+1):
        
        _, index_val, vocab_val = word_embedding('valid.txt')
        data_val, valX, valY = get_batch('valid.txt', args.batch_size, args.seq_len)
        val_input = valX[:, i*args.seq_len:(i+1)*args.seq_len, :]
        val_target = Variable(valY[:, i*args.seq_len:(i+1)*args.seq_len]).contiguous().view(-1).cuda()
        
        valid_out, states = model(val_input, states)
        valid_loss += nn.NLLLoss(F.softmax(valid_out).view(-1, len(vocab_val)), val_target)
        ppl_ = torch.exp(valid_loss / len(valX))
        ppl_valid.append(ppl_)
        
    print(valid_loss, ppl_valid)
 
def test(args):       
    # for test
    model.eval()
    test_loss = 0
    ppl_test= []
    
    for epoch in range(1, args.epoch+1):
        
        _, index_te, vocab_te = word_embedding('test.txt')
        data_te, teX, teY= get_batch('test.txt', args.batch_size, args.seq_len)
        test_input = teX[:, i*args.seq_len:(i+1)*args.seq_len, :]
        test_target = Variable(teY[:, i*args.seq_len:(i+1)*args.seq_len]).contiguous().view(-1).cuda()
        
        test_out, states = model(test_input, states)
        test_loss += nn.NLLLoss(F.softmax(test_out).view(-1, len(vocab_te)), test_target)       
        ppl = torch.exp(test_loss / len(teX))
        ppl_test.append(ppl)        
        
    print(test_loss, ppl_test)
    
if __name__ == '__main__':
    main()
    
