# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:01:14 2018

@author: Yookyung Koh
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import urllib.request
import nltk
import torch
import torch.nn as nn
from torch.autograd import Variable


main_path = os.getcwd()
data_path = main_path + '/data'

def word_tokenize(filename):
    """ word-level tokenize """
    
    content = open(os.path.join(data_path, filename)).read()
    token = content.split() 

    token = [w.replace('<unk>', 'A') for w in token]
    # to generate an embedded vector for <unk> token later on
    
    seen = set()
    word_dic = []
    for w in token:
        if w not in seen:
            word_dic.append(w)
            seen.add(w)
            
    return token, word_dic

def max_word_length(token):
    max_word_len = np.max([len(word) for word in token]) # 19

    return max_word_len

def char_embedding(filename):
    """ character-level tokenize """
    token, word_dic = word_tokenize(filename)

    char_list = [char for word in token for char in word]
    
    # calculate vocab size
    seen = set()
    vocab_list = []
    for x in char_list:
        if x not in seen:
            vocab_list.append(x)

            seen.add(x)

    # add [S, E] S: start-of-word, E: end-of-word
    vocab_list.append('S')
    vocab_list.append('E')
    vocab_list.append(' ')
    vocab_list.sort()
    
    """ character embedding """
    char_to_idx = {char: i for i, char in enumerate(vocab_list)} # character dictionary
    
    return char_to_idx

def word_embedding(filename):
    token, word_dic = word_tokenize(filename)
    max_word_len = max_word_length(token)
    char_to_idx = char_embedding('train.txt')       
    
    '''<sow> + <eow> + padding'''
    for i in range(len(token)):
        token[i] = 'S' + token[i] + 'E'
        # S = start-of-word, E = end-of-word   
        l = len(token[i])
        if l < max_word_len + 2:
            token[i] = token[i] + ' '*(max_word_len+2-l)
            # zero padding so the number of columns is constant (= max_word_len)
    
    ''' word dictionary '''
    seen = set()
    word_dic = []
    for w in token:
        if w not in seen:
            word_dic.append(w)
            seen.add(w)
    
    word_to_idx = {word: w for w, word in enumerate(word_dic)} # word dictionary
    
    ''' character indexing for words'''
    word2idx = []
    for i in range(len(token)):
        idxs = []
        for j in range(len(token[i])):
            # char_tokenize for each word
            idxs.append(char_to_idx[token[i][j]])
        word2idx.append(idxs)
        
    word_index = [word_to_idx[token[i]] for i in range(len(token))]
    
    return torch.LongTensor(word2idx), torch.LongTensor(word_index), word_to_idx

def get_batch(filename, args):
    
    bsz = args.batch_size
    seq_len = args.seq_len
    max_len = args.max_word_len

    data, index, word_to_idx = word_embedding(filename)
        
    nbatch = data.size(0) // (bsz * seq_len)
    data = data[: bsz * seq_len * nbatch]
    index = index[1: bsz * seq_len * nbatch + 1]

    # data: (886900 x 21) , index: (886900) , word_to_idx: (9999)
    
    num = data.size(0) // bsz
    dataX = torch.LongTensor(bsz, num, max_len)
    dataY = torch.LongTensor(bsz, num)

    for i in range(bsz):
        dataX[i] = data[i*num : (i+1)*num]
        dataY[i] = index[i*num : (i+1)*num]

    return data, dataX, dataY



