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
import torch
from torch.autograd import Variable


main_path = os.getcwd()
data_path = main_path + '/data'

def word_embedding(filename, args):
    max_word_len = args.max_word_len
    
    """ word tokenize """    
    content = open(os.path.join(data_path, filename)).read()
    token = content.split() 

    word_list = [w.replace('<unk>', 'A') for w in token]
    # to generate an embedded vector for <unk> token later on
    
    """ character tokenize """
    char_list = [char for word in word_list for char in word]
    
    seen_c = set()
    char_dict = []
    for x in char_list:
        if x not in seen_c:
            char_dict.append(x)
            seen_c.add(x)

    # add [S, E] S: start-of-word, E: end-of-word
    char_dict.append('S')
    char_dict.append('E')
    char_dict.append(' ')
    char_dict.sort()
    
    # character dictionary
    char_to_idx = {char: i for i, char in enumerate(char_dict)} 

    '''<sow> + <eow> + padding'''
    for i in range(len(word_list)):
        word_list[i] = 'S' + word_list[i] + 'E'
        # S = start-of-word, E = end-of-word   
        l = len(word_list[i])
        if l < max_word_len:
            word_list[i] = word_list[i] + ' '*(max_word_len-l)
            # zero padding so the number of columns is constant (= max_word_len)
    
    # word to character idx
    word2idx = []
    for i in range(len(word_list)):
        idxs = []
        for j in range(len(word_list[i])):
            # char_tokenize for each word
            idxs.append(char_to_idx[word_list[i][j]])
        word2idx.append(idxs)
    
    seen_w = set()
    word_dict = []
    for w in word_list:
        if w not in seen_w:
            word_dict.append(w)
            seen_w.add(w)
            
    # word dictionary
    word_to_idx = {word: w for w, word in enumerate(word_dict)}   
        
    word_index = [word_to_idx[word_list[i]] for i in range(len(word_list))]

    return torch.LongTensor(word2idx), torch.LongTensor(word_index), word_dict

def max_word_length(token):
    max_word_len = np.max([len(word) for word in token]) # 19
    return max_word_len

def get_batch(filename, args):
    
    bsz = args.batch_size
    seq_len = args.seq_len
    max_len = args.max_word_len

    data, index, _ = word_embedding(filename,args)
        
    nbatch = data.size(0) // (bsz * seq_len)
    data = data[: bsz * seq_len * nbatch]
    index = index[1: bsz * seq_len * nbatch + 1]

    # data: (886900 x 21) , index: (886900) , word_dict: (9999)
    
    num = data.size(0) // bsz
    dataX = torch.LongTensor(bsz, num, max_len)
    dataY = torch.LongTensor(bsz, num)

    for i in range(bsz):
        dataX[i] = data[i*num : (i+1)*num]
        dataY[i] = index[i*num : (i+1)*num]
    
    return data, dataX, dataY



