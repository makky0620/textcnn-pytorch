# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

dtype = torch.FloatTensor


class TextCNN(nn.Module):
    def __init__(self, embedding_size, sequence_length, num_classes, filter_sizes, num_filters, vocab_size):
        super(TextCNN, self).__init__()
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.vocab_size = vocab_size

        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = nn.Parameter(torch.empty(vocab_size, embedding_size).uniform_(-1, 1)).type(dtype)
        self.Weight = nn.Parameter(torch.empty(self.num_filters_total, num_classes).uniform_(-1, 1)).type(dtype)
        self.Bias = nn.Parameter(0.1 * torch.ones([num_classes])).type(dtype)
        self.conv1 = nn.Conv2d(1, self.num_filters, (2, self.embedding_size), bias=True)
        self.embed = nn.Embedding(vocab_size, embedding_size)

    def forward(self, X):
        embedded_chars = self.embed(X).unsqueeze(1)
        pooled_outputs = []
        for filter_size in self.filter_sizes:
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            conv = self.conv1(embedded_chars) 
            h = F.relu(conv)
            # mp : ((filter_height, filter_width))
            mp = nn.MaxPool2d((self.sequence_length - filter_size + 1, 1))
            
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes)) # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        model = torch.mm(h_pool_flat, self.Weight) + self.Bias # [batch_size, num_classes]
        return model