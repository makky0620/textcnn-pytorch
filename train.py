# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchtext import data, datasets
import numpy as np
from pprint import pprint

from model import TextCNN

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3 words sentences (=sequence_length is 3)
    TEXT = data.Field(sequential=True, batch_first=True)
    LABEL = data.Field(sequential=False, batch_first=True)

    train = data.TabularDataset(
        path="train.tsv",
        format="tsv",
        fields=[
            ("text", TEXT),
            ("label", LABEL)
        ]
    )

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iter = data.BucketIterator(
        dataset=train,
        batch_size=3,
        repeat=False
    )

    # Text-CNN Parameter
    embedding_size = 2 # n-gram
    sequence_length = 3
    num_classes = len(LABEL.vocab)  # 0 or 1
    vocab_size = len(TEXT.vocab)
    filter_sizes = [2, 2, 2] # n-gram window
    num_filters = 3

    batch = next(iter(train_iter))
    input_batch = batch.text.to(device)
    target_batch = batch.label.to(device)

    model = TextCNN(embedding_size, sequence_length, num_classes, filter_sizes, num_filters, vocab_size).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()
