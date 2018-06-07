import torch
import torch.nn as nn
import unicodedata
import re
import time
import math
import random
import os
import csv

from torch.autograd import Variable
from torch import optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


'''
*****************************
read data
*****************************
'''

label = list()
sent1 = list()
sent2 = list()

with open('/home/dw1215/PycharmProjects/first_pytorch/data/stsbenchmark/sts-test.csv', 'r') as file:
    sts_data = csv.reader(file, delimiter='\t')
    for s in sts_data:
        label.append(s[4])
        sent1.append(s[5])
        sent2.append(s[6])

use_cuda = torch.cuda.is_available()

MAX_LENGTH = 10
hidden_size = 300

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

GLOVE_DIR = '/media/dw1215/Di_files/data/glove.6B'
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    s = s[:-1]

    return s


def filterPair(p):
    return 2 < len(p.split(' ')) < MAX_LENGTH


def filterPairs(sentences):
    return [sentence for sentence in sentences if filterPair(sentence)]


# input_sentences_class, input_sentences = prepareData('eng', 'fra')
# print(random.choice(input_sentences))

sent1 = [normalize_string(l) for l in sent1]
sent2 = [normalize_string(l) for l in sent2]
all_sent = sent1 + sent2
all_class = Lang('all')
for sentence in sent1:
    all_class.addSentence(sentence)
for sentence in sent2:
    all_class.addSentence(sentence)
sent1_class = Lang('sent1')
for sentence in sent1:
    sent1_class.addSentence(sentence)
sent2_class = Lang('sent2')
for sentence in sent2:
    sent2_class.addSentence(sentence)

embedding_matrix = np.zeros((all_class.num_words, hidden_size))
for word, i in all_class.word2index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

embedding = nn.Embedding(all_class.num_words, hidden_size)
embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
embedding = embedding.cuda()

training_sentences = [variableFromSentence(all_class, all_sent[k]) for k in
                      range(len(all_sent))]

file = open('/home/dw1215/PycharmProjects/first_pytorch/data/encoded_average_embedding', 'w')

for iter in range(len(all_sent)):
    training_sentence = training_sentences[iter]
    input_variable = training_sentence

    input_length = input_variable.size()[0]

    input_variable_embedded = []
    for i in range(input_length):
        word_embedded = embedding(input_variable[i]).view(1, 1, -1)
        input_variable_embedded.append(word_embedded)
    input_variable_embedded = torch.cat(input_variable_embedded, 1)

    sentence_average_mean = torch.mean(input_variable_embedded, 1)
    current_representation = sentence_average_mean.view(1, 1, -1)

    current_sentence = []
    for w in input_variable.data.cpu().numpy():
        current_sentence.append(all_class.index2word[int(w)])

    file.write(' '.join(current_sentence))
    file.write(str(current_representation.data.cpu().numpy()) + '\n')

