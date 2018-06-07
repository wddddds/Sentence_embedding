import torch
import torch.nn as nn
import unicodedata
import re
import time
import math
import random
import os

from torch.autograd import Variable
from torch import optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


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


class DetectorRNN(nn.Module):
    def __init__(self, hidden_size):
        super(DetectorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(2*hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, 1)

    def forward(self, word1, word2, hidden):
        word1_relu = self.W1(word1).clamp(min=0)
        word2_relu = self.W2(word2).clamp(min=0)
        word_concat = torch.cat([word1_relu, word2_relu], dim=1).view(1, 1, -1)
        output, hidden = self.gru(word_concat, hidden)
        output = self.W3(output).clamp(min=0)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class Detector(nn.Module):
    def __init__(self, hidden_size):
        super(Detector, self).__init__()
        self.hidden_size = hidden_size

        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.W3 = nn.Linear(2*hidden_size, 1)

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, word1, word2):
        word1_relu = self.W1(word1).clamp(min=0)
        word2_relu = self.W2(word2).clamp(min=0)
        combined = torch.cat([word1_relu, word2_relu], dim=1)
        output = self.W3(combined).clamp(min=0)
        return output


class CombinerRNN(nn.Module):
    def __init__(self, hidden_size):
        super(CombinerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(2*hidden_size, hidden_size)

    def forward(self, word1, word2, hidden):
        word1_relu = self.W1(word1).clamp(min=0)
        word2_relu = self.W2(word2).clamp(min=0)
        word_concat = torch.cat([word1_relu, word2_relu], dim=1).view(1, 1, -1)
        output, hidden = self.gru(word_concat, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class Combiner(nn.Module):
    def __init__(self, hidden_size):
        super(Combiner, self).__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, word1, word2):
        word_concat = torch.cat([word1, word2], dim=1)
        output = self.W(word_concat).clamp(min=0)
        return output


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size*2)

    def forward(self, word, hidden, split=False):
        word = word.view(1, 1, -1)
        if not split:
            embedding, hidden = self.gru(word, hidden)
            return hidden
        else:
            embedding, hidden = self.gru(word, hidden)
            extended_embedding = self.W3(embedding).clamp(min=0)
            cluster1 = extended_embedding[:, :, :hidden_size]
            cluster2 = extended_embedding[:, :, hidden_size:]
            cluster1 = self.W1(cluster1).clamp(min=0)
            cluster2 = self.W2(cluster2).clamp(min=0)
            return cluster1, cluster2

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(hidden_size, hidden_size*2)

    def forward(self, word_embedding):
        extended_embedding = self.W(word_embedding).clamp(min=0)
        cluster1 = extended_embedding[:, :hidden_size]
        cluster2 = extended_embedding[:, hidden_size:]
        return cluster1, cluster2


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


def read_languages(lang1, lang2):
    print('reading lines')

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    english_sentences = [[normalize_string(s) for s in l.split('\t')][0] for l in lines]

    input_ = Lang(lang1)

    return input_, english_sentences


def filterPair(p):
    return 2 < len(p.split(' ')) < MAX_LENGTH


def filterPairs(sentences):
    return [sentence for sentence in sentences if filterPair(sentence)]


def prepareData(lang1, lang2):
    data_class, input_sentences = read_languages(lang1, lang2)
    print("Read %s sentences" % len(input_sentences))
    input_sentences = filterPairs(input_sentences)
    print("Trimmed to %s sentences" % len(input_sentences))
    print("Counting words...")
    for sentence in input_sentences:
        data_class.addSentence(sentence)
    print("Counted words:")
    print(data_class.name, data_class.num_words)
    return data_class, input_sentences


input_sentences_class, input_sentences = prepareData('eng', 'fra')
print(random.choice(input_sentences))

embedding_matrix = np.zeros((input_sentences_class.num_words, hidden_size))
for word, i in input_sentences_class.word2index.items():
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


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()


def train(input_variable, embedding, detector, combiner, decoder, detector_optimizer, combiner_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    combiner_optimizer.zero_grad()
    detector_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    detector_hidden = detector.initHidden()
    combiner_hidden = combiner.initHidden()
    decoder_hidden = decoder.initHidden()

    input_length = input_variable.size()[0]

    input_variable_embedded = []
    for i in range(input_length):
        word_embedded = embedding(input_variable[i]).view(1, 1, -1)
        input_variable_embedded.append(word_embedded)
    input_variable_embedded = torch.cat(input_variable_embedded, 1)

    loss = 0

    path = []

    current_sentence = input_variable_embedded

    for i in range(input_length-1):
        scores = []

        if current_sentence.size()[1] != 2:
            for j in range(current_sentence.size()[1]-1):
                score, detector_hidden = detector(current_sentence[:, j, :], current_sentence[:, j + 1, :], detector_hidden)
                score = score.data[0, 0, 0]
                scores.append(score)
            combined_index = scores.index(max(scores))
            path.append(combined_index)
            combined_word, combiner_hidden = \
                combiner(current_sentence[:, combined_index, :], current_sentence[:, combined_index+1, :], combiner_hidden)
            combined_word = combined_word.view(1, 1, -1)

            if combined_index == 0:
                current_sentence = torch.cat([combined_word, current_sentence[:, combined_index + 2:, :]], 1)
            elif combined_index == current_sentence.size()[1]-2:
                current_sentence = torch.cat([current_sentence[:, :combined_index, :], combined_word], 1)
            else:
                current_sentence = torch.cat([current_sentence[:, :combined_index, :], combined_word,
                                              current_sentence[:, combined_index + 2:, :]], 1)
        else:
            path.append(0)
            current_sentence, combiner_hidden = combiner(current_sentence[:, 0, :], current_sentence[:, 1, :], combiner_hidden)

    current_representation = current_sentence.view(1, 1, -1)

    for i in path[::-1]:
        for j in range(int(i)):
            decoder_hidden = decoder(current_representation[:, j, :], decoder_hidden)
        word_to_be_separated = current_representation[:, i, :]
        word_in_front, word_behind = decoder(word_to_be_separated, decoder_hidden, split=True)
        word_in_front = word_in_front.view(1, 1, -1)
        word_behind = word_behind.view(1, 1, -1)
        if i == 0:
            try:
                current_representation = torch.cat([word_in_front, word_behind, current_representation[:, i+1:, :]], 1)
            except ValueError:
                current_representation = torch.cat([word_in_front, word_behind], 1)
        elif i == current_representation.size()[1]-1:
            try:
                current_representation = torch.cat([current_representation[:, :i, :], word_in_front, word_behind], 1)
            except ValueError:
                current_representation = torch.cat([word_in_front, word_behind], 1)
        else:
            current_representation = torch.cat([current_representation[:, :i, :], word_in_front, word_behind,
                                                current_representation[:, i+1:, :]], 1)

    for i in range(input_length):
        loss += criterion(current_representation[:, i, :], input_variable_embedded[:, i, :])

    loss.backward()

    detector_optimizer.step()
    combiner_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / input_length, path


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(detector, combiner, decoder, n_iters, n_per_iter, print_every=500, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    detector_optimizer = optim.SGD(detector.parameters(), lr=learning_rate)
    combiner_optimizer = optim.SGD(combiner.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = mse_loss

    print('training...')

    embedding = nn.Embedding(input_sentences_class.num_words, hidden_size)
    embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
    embedding = embedding.cuda()

    for i in range(n_iters):

        training_sentences = [variableFromSentence(input_sentences_class, random.choice(input_sentences)) for _ in
                              range(n_per_iter)]

        for iter in range(1, n_per_iter + 1):
            training_sentence = training_sentences[iter - 1]
            input_variable = training_sentence

            loss, path = train(input_variable, embedding, detector, combiner,
                               decoder,  detector_optimizer, combiner_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / (n_iters * n_per_iter)),
                                             iter + i * n_per_iter, (iter + i * n_per_iter)/(n_iters * n_per_iter) * 100, print_loss_avg))
                current_sentence = []
                for w in input_variable.data.cpu().numpy():
                    current_sentence.append(input_sentences_class.index2word[int(w)])
                print(current_sentence, path)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

# detector1 = Detector(hidden_size)
# combiner1 = Combiner(hidden_size)
# decoder1 = Decoder(hidden_size)

detector1 = DetectorRNN(hidden_size)
combiner1 = CombinerRNN(hidden_size)
decoder1 = DecoderRNN(hidden_size)

if use_cuda:
    detector1 = detector1.cuda()
    combiner1 = combiner1.cuda()
    decoder1 = decoder1.cuda()

trainIters(detector1, combiner1, decoder1, 100, 100000, print_every=200)

