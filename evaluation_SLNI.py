import re
import numpy as np
import pickle
import csv
import torch

Embedding_saving_path = '/home/dw1215/PycharmProjects/first_pytorch/data/' \
                        'cleaned_embedding_SLNI_normalized_20000.pkl'


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
# ====================================================
# Clean and save embeddings
# ====================================================
with open('/home/dw1215/PycharmProjects/first_pytorch/data/encoded_SNLI_normalized2') as f:
    embeddings = list()
    for line in f:
        start = '[' in line
        end = ']' in line
        l1 = re.sub(' +', ' ', line)
        if start:
            s = l1.split('[[[')
            s = s[1]
            s = s.split(' ')
            if '' in s:
                s.remove('')
            embedding = []
            for value in s:
                embedding.append(float(value))
        elif end:
            s = re.sub(']', '', l1)
            s = s.split(' ')
            if '' in s:
                s.remove('')
            if '\n' in s:
                s.remove('\n')
            for value in s:
                embedding.append(float(value))
            embeddings.append(np.asarray(embedding, dtype='float32'))
        else:
            l = l1.split(' ')
            if '' in l:
                l.remove('')
            for value in l:
                embedding.append(float(value))


with open(Embedding_saving_path, 'wb') as f:
    pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)

# with open('/home/dw1215/PycharmProjects/first_pytorch/data/'
#           'cleaned_embedding.pkl', 'rb') as f:
#     embeddings = pickle.load(f)
#
# print(embeddings[:20])
# for i in range(20):
#     print(len(list(embeddings[i])))

# ====================================================
# Read labels
# ====================================================
with open('/home/dw1215/PycharmProjects/first_pytorch/data/'
          'stanford-natural-language-inference-corpus/snli_1.0_train.csv', 'r') as fp:
    reader = csv.reader(fp, delimiter=',', quotechar='"')
    # next(reader, None)  # skip the headers
    data_read = [row for row in reader]
    # print(data_read[:3])

sent1 = list()
sent2 = list()
labels = list()
count = 0
for l in data_read[1:]:
    # l = line[:-5].replace('"', '')
    # print(line[:-5])
    if len(l[-5].split(' ')) > 20 or len(l[-4].split(' ')) > 20:
        count += 1
    else:
        sent1.append(l[-9])
        sent2.append(l[-8])
        labels.append(l[-5])

# ====================================================
# Read embeddings
# ====================================================
with open(Embedding_saving_path, 'rb') as f:
    embeddings = pickle.load(f)

sentences1 = embeddings[:int(len(embeddings)/2)]
sentences2 = embeddings[int(len(embeddings)/2):]
