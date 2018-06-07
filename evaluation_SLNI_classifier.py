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
with open('/home/dw1215/PycharmProjects/first_pytorch/data/encoded_SNLI_second2') as f:
    embeddings = list()
    for line in f:
        start = '[' in line
        end = ']' in line
        l1 = re.sub(' +', ' ', line)
        if start:
            s = l1.split('[[[')
            embeddings.append(s[0])

# with open(Embedding_saving_path, 'rb') as f:
#     embeddings = pickle.load(f)

s1 = embeddings[:(int(len(embeddings)/2))]
s2 = embeddings[(int(len(embeddings)/2)):]

count = 0
for x, y in zip(s1, s2):
    if count < 1000:
        print(x, ' **** ', y)
        count += 1
    else:
        break
