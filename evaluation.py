import pickle
import csv
import numpy
from scipy import spatial
import re


label = list()
with open('/home/dw1215/PycharmProjects/first_pytorch/data/stsbenchmark/sts-test.csv', 'r') as file:
    sts_data = csv.reader(file, delimiter='\t')
    for s in sts_data:
        label.append(float(s[4]))

with open('/home/dw1215/PycharmProjects/first_pytorch/data/'
          'cleaned_embedding_seq2seq.pkl', 'rb') as f:
    embeddings = pickle.load(f)

st = []
with open('/home/dw1215/PycharmProjects/first_pytorch/data/encoded_upd_every_node1') as f:
    for line in f:
        start = '[' in line
        end = ']' in line
        l1 = re.sub(' +', ' ', line)
        if start:
            s = l1.split('[[[')
            st.append(s[0])

print(len(embeddings))

sentences1 = embeddings[:int(len(embeddings)/2)]
sentences2 = embeddings[int(len(embeddings)/2):]

similarity_predictions = list()
for s1, s2 in zip(sentences1, sentences2):
    s1_list = s1.tolist()
    s2_list = s2.tolist()
    result = 1 - spatial.distance.cosine(s1_list, s2_list)
    # result = result * 5
    similarity_predictions.append(result)

print(len(label))
print(len(similarity_predictions))
print(numpy.corrcoef(label, similarity_predictions))

s1 = st[:int(len(st)/2)]
s2 = st[int(len(st)/2):]
for p, l, ss1, ss2 in zip(similarity_predictions, label, s1, s2):
    print(p, '***', l, '=======', ss1, '***', ss2)

