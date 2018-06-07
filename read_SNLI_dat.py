import csv

# with open('/home/dw1215/PycharmProjects/first_pytorch/data/'
#           'stanford-natural-language-inference-corpus/snli_1.0_train.csv') as file:
#     data = file.readlines()

# Read CSV file
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

print(sent1[:5])
print(sent2[:5])
print(labels[:5])
print(count)
