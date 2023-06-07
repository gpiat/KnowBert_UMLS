import csv
from collections import OrderedDict

embeddings = OrderedDict()
with open("UMLS_embeddings.csv", 'r') as f:
    emb_reader = csv.reader(f)
    for row in emb_reader:
        embeddings[row[0]] = [float(n) for n in row[1:]]

with open("UMLS_embeddings.ssv", 'w') as f:
    for k, v in embeddings.items():
        row = ' '.join([k] + [str(n) for n in v])
        print(row, file=f)

