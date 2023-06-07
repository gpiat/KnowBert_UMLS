import re
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# entity_file = "/home/gpiat/Documents/Datasets/UMLS/MRCONSO_SHORT.RRF"
entity_file = "/home/users/gpiat/Documents/Datasets/UMLS/MRCONSO_SHORT.RRF"
cui_list = []
str_list = []
with open(entity_file, 'r') as f:
    for l in f:
        c, s = l.split('|')
        cui_list.append(c)
        str_list.append(s.strip())

vectorizer = CountVectorizer(strip_accents='ascii', ngram_range=(3,3), analyzer='char_wb')

tf_idf_matrix = vectorizer.fit_transform(str_list)

with open("/home/users/gpiat/Documents/Datasets/UMLS/tfidfmatfile.pkl", 'wb') as f:
    pickle.dump(tf_idf_matrix, f)

with open("/home/users/gpiat/Documents/Datasets/UMLS/tfidfmatfile.pkl", 'rb') as f:
    tf_idf_matrix = pickle.load(f)
