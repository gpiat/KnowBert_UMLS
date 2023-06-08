import nltk
import os

nltk.download("stopwords")
nltk.download('punkt')

bashCmd = "python -m spacy download en"
os.system(bashCmd)

# bergamote
# umls_installation_path = "/scratch_global/DATASETS/umls/2019AA/META/"
# destination_path = "/scratch_global/gpiat/QuickUMLS/"
# factoryIA
umls_installation_path = "/home/data/dataset/UMLS/2019AA/META/"
destination_path = "/home/users/gpiat/Software/QuickUMLS/"

os.system(f"python -m quickumls.install {umls_installation_path} {destination_path}")

# decompressing umls vocabulary
os.system("tar xvzf tests/fixtures/umls_vocab.tar.gz")
# code for QuickUMLS

# download bert model
# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
from pytorch_pretrained_bert.modeling import BertForPreTraining
BertForPreTraining.from_pretrained('bert-base-uncased')
from pytorch_pretrained_bert.tokenization import BertTokenizer
BertTokenizer.from_pretrained('bert-base-uncased')

# downloading the scispacy entity linker
import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

nlp = spacy.load("en_core_sci_sm")
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
linker = EntityLinker(
    resolve_abbreviations=True,
    name="umls")
nlp.add_pipe(linker)
