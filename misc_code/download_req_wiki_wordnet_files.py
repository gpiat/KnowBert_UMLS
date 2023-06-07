import kb.include_all
from kb.common import EntityEmbedder

from allennlp.common.params import Params
from allennlp.data import DatasetReader
from allennlp.data import TokenIndexer
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer, TrainerPieces


par = Params.from_file("training_config/pretraining/knowbert_wordnet_linker.jsonnet")
pieces = TrainerPieces.from_params(par, './shaz/', False)
# Vocabulary.from_params(par['vocabulary'])
# dataset_reader = DatasetReader.from_params(par["dataset_reader"])
# EntityEmbedder.from_params(par["model"]["soldered_kgs"]["wordnet"]["entity_linker"]["concat_entity_embedder"])
# train_data_path

par = Params.from_file("training_config/pretraining/knowbert_wiki_linker.jsonnet")
pieces = TrainerPieces.from_params(par, './shaz/', False)
# Vocabulary.from_params(par['vocabulary'])
# DatasetReader.from_params(par["dataset_reader"])
# Embedding.from_params(par["model"]["soldered_kgs"]["wiki"]["entity_linker"]["entity_embedding"])
# "train_data_path": "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/aida_train.txt",
# "validation_data_path": "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/aida_dev.txt",
