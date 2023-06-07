from allennlp.data import DatasetReader, Vocabulary, DataIterator, TokenIndexer
from allennlp.common import Params
from allennlp.modules import TokenEmbedder
from allennlp.models import Model
from kb.include_all import *

import torch
from os.path import expanduser


def get_bert_test_fixture():
    embedder_params = {
        "type": "bert-pretrained",
        "pretrained_model": "tests/fixtures/bert/bert_test_fixture.tar.gz",
        "requires_grad": True,
        "top_layer_only": True,
    }
    embedder_params_copy = dict(embedder_params)
    embedder = TokenEmbedder.from_params(Params(embedder_params))

    indexer_params = {
        "type": "bert-pretrained",
        "pretrained_model": "tests/fixtures/bert/vocab.txt",
        "do_lowercase": True,
        "use_starting_offsets": True,
        "max_pieces": 512,
    }
    indexer_params_copy = dict(indexer_params)
    indexer = TokenIndexer.from_params(Params(indexer_params))

    return {'embedder': embedder, 'embedder_params': embedder_params_copy,
            'indexer': indexer, 'indexer_params': indexer_params_copy}


def get_wsd_reader(is_training, use_bert_indexer=False, wordnet_entity_file=None):
    if wordnet_entity_file is None:
        wordnet_entity_file = "tests/fixtures/wordnet/entities_cat_hat.jsonl"

    if use_bert_indexer:
        bert_fixtures = get_bert_test_fixture()
        indexer_params = bert_fixtures["indexer_params"]
    else:
        indexer_params = {"type": "single_id", "lowercase_tokens": True}

    reader_params = {
        "type": "wordnet_fine_grained",
        "wordnet_entity_file": wordnet_entity_file,
        "token_indexers": {
            "tokens": indexer_params,
        },
        "entity_indexer": {
            "type": "characters_tokenizer",
            "tokenizer": {
                "type": "word",
                "word_splitter": {"type": "just_spaces"},
            },
            "namespace": "entity"
        },
        "is_training": is_training,
        "use_surface_form": False
    }
    reader = DatasetReader.from_params(Params(reader_params))

    vocab_params = {
        "directory_path": "tests/fixtures/wordnet/cat_hat_vocabdir"
    }
    vocab = Vocabulary.from_params(Params(vocab_params))

    iterator = DataIterator.from_params(Params({"type": "basic"}))
    iterator.index_with(vocab)

    return reader, vocab, iterator


def get_wsd_fixture_batch(is_training, use_bert_indexer=False):
    wsd_file = 'tests/fixtures/wordnet/wsd_dataset.json'
    reader, vocab, iterator = get_wsd_reader(
        is_training, use_bert_indexer=use_bert_indexer)
    instances = reader.read(wsd_file)

    for batch in iterator(instances, shuffle=False, num_epochs=1):
        break
    return batch


def get_bert_pretraining_reader_with_kg(
        mask_candidate_strategy='none', masked_lm_prob=0.15, include_wiki=False
):
    params = {
        "type": "bert_pre_training",
        "tokenizer_and_candidate_generator": {
            "type": "bert_tokenizer_and_candidate_generator",
            "entity_candidate_generators": {
                "wordnet": {"type": "wordnet_mention_generator",
                            "entity_file": "tests/fixtures/wordnet/entities_fixture.jsonl"}
            },
            "entity_indexers": {
                "wordnet": {
                    "type": "characters_tokenizer",
                    "tokenizer": {
                        "type": "word",
                        "word_splitter": {"type": "just_spaces"},
                    },
                    "namespace": "entity"
                }
            },
            "bert_model_type": "tests/fixtures/bert/vocab.txt",
            "do_lower_case": True,
        },
        "mask_candidate_strategy": mask_candidate_strategy,
        "masked_lm_prob": masked_lm_prob
    }

    if include_wiki:
        params["tokenizer_and_candidate_generator"]["entity_candidate_generators"]["wiki"] = {
            "type": "wiki",
            "candidates_file": "tests/fixtures/linking/priors.txt",
        }
        params["tokenizer_and_candidate_generator"]["entity_indexers"]["wiki"] = {
            "type": "characters_tokenizer",
            "tokenizer": {
                "type": "word",
                "word_splitter": {"type": "just_spaces"},
            },
            "namespace": "entity_wiki"
        }
        params["tokenizer_and_candidate_generator"]["entity_indexers"]["wordnet"]["namespace"] = "entity_wordnet"

    return DatasetReader.from_params(Params(params))


def get_bert_retrain_pregen_reader_with_kg(
        mask_candidate_strategy='none', masked_lm_prob=0.15
):
    """ This class loads a BertPreTrainingPreGenCandidatesReader from
        kb.bert_pretraining_reader for unit testing purposes.
    """
    params = {
        "type": "bert_pre_training_pregen_cand",
        "tokenizer_and_candidate_generator": {
            "type": "bert_tokenizer_pregen_candidates",
            "candidate_type_name": "umls",
            # "entity_candidate_generators": {
            #     "umls": {
            #         "type": "umls_mention_generator",
            #         "cui_count_file": "umls_data/cui_counts.json",
            #         "max_entity_length": 7,
            #         "max_number_candidates": 30,
            #         "count_smoothing": 1,
            #         "qUMLS_fp": expanduser("~") + "/Software/QuickUMLS/",
            #         "qUMLS_thresh": 0.7,
            #         "similarity_name": "cosine"
            #     },
            # },
            "entity_indexers": {
                "umls": {
                    "type": "characters_tokenizer",
                    "tokenizer": {
                        "type": "word",
                        "word_splitter": {"type": "just_spaces"},
                    },
                    "namespace": "entity"
                }
            },
            "bert_model_type": "tests/fixtures/bert_pretraining/vocab_pmc.txt",
            "do_lower_case": True,
        },
        "candidate_directory": "tests/fixtures/bert_pretraining/cand_files/",
        "mask_candidate_strategy": mask_candidate_strategy,
        "masked_lm_prob": masked_lm_prob
    }

    return DatasetReader.from_params(Params(params))


def get_sharded_reader():
    """ This class loads a BertPreTrainingPreGenCandidatesReader from
        kb.bert_pretraining_reader for unit testing purposes.
    """
    params = {
        "type": "sharded",
        "base_reader": {
            "type": "bert_pre_training_pregen_cand",
            "tokenizer_and_candidate_generator": {
                "type": "bert_tokenizer_pregen_candidates",
                "candidate_type_name": "umls",
                "entity_indexers": {
                    "umls": {
                        "type": "characters_tokenizer",
                        "tokenizer": {
                            "type": "word",
                            "word_splitter": {"type": "just_spaces"},
                        },
                        "namespace": "entity"
                    }
                },
                "bert_model_type":
                    "tests/fixtures/bert_pretraining/vocab_pmc.txt",
                "do_lower_case": True,
            },
            "candidate_directory": "tests/fixtures/bert_pretraining/cand_files/",
            "mask_candidate_strategy": 'none',
            "masked_lm_prob": 0,
            "lazy": True,
        },
        "shuffle": False
    }

    return DatasetReader.from_params(Params(params))
