// Warning : The value of this variable depends on the location this file is CALLED from
local dir = std.extVar("PWD") + "/";

local batch_size_per_gpu = 32;

{
    "vocabulary": {
        "directory_path": dir + "tests/fixtures/umls_vocab/",
        // The vocab follows the model provided by KnowBert. It contains a non_padded_namespaces.txt file (same as wiki and wordnet vocabs) and an entity.txt file which contains one string per CUI (the first for each CUI found in MRCONSO.RRF as it seems to be the "default" string), + UNK, NULL and MASK tokens (in the same places as they are in wiki and wordnet vocabs)
    },

    "dataset_reader": {
        "type": "umls_entity_linking",
        "mention_generator": {
            "type": "scispacy_mention_generator",
            // file containing CUI occurrence counts in MedMentions to be used as an estimate for CUI prior probabilities.
            "cui_count_file": dir + "umls_data/cui_counts.json",
            // determines how many candidate CUIs to keep for each candidate span.
            "max_number_candidates": 30,
            // Is added to CUI counts for the purposes of computing priors. It is not advised to use smoothing < 1. Smoothing of 0 will consider that CUIs that do not appear in MedMentions have a prior probability of 0. Smoothing of +inf will consider that all concepts are equiprobable.
            "count_smoothing": 1,
            // directory where the QuickUMLS data files are installed.
            "sim_thresh": 0.5,
            "spaCy_name": "en_core_sci_sm"
        },
        "token_indexers": {
            "tokens": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-uncased",
                "do_lowercase": true,
                "use_starting_offsets": true,
                "max_pieces": 512,
            },
        },
        "entity_indexer": {
           "type": "characters_tokenizer",
           "tokenizer": {
               "type": "word",
               "word_splitter": {"type": "just_spaces"},
           },
           "namespace": "entity"
        },
        "should_remap_span_indices": false
    },

    // is passed to dataset_reader, points to MedMentions
    "train_data_path": "/home/data/dataset/MedMentions/full/data/corpus_pubtator_train.txt",
    // "train_data_path": "/home/data/dataset/MedMentions/full/data/corpus_pubtator_debug.txt",

    "iterator": {
        "type": "cross_sentence_linking",
        "batch_size": batch_size_per_gpu,
        "entity_indexer": {
            "type": "characters_tokenizer",
            "tokenizer": {
                "type": "word",
                "word_splitter": {"type": "just_spaces"},
            },
            "namespace": "entity"
        },
        "bert_model_type": "bert-base-uncased",
        "do_lower_case": true,
        // GP: the following is not my comment
        // this is ignored
        "mask_candidate_strategy": "none",
        "max_predictions_per_seq": 0,
        "id_type": "umls",
        "use_nsp_label": false,
    },

    "model": {
        "type": "knowbert",
        "bert_model_name": "bert-base-uncased",
        "mode": "entity_linking",
        "soldered_layers": {"umls": 9},
        "soldered_kgs": {
            "umls": {
                "type": "soldered_kg",
                "entity_linker": {
                    "type": "entity_linking_with_candidate_mentions",
                    "loss_type": "softmax",
                    "entity_embedding": {
                        "vocab_namespace": "entity",
                        "embedding_dim": 50,
                        "pretrained_file": "/home/data/dataset/UMLS/UMLS_embeddings.gz",
                        "trainable": false,
                        "sparse": false
                    },
                    "contextual_embedding_dim": 768,
                    "span_encoder_config": {
                        "hidden_size": 50,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 5,
                        "intermediate_size": 1024
                    },
                },
                "span_attention_config": {
                    "hidden_size": 50,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 5,
                    "intermediate_size": 1024
                },
            },
        },
    },

    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            "lr": 1e-3,
            "t_total": -1,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },
        "num_epochs": 5,

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 5,
            // semcor + examples batch size=32
            "num_steps_per_epoch": 2470,
        },
        "num_serialized_models_to_keep": 1,
        "should_log_learning_rate": true,
        "cuda_device": 0,
    }

}
