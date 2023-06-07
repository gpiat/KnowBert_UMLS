// Warning : The value of this variable depends on the location this file is CALLED from
local dir = std.extVar("PWD") + "/";

local batch_size_per_gpu = 32;

{
    "vocabulary": {
        "directory_path": dir + "tests/fixtures/umls_vocab/",
    },

    "dataset_reader": {
        "type": "multitask_reader",
        "datasets_for_vocab_creation": [],
        "dataset_readers": {
            "language_modeling": {
                "type": "multiprocess",
                "base_reader": {
                    "type":  "bert_pre_training",
                    "tokenizer_and_candidate_generator": {
                        "type": "bert_tokenizer_and_candidate_generator",
                        "entity_candidate_generators": {
                            "umls": {
                                "type": "umls_mention_generator",
                                // file containing CUI occurrence counts in MedMentions to be used as an estimate for CUI prior probabilities.
                                "cui_count_file": dir + "umls_data/cui_counts.json",
                                // max candidate span length. Increasing this dramatically increases compute time. Decreasing this under 7 dramatically decreases candidate span recall.
                                "max_entity_length": 7,
                                // as of now does nothing but will eventually determine how many candidate CUIs to keep for each candidate span.
                                "max_number_candidates": 30,
                                // Is added to CUI counts for the purposes of computing priors. It is not advised to use smoothing < 1. Smoothing of 0 will consider that CUIs that do not appear in MedMentions have a prior probability of 0. Smoothing of +inf will consider that all concepts are equiprobable.
                                "count_smoothing": 1,
                                // directory where the QuickUMLS data files are installed.
                                "qUMLS_fp": "/scratch_global/gpiat/QuickUMLS/",
                                // similarity threshold that quickUMLS should use. Lower => more compute time, lower Accuracy, higher recall.
                                "qUMLS_thresh": 0.7,
                                // choose between cosine, dice, and jaccard. Each is faster than the previous but also has lower recall for a given threshold.
                                "similarity_name": "cosine"
                            },
                        },
                        "entity_indexers":  {
                            "umls": {
                                "type": "characters_tokenizer",
                                "tokenizer": {
                                    "type": "word",
                                    "word_splitter": {"type": "just_spaces"},
                                },
                                "namespace": "entity"
                            }
                        },
                        "bert_model_type": "bert-base-uncased",
                        "do_lower_case": true,
                    },
                    "lazy": true,
                    "mask_candidate_strategy": "full_mask",
                },
                "num_workers": 1,
            },
            "entity_linking": {
                "type": "umls_entity_linking",
                "mention_generator": {
                    "type": "umls_mention_generator",
                    // file containing CUI occurrence counts in MedMentions to be used as an estimate for CUI prior probabilities.
                    "cui_count_file": dir + "umls_data/cui_counts.json",
                    // max candidate span length. Increasing this dramatically increases compute time. Decreasing this under 7 dramatically decreases candidate span recall.
                    "max_entity_length": 7,
                    // as of now does nothing but will eventually determine how many candidate CUIs to keep for each candidate span.
                    "max_number_candidates": 30,
                    // Is added to CUI counts for the purposes of computing priors. It is not advised to use smoothing < 1. Smoothing of 0 will consider that CUIs that do not appear in MedMentions have a prior probability of 0. Smoothing of +inf will consider that all concepts are equiprobable.
                    "count_smoothing": 1,
                    // directory where the QuickUMLS data files are installed.
                    "qUMLS_fp": "/scratch_global/gpiat/QuickUMLS/",
                    // similarity threshold that quickUMLS should use. Lower => more compute time, lower Accuracy, higher recall.
                    "qUMLS_thresh": 0.7,
                    // choose between cosine, dice, and jaccard. Each is faster than the previous but also has lower recall for a given threshold.
                    "similarity_name": "cosine"
                },
                "entity_indexer": {
                   "type": "characters_tokenizer",
                   "tokenizer": {
                       "type": "word",
                       "word_splitter": {"type": "just_spaces"},
                   },
                   "namespace": "entity"
                },
                // "is_training": true,
                "should_remap_span_indices": false,
            },
        }
    },

    "iterator": {
        "type": "multitask_iterator",
        "names_to_index": ["language_modeling", "entity_linking"],
        "iterate_forever": true,

        "sampling_rates": [0.8, 0.2],

        "iterators": {
            "language_modeling": {
                "type": "multiprocess",
                "base_iterator": {
                    "type": "self_attn_bucket",
                    "batch_size_schedule": "base-24gb-fp32",
                    "iterator": {
                        "type": "bucket",
                        "batch_size": batch_size_per_gpu,
                        "sorting_keys": [["tokens", "num_tokens"]],
                        "max_instances_in_memory": 2500,
                    }
                },
                "num_workers": 1,
            },
            "entity_linking": {
                "type": "self_attn_bucket",
                "batch_size_schedule": "base-24gb-fp32",
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
                    // this is ignored
                    "mask_candidate_strategy": "none",
                    "max_predictions_per_seq": 0,
                    "iterate_forever": true,
                    "id_type": "umls",
                    "use_nsp_label": true,
                }
            },
        },
    },

    "train_data_path": {
        // TODO: Run bin/create_pretraining_data_for_bert.py to group the sentences by length, do the NSP sampling, and write out files for training.
        //      python bin/create_pretraining_data_for_bert.py "/scratch_global/DATASETS/pmc/dev/jtourille/oa_bulk_chunks_50Mb/000*.txt.gz" /scratch_global/DATASETS/pmc/dev/jtourille/oa_bulk_bert/ 9
        // "language_modeling": "/scratch_global/DATASETS/pmc/dev/jtourille/oa_bulk_chunks_50Mb/0001.txt.gz",
        "language_modeling": "/scratch_global/DATASETS/pmc/dev/jtourille/oa_bulk_bert/*",
        "entity_linking": "/scratch_global/DATASETS/MedMentions/full/data/corpus_pubtator_debug.txt",
    },
    "validation_data_path": {
        // TODO: Run bin/create_pretraining_data_for_bert.py to group the sentences by length, do the NSP sampling, and write out files for training.
        //      python bin/create_pretraining_data_for_bert.py "/scratch_global/DATASETS/pmc/dev/jtourille/oa_bulk_chunks_50Mb/000*.txt.gz" /scratch_global/DATASETS/pmc/dev/jtourille/oa_bulk_bert/ 9
        // "language_modeling": "/scratch_global/DATASETS/pmc/dev/jtourille/oa_bulk_chunks_50Mb/0001.txt.gz",
        "language_modeling": "/scratch_global/DATASETS/pmc/dev/jtourille/oa_bulk_bert/*",
        "entity_linking": "/scratch_global/DATASETS/MedMentions/full/data/corpus_pubtator_debug.txt",
    },
    

    "model": {
        "type": "knowbert",
        "bert_model_name": "bert-base-uncased",
        "model_archive": dir + "umls_data/kar_train/model.tar.gz",
        "soldered_layers": {"umls": 9},
        "soldered_kgs": {
            "umls": {
                "type": "soldered_kg",
                "entity_linker": {
                    "type": "entity_linking_with_candidate_mentions",
                    "loss_type": "softmax",
                    // EITHER concat_entity_embedder OR entity_embedding
                    // "concat_entity_embedder": {
                    //     "type": "umls_embeddings",
                    //     "entity_dim": 50,
                    //     // found from https://github.com/r-mal/umls-embeddings/
                    //     "embedding_file": "/scratch_global/gpiat/UMLS_embeddings.csv",
                    //     // "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5",
                    //     "include_null_embedding": true,
                    // },
                    // TODO: Hopefully this uses a Null embedding
                    "entity_embedding": {
                        "vocab_namespace": "entity",
                        "embedding_dim": 50,
                        "pretrained_file": "/scratch_global/gpiat/UMLS_embeddings.gz",
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
            "lr": 1e-4,
            "t_total": -1,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,

            "parameter_groups": [
                // all layer norm and bias in bert have no weight decay and small lr
                [["pretrained_bert.*embeddings.*LayerNorm",
                  "pretrained_bert.*encoder.layer.[0-9]\\..*LayerNorm", "pretrained_bert.*encoder.layer.[0-9]\\..*bias",
                  "pretrained_bert.*cls.*LayerNorm", "pretrained_bert.*cls.*bias",
                  "pretrained_bert.*pooler.*bias"], {"lr": 2e-5, "weight_decay": 0.0}],
                // remaining parameters have small lr
                [["pretrained_bert.*embeddings[^L]+$", "pretrained_bert.*pooler.*weight", "pretrained_bert.*cls[^L]+weight",
                  "pretrained_bert.*encoder.layer.[0-9]\\.[^L]+weight"], {"lr": 2e-5, "weight_decay": 0.01}],
                [[
                  "pretrained_bert.*encoder.layer.1[0-1].*LayerNorm", "pretrained_bert.*encoder.layer.1[0-1].*bias"],
                  {"lr": 5e-5, "weight_decay": 0.0}],
                [[
                  "pretrained_bert.*encoder.layer.1[0-1][^L]+weight"],
                  {"lr": 5e-5, "weight_decay": 0.01}],
                // other bias and layer norm have no weight decay
                [["soldered_kg.*LayerNorm", "soldered_kg.*layer_norm", "soldered_kg.*bias"],
                  {"weight_decay": 0.0}],
            ],
        },
        "gradient_accumulation_batch_size": batch_size_per_gpu,
        "num_epochs": 1,
        "num_steps_reset_metrics": 5000,

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 1,
            "num_steps_per_epoch": 240000,
            "cut_frac": 0.025,
        },
        "num_serialized_models_to_keep": 2,
        "model_save_interval": 600,
        "should_log_learning_rate": true,
        "cuda_device": 0,
    }
}
