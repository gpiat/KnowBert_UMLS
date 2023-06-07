local batch_size = 32;

{
    "vocabulary": {
        "directory_path": "/home/users/gpiat/Documents/projects/KnowBert-UMLS/tests/fixtures/umls_vocab_stid/",
    },

    "dataset_reader": {
        "type": "multitask_reader",
        "datasets_for_vocab_creation": [],
        "dataset_readers": {
            "language_modeling": {
                "type": "sharded",
                "base_reader": {
                    "type": "bert_pre_training_pregen_cand",
                    "tokenizer_and_candidate_generator": {
                        "type": "bert_tokenizer_pregen_candidates",
                        "candidate_type_name": "umls",
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
                    "candidate_directory": "/media/pmc/candfiles/en_core_sci_lg-0.5-10/",
                    "max_predictions_per_seq": 10,
                    "lazy": true,
                    "mask_candidate_strategy": "full_mask",
                },
                "buffer_size": 1000,
                "shuffle": false
            },
            "entity_linking": {
                "type": "umls_entity_linking",
                "mention_generator": {
                    "type": "scispacy_mention_generator",
                    // file containing CUI occurrence counts in MedMentions to be used as an estimate for CUI prior probabilities.
                    "cui_count_file": "/home/users/gpiat/Documents/projects/KnowBert-UMLS/umls_data/stid_counts.json",
                    // determines how many candidate CUIs to keep for each candidate span.
                    "max_number_candidates": 10,
                    // Is added to CUI counts for the purposes of computing priors. It is not advised to use smoothing < 1. Smoothing of 0 will consider that CUIs that do not appear in MedMentions have a prior probabili$
                    "count_smoothing": 1,
                    // directory where the QuickUMLS data files are installed.
                    "sim_thresh": 0.5,
                    "spaCy_name": "en_core_sci_lg",
                    "target": "semtype"
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
                "multi_label_policy": "max_prior"
            },
        }
    },

    "iterator": {
        "type": "multitask_iterator",
        "names_to_index": ["language_modeling", "entity_linking"],
        "iterate_forever": true,
        //false,

        "sampling_rates": [0.7, 0.3],

        "iterators": {
            "language_modeling": {
                "type": "self_attn_bucket",
                "batch_size_schedule": "base-24gb-fp32",
                "iterator": {
                    "type": "bucket",
                    "batch_size": batch_size,
                    "sorting_keys": [["tokens", "num_tokens"]],
                    # TODO: CHANGE THIS VALUE?
                    "max_instances_in_memory": 2500,
                }
            },
            "entity_linking": {
                "type": "self_attn_bucket",
                "batch_size_schedule": "base-24gb-fp32",
                "iterator": {
                    "type": "cross_sentence_linking",
                    "batch_size": batch_size,
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
                    "max_predictions_per_seq": 10,
                    "iterate_forever": true,
                    "id_type": "umls",
                    "use_nsp_label": true,
                }
            },
        },
    },

    "train_data_path": {
        "language_modeling": "/media/pmc/oa_bulk_bert_512/0[0-7][0-9].txt",
        "entity_linking": "/sscratch/gpiat/MedMentions/full/data/corpus_pubtator_train.txt",
    },

    "model": {
        "type": "knowbert",
        "bert_model_name": "bert-base-uncased",
        "model_archive": "/home/users/gpiat/Documents/projects/KnowBert-UMLS/umls_data/2022_06_09_kar_train_stid_singletarget_maxprior_10ep/model.tar.gz",
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
                    //     "embedding_file": "/sscratch_global/gpiat/UMLS_embeddings.csv",
                    //     // "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5",
                    //     "include_null_embedding": true,
                    // },
                    // TODO: Hopefully this uses a Null embedding
                    "entity_embedding": {
                        "vocab_namespace": "entity",
                        "embedding_dim": 50,
                        // from https://github.com/r-mal/umls-embeddings
                        "pretrained_file": "/sscratch/gpiat/UMLS/UMLS_embeddings.gz",
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
        "gradient_accumulation_batch_size": batch_size,
        "num_epochs": 1,
        "num_steps_reset_metrics": 5000,

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 1,
            "num_steps_per_epoch": 24000000 / batch_size,
            "cut_frac": 0.025,
        },
        "num_serialized_models_to_keep": 2,
        "model_save_interval": 600,
        "should_log_learning_rate": true,
        "cuda_device": 0,
    }
}
