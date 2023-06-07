{
    "dataset_reader": {
        "type": "iob2_ner_reader",
        "tokenizer_and_candidate_generator": {
            "type": "bert_tokenizer_and_candidate_generator",
            "entity_candidate_generators": {
                "umls": {
                    "type": "scispacy_mention_generator",
                    // file containing CUI occurrence counts in MedMentions to be used as an estimate for CUI prior probabilities.
                    "cui_count_file": "/home/users/gpiat/Documents/projects/KnowBert-UMLS/umls_data/stid_counts.json",
                    // as of now does nothing but will eventually determine how many candidate CUIs to keep for each candidate span.
                    "max_number_candidates": 10,
                    // Is added to CUI counts for the purposes of computing priors. It is not advised to use smoothing < 1. Smoothing of 0 will consider that CUIs that do not appear in MedMentions have a prior p$
                    "count_smoothing": 1,
                    // similarity threshold that SciSpaCy should use. Lower => more compute time, lower Accuracy, higher recall.
                    "sim_thresh": 0.5,
                    "spaCy_name": "en_core_sci_lg",
                    "target": "semtype"
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
            // there's no issue with whitespace tokenization in i2b2
            "whitespace_tokenize": true
            /*
            // TODO: is bert-base-cased an acceptable model type?
            // It should be, if that just defines a vocabulary
            "bert_model_type": "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/bert-base-uncased-tacred-entity-markers-vocab.txt",
            // TODO: if candidates end up needing to be pregenerated
            "type": "bert_tokenizer_pregen_candidates",
            "candidate_type_name": "umls",
            */
        }
    },
    "iterator": {
        "type": "self_attn_bucket",
        "iterator": {
            "type": "basic",
            "batch_size": 32
        },
        "batch_size_schedule": "base-12gb-fp32"
    },
    "model": {
        "type": "simple-classifier",
        "model": {
            "type": "from_archive",
            "archive_file": "umls_data/99_percent_data/2022_06_03_retrain_stid_singletarget_maxprior_fix/model.tar.gz",
        },
        "bert_dim": 768,
        "concat_word_a": false,
        "include_cls": false,
        "metric_a": {
            "type": "seqeval",
            "label_map": {
                "O": 0,
                "B-test": 1,
                "I-test": 2,
                "B-problem": 3,
                "I-problem": 4,
                "B-treatment": 5,
                "I-treatment": 6,
                "[PAD]": 7
            }
        },
        // Test/problem/treatment ; I/O/B + padding for wordpieces
        "num_labels": 8,
        "task": "word_classification",
        /* Hypothetically, if my understading is correct, Binary Cross
           Entropy is for cases where one sample cand be part of multiple
           classes.*/
        "use_bce_loss": false
    },
    "train_data_path": "/sscratch/gpiat/n2c2/train.tsv",
    "validation_data_path": "/sscratch/gpiat/n2c2/dev.tsv",
    "test_data_path": "/sscratch/gpiat/n2c2/test.tsv",
    "trainer": {
        "cuda_device": 0,
        "gradient_accumulation_batch_size": 32,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 10,
            "num_steps_per_epoch": 62.5
        },
        "num_epochs": 10,
        // TODO: 5 here?
        "num_serialized_models_to_keep": 1,
        "optimizer": {
            "type": "bert_adam",
            "b2": 0.98,
            "lr": 3e-05,
            "max_grad_norm": 1,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.bias",
                        "LayerNorm.weight",
                        "layer_norm.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "t_total": -1,
            "weight_decay": 0.01
        },
        "should_log_learning_rate": true,
        "validation_metric": "+f1"
    },
    "vocabulary": {
        "directory_path": "/home/users/gpiat/Documents/projects/KnowBert-UMLS/tests/fixtures/umls_vocab_stid/"
    }
}
