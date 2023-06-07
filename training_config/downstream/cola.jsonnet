local batch_size = 32;
local num_epochs = 10;
local num_documents = 8000;

{
    "dataset_reader": {
        "type": "cola_reader",
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
            "do_lower_case": true
        }
    },
    "iterator": {
        "type": "self_attn_bucket",
        "iterator": {
            "type": "basic",
            "batch_size": batch_size
        },
        "batch_size_schedule": "base-12gb-fp32"
    },
    "model": {
        "type": "simple-classifier",
        "model": {
            "type": "from_archive",
            "archive_file": "umls_data/SUCCESS/retrained.tar.gz",
        },
        "bert_dim": 768,
        // Concatenates representations of words indexed by index_a and
        //   index_b in dataset reader to pooled representation for prediction.
        "concat_word_a_b": false,
        "include_cls": true,
        "metric_a": {
            "type": "microf1",
            "negative_label": 0
        },
        "num_labels": 2,
        "task": "classification"
    },
    "train_data_path": "/home/data/dataset/cola/train.tsv",
    "validation_data_path": "/home/data/dataset/cola/dev.tsv",
    "test_data_path": "/home/data/dataset/cola/test.tsv",
    "trainer": {
        "cuda_device": 0,
        "gradient_accumulation_batch_size": batch_size,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": num_epochs,
            "num_steps_per_epoch": num_documents / batch_size
        },
        "num_epochs": num_epochs,
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
        "validation_metric": "+micro_f1"
    },
    "vocabulary": {
        "directory_path": "/home/users/gpiat/Documents/projects/KnowBert-UMLS/tests/fixtures/umls_vocab_stid/"
    }
}
