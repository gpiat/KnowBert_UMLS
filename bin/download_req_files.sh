#!/bin/bash
# wordnet linker
curl -o vocabulary_wordnet.tar.gz https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/vocabulary_wordnet.tar.gz
curl -o entities.jsonl https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/entities.jsonl
curl -o semcor_and_wordnet_examples.json https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/semcor_and_wordnet_examples.json
curl -o wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5 https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5
curl -o wordnet_synsets_mask_null_vocab.txt https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/wordnet_synsets_mask_null_vocab.txt

# wiki linker
curl -o vocabulary_wiki.tar.gz https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/vocabulary_wiki.tar.gz
curl -o aida_train.txt https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/aida_train.txt
curl -o aida_dev.txt https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/aida_dev.txt
curl -o entities_glove_format.gz https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/entities_glove_format.gz



