testsesh
rm -r umls_data/kar_train/*
ls umls_data/kar_train/
python -m pdb ~/.conda/envs/knowbert/bin/allennlp train -s `pwd`'/umls_data/kar_train/' --file-friendly-logging --include-package kb.include_all training_config/pretraining/knowbert_umls_linker.jsonnet
b /home/users/gpiat/.conda/envs/knowbert/lib/python3.6/site-packages/allennlp/commands/train.py:182
cont

OUTPUT_DIRECTORY=`pwd`'/umls_data/knowbert_umls_finetune_i2b2_debug_2/'
rm -r $OUTPUT_DIRECTORY*
python -m pdb ~/.conda/envs/knowbert/bin/allennlp train -s $OUTPUT_DIRECTORY --file-friendly-logging --include-package kb.include_all training_config/downstream/i2b2_ner_test.jsonnet
b kb/evaluation/classification_model.py:74