from kb.testing import get_sharded_reader as get_reader

from allennlp.common import Params
from allennlp.data import DataIterator
from allennlp.data import Vocabulary

import numpy as np

import os
import pickle


reader = get_reader()
np.random.seed(5)
instances = reader.read("tests/fixtures/bert_pretraining/shard*.txt")
vocab = Vocabulary.from_params(Params({
    "directory_path": "tests/fixtures/pregen"
}))
iterator = DataIterator.from_params(Params({"type": "basic"}))
iterator.index_with(vocab)
instance_list = []
# unt 25
# b /home/users/gpiat/.conda/envs/knowbert/lib/python3.6/site-packages/allennlp/data/iterators/data_iterator.py:144
# b /home/users/gpiat/.conda/envs/knowbert/lib/python3.6/site-packages/allennlp/data/iterators/basic_iterator.py:23
# b /home/users/gpiat/.conda/envs/knowbert/lib/python3.6/site-packages/allennlp/data/iterators/data_iterator.py:216
# b /home/users/gpiat/.conda/envs/knowbert/lib/python3.6/site-packages/allennlp/common/tqdm.py:43

for instance in iterator(instances, num_epochs=1, shuffle=False):
    instance_list.append(instance)

os.makedirs("test_results", exist_ok=False)
with open("test_results/batchtestfile.pkl", 'wb') as f:
    pickle.dump(instance_list, f)


