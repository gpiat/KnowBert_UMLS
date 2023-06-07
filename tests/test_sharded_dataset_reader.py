from kb.testing import get_sharded_reader as get_reader

from allennlp.common import Params
from allennlp.data import DataIterator
from allennlp.data import Vocabulary

import numpy as np
import unittest


class TestShardedDatasetReader(unittest.TestCase):
    def test_reader(self):
        reader = get_reader()
        np.random.seed(5)
        instances = reader.read("tests/fixtures/bert_pretraining/shard*.txt")
        vocab = Vocabulary.from_params(Params({
            "directory_path": "tests/fixtures/pregen"
        }))
        iterator = DataIterator.from_params(Params({"type": "basic"}))
        iterator.index_with(vocab)
        instance_list = []
        for instance in iterator(instances, num_epochs=1, shuffle=False):
            instance_list.append(instance)
        self.assertEqual(len(instance_list), 1)
        self.assertEqual(set(instance_list[0].keys()),
                         {'tokens', 'segment_ids',
                          'candidates', 'lm_label_ids',
                          'next_sentence_label'})
        self.assertEqual(tuple(instance_list[0]['tokens']['tokens'].shape),
                         (4, 12))


if __name__ == '__main__':
    unittest.main()
