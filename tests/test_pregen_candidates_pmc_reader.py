from kb.testing import get_bert_retrain_pregen_reader_with_kg as get_reader

from allennlp.common import Params
from allennlp.data import DataIterator
from allennlp.data import Vocabulary

import numpy as np
import unittest
import torch

# from kb.bert_pretraining_reader import BertPreTrainingPreGenCandidatesReader
# from kb.bert_tokenizer_and_candidate_generator import BertTokenizerAndCandidateGenerator
# from kb.umls import UMLSCandidateMentionGenerator
# from allennlp.data import DatasetReader


class TestBertPretrainingPregenReader(unittest.TestCase):
    def test_create_masked_lm_predictions(self):
        reader = get_reader(masked_lm_prob=0.5)
        np.random.seed(5)

        tokens, lm_labels = reader._tokenizer_masker.create_masked_lm_predictions(
            "The original tokens in the sentence .".split()
        )

        expected_tokens = ['The', '[MASK]', '[MASK]',
                           'in', '[MASK]', 'sentence', '[MASK]']
        expected_lm_labels = ['[PAD]', 'original',
                              'tokens', '[PAD]', 'the', '[PAD]', '.']

        self.assertEqual(expected_tokens, tokens)
        self.assertEqual(expected_lm_labels, lm_labels)

    def test_reader_can_run_with_full_mask_strategy(self):
        reader = get_reader('full_mask', masked_lm_prob=0.5)
        instances = reader.read("tests/fixtures/bert_pretraining/pmc_test_sentences.txt")
        self.assertEqual(len(instances), 2)

    def test_reader(self):
        reader = get_reader(masked_lm_prob=0.15)

        np.random.seed(5)
        instances = reader.read("tests/fixtures/bert_pretraining/pmc_test_sentences.txt")

        vocab = Vocabulary.from_params(Params({
            "directory_path": "tests/fixtures/pregen"
        }))
        iterator = DataIterator.from_params(Params({"type": "basic"}))
        iterator.index_with(vocab)

        for batch in iterator(instances, num_epochs=1, shuffle=False):
            break

        actual_tokens_ids = batch['tokens']['tokens']
        # TODO: execute up to here, load candidate file with pickle and look
        # up the test sentences in there. Remove unneeded candidates.
        # TODO: figure out if I'm even using entity vocab properly in main project
        """
        import pickle
        from hashlib import md5
        with open('tests/fixtures/bert_pretraining/cand_files/pmc_test_sentences.cand', 'rb') as f:
            cands=pickle.load(f)
        instance_tokens = [instances[i]['tokens'].tokens for i in range(3)]
        instance_strs = [i.text for i in instance_tokens]
        with open("tests/fixtures/bert_pretraining/pmc_test_sentences.txt", 'r') as f:
            sentences = [s for line in f for s in line.strip().split('\t')[1:]]
        hashes = [md5(sent.encode()).hexdigest() for sent in sentences]
        [cands[hash] for hash in hashes]
        """
        expected_tokens_ids = torch.tensor(
            [[94, 68, 80, 78, 18, 76, 67,  4,  8, 74, 72, 96, 18, 17, 75, 25, 95,  2,
              18,  9, 79, 47,  2, 96, 20, 96, 96, 25, 96, 31, 18, 64, 72, 71, 38, 96,
              96, 96, 81,  1, 22, 41, 55, 51, 87, 35, 91,  1, 65, 31, 51, 24, 35, 59,
              14,  1, 57, 82, 89, 31, 14, 51, 62, 35, 96,  1, 43, 25, 95,  0,  0,  0,
               0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [94, 68, 96, 36, 92, 18, 86, 70, 34, 81, 69, 23, 93,  3, 31, 69, 23, 93,
              48, 31, 42, 23, 93,  3, 96, 14, 42, 23, 93, 48,  1, 53, 44, 21, 49, 96,
               1, 77, 49, 87, 14, 66, 49, 31,  1,  6, 88,  1, 81, 96, 23, 93, 90, 31,
              69, 96, 93, 16, 31, 42, 96, 93, 90, 31, 14, 42, 23, 93, 81, 25, 95,  1,
              52,  1, 27, 23, 18, 13,  5, 96, 29, 14, 63, 33,  1, 95]])

        self.assertEqual(actual_tokens_ids.tolist(),
                         expected_tokens_ids.tolist())

        actual_entities = batch['candidates']['umls']['candidate_entities']['ids']
        expected_entities = torch.tensor(
            [[[ 58, 120,   0,   0],
              [ 83, 109,   0,   0],
              [ 88,   0,   0,   0],
              [ 37,  69, 112,   0],
              [ 97,   0,   0,   0],
              [ 13, 134,   0,   0],
              [ 40,  16,  70,  92],
              [  4,   0,   0,   0],
              [ 56,   0,   0,   0],
              [  4,   0,   0,   0],
              [ 76, 128, 137,  79],
              [ 32,   0,   0,   0]],

             [[ 50,   6,   0,   0],
              [ 50,   6,   0,   0],
              [111,  17, 118,   0],
              [118, 111,  17,   0],
              [ 50,   6,   0,   0],
              [  6,  50,   0,   0],
              [ 31,  87,   0,   0],
              [ 15, 116, 114,   0],
              [104,  65,  64,   0],
              [122,   0,   0,   0],
              [ 54,   0,   0,   0],
              [ 39,  82,   0,   0]]])
        self.assertEqual(actual_entities.tolist(), expected_entities.tolist())

        expected_spans = torch.tensor(
            [[[ 5,  5], [ 6,  7], [ 6,  8], [ 8,  8],
              [10, 10], [14, 15], [48, 49], [56, 56],
              [57, 57], [64, 64], [65, 65], [66, 67]],

             [[10, 14], [15, 19], [20, 24], [26, 29],
              [49, 53], [54, 58], [74, 74], [77, 77],
              [78, 80], [79, 80], [80, 80], [82, 82]]])
        actual_spans = batch['candidates']['umls']['candidate_spans']
        self.assertEqual(actual_spans.tolist(), expected_spans.tolist())

        expected_lm_labels = torch.tensor(
            [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 60,  0,  0,  0,  0,  0,  0,
               0,  0,  0,  0,  0, 54,  0, 25, 24,  0, 24,  0,  0,  0,  0,  0,  0, 12,
               2,  1,  0,  0,  0,  0,  0,  0,  0, 35,  0,  0,  0,  0,  0,  0,  0,  0,
               0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 57,  0,  0,  0,  0,  0,  0,  0,
               0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
               0,  0,  0,  0, 93,  0, 31,  0,  0, 23,  0,  0,  0,  0,  0,  0,  0, 31,
               0,  0,  0, 31,  0,  0,  0,  0,  0,  0,  0,  0,  0, 69,  0, 93,  0,  0,
               0, 23,  0,  0,  0,  0, 23,  0,  0,  0,  0,  0,  0,  0, 16,  0,  0,  0,
               0,  0,  0, 61,  0,  0,  0, 37,  0,  0,  0,  0,  0,  0]])
        actual_lm_labels = batch['lm_label_ids']['lm_labels']
        self.assertEqual(actual_lm_labels.tolist(),
                         expected_lm_labels.tolist())

        expected_segment_ids = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        self.assertEqual(batch['segment_ids'].tolist(),
                         expected_segment_ids.tolist())
        self.assertTrue(batch['segment_ids'].dtype == torch.long)


if __name__ == '__main__':
    unittest.main()
