from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.data import Token

from kb.umls import UMLSCandidateMentionGenerator
from kb.umls import UMLSSpacyPreprocessor
from kb.umls import pubtator_seq_generator

import numpy as np
import unittest


mg = UMLSCandidateMentionGenerator(
    cui_count_file="umls_data/cui_counts.json",
    qUMLS_fp="/home/data/dataset/UMLS/QuickUMLS/")


class TestUMLSSpacyPreprocessor(unittest.TestCase):
    def test_call(self):
        ppcsr = UMLSSpacyPreprocessor()
        input_text = ("DCTN4 as a modifier of chronic Pseudomonas aeruginosa "
                      "infection in cystic fibrosis")
        output = [(token.text, token.i, token.idx)
                  for token in ppcsr(input_text)]
        expected_output = [
            ('DCTN4', 0, 0), ('as', 1, 6), ('a', 2, 9), ('modifier', 3, 11),
            ('of', 4, 20), ('chronic', 5, 23), ('Pseudomonas', 6, 31),
            ('aeruginosa', 7, 43), ('infection', 8, 54), ('in', 9, 64),
            ('cystic', 10, 67), ('fibrosis', 11, 74)]
        self.assertEqual(output, expected_output)

    def test_char_to_token_index(self):
        ppcsr = UMLSSpacyPreprocessor()
        text = ("DCTN4 as a modifier of chronic Pseudomonas aeruginosa "
                "infection in cystic fibrosis")
        tokenized = ppcsr(text)

        candidate_spans = [(11, 19), (23, 30), (23, 63), (31, 42), (31, 53),
                           (31, 63), (31, 73), (54, 63), (67, 82), (74, 82)]

        output = UMLSSpacyPreprocessor.char_to_token_index(candidate_spans,
                                                           tokenized)
        expected_output = {
            (11, 19): (3, 3), (23, 30): (5, 5), (23, 63): (5, 8),
            (31, 42): (6, 6), (31, 53): (6, 7), (31, 63): (6, 8),
            (31, 73): (6, 10), (54, 63): (8, 8), (67, 82): (10, 11),
            (74, 82): (11, 11)}
        self.assertEqual(output, expected_output)


class TestUmlsCandidateGenerator(unittest.TestCase):
    def test_get_cui_priors(self):
        cuis = ['C2712105', 'C4082764', 'C0851162', 'C0012683']
        expected_output = [1 / 8, 1 / 8, 1 / 8, 5 / 8]
        output = mg.get_cui_priors(cuis)
        self.assertEqual(output, expected_output)

        # test count smoothing
        mgsmooth = UMLSCandidateMentionGenerator(
            cui_count_file="umls_data/cui_counts.json",
            qUMLS_fp="/home/data/dataset/UMLS/QuickUMLS/",
            count_smoothing=3)
        expected_output = [3 / 16, 3 / 16, 3 / 16, 7 / 16]
        output = mgsmooth.get_cui_priors(cuis)
        self.assertEqual(output, expected_output)

    def test_get_mentions_raw_text(self):
        text = "modifier of Pseudomonas"
        expected_output = {
            'tokenized_text': text.split(),
            'candidate_spans': [(0, 0), (2, 2)],
            'candidate_entities': [['C0454144', 'C3542952', 'C4284280'],
                                   ['C0033817', 'C0152972', 'C0854322',
                                    'C1450431', 'C1955913', 'C2005647',
                                    'C4087238', 'C4553236']
                                   ],
            'candidate_entity_priors': [
                [1 / 3] * 3,
                [1 / 8] * 8]}
        output = mg.get_mentions_raw_text(text)
        # sorting because candidate lists are generated in random order
        output['candidate_entities'] = [sorted(candlist) for candlist
                                        in output['candidate_entities']]
        self.assertEqual(output, expected_output)

    def test_get_candidates_given_gold_spans(self):
        gold_annotations = {
            'tokenized_text': ("DCTN4 as a modifier of chronic Pseudomonas"
                               " aeruginosa infection in cystic "
                               "fibrosis".split()),
            'gold_spans': [(0, 0), (5, 8)]
        }

        expected_strict_output = {
            'tokenized_text': gold_annotations['tokenized_text'],
            'candidate_spans': [(0, 0), (5, 8)],
            'candidate_entities': [[], []],
            'candidate_entity_priors': [[], []]}
        strict_output = mg.get_candidates_given_gold_spans(gold_annotations)
        self.assertEqual(strict_output, expected_strict_output)

        expected_lax_output = {
            'tokenized_text': gold_annotations['tokenized_text'],
            'candidate_spans': [(0, 0), (5, 8)],
            'candidate_entities': [
                [],
                ['C0008679', 'C0023474', 'C0023977', 'C0033817', 'C0149960',
                 'C0150055', 'C0241838', 'C0262421', 'C0268108', 'C0276075',
                 'C0276076', 'C0310849', 'C0581862', 'C0678355', 'C0694539',
                 'C0730525', 'C0744270', 'C0748095', 'C0854135', 'C0856722',
                 'C0867389', 'C1096223', 'C1274350', 'C1547296', 'C1555457',
                 'C1737018', 'C2029373', 'C2242816', 'C3273552', 'C3830520',
                 'C4050258', 'C4087435', 'C4303270', 'C4522269', 'C4536205']
            ],
            'candidate_entity_priors': [
                [],
                [i / 104 for i in
                 [43, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  19, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]}
        lax_output = mg.get_candidates_given_gold_spans(gold_annotations,
                                                        strict=False)

        # sorting because candidate lists are generated in random order
        sorted_orders = [np.argsort(candlist)
                         for candlist in lax_output['candidate_entities']]
        lax_output['candidate_entities'] = [
            [candlist[j] for j in sorted_orders[i]] for i, candlist in
            enumerate(lax_output['candidate_entities'])]
        lax_output['candidate_entity_priors'] = [
            [priorlist[j] for j in sorted_orders[i]] for i, priorlist in
            enumerate(lax_output['candidate_entity_priors'])]

        self.assertEqual(lax_output, expected_lax_output)

    def test_get_candidate_cuis_for_span(self):
        text = "chronic Pseudomonas aeruginosa infection".split()

        expected_strict_output = []
        strict_output = mg.get_candidate_cuis_for_span(text, strict=True)
        self.assertEqual(strict_output, expected_strict_output)

        expected_lax_output = [
            'C0008679', 'C0023474', 'C0023977', 'C0033817', 'C0149960',
            'C0150055', 'C0241838', 'C0262421', 'C0268108', 'C0276075',
            'C0276076', 'C0310849', 'C0581862', 'C0678355', 'C0694539',
            'C0730525', 'C0744270', 'C0748095', 'C0854135', 'C0856722',
            'C0867389', 'C1096223', 'C1274350', 'C1547296', 'C1555457',
            'C1737018', 'C2029373', 'C2242816', 'C3273552', 'C3830520',
            'C4050258', 'C4087435', 'C4303270', 'C4522269', 'C4536205']
        lax_output = mg.get_candidate_cuis_for_span(text, strict=False)
        # sorting because candidate lists are generated in random order
        lax_output.sort()
        self.assertEqual(lax_output, expected_lax_output)


class TestUMLSEntityLinkingReader(unittest.TestCase):
    def test_read(self):
        params = {
            "type": "umls_entity_linking",
            "mention_generator": {
                "type": "quick_umls_mention_generator",
                "cui_count_file": "umls_data/cui_counts.json",
                "max_entity_length": 7,
                "max_number_candidates": 30,
                "count_smoothing": 1,
                "qUMLS_fp": "/home/data/dataset/UMLS/QuickUMLS/",
                "qUMLS_thresh": 0.7,
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
            "should_remap_span_indices": False,
        }
        reader = DatasetReader.from_params(Params(params))
        corpus_file = "tests/fixtures/pubtator_test/pubtator_test.txt"
        instances = reader._read(corpus_file)

        # can't use instance or fields directly because assertEqual will
        # just compare objects instead of contents, so we need to extract
        # data.
        output = next(instances).fields
        output["tokens"] = [tok.text for tok in output["tokens"].tokens]
        output['candidate_entities'] = [
            candlist_as_token.text.split()
            for candlist_as_token in
            output['candidate_entities'].tokens]
        output['candidate_entity_prior'] =\
            output['candidate_entity_prior'].array.tolist()
        # removing padding because arrays are of consistent dimensions
        output['candidate_entity_prior'] = [
            # the first N elements where N is the length of
            # the corresponding candidate list
            priorlist[:len(output['candidate_entities'][i])]
            for i, priorlist in enumerate(output['candidate_entity_prior'])]
        output['candidate_segment_ids'] =\
            output['candidate_segment_ids'].array.tolist()
        output["gold_entities"] = [tok.text for tok in
                                   output["gold_entities"].tokens]
        output['candidate_spans'] = output['candidate_spans'].as_tensor(
            output['candidate_spans'].get_padding_lengths()).tolist()
        # sorting because candidate lists are generated in random order
        candidate_entities_sort = [
            np.argsort(candlist) for candlist in
            output['candidate_entities']]
        output['candidate_entities'] = [
            [candlist[j] for j in candidate_entities_sort[i]]
            for i, candlist in enumerate(output['candidate_entities'])
        ]
        output['candidate_entity_prior'] = [
            [candlist[j] for j in candidate_entities_sort[i]]
            for i, candlist in enumerate(output['candidate_entity_prior'])
        ]

        candidate_entities = [
            [], [],
            ['C0010674', 'C0016034', 'C0239946', 'C0260930', 'C0392164',
             'C0420028', 'C0428295', 'C0455371', 'C0494350', 'C0546982',
             'C0809945', 'C0948452', 'C1135187', 'C1527396', 'C1578972',
             'C1859047', 'C1997690', 'C2711060', 'C2911653', 'C3669165',
             'C3669166', 'C3825312', 'C4020619', 'C4485507', 'C4510747',
             'C4546076', 'C4546077', 'C4546078', 'C4698912']]
        token_field = (
            'DCTN 4 as a modifier of chronic Pseudomonas aeruginosa'
            ' infection in cystic fibrosis'.split())
        expected_output = {
            "tokens": token_field,
            "candidate_entities": candidate_entities,
            "candidate_entity_prior": [
                [], [],
                [n / 58 for n in
                 [20, 4, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            ],
            "candidate_segment_ids": [0] * len(candidate_entities),
            "gold_entities": ['C4308010', 'C0854135', 'C0010674'],
            "candidate_spans": [[0, 1], [6, 9], [11, 12]]
        }
        self.assertEqual(output, expected_output)

    def test_pubtator_seq_generator(self):
        corpus_file = "tests/fixtures/pubtator_test/pubtator_test.txt"
        output = next(pubtator_seq_generator(corpus_file))
        expected_output = {
            'doc_id': '25763772',
            'tokenized_text': ['DCTN', '4', 'as', 'a', 'modifier', 'of',
                               'chronic', 'Pseudomonas', 'aeruginosa',
                               'infection', 'in', 'cystic', 'fibrosis'],
            'gold_spans': [[0, 1], [6, 9], [11, 12]],
            'gold_cuis': ['C4308010', 'C0854135', 'C0010674']}
        self.assertEqual(output, expected_output)


if __name__ == '__main__':
    unittest.main()
