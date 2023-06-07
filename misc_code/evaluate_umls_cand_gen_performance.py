from allennlp.data import Token
from collections import OrderedDict
from umls_query import find_candidate_cuis
from umls_query import jarowinkler_equivalent
# from umls_query import umls_concept_loader

from collections import defaultdict
from quickumls import QuickUMLS
from sys import argv
from typing import List

import json
import spacy
import time


# entity_file = "/scratch_global/DATASETS/umls/2019AA/META/MRCONSO.RRF"
# entity_file = "/home/gpiat/Documents/Datasets/UMLS/MRCONSO_SHORT.RRF"
entity_file = "/scratch_global/gpiat/MRCONSO_SHORT.RRF"

with open(entity_file, 'r') as f:
    _candidate_list = [tuple(l.split('|')) for l in f]
_candidate_list = [(c, s.strip()) for c, s in _candidate_list]
# _candidate_list = list(umls_concept_loader(mrconso_file=entity_file))
cui_count_file = "../cui_counts.json"
_cui_counts = defaultdict(lambda: 0)
with open(cui_count_file, 'r') as f:
    _cui_counts.update(json.load(f))
_max_entity_length = 5


class WordNetSpacyPreprocessor:
    """
    A "preprocessor" that really does POS tagging and lemmatization
    using spacy, plus some hand crafted rules.

    allennlp tokenizers take strings and return lists of Token classes.
    we'll run spacy first, then modify the POS / lemmas as needed, then
    return a new list of Token
    """

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm',
                              disable=['tagger', 'parser', 'ner', 'textcat'])

        self.spacy_to_wordnet_map = {
            'PROPN': 'NOUN'
        }

    def __call__(self, text: str) -> List[Token]:
        spacy_doc = self.nlp(text)

        # create allennlp tokens
        normalized_tokens = [
            Token(spacy_token.text,
                  pos_=self.spacy_to_wordnet_map.get(spacy_token.pos_,
                                                     spacy_token.pos_),
                  lemma_=spacy_token.lemma_)

            for spacy_token in spacy_doc
            if not spacy_token.is_space
        ]

        return normalized_tokens


_raw_data_processor = WordNetSpacyPreprocessor()


def get_candidate_cuis(text: List[str], eq_function=None, best_match=False):
    candidate_str = ' '.join([t for t in text if t != '-'])
    if eq_function is None:
        candidates = find_candidate_cuis(_candidate_list, candidate_str)
    elif eq_function == quickmatcher:
        matches = quickmatcher.match(candidate_str, best_match=best_match)
        # print(matches)
        candidates = {c['cui'] for candset in matches for c in candset}
        if len(candidates) > 0:
            print("quickUMLS candidates for string: " + candidate_str)
            print("extracted CUIs:")
            print(candidates)
    else:
        candidates = find_candidate_cuis(_candidate_list,
                                         candidate_str,
                                         eq_function)
    return list(candidates)


def get_cui_priors(cuis):
    _count_smoothing = 1
    cui_counts = [(_cui_counts[c] + _count_smoothing)
                  for c in cuis]
    total_count = sum(cui_counts)
    return [p / total_count for p in cui_counts]


def get_mentions_pretokenized(text: List[str], **kwargs):
    """ returns:
            {'tokenized_text': List[str],
             'candidate_spans': List[List[int]] list of (start, end)
                    indices for candidates, where span is
                    tokenized_text[start:(end + 1)]
             'candidate_entities': List[List[str]] = for each entity,
                    the candidates to link to. value is synset id, e.g
                    able.a.02 or hot_dog.n.01
             'candidate_entity_priors': List[List[float]]
        }
    """
    n = len(text)

    candidate_spans = []
    candidate_entities = []
    candidate_entity_priors = []
    # This nested loop creates EVERY text span of acceptable length,
    # i.e. every (start, end) combination such that
    # tokenized_text[start, end] < _max_entity_length
    for start in range(n):
        for end in range(start,
                         min(n, start + _max_entity_length - 1)):
            # only consider strings that don't begin/end with '-'
            # and surface forms that are different from lemmas
            if text[start] == '-' or text[end] == '-':
                continue

            candidate_cuis = get_candidate_cuis(text[start:(end + 1)],
                                                **kwargs)

            # ignore span if no candidate concepts
            if len(candidate_cuis) == 0:
                continue

            candidate_spans.append([start, end])
            candidate_entities.append(candidate_cuis)

            candidate_entity_priors.append(
                get_cui_priors(candidate_cuis))

    ret = {'tokenized_text': text,
           'candidate_spans': candidate_spans,
           'candidate_entities': candidate_entities,
           'candidate_entity_priors': candidate_entity_priors}

    return ret


def get_mentions_raw_text(text: str, **kwargs):
    """ Wrapper for get_mentions_pretokenized when input text
        is a str.
    """
    tokenized = _raw_data_processor(text)

    # tokenized : list of Token objects
    tokenized_text = [token.text for token in tokenized]
    # tokenized_text : list of strings
    normed_text = [tok.lower().replace('.', '') for tok in tokenized_text]

    return get_mentions_pretokenized(normed_text, **kwargs)


def candidate_set_to_dict(candidate_set):
    candidate_dict = OrderedDict()
    for beg, end, _, cui in sorted(candidate_set):
        if (beg, end) in candidate_dict.keys():
            candidate_dict[(beg, end)].append(cui)
        else:
            candidate_dict[(beg, end)] = [cui]
    for k in candidate_dict:
        candidate_dict[k].sort()
    return candidate_dict


def do_full_quickumls(matcher, in_str, expected, expected_spans):
    start = time.time()
    candidate_set = {(c['start'], c['end'], c['ngram'], c['cui']) for candset
                     # best_match is set to False to return a greater variety
                     #     of candidate spans (+ recall, - precision)
                     in matcher.match(in_str, best_match=False)
                     for c in candset}
    candidate_spans = {(s, e) for s, e, _, _ in candidate_set}

    spans_tp = len(candidate_spans.intersection(expected_spans))
    spans_fp = len(candidate_spans.difference(expected_spans))
    spans_fn = len(expected_spans.difference(candidate_spans))

    cand_tp = len(candidate_set.intersection(expected))
    cand_fp = len(candidate_set.difference(expected))
    cand_fn = len(expected.difference(candidate_set))

    print("\tTP\tFP\tFN\tP\tR")
    print(f"Spans:\t{spans_tp}\t{spans_fp}\t{spans_fn}\t"
          f"{round(spans_tp / (spans_tp + spans_fp), 2)}\t"
          f"{round(spans_tp / (spans_tp + spans_fn), 2)}")
    print(f"CUIs:\t{cand_tp}\t{cand_fp}\t{cand_fn}\t"
          f"{round(cand_tp / (cand_tp + cand_fp), 2)}\t"
          f"{round(cand_tp / (cand_tp + cand_fn), 2)}")
    print("Time taken: {} sec\n".format(round(time.time() - start, 2)))
    print(candidate_set_to_dict(candidate_set))


def frange(start, end, step):
    while start <= end:
        yield start
        start += step


if __name__ == '__main__':
    expected = {(0, 37, 'Pseudomonas aeruginosa (Pa) infection', 'C0854135'),
                (41, 56, 'cystic fibrosis', 'C0010674'),
                (58, 60, 'CF', 'C0010674'),
                (106, 123, 'pulmonary disease', 'C0024115'),
                (150, 170, 'chronic Pa infection', 'C0854135'),
                (172, 175, 'CPA', 'C0854135'),
                (219, 246, 'faster rate of lung decline', 'C3160731'),
                (267, 280, 'exacerbations', 'C4086268'),
                (312, 328, 'exome sequencing', 'C3640077'),
                (386, 394, 'isoforms', 'C0597298'),
                (398, 408, 'dynactin 4', 'C4308010'),
                (410, 415, 'DCTN4', 'C4308010'),
                (431, 443, 'Pa infection', 'C0854135'),
                (447, 449, 'CF', 'C0010674'),
                (468, 487, 'respiratory disease', 'C0035204'),
                (509, 514, 'study', 'C2603343'),
                (546, 551, 'DCTN4', 'C4308010'),
                (561, 569, 'variants', 'C0597298'),
                (573, 585, 'Pa infection', 'C0854135'),
                (610, 622, 'Pa infection', 'C0854135'),
                (627, 647, 'chronic Pa infection', 'C0854135'),
                (663, 669, 'cohort', 'C0599755'),
                (679, 681, 'CF', 'C0010674'),
                (705, 711, 'centre', 'C0475309'),
                (713, 738, 'Polymerase chain reaction', 'C0032520'),
                (743, 760, 'direct sequencing', 'C3899368'),
                (781, 792, 'DNA samples', 'C0444245'),
                (797, 802, 'DCTN4', 'C4308010'),
                (803, 811, 'variants', 'C0597298'),
                (834, 836, 'CF', 'C0010674'),
                (855, 880, 'Cochin Hospital CF centre', 'C0019994'),
                (926, 930, 'CFTR', 'C1413365'),
                (965, 984, 'pulmonary infection', 'C0876973'),
                (990, 992, 'Pa', 'C0033809'),
                (1022, 1025, 'CPA', 'C0854135'),
                (1027, 1032, 'DCTN4', 'C4308010'),
                (1033, 1041, 'variants', 'C0597298'),
                (1074, 1076, 'CF', 'C0010674'),
                (1091, 1103, 'Pa infection', 'C0854135'),
                (1127, 1129, 'CF', 'C0010674'),
                (1147, 1159, 'Pa infection', 'C0854135')}
    expected_spans = {(s, e) for s, e, _, _ in expected}

    in_str = ("Pseudomonas aeruginosa (Pa) infection in cystic fibrosis (CF) "
              "patients is associated with worse long-term pulmonary disease "
              "and shorter survival, and chronic Pa infection (CPA) is "
              "associated with reduced lung function, faster rate of lung "
              "decline, increased rates of exacerbations and shorter survival."
              " By using exome sequencing and extreme phenotype design, it was"
              " recently shown that isoforms of dynactin 4 (DCTN4) may "
              "influence Pa infection in CF, leading to worse respiratory "
              "disease. The purpose of this study was to investigate the role "
              "of DCTN4 missense variants on Pa infection incidence, age at "
              "first Pa infection and chronic Pa infection incidence in a "
              "cohort of adult CF patients from a single centre. Polymerase "
              "chain reaction and direct sequencing were used to screen DNA "
              "samples for DCTN4 variants. A total of 121 adult CF patients "
              "from the Cochin Hospital CF centre have been included, all of"
              " them carrying two CFTR defects: 103 developed at least 1 "
              "pulmonary infection with Pa, and 68 patients of them had CPA. "
              "DCTN4 variants were identified in 24% (29/121) CF patients "
              "with Pa infection and in only 17% (3/18) CF patients with no "
              "Pa infection.")

    def quickmatchers():
        for sim_name in ["jaccard", "dice", "cosine"]:
            for i in frange(0.2, 0.9, 0.1):
                yield QuickUMLS(
                    quickumls_fp="/scratch_global/gpiat/QuickUMLS/",
                    threshold=round(i, 2),
                    similarity_name=sim_name,
                    window=7)

    for matcher in quickmatchers():
        print(f"QuickUMLS {matcher.similarity_name} at {matcher.threshold}:")
        do_full_quickumls(matcher, in_str, expected, expected_spans)
