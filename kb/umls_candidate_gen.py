from typing import List
from kb.common import MentionGenerator
from quickumls import QuickUMLS

from kb.umls_candgen_utils import construct_cui_counts
from kb.umls_candgen_utils import flatten_n_filter_cand_ids_for_span
from kb.umls_candgen_utils import get_cui_priors
from kb.umls_candgen_utils import process_matcher_output
from kb.umls_candgen_utils import SpaCyMatcher
from kb.umls_candgen_utils import UMLSSpacyPreprocessor


@MentionGenerator.register("scispacy_mention_generator")
class SciSpaCyUMLSCandidateMentionGenerator(MentionGenerator):
    def __init__(self, cui_count_file: str,
                 max_number_candidates: int = 30,
                 count_smoothing: int = 1,
                 sim_thresh: float = 0.4,
                 spaCy_name: str = "en_core_sci_sm",
                 target: str = "cui"):
        assert target in ["cui", "semtype"], (
            "SciSpaCyUMLSCandidateMentionGenerator target invalid. "
            "Should be 'cui' or 'semtype'.")
        self.target = target
        self._raw_data_processor = UMLSSpacyPreprocessor()
        self._raw_data_processor_whitespace = UMLSSpacyPreprocessor(
            whitespace_tokenize_only=True
        )
        self._cui_counts = construct_cui_counts(cui_count_file)
        self._max_number_candidates = max_number_candidates
        self._count_smoothing = count_smoothing
        self.matcher = SpaCyMatcher(spaCy_name, sim_thresh,
                                    max_number_candidates)

    def get_mentions_raw_text(self, text: str,
                              whitespace_tokenize: bool = False,
                              allow_empty_candidates: bool = False):
        candidate_sets = self.matcher.match(text)

        if whitespace_tokenize:
            tokenizer = self._raw_data_processor_whitespace
        else:
            tokenizer = self._raw_data_processor

        return process_matcher_output(
            candidate_sets,
            text,
            tokenizer,
            cui_counts=self._cui_counts,
            smoothing=self._count_smoothing,
            max_number_candidates=self._max_number_candidates,
            allow_empty_candidates=allow_empty_candidates,
            target=self.target)

    def get_candidates_given_gold_spans(self, gold_annotations, strict=True):
        tokenized_text = gold_annotations['tokenized_text']

        candidates = {
            'tokenized_text': tokenized_text,
            'candidate_spans': gold_annotations['gold_spans'],
            'candidate_entities': [],
            'candidate_entity_priors': []
        }

        for start, end in gold_annotations['gold_spans']:
            entities = self._get_candidate_ents_for_span(
                tokenized_text[start:(end + 1)],
                strict)
            candidates['candidate_entities'].append(entities)
            candidates['candidate_entity_priors'].append(
                get_cui_priors(entities, self._cui_counts,
                               self._count_smoothing))

        return candidates

    def _get_candidate_ents_for_span(self, text: List[str], strict=True):
        candidate_str = ' '.join([t for t in text if t != '-'])
        candidate_sets = self.matcher.match(candidate_str)
        return flatten_n_filter_cand_ids_for_span(
            text, candidate_sets, strict, self.target)


@MentionGenerator.register("quick_umls_mention_generator")
class QuickUMLSCandidateMentionGenerator(MentionGenerator):
    """ Generate lists of candidate entities. Provides several methods that
        process input text of various format to produce mentions.

        Each text is represented by:
                {'tokenized_text': List[str],
                 'candidate_spans': List[List[int]] list of (start, end)
                        indices for candidates, where span is
                        tokenized_text[start:(end + 1)]
                 'candidate_entities': List[List[str]] = for each entity,
                        the candidates to link to. value is synset id, e.g
                        able.a.02 or hot_dog.n.01
                 'candidate_entity_priors': List[List[float]]
            }
        Args:
    """

    def __init__(self, cui_count_file: str,
                 max_entity_length: int = 7,
                 max_number_candidates: int = 30,
                 count_smoothing: int = 1,
                 qUMLS_fp: str = "/home/data/dataset/UMLS/QuickUMLS/",
                 qUMLS_thresh: float = 0.7,
                 similarity_name: str = "cosine"):

        self._raw_data_processor = UMLSSpacyPreprocessor()
        self._raw_data_processor_whitespace = UMLSSpacyPreprocessor(
            whitespace_tokenize_only=True
        )

        self._cui_counts = construct_cui_counts(cui_count_file)
        self._max_entity_length = max_entity_length
        self._max_number_candidates = max_number_candidates
        self._count_smoothing = count_smoothing

        self.quickmatcher = QuickUMLS(
            quickumls_fp=qUMLS_fp,
            threshold=qUMLS_thresh,
            similarity_name=similarity_name,
            window=max_entity_length)

    def get_mentions_raw_text(self, text: str,
                              whitespace_tokenize: bool = False,
                              allow_empty_candidates: bool = False):
        """
        """
        # best_match is set to False to return a greater variety
        # of candidate spans (+ recall, - precision)
        candidate_sets = self.quickmatcher.match(text, best_match=False)

        if whitespace_tokenize:
            tokenizer = self._raw_data_processor_whitespace
        else:
            tokenizer = self._raw_data_processor

        return process_matcher_output(
            candidate_sets,
            text,
            tokenizer,
            cui_counts=self._cui_counts,
            smoothing=self._count_smoothing,
            max_number_candidates=self._max_number_candidates,
            allow_empty_candidates=allow_empty_candidates)

    def get_candidates_given_gold_spans(self, gold_annotations, strict=True):
        tokenized_text = gold_annotations['tokenized_text']

        candidates = {
            'tokenized_text': tokenized_text,
            'candidate_spans': gold_annotations['gold_spans'],
            'candidate_entities': [],
            'candidate_entity_priors': []
        }

        for start, end in gold_annotations['gold_spans']:
            cuis = self._get_candidate_cuis_for_span(
                tokenized_text[start:(end + 1)],
                strict)
            candidates['candidate_entities'].append(cuis)
            candidates['candidate_entity_priors'].append(
                get_cui_priors(cuis, self._cui_counts, self._count_smoothing))

        return candidates

    def _get_candidate_cuis_for_span(self, text: List[str], strict=True):
        candidate_str = ' '.join([t for t in text if t != '-'])
        candidate_sets = self.quickmatcher.match(candidate_str,
                                                 best_match=True)
        return flatten_n_filter_cand_ids_for_span(
            text, candidate_sets, strict)
