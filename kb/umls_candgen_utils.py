import json
import numpy as np
import spacy

from collections import defaultdict
from collections import OrderedDict
from typing import List

from kb.common import WhitespaceTokenizer
from kb.common import get_empty_candidates

# These may look unused, but are required for abbreviation detection
# and "Entity Linking" (candidate generation) in SpaCy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker


def construct_cui_counts(cui_count_file):
    cui_counts = defaultdict(lambda: 0)
    with open(cui_count_file, 'r') as f:
        cui_counts.update(json.load(f))
    return cui_counts


def get_cui_priors(cuis, cui_counts, smoothing):
    cui_counts = [(cui_counts[c] + smoothing)
                  for c in cuis]
    total_count = sum(cui_counts)
    return [p / total_count for p in cui_counts]


class UMLSSpacyPreprocessor:
    """
    A "preprocessor" that really does POS tagging and lemmatization
    using spacy, plus some hand crafted rules.

    allennlp tokenizers take strings and return lists of Token classes.
    we'll run spacy first, then modify the POS / lemmas as needed, then
    return a new list of Token
    """

    def __init__(self, whitespace_tokenize_only: bool = False):
        self.nlp = spacy.load('en_core_web_sm',
                              disable=['tagger', 'parser', 'ner', 'textcat'])
        if whitespace_tokenize_only:
            self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)

    def __call__(self, text: str) -> List[spacy.tokens.token.Token]:
        spacy_doc = self.nlp(text)

        # create allennlp tokens
        normalized_tokens = [
            spacy_token
            for spacy_token in spacy_doc
            if not spacy_token.is_space
        ]

        return normalized_tokens

    @staticmethod
    def char_to_token_index(char_spans, tokens):
        """ This function takes in a list of pairs of character indices for a
            given text and maps it to corresponding pairs of token
            indices.
            Edge cases:
                if a character span starts or ends within a token, the span is
                    stretched to include the entire token.
                if a span ends before it starts, rather than failing, the span
                    end will become the end of the text.
            Args:
                char_spans: a list of pairs of character indices. The pairs
                    can represented by lists or tuples.
                tokens: a list of spacy tokens
            Return:
                char_to_tok_spans: a dict mapping (char_start, char_end)
                    to (tok_start, tok_end) or None if the span is invalid.
        """
        char_spans = [tuple(span) for span in char_spans]
        char_to_tok_spans = dict.fromkeys(char_spans)
        # token span ends are inclusive
        for start, end in char_spans:
            for i, tok in enumerate(tokens):
                if tok.idx <= start:
                    tok_start = tok.i
                # if current token starts after the end of the span
                elif tok.idx > end:
                    # set the token-based end of the span to the current
                    # token - 1 since the end index is inclusive
                    # (yes, it really is, I've checked)
                    char_to_tok_spans[start, end] = (tok_start, tok.i - 1)
                    break
                # this handles the case where the span ends after the
                # beginning of the last token. This is not mutually
                # exclusive with that token being the starting token.
                if i == len(tokens) - 1:
                    char_to_tok_spans[start, end] = (tok_start, tok.i)
        return char_to_tok_spans


class SpaCyMatcher:
    def __init__(self, spaCy_name, threshold, max_candidates):
        self.nlp = spacy.load(spaCy_name)
        abbreviation_pipe = AbbreviationDetector(self.nlp)
        self.nlp.add_pipe(abbreviation_pipe)
        self.linker = EntityLinker(
            resolve_abbreviations=True,
            name="umls",
            threshold=threshold,
            max_entities_per_mention=max_candidates)
        self.nlp.add_pipe(self.linker)

    def match(self, text):
        processed = self.nlp(text)
        candidate_sets = []
        for entity in processed.ents:
            candidate_sets.append([
                {'cui': cui,
                 'start': entity.start_char,
                 'end': entity.end_char,
                 'similarity': similarity,
                 'term': self.linker.kb.cui_to_entity[cui].canonical_name,
                 'semtype': self.linker.kb.cui_to_entity[cui].types}
                for cui, similarity in entity._.kb_ents]
            )
        return candidate_sets


def process_matcher_output(candidate_sets: list,
                           text: str,
                           tokenizer: UMLSSpacyPreprocessor,
                           cui_counts: list,
                           smoothing: int = 1,
                           max_number_candidates: int = 30,
                           allow_empty_candidates: bool = False,
                           target: str = 'cui'):
    assert target in ['cui', 'semtype'], (
        "Bad target argument, should be 'cui' or 'semtype'")
    tokenized = tokenizer(text)

    # getting what we care about and flattening
    if target == 'semtype':
        candidates = {(c['start'], c['end'], stid)
                      for candset in candidate_sets
                      for c in candset
                      for stid in c[target]}
    else:
        candidates = {(c['start'], c['end'], c[target])
                      for candset in candidate_sets
                      for c in candset}

    # getting a sorted list of only the candidate spans.
    candidate_spans = list({(c['start'], c['end']) for candset
                            in candidate_sets
                            for c in candset})
    candidate_spans.sort()

    char_to_token_idx_lookup = UMLSSpacyPreprocessor.char_to_token_index(
        candidate_spans, tokenized)
    rejected_spans = [span for span in candidate_spans
                      if char_to_token_idx_lookup[span] is None]
    # we turn the spans in the list of candidate spans from character-based
    # indexing to token-based indexing. This may cause duplicate spans.
    candidate_spans = [char_to_token_idx_lookup[span]
                       for span in candidate_spans
                       if char_to_token_idx_lookup[span] is not None]

    if rejected_spans != []:
        print(f"Warning: found rejected spans: {rejected_spans}")
        print(f"for text: {text}")

    # isolating a list of candidate CUIs for each span,
    # whilst respecting ordering
    candidate_entities = OrderedDict()
    for cs in candidate_spans:
        candidate_entities[cs] = []
    for c in candidates:
        if (c[0], c[1]) not in rejected_spans:
            ent = candidate_entities[
                char_to_token_idx_lookup[c[0], c[1]]
            ]
            ent.append(c[2])

    # removing duplicate candidate spans (as dict keys are unique)
    candidate_spans = list(candidate_entities.keys())
    # candidate_spans now has the keys, all we need to keep is the
    # list of values. Since candidate_entities is an ordered dict,
    # there is a guaranteed 1-to-1 correspondence.
    candidate_entities = list(candidate_entities.values())

    # iterating through lists of candidate CUIs and getting relevant priors
    candidate_entity_priors = [
        get_cui_priors(candidate_cuis, cui_counts, smoothing)
        for candidate_cuis in candidate_entities
    ]

    # removing candidates past the max_number_candidates
    for i in range(len(candidate_entity_priors)):
        if len(candidate_entity_priors[i]) > max_number_candidates:
            candidate_order = np.argsort(candidate_entity_priors[i])
            # keeping only the top N candidates
            candidate_order =\
                candidate_order[-max_number_candidates:]
            # extracting priors and corresponding entities
            candidate_entity_priors[i] =\
                [candidate_entity_priors[i][j] for j in candidate_order]
            candidate_entities[i] =\
                [candidate_entities[i][j] for j in candidate_order]
            # rescaling the prior probabilities so they add to 1
            coeff = 1 / sum(candidate_entity_priors[i])
            candidate_entity_priors[i] =\
                [j * coeff for j in candidate_entity_priors[i]]

    ret = {'tokenized_text': [token.text for token in tokenized],
           'candidate_spans': candidate_spans,
           'candidate_entities': candidate_entities,
           'candidate_entity_priors': candidate_entity_priors}

    if not allow_empty_candidates and len(candidate_spans) == 0:
        # no candidates found, substitute the padding entity id
        ret.update(get_empty_candidates())
    return ret


def flatten_n_filter_cand_ids_for_span(text: List[str],
                                       candidate_sets: List,
                                       strict: bool = True,
                                       target: str = 'cui'):
    """ Helper function which, given a tokenized text and a list of
        character-indexed candidates for sub-spans, returns a flat list of
        candidate CUIs and may do some filtering.
        Args:
            text: list[str]
                tokenized text
            candidate_sets: iterable[iterable[dict]]
                candidate sets, typically grouped by sub-span. Each candidate
                is a dict with keys 'cui', 'start', and 'end'. 'start', and
                'end' should be character indexed based on the un-tokenized
                text string.
            strict: bool = True
                If True, will filter out candidates for sub-spans that do not
                include all of the tokens.
            target: str = 'cui'
                'cui' or 'semtype', depending on whether we are linking
                concepts or typing with STIDs
    """
    if strict:
        # getting CUIs for candidate sets which include all of the
        # tokens. We implement a tolerance such that if a non-zero
        # number of characters from the first and last token are
        # included in the span, it is considered correct. This corrects
        # for minor tokenization inconsistencies such as punctuation.
        str_without_last_tok = ' '.join([t for t in text[:-1] if t != '-'])

    def _verif(c):
        return (not strict or
                (c['start'] < len(text[0]) and
                 c['end'] > len(str_without_last_tok))
                )

    if target == 'cui':
        return list({c[target]
                     for candset in candidate_sets
                     for c in candset
                     if _verif(c)})
    else:
        # semtypes are nested one level deeper. I tried to be smart about this
        # but I can't seem to find a way to not use an if/else.
        return list({stid
                     for candset in candidate_sets
                     for c in candset
                     for stid in c[target]
                     if _verif(c)})
