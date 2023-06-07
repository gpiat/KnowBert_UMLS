import numpy as np
import spacy

from overrides import overrides
from random import choice
from time import sleep
from typing import Dict
from typing import List

from allennlp.data import DatasetReader
from allennlp.data import Token
from allennlp.data.fields import ArrayField
from allennlp.data.fields import ListField
from allennlp.data.fields import SpanField
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers.word_splitter import WordSplitter

from kb.common import MentionGenerator
from kb.entity_linking import EntityLinkingReader
from kb.umls_candgen_utils import get_cui_priors

from pubtatortool import PubTatorCorpus


@DatasetReader.register("umls_entity_linking")
class UMLSEntityLinkingReader(EntityLinkingReader):
    """ Dataset reader for pubtator-formatted UMLS Entity Linking corpora.
        Each instance is a context of text, gold mention spans, and gold
        entity or type id.

        This is converted to tensors:
            tokens: dict -> token id from the token indexer
                (batch_size, num_times)
            candidate_spans: -> list of (start, end) indices of each span to
                make a prediction for, (batch_size, num_spans, 2)
            candidate_entites: -> list of candidate entity ids for each span,
                (batch_size, num_spans, num_candidates)
            gold_entities: list of gold CUIs for span
                (batch_size, num_spans, 1)

        The model makes a prediction for each span in candidate_spans.
        Depending on whether the model is expected to learn Mention Detection,
        one can make the model make predictions on gold spans candidate spans.

        tokens is a TextField
        candidate_spans is a spanfield
        candidate_entities is a TextField that we use a vocabulary to
            do the indexing
        gold_entities is a text field
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 entity_indexer: TokenIndexer,
                 mention_generator: MentionGenerator = None,
                 should_remap_span_indices: bool = True,
                 multi_label_policy: str = None):
        """ Attr:
                token_indexers: Dict[str, TokenIndexer]
                entity_indexer: TokenIndexer
                mention_generator: MentionGenerator = None
                should_remap_span_indices: bool = True
                multi_label_policy: str = None:
                    needn't be specified if mentions have a single
                    label or if multiple labels are to be used with
                    binary cross entropy loss.
                    - "random" chooses a random label
                    - more modes to come
        """

        super().__init__(False)

        self.mention_generator = mention_generator
        self.token_indexers = token_indexers
        self.entity_indexer = {"ids": entity_indexer}
        self.should_remap_span_indices = should_remap_span_indices
        self.multi_label_policy = multi_label_policy

    def _read(self, file_path: str):
        # depending on the mention generator's target (cui or semtype), we
        # may want the CUIs or STIDs of the mentions.
        target_picker = {'cui': 'gold_cuis',
                         'semtype': 'gold_stids'}
        # for gold_annotations in iob2_seq_generator(file_path):
        for gold_annotations in pubtator_seq_generator(file_path):
            candidates =\
                self.mention_generator.get_candidates_given_gold_spans(
                    gold_annotations)

            # depending on the mention generator's target (cui or semtype),
            # we may want the CUIs or STIDs of the mentions
            k = target_picker[self.mention_generator.target]
            gold_entities = gold_annotations[k]

            if len(gold_entities) and isinstance(gold_entities[0], list):
                # gold_entities contains one element per mention. These can be
                # strings or lists depending on whether there are multiple
                # valid labels. If we happen to have multiple labels, we check
                # the multi-label policy and may pick only one of the correct
                # labels before carrying on in single label mode.
                if self.multi_label_policy == 'random':
                    gold_entities = [choice(entlist)
                                     for entlist in gold_entities]
                elif self.multi_label_policy == 'max_prior':
                    gold_entities = [entlist[
                        np.argmax(get_cui_priors(
                            entlist,
                            self.mention_generator._cui_counts,
                            self.mention_generator._count_smoothing))]
                        for entlist in gold_entities]

            if len(candidates['candidate_spans']) > 0:
                yield self.text_to_instance(
                    tokens=gold_annotations['tokenized_text'],
                    candidate_entities=candidates['candidate_entities'],
                    candidate_spans=candidates['candidate_spans'],
                    candidate_entity_prior=candidates[
                        'candidate_entity_priors'],
                    document_id=gold_annotations['doc_id'],
                    gold_entities=gold_entities,
                )

    def text_to_instance(self,
                         tokens: List[str],
                         candidate_entities: List[List[str]],
                         candidate_spans: List[List[int]],
                         candidate_entity_prior: List[List[float]],
                         document_id: str,
                         gold_entities: List[str] = None):

        # prior needs to be 2D and full
        # can look like [[0.2, 0.8], [1.0]]  if one candidate for second
        # candidate span and two candidates for first
        max_cands = max(len(p) for p in candidate_entity_prior)
        for p in candidate_entity_prior:
            if len(p) < max_cands:
                p.extend([0.0] * (max_cands - len(p)))
        np_prior = np.array(candidate_entity_prior)

        fields = {
            "tokens": TextField([Token(t) for t in tokens],
                                token_indexers=self.token_indexers),

            # join by space, then retokenize in the "character indexer"
            "candidate_entities": TextField(
                [Token(" ".join(candidate_list))
                 for candidate_list in candidate_entities],
                token_indexers=self.entity_indexer),
            "candidate_entity_prior": ArrayField(np.array(np_prior)),
            # only one sentence
            "candidate_segment_ids": ArrayField(
                np.array([0] * len(candidate_entities)), dtype=np.int
            )
        }

        if gold_entities is not None:
            # `label` may be a list of semantic types or a CUI.
            # You can't create a Token from a list, so if applicable
            # we do the space-joining-and-retokenizing trick.
            fields["gold_entities"] =\
                TextField([Token(label) if isinstance(label, str) else
                           Token(" ".join(label))
                           for label in gold_entities],
                          token_indexers=self.entity_indexer)

        span_fields = []
        for span in candidate_spans:
            span_fields.append(SpanField(span[0], span[1], fields['tokens']))
        fields['candidate_spans'] = ListField(span_fields)

        r = Instance(fields,
                     should_remap_span_indices=self.should_remap_span_indices)
        r.doc_id = document_id
        return r


@WordSplitter.register('semtype_tokenifier')
class JustSpacesWordSplitter(WordSplitter):
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        return [Token(t) for t in sentence.split()]

    def _remove_spaces(
        tokens: List[spacy.tokens.Token]
    ) -> List[spacy.tokens.Token]:
        return [token for token in tokens if not token.is_space]


def pubtator_seq_generator(fname: str):
    """ Args:
            fname: file name to read from
        Yield:
            dict:
                {'gold_spans': [[start_mention_1, end_mention_1],
                               [start_mention_2, end_mention_2]...],
                 'gold_cuis': [CUI_1, CUI_2...],
                 'tokenized_text': ['token_1', 'token_2'...]}
                the end of the mention index is inclusive
    """
    corpus = PubTatorCorpus([fname])
    for doc in corpus.document_list:
        if doc.pmid == '28443481':
            # At this document, if some specific unrelated settings are tweaked
            # a certain way, this document's token_to_char_lookup will have
            # a missing value at character 1296. This does not happen in debug
            # mode. I have studied this dark magic, and my best guess is that
            # tqdm is trying to log something when this character is loaded in
            # a spacy process outside the GIL, which causes this issue. In any
            # case, sleeping tqdm's logging interval time fixes the issue.
            sleep(11)
        # this code should be deleted if the ctkl[m.stop_idx - 1] fix works
        # # ctkl = [-1] * len(doc.raw_text)
        # # for tok_idx, charlist in doc.token_to_char_lookup.items():
        # #     for char_idx in charlist:
        # #         ctkl[char_idx] = tok_idx
        # # for char_idx, tok_idx in enumerate(ctkl):
        # #     if tok_idx == -1:
        # #         ctkl[char_idx] = ctkl[char_idx - 1]
        # The doc object has a dictionary that converts token indices to the
        # list of corresponding char indice. We need the opposite, a dict of
        # char indices corresponding to the proper token index. ctkl stands
        # for char_to_token_lookup
        ctkl = {
            char: tok
            for tok, charlist in doc.token_to_char_lookup.items()
            for char in charlist}

        for sentence, (s_start, s_end) in \
                zip(doc.sentences, doc.sent_start_end_indices):
            gold_annotations = {'doc_id': doc.pmid, 'tokenized_text': sentence}
            relevant_mentions = [m for m in doc.umls_entities if
                                 s_start <= m.start_idx <= m.stop_idx <= s_end]

            # span indices are at character level but token level is
            # expected since we work with the tokenized text later.
            # ctkl does the conversion for us.
            tok_s_start = ctkl[s_start]
            # the -1 in ctkl[m.stop_idx - 1] comes from the fact that mentions
            # end indices are exclusive and thus line up with spaces which
            # aren't part of a token.

            # TODO: remove try/except and reenable listcomp
            # gold_annotations['gold_spans'] = [
            #     [ctkl[m.start_idx] - tok_s_start,
            #      ctkl[m.stop_idx - 1] - tok_s_start]
            #     for m in relevant_mentions]
            try:
                gold_annotations['gold_spans'] = []
                for m in relevant_mentions:
                    tok_start = ctkl[m.start_idx] - tok_s_start
                    tok_end = ctkl[m.stop_idx - 1] - tok_s_start
                    gold_annotations['gold_spans'].append([tok_start, tok_end])
            except KeyError as e:
                print(f"gold_annotations: {gold_annotations}")
                print(f"ctkl: {ctkl}")
                print(f"offending mention: {m}")
                print("document's token-to-char lookup: ".format(
                    doc.token_to_char_lookup))
                import pickle
                stuff = {
                    'corpus': corpus,
                    'doc': doc,
                    'gold_annotations': gold_annotations,
                    "ctkl": ctkl,
                    "m": m,
                }
                with open('stid_linking_fail.pkl', 'wb') as f:
                    pickle.dump(stuff, f)
                raise e
            gold_annotations['gold_cuis'] = [m.cui for m in relevant_mentions]
            gold_annotations['gold_stids'] = [m.semantic_type_ID.split(',')
                                              for m in relevant_mentions]
            yield gold_annotations
