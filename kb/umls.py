"""
Notes on wordnet ids:
    in KG embeddings, have both synset and lemma nodes:
        synsets are keyed by something like
            able.a.01register("wordnet_mention_generator")
        each synset has a number of lemmas, keyed by something like
            able%3:00:00::

    In WSD task, you are given (lemma, pos) and asked to predict the lemma
        key, e.g. (able, adj) -> which synset do we get?

    Internally, we use the able.a.01 key for synsets, but maintain a map
    from (lemma, pos, internal key) -> external key for evaluation with semcor.
"""


import json
import numpy as np
import spacy

from collections import defaultdict
from collections import OrderedDict
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

from kb.common import MentionGenerator
from kb.common import WhitespaceTokenizer
from kb.common import get_empty_candidates

from kb.entity_linking import EntityLinkingReader

from quickumls import QuickUMLS
from pubtatortool import PubTatorCorpus


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

# Unsupervised setting for LM:
#   raw data -> use spacy to get lemma -> look up all candidates normalizing
#       - and _
#
# With annotated data:
#       at train time:
#           given gold spans and entity ids:
#               map semcor tokens to flat token sequence + gold ids +
#                   gold spans
#               look up all candidate spans using raw data approach
#                   ignoring POS and lemma
#               remove generic entity types
#               restrict candidate spans to just those that have
#                   annotated senses
#               compute the recall of gold span / entity from pruned
#                   candidate lsit (for MWE separate from single words)
#
#       at test time:
#           given gold POS and lemma, get candidates.
#           for generic entity types, use heuristic to restrict candidates
#           should have near 100% recall of gold span
#           and first sense baseline should be high


#@MentionGenerator.register("umls_mention_generator")
class UMLSCandidateMentionGenerator(MentionGenerator):
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

        self._cui_counts = defaultdict(lambda: 0)
        with open(cui_count_file, 'r') as f:
            self._cui_counts.update(json.load(f))

        self._max_entity_length = max_entity_length
        self._max_number_candidates = max_number_candidates
        self._count_smoothing = count_smoothing

        self.quickmatcher = QuickUMLS(
            quickumls_fp=qUMLS_fp,
            threshold=qUMLS_thresh,
            similarity_name=similarity_name,
            window=max_entity_length)

    def get_cui_priors(self, cuis):
        cui_counts = [(self._cui_counts[c] + self._count_smoothing)
                      for c in cuis]
        total_count = sum(cui_counts)
        return [p / total_count for p in cui_counts]

    def process_qumls_output(self, candidate_sets: list,
                             text: str,
                             whitespace_tokenize: bool = False,
                             allow_empty_candidates: bool = False):
        if whitespace_tokenize:
            tokenized = self._raw_data_processor_whitespace(text)
        else:
            tokenized = self._raw_data_processor(text)

        # getting what we care about and flattening
        candidates = {(c['start'], c['end'], c['cui'])
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
                candidate_entities[
                    char_to_token_idx_lookup[c[0], c[1]]
                ].append(c[2])

        # removing duplicate candidate spans (as dict keys are unique)
        candidate_spans = list(candidate_entities.keys())
        # candidate_spans now has the keys, all we need to keep is the
        # list of values. Since candidate_entities is an ordered dict,
        # there is a guaranteed 1-to-1 correspondence.
        candidate_entities = list(candidate_entities.values())

        # iterating through lists of candidate CUIs and getting relevant priors
        candidate_entity_priors = [self.get_cui_priors(candidate_cuis)
                                   for candidate_cuis in candidate_entities]

        # removing candidates past the max_number_candidates
        for i in range(len(candidate_entity_priors)):
            if len(candidate_entity_priors[i]) > self._max_number_candidates:
                candidate_order = np.argsort(candidate_entity_priors[i])
                # keeping only the top N candidates
                candidate_order =\
                    candidate_order[-self._max_number_candidates:]
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

    def get_mentions_raw_text(self, text: str,
                              whitespace_tokenize: bool = False,
                              allow_empty_candidates: bool = False):
        """
        """
        # best_match is set to False to return a greater variety
        # of candidate spans (+ recall, - precision)
        candidate_sets = self.quickmatcher.match(text, best_match=False)
        return self.process_qumls_output(candidate_sets, text,
                                         whitespace_tokenize,
                                         allow_empty_candidates)

    def get_candidates_given_gold_spans(self, gold_annotations, strict=True):
        """ use for training with semcor -- it will use the full unrestricted
            generator, but restrict to just the gold annotation spans.
        """
        tokenized_text = gold_annotations['tokenized_text']

        candidates = {
            'tokenized_text': tokenized_text,
            'candidate_spans': gold_annotations['gold_spans'],
            'candidate_entities': [],
            'candidate_entity_priors': []
        }

        for start, end in gold_annotations['gold_spans']:
            cuis = self.get_candidate_cuis_for_span(
                tokenized_text[start:(end + 1)],
                strict)
            candidates['candidate_entities'].append(cuis)
            candidates['candidate_entity_priors'].append(
                self.get_cui_priors(cuis))

        return candidates

    def get_candidate_cuis_for_span(self, text: List[str], strict=True):
        candidate_str = ' '.join([t for t in text if t != '-'])
        candidate_sets = self.quickmatcher.match(candidate_str,
                                                 best_match=True)

        if strict:
            str_without_last_tok = ' '.join([t for t in text[:-1] if t != '-'])
            # getting CUIS for candidate sets which are include all of the
            # tokens (even if a character like a parenthasis is missing)
            return list({c['cui']
                         for candset in candidate_sets
                         for c in candset
                         if c['start'] < len(text[0]) and
                         c['end'] > len(str_without_last_tok)})
        else:
            return list({c['cui']
                         for candset in candidate_sets
                         for c in candset})


#@DatasetReader.register("umls_entity_linking")
class UMLSEntityLinkingReader(EntityLinkingReader):
    """ Dataset reader for pubtator-formatted UMLS Entity Linking corpora.
        Each instance is a context of text, gold mention spans, and gold
        entity id.

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
                 should_remap_span_indices: bool = True
                 # extra_candidate_generators:
                 # Dict[str, MentionGenerator] = None
                 ):

        super().__init__(False)

        self.mention_generator = mention_generator
        self.token_indexers = token_indexers
        self.entity_indexer = {"ids": entity_indexer}
        self.should_remap_span_indices = should_remap_span_indices

        # self.extra_candidate_generators = extra_candidate_generators

    def _read(self, file_path: str):
        # for gold_annotations in iob2_seq_generator(file_path):
        for gold_annotations in pubtator_seq_generator(file_path):
            # TODELETE: this is leftover code from wordnet, should not be
            # necessary as we retrieve CUIs directly from gold_annotations
            # gold_span_to_entity_id = {
            #     tuple(span): cui
            #     for span, cui in zip(
            #         gold_annotations['gold_spans'],
            #         gold_annotations['gold_cuis']
            #     )
            # }
            candidates =\
                self.mention_generator.get_candidates_given_gold_spans(
                    gold_annotations)

            # TODELETE: this is leftover code from wordnet, should not be
            # necessary as we retrieve CUIs directly from gold_annotations
            # This will not return an error for false positive candidate
            # spans as the candidate spans are the same as gold spans
            # gold_entities = [
            #     gold_span_to_entity_id[tuple(candidate_span)]
            #     for candidate_span in candidates['candidate_spans']
            # ]
            gold_entities = gold_annotations['gold_cuis']

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

    # Wordnet use a gold_data_ids argument, not sure what that does.
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
            fields["gold_entities"] =\
                TextField([Token(entity) for entity in gold_entities],
                          token_indexers=self.entity_indexer)

        span_fields = []
        for span in candidate_spans:
            span_fields.append(SpanField(span[0], span[1], fields['tokens']))
        fields['candidate_spans'] = ListField(span_fields)

        # if self.extra_candidate_generators:
        #     extra_candidates = {
        #         key: generator.get_mentions_pretokenized(tokens)
        #         for key, generator in self.extra_candidate_generators.items()
        #     }
        #     fields['extra_candidates'] = MetadataField(extra_candidates)

        r = Instance(fields,
                     should_remap_span_indices=self.should_remap_span_indices)
        r.doc_id = document_id
        return r


'''
    @EntityEmbedder.register('umls_embeddings')
    class UmlsEmbeddings(torch.nn.Module, EntityEmbedder):
        """ Loads pretrained UMLS embeddings as per
            https://github.com/r-mal/umls-embeddings
            Given entity candidate list:
                - get list of unique entity ids
                - look up
                - concat embeddings
                - linear project
                - remap to candidate embedding shape
        """
        """ kb.knowbert.EntityDisambiguator uses this as self.entity_embeddings
            attribute like this:
            OPTIONAL:
                null_embedding = self.entity_embeddings.get_null_embedding()
            candidate_entity_embeddings =
                self.kg_layer_norm(
                    self.entity_embeddings(candidate_entities)
                )
            Here the UmlsAllEmbedding object is called, which is basically just
            calling `forward`.
        """

        def __init__(self,
                     embedding_file: str,
                     entity_dim: int = 50,
                     dropout: float = 0.1,
                     include_null_embedding: bool = False):
            """
            """

            super().__init__()

            self.entities = []
            entity_embeddings = []
            with open(embedding_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.entities.append(row[0])
                    entity_embeddings.append([float(i) for i in row[1:]])
            entity_embeddings = torch.tensor(entity_embeddings)

            self.entity_embeddings = torch.nn.Embedding(
                entity_embeddings.shape[0], entity_embeddings.shape[1],
                padding_idx=0
            )
            self.entity_embeddings.weight.data.copy_(
                entity_embeddings.contiguous())

            concat_dim = entity_embeddings.shape[1]

            self.proj_feed_forward = torch.nn.Linear(concat_dim, entity_dim)
            init_bert_weights(self.proj_feed_forward, 0.02)

            self.dropout = torch.nn.Dropout(dropout)

            self.entity_dim = entity_dim

            self.include_null_embedding = include_null_embedding
            if include_null_embedding:
                self.entities = (["@@PADDING@@", "@@UNKNOWN"] + self.entities +
                                 ["@@MASK@@", "@@NULL@@"])
                self.null_id = entities.index("@@NULL@@")
                self.null_embedding = torch.nn.Parameter(torch.zeros(entity_dim))
                self.null_embedding.data.normal_(mean=0.0, std=0.02)

        def get_output_dim(self):
            return self.entity_dim

        def get_null_embedding(self):
            return self.null_embedding

        def forward(self, entity_ids):
            """
            entity_ids = LongTensor containing entity ids of shape
                (batch_size, num_candidates, num_entities) array of entity

            # Ultimately the entity candidate IDs come from knowbert.py
            # line 622 in  EntityLinkingWithCandidateMentions.forward().
            # Before that, the entity candidate dict-like comes from
            #     the SolderedKG.
            # Line 736 in SolderedKG.forward(), candidate_entities (supposedly
            #     a Tensor passed as argument) is passed to its "entity_linker"
            #     attribute's forward function.
            # SolderedKG is an attribute of KnowBert model.
            # SolderedKG.forward() is called line 915 in KnowBert.forward()
            #     with the **soldered_kwargs argument which contains the
            #     candidate_entities argument, and is derived from **kwargs.
            # **kwargs is passed to KnowBert.forward() by the allennlp training
            #     script.
            # This argument is included in the dict created by the batch
            #   creator... which is CrossSentenceLinking in entity_linking.py?
            #     _set_entity_candidate_generator_cache seems to be a relevant
            #     function in CrossSentenceLinking but maybe not

            # >>> batch['candidates'].keys()
            # dict_keys(['wiki', 'wordnet'])
            # >>> batch['candidates']['wordnet'].keys()
            # dict_keys(['candidate_entity_priors', 'candidate_entities',
            #            'candidate_spans', 'candidate_segment_ids'])

            # >>> batch['candidates']['wordnet']['candidate_entity_priors'].shape
            # torch.Size([2, 5, 14])  /
            #       [n_sentences, max_n_candidate_spans, max_n_entity_candidates]
            #       Default prior: 0
            # >>> batch['candidates']['wordnet']['candidate_entity_priors'].type()
            # torch.FloatTensor

            # >>> batch['candidates']['wordnet']['candidate_entities'].keys()
            # dict_keys(['ids'])
            # >>> batch['candidates']['wordnet']['candidate_entities']['ids'].shape
            # torch.Size([2, 5, 14])  /
            #       [n_sentences, max_n_candidate_spans, max_n_entity_candidates]
            #       Default ID: 0
            # >>> batch['candidates']['wordnet']
            #          ['candidate_entities']['ids'].type()
            # 'torch.LongTensor'

            # >>> batch['candidates']['wordnet']['candidate_spans'].shape
            # torch.Size([2, 5, 2])  / contains INTs (indices in the sentence)
            #       [n_sentences, max_n_candidate_spans, (start, end))]
            #       Default span: -1, -1

            # >>> batch['candidates']['wordnet']['candidate_segment_ids']
            # 0 tensor of shape (n_sentences, max_n_candidate_spans)

            # This is generated in KnowBertBatchifier.iter_batches at
            # knowbert_utils.py. Candidates are thus generated:
            # self.tokenizer_and_candidate_generator.\
            #     tokenize_and_generate_candidates(input_sentences)
            # where tokenizer_and_candidate_generator is a
            # BertTokenizerAndCandidateGenerator from
            # bert_tokenizer_and_candidate_generator.py.
            #     > thus UMLS_tokenizer_and_candidate_generator

            returns FloatTensor of entity embeddings with shape
                (batch_size, num_candidates, num_entities, embed_dim)
            """
            # get list of unique entity ids
            unique_ids, unique_ids_to_entity_ids = torch.unique(
                entity_ids, return_inverse=True)
            # unique_ids[unique_ids_to_entity_ids].reshape(entity_ids.shape)
            # unique_ids is a 1D tensor, basically
            #     tensor(list(set(flatten(entity_ids))))
            # unique_ids_to_entity_ids is a tensor of same shape as entity_ids
            #     with the values replaced by the index that they find
            #     themselves at unique_ids

            # look up (num_unique_embeddings, full_entity_dim)
            # figure out what self.entity_embeddings is/does and modify
            #   > defined in constructor, is an nn.Embedding and
            #        related to h5py
            #   > h5py is a lib that parses large files as numpy arrays.
            #   > the embeddings are stored in one such large file
            #   > apparently forward() selects the IDs?
            embeddings = self.entity_embeddings(
                unique_ids.contiguous()).contiguous()

            # run the ff
            # (num_embeddings, entity_dim)
            projected_embeddings = self.dropout(
                # this projects embeddings thru BERT weights or something
                self.proj_feed_forward(embeddings.contiguous()))

            # replace null if needed
            if self.include_null_embedding:
                null_mask = unique_ids == self.null_id
                projected_embeddings[null_mask] = self.null_embedding

            # remap to candidate embedding shape
            return projected_embeddings[unique_ids_to_entity_ids].contiguous()
'''


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
            gold_annotations['gold_spans'] = [
                [ctkl[m.start_idx] - tok_s_start,
                 ctkl[m.stop_idx - 1] - tok_s_start]
                for m in relevant_mentions]
            gold_annotations['gold_cuis'] = [m.cui for m in relevant_mentions]
            yield gold_annotations
