
"""
 Entity linking:
   Input = sequence of tokens
   Output = list of spans + entity id of linked entity
       span_indices = (batch_size, max_num_spans, 2)
       entity_id = (batch_size, max_num_spans)

 Proceeds in two steps:
   (1) candidate mention generation = generate a list of spans and possible
       candidate entitys to link to
   (2) disambiguated entity to predict


Model component is split into several sub-components.

 Candidate mention generation is off loaded to data generators, and uses
 pre-processing, dictionaries, and rules.

 EntityDisambiguation: a module that takes contextualized vectors, candidate
   spans (=mention boundaries, and candidate entity ids to link to),
   candidate entity priors and returns predicted probability for each candiate.

 EntityLinkingWithCandidateMentions: a Model that encapusulates:
    a LM that contextualizes token ids
    a EntityDisambiguation that predicts candidate mentions from LM context
        vectors
    a loss calculation for optimizing
    (optional) a KG embedding model that can be used for multitasking entity
        embeddings and the entity linker
"""


# tokenization notes:
#
# BERT tokenization:
#    - apply word tokenization
#    - apply subword tokenization
#    - truncate length
#    - add [CLS] text a [SEP]   OR   [CLS] text a [SEP] text b [SEP]
#
# For entity candidate generation, or annotated spans for entity linking
#   or NER or ...
#
# Original data is word tokenized starting with index 0 as first word.
# Original data can have [SEP] in middle but SHOULD NOT have [CLS] or
# terminal [SEP].
#
# Then use "bert-pretrained" token indexer from allennlp with
# "use_starting_offsets": True:
# {
#        "type": "bert-pretrained",
#        "pretrained_model": "tests/fixtures/bert/vocab.txt",
#        "do_lowercase": True,
#        "use_starting_offsets": True,
#        "max_pieces": 512,
#    }
# This will add [CLS] and [SEP] to original data, so first offset index will
#   be 1
#
# Should ideally do the original tokenization with the bert word splitter,
# or if orginal annotation is word split already re-tokenize it.


import torch

import copy
import random

from allennlp.common.registrable import Registrable
from allennlp.data import DatasetReader
from allennlp.data import Token
from allennlp.data import Tokenizer
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import LabelField
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.models import Model
from allennlp.nn.regularizers import RegularizerApplicator

from kb.bert_pretraining_reader import BertTokenizerCandidateGeneratorMasker
from kb.bert_tokenizer_and_candidate_generator import\
    BertTokenizerAndCandidateGenerator
from kb.common import F1Metric
from kb.common import get_dtype_for_module
from kb.dict_field import DictField


KNOWLEDGEBASES = ('wordnet', 'wiki', 'umls')


@TokenIndexer.register("characters_tokenizer")
class TokenCharactersIndexerTokenizer(TokenCharactersIndexer):

    @classmethod
    def from_params(cls, params):
        tokenizer = Tokenizer.from_params(params.pop("tokenizer"))
        ret = TokenCharactersIndexer.from_params(params)
        ret._character_tokenizer = tokenizer
        return ret


class EntityLinkingReader(DatasetReader, Registrable):
    """
    Each instance is a context of text, gold mention spans, and gold entity id.

    This is converted to tensors:
        tokens: dict -> token id from the token indexer (batch_size, num_times)
        candidate_spans: -> list of (start, end) indices of each span to make
            a prediction for, (batch_size, num_spans, 2)
        candidate_entites: -> list of candidate entity ids for each span,
            (batch_size, num_spans, num_candidates)
        gold_entities: list of gold entity id for span
            (batch_size, num_spans, 1)

    The model makes a prediction for each span in candidate_spans.
    Depending on whether it's desirable to use gold entity spans or have
    the model predict spans will determine whether to pass gold spans as
    candidate_spans or pass many candidate spans that have NULL entity.


    tokens is a TextField
    candidate_spans is a spanfield
    candidate_entities is a TextField that we use a vocabulary to
        do the indexing
    gold_entities is a text field
    """
    pass


class BaseEntityDisambiguator(Registrable):
    pass


class EntityLinkingBase(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 margin: float = 0.2,
                 decode_threshold: float = 0.0,
                 loss_type: str = 'margin',
                 namespace: str = 'entity',
                 regularizer: RegularizerApplicator = None):

        super().__init__(vocab, regularizer)

        if loss_type == 'margin':
            self.loss = torch.nn.MarginRankingLoss(margin=margin)
            self.decode_threshold = decode_threshold
        elif loss_type == 'softmax':
            self.loss = torch.nn.NLLLoss(ignore_index=-100)
            # set threshold to small value so we just take argmax
            self._log_softmax = torch.nn.LogSoftmax(dim=-1)
            self.decode_threshold = -990
        elif loss_type == "bce":
            self.loss = torch.nn.BCEWithLogitsLoss()
            self.decode_threshold = decode_threshold
        else:
            raise ValueError("invalid loss type, got {}".format(loss_type))
        self.loss_type = loss_type

        self.null_entity_id = self.vocab.get_token_index('@@NULL@@', namespace)
        assert self.null_entity_id != self.vocab.get_token_index('@@UNKNOWN@@',
                                                                 namespace)
        self.namespace = namespace
        self._f1_metric = F1Metric()
        self._f1_metric_untyped = F1Metric()
        # TODO: REMOVE
        self.debugprint = True

    def _compute_f1(self, linking_scores,
                    candidate_spans,
                    candidate_entities,
                    gold_entities):
        # will call F1Metric with predicted and gold entities encoded as
        # [(start, end), entity_id]

        predicted_entities = self._decode(
            linking_scores, candidate_spans, candidate_entities
        )

        # make a mask of valid predictions and non-null
        #   entities to select ids and spans
        # shape of gold_entities:
        #   (batch_size, num_spans, n_targets)
        # desired shape of gold_mask:
        #   (batch_size, num_spans, 1)
        gold_ent_mask = ((gold_entities > 0) &
                         (gold_entities != self.null_entity_id))
        # current shape of gold_ent_mask:
        #   (batch_size, num_spans, n_targets)
        # If there is a single valid target in the n_targets-length array at
        # every (batch, span) position, the mention span is valid. We therefore
        # apply an AND over the 3rd dimension of the tensor. This gives us a
        # (batch_size, num_spans) shape which we pull into a 3rd dim of len 1
        gold_spans_mask = sum([gold_ent_mask[:, :, i]
                               for i in range(gold_ent_mask.shape[-1])]
                              )[:, :, None]

        # Spans are of shape (batch_size, num_spans, 2) as each span has a
        # start and end index. We don't want tp separate these indices so we
        # duplicate the mask along its 3rd dimension.
        valid_gold_entity_spans = candidate_spans[
            torch.cat([gold_spans_mask, gold_spans_mask], dim=-1)
        ].view(-1, 2).tolist()
        # shape of valid_gold_entity_spans:
        #   (real_num_spans, 2) = (batch_size * num_spans - n_padding_spans, 2)

        # TODO: implement multi-label compatibility, probably by making
        #   valid_gold_entity_id a nested list
        # valid_gold_entity_id is a flat list of all valid entity IDs
        valid_gold_entity_id = gold_entities[gold_ent_mask].tolist()

        batch_size, num_spans, _ = linking_scores.shape
        batch_indices =\
            torch.arange(
                batch_size).unsqueeze(-1).repeat([1, num_spans])[
                    gold_spans_mask.squeeze(-1).cpu()]

        gold_entities_for_f1 = []
        predicted_entities_for_f1 = []
        gold_spans_for_f1 = []
        predicted_spans_for_f1 = []
        for k in range(batch_size):
            gold_entities_for_f1.append([])
            predicted_entities_for_f1.append([])
            gold_spans_for_f1.append([])
            predicted_spans_for_f1.append([])

        # TODO: This assumes len(valid_gold_entity_id) ==\
        #   len(valid_gold_entity_spans) which may not be true with multi-label
        for gi, gs, g_batch_index in zip(valid_gold_entity_id,
                                         valid_gold_entity_spans,
                                         batch_indices.tolist()):
            gold_entities_for_f1[g_batch_index].append((tuple(gs), gi))
            gold_spans_for_f1[g_batch_index].append((tuple(gs), "ENT"))

        for p_batch_index, ps, pi in predicted_entities:
            span = tuple(ps)
            predicted_entities_for_f1[p_batch_index].append((span, pi))
            predicted_spans_for_f1[p_batch_index].append((span, "ENT"))

        self._f1_metric_untyped(predicted_spans_for_f1, gold_spans_for_f1)
        self._f1_metric(predicted_entities_for_f1, gold_entities_for_f1)

    def _decode(self, linking_scores, candidate_spans, candidate_entities):
        # returns [[batch_index1, (start1, end1), eid1],
        #          [batch_index2, (start2, end2), eid2], ...]

        # Note: We assume that linking_scores has already had the mask
        # applied such that invalid candidates have very low score.
        # As a result, we don't need to worry about masking the valid
        # candidate spans here, since their score will be very low, and
        # won't exceed the threshold.

        # find maximum candidate entity score in each valid span
        # (batch_size, num_spans), (batch_size, num_spans)
        max_candidate_score, max_candidate_indices = linking_scores.max(dim=-1)

        # get those above the threshold
        above_threshold_mask = max_candidate_score > self.decode_threshold

        # for entities with score > threshold:
        #       get original candidate span
        #       get original entity id
        # (num_extracted_spans, 2)
        extracted_candidates = candidate_spans[above_threshold_mask]
        # (num_extracted_spans, num_candidates)
        candidate_entities_for_extracted_spans =\
            candidate_entities[above_threshold_mask]
        extracted_indices = max_candidate_indices[above_threshold_mask]
        # the batch number (num_extracted_spans, )
        batch_size, num_spans, _ = linking_scores.shape

        unfilt_BIs = torch.arange(batch_size).unsqueeze(-1).repeat(
            [1, num_spans])
        batch_indices = unfilt_BIs[above_threshold_mask.cpu()]

        extracted_entity_ids = []
        for k, ind in enumerate(extracted_indices):
            extracted_entity_ids.append(
                candidate_entities_for_extracted_spans[k, ind])

        # make tuples [(span start, span end), id], ignoring the null entity
        ret = []
        for start_end, eid, batch_index in zip(
            extracted_candidates.tolist(),
            extracted_entity_ids,
            batch_indices.tolist()
        ):
            entity_id = eid.item()
            if entity_id != self.null_entity_id:
                ret.append((batch_index, tuple(start_end), entity_id))

        return ret

    def get_metrics(self, reset: bool = False):
        precision, recall, f1_measure = self._f1_metric.get_metric(reset)
        (precision_span,
         recall_span,
         f1_measure_span) = self._f1_metric_untyped.get_metric(reset)
        metrics = {
            'el_precision': precision,
            'el_recall': recall,
            'el_f1': f1_measure,
            'span_precision': precision_span,
            'span_recall': recall_span,
            'span_f1': f1_measure_span
        }

        return metrics

    def _compute_loss(self,
                      candidate_entities,
                      candidate_spans,
                      linking_scores,
                      gold_entities):

        if self.loss_type == 'margin':
            loss = self._compute_margin_loss(candidate_entities,
                                             candidate_spans,
                                             linking_scores,
                                             gold_entities)
        elif self.loss_type == 'softmax':
            loss = self._compute_softmax_loss(candidate_entities,
                                              candidate_spans,
                                              linking_scores,
                                              gold_entities)
        elif self.loss_type == 'bce':
            loss = self._compute_bce_loss(candidate_entities,
                                          candidate_spans,
                                          linking_scores,
                                          gold_entities)
        # metrics
        self._compute_f1(linking_scores, candidate_spans,
                         candidate_entities,
                         gold_entities)
        return loss

    def _compute_margin_loss(self,
                             candidate_entities,
                             candidate_spans,
                             linking_scores,
                             gold_entities):

        # compute loss
        # in End-to-End Neural Entity Linking
        # loss = max(0, gamma - score) if gold mention
        # loss = max(0, score) if not gold mention
        #
        # torch.nn.MaxMarginLoss(x1, x2, y) = max(0, -y * (x1 - x2) + gamma)
        #   = max(0, -x1 + x2 + gamma)  y = +1
        #   = max(0, gamma - x1) if x2 == 0, y=+1
        #
        #   = max(0, x1 - gamma) if y==-1, x2=0

        candidate_mask = candidate_entities > 0
        # (num_entities, )
        non_masked_scores = linking_scores[candidate_mask]

        # broadcast gold ids to all candidates
        num_candidates = candidate_mask.shape[-1]
        # (batch_size, num_spans, num_candidates)
        broadcast_gold_entities = gold_entities.repeat(
            1, 1, num_candidates
        )
        # compute +1 / -1 labels for whether each candidate is gold
        positive_labels = (broadcast_gold_entities ==
                           candidate_entities).long()
        negative_labels = (broadcast_gold_entities !=
                           candidate_entities).long()
        labels = (positive_labels - negative_labels).to(
            dtype=get_dtype_for_module(self))
        # finally select the non-masked candidates
        # (num_entities, ) with +1 / -1
        non_masked_labels = labels[candidate_mask]

        loss = self.loss(
            non_masked_scores, torch.zeros_like(non_masked_labels),
            non_masked_labels
        )
        return {'loss': loss}

    def _compute_softmax_loss(self,
                              candidate_entities,
                              candidate_spans,
                              linking_scores,
                              gold_entities):
        # {'candidate_entities': torch.Size([32, 27, 30]),
        #  'candidate_spans': torch.Size([32, 27, 2]),
        #  'linking_scores': torch.Size([32, 27, 30]),
        #  'gold_entities': torch.Size([32, 27, 1])}
        # 32 = LM batch_size = EL batch_size = gradient_accumulation_batch_size
        # 30 = num_candidates = max_predictions_per_seq?
        # 27 = tokens? num_spans?
        #  2 = start + end indices

        # compute log softmax
        # linking scores is already masked with -1000 in invalid locations
        # (batch_size, pregen, max_num_candidates)
        log_prob = self._log_softmax(linking_scores)
        # log_prob is same shape as linking_scores

        # get the valid scores.
        # needs to be index into the last time of log_prob, with -100
        # for missing values
        num_candidates = log_prob.shape[-1]
        # gold_entities.shape          =(batch_size, num_spans, 1)
        # broadcast_gold_entities.shape=(batch_size, num_spans, num_candidates)
        broadcast_gold_entities = gold_entities.repeat(
            1, 1, num_candidates
        )

        # location of the positive label
        # positive_labels.shape = broadcast_gold_entities.shape
        #                       = candidate_entities.shape
        # candidate_entities[x][y] is a vector of candidate entities for a
        #    candidate span
        # positive_labels[x][y] is a 1-hot or 0s vector, same shape as
        #    candidate_entities, with 0s where the candidates are wrong and 1s
        #    if and where the candidate is correct.
        positive_labels = (broadcast_gold_entities ==
                           candidate_entities).long()
        # index of the positive class in every positive_labels[x][y]
        # therefore targets.shape = (batch_size, num_spans?)
        # is 0 if there is no positive class
        targets = positive_labels.argmax(dim=-1)

        # fill in the ignore class
        # DANGER: we assume that each instance has exactly one gold
        # label, and that padded instances are ones for which all
        # candidates are invalid
        # invalid_prediction_mask shape = (batch_size, num_spans)
        # invalid_prediction_mask contains True where the corresponding span
        #   has only 0s for candidate concepts, i.e. the span was not found by
        #   the candidate generator or is padding.
        invalid_prediction_mask = (
            candidate_entities != 0
        ).long().sum(dim=-1) == 0
        targets[invalid_prediction_mask] = -100

        # This would create a mask for all candidate spans that have no
        #    correct candidate concepts
        # no_mask = (
        #     positive_labels != 0
        # ).long().sum(dim=-1) == 0

        # size:      (batch_size x num_spans, num_candidates)
        # 1                                              batch_size x num_spans
        loss = self.loss(log_prob.view(-1, num_candidates), targets.view(-1, ))
        return {'loss': loss}

    def _compute_bce_loss(self,
                          candidate_entities,
                          candidate_spans,
                          linking_scores,
                          gold_entities):
        # self.vocab.get_token_index(, namespace)
        # TODO: REMOVE
        if self.debugprint:
            import pickle
            from pathlib import Path

            Path("experiments/last_logs").mkdir(parents=True, exist_ok=True)
            with open("experiments/last_logs/semtype_loss_args.pkl", 'wb') as f:
                pickle.dump({"candidate_entities": candidate_entities,
                             "candidate_spans": candidate_spans,
                             "linking_scores": linking_scores,
                             "gold_entities": gold_entities,
                             "loss": self.loss},
                            f)
            self.debugprint = False
        # like cross entropy loss, linking_scores should have shape
        #   [batch_size, num_spans, num_candidates]
        # but several candidates can be correct.

        num_candidates = linking_scores.shape[-1]

        # In regular CE (softmax) loss, gold_entities has shape
        #   (batch_size x n_spans x 1) (there is 1 correct label for each span)
        # here, gold_entities may has shape (batch_size x n_spans x n_labels).
        # => Inserting a dimension to the gold_entities tensor
        # Its new shape will be (batch_size x n_spans x 1 x n_labels)
        gold_entities = gold_entities[:, :, None, :]

        # Duplicating the gold entity list for each candidate concept.
        # Shape of broadcast_gold_entities is
        #   (batch_size x n_spans x num_candidates x n_labels)
        broadcast_gold_entities = gold_entities.repeat(
            1, 1, num_candidates, 1
        )

        # Each broadcast_gold_entities[:, :, :, i] slice has shape
        #   (batch_size x n_spans x num_candidates) where
        #   broadcast_gold_entities[j, k, :, i] is a repeating vector of
        #   size num_candidates with one of the correct labels repeated.
        # Each slice has the same shape as candidate_entities. We generate
        #   a binary tensor such that there are 1s where the candidates are
        #   equal to the repeated label of each i, and take the sum.
        # Labels occur only once per span, therefore the sum should remain
        #   a binary tensor combining the 1s of all the generated tensors.
        # The result is a binary tensor with ones in the cells corresponding
        #   to the correct types for the span and 0s elsewhere.
        targets = sum(
            [(broadcast_gold_entities[:, :, :, i] ==
              candidate_entities).long()
             for i in range(broadcast_gold_entities.shape[-1])]
        )

        # invalid_prediction_mask contains True where the corresponding span
        #   has only 0s for candidate concepts, i.e. the span was not found by
        #   the candidate generator or is padding.
        invalid_prediction_mask = (
            candidate_entities != 0
        ).long().sum(dim=-1) == 0
        targets[invalid_prediction_mask] = -100

        loss = self.loss(linking_scores, targets.float())
        return {'loss': loss}


class StubbedCandidateGenerator:
    # this is dangerous, we will stub out the candidate generator to do a look
    # up operation from the known candidates

    def set_cache(self, candidates):
        self.cached_candidates = candidates

    def get_mentions_raw_text(self, text, whitespace_tokenize=True):
        return copy.deepcopy(self.cached_candidates[text])


@DataIterator.register("cross_sentence_linking")
class CrossSentenceLinking(DataIterator):
    """
    Assumes the dataset reader is not lazy
    """

    def __init__(self,
                 batch_size: int,
                 entity_indexer: TokenIndexer,
                 bert_model_type: str,
                 do_lower_case: bool,
                 mask_candidate_strategy: str = 'none',
                 dataset_index: int = None,
                 iterate_forever: bool = False,
                 id_type: str = 'wordnet',
                 max_predictions_per_seq: int = 20,
                 use_nsp_label: bool = True,
                 max_word_piece_sequence_length: int = 512,
                 extra_id_type: str = None,
                 extra_entity_indexer: TokenIndexer = None):

        super().__init__(batch_size)

        self.batch_size = batch_size
        self.iterate_forever = iterate_forever
        self.id_type = id_type
        self.entity_indexer = {"ids": entity_indexer}
        self.dataset_index = dataset_index
        self.use_nsp_label = use_nsp_label

        assert id_type in KNOWLEDGEBASES
        entity_candidate_generators = {
            self.id_type: StubbedCandidateGenerator()}
        entity_indexers = {self.id_type: entity_indexer}

        self.extra_id_type = extra_id_type
        self.extra_entity_indexer = {"ids": extra_entity_indexer}
        if self.extra_id_type is not None:
            assert self.extra_id_type in KNOWLEDGEBASES
            entity_candidate_generators[self.extra_id_type] =\
                StubbedCandidateGenerator()
            entity_indexers[self.extra_id_type] = extra_entity_indexer

        # we will use an instance of BertTokenizerCandidateGeneratorMasker
        # to combine the instances and tokenize and mask
        tokenizer_and_candidate_generator = BertTokenizerAndCandidateGenerator(
            entity_candidate_generators=entity_candidate_generators,
            entity_indexers=entity_indexers,
            bert_model_type=bert_model_type,
            do_lower_case=do_lower_case,
            whitespace_tokenize=True,
            max_word_piece_sequence_length=max_word_piece_sequence_length
        )

        self.tokenizer_and_masker = BertTokenizerCandidateGeneratorMasker(
            tokenizer_and_candidate_generator,
            mask_candidate_strategy=mask_candidate_strategy,
            max_predictions_per_seq=max_predictions_per_seq
        )

    def _get_document_id_umls(self, instance):
        return instance.doc_id

    def _get_document_id_wordnet(self, instance):
        gold_key_ids = instance['gold_data_ids']
        if gold_key_ids[0][:18] == 'example_definition':
            # key is the full thing like 'example_definition.55'
            document_key = gold_key_ids[0]
        else:
            # key is d000.s000.t000
            document_key = gold_key_ids[0].partition('.')[0]
        return document_key

    def _get_document_id_aida(self, instance):
        return instance['doc_id'].metadata

    def _group_instances_into_documents(self, instances):
        # we need negative samples for each document, so make dict
        # documents, where each document is a list of sentences
        documents = {}
        for instance in instances:
            if self.id_type == 'wordnet':
                document_key = self._get_document_id_wordnet(instance)
            elif self.id_type == 'wiki':
                document_key = self._get_document_id_aida(instance)
            elif self.id_type == 'umls':
                document_key = self._get_document_id_umls(instance)
            if document_key not in documents:
                documents[document_key] = []
            documents[document_key].append(instance)
        return documents

    def _set_entity_candidate_generator_cache(self, instances):
        # set the cache on the entity candidate generator
        # also keep track of the gold entities.  some text is duplicated
        # with inconsistent annotations, so we'll make the assumption
        # to just keep the first annotations for each unique text string.
        cache = {}
        extra_cache = {}
        gold_cache = {}
        for instance in instances:
            text = ' '.join([t.text for t in instance['tokens'].tokens])
            if text not in cache:
                candidate_spans = [[span.span_start, span.span_end]
                                   for span in instance['candidate_spans']]
                candidate_entities = [
                    t.text.split() for t in
                    instance.fields['candidate_entities'].tokens
                ]
                candidate_entity_prior = [
                    p[p > 0].tolist()
                    for p in instance['candidate_entity_prior'].array
                ]
                candidate_segment_ids =\
                    instance.fields['candidate_segment_ids'].array.tolist()

                is_sorted = sorted(candidate_spans) == candidate_spans
                if not is_sorted:
                    sort_indices = [ele[0] for ele in sorted(
                        enumerate(candidate_spans), key=lambda x: x[1])]
                else:
                    sort_indices = list(range(len(candidate_spans)))

                sorted_candidate_spans = [candidate_spans[i]
                                          for i in sort_indices]
                sorted_candidate_entities = [
                    candidate_entities[i] for i in sort_indices]
                sorted_candidate_entity_prior = [
                    candidate_entity_prior[i] for i in sort_indices]
                sorted_candidate_segment_ids = [
                    candidate_segment_ids[i] for i in sort_indices]
                candidate = {
                    'candidate_spans': sorted_candidate_spans,
                    'candidate_entities': sorted_candidate_entities,
                    'candidate_entity_priors': sorted_candidate_entity_prior,
                    'candidate_segment_ids': sorted_candidate_segment_ids}

                cache[text] = candidate

                gold_cache[text] = copy.deepcopy(
                    [instance['gold_entities'].tokens[i].text
                     for i in sort_indices]
                )

                # extra candidate generator
                if self.extra_id_type is not None:
                    extra_candidates =\
                        instance['extra_candidates'].metadata[
                            self.extra_id_type]
                    e_candidate = {
                        'candidate_spans': extra_candidates['candidate_spans'],
                        'candidate_entities':
                        extra_candidates['candidate_entities'],
                        'candidate_entity_priors':
                        extra_candidates['candidate_entity_priors'],
                        'candidate_segment_ids':
                        [0] * len(extra_candidates['candidate_spans']),
                    }
                    extra_cache[text] = e_candidate

        tcg = self.tokenizer_and_masker.tokenizer_and_candidate_generator
        tcg.candidate_generators[self.id_type].set_cache(cache)
        if self.extra_id_type is not None:
            tcg.candidate_generators[self.extra_id_type].set_cache(
                extra_cache)

        return gold_cache

    def _combine_instances(self, instance_a, instance_b,
                           nsp_label, gold_cache):
        text_a = ' '.join([t.text for t in instance_a['tokens'].tokens])
        text_b = ' '.join([t.text for t in instance_b['tokens'].tokens])

        fields = self.tokenizer_and_masker.tokenize_candidates_mask(
            text_a, text_b)
        candidate_spans = [
            [s.span_start, s.span_end]
            for s in fields[
                'candidates'].field_dict[
                self.id_type].field_dict[
                'candidate_spans'].field_list
        ]
        assert sorted(candidate_spans) == candidate_spans

        # combine the gold entities
        golds = []
        for text in [text_a, text_b]:
            golds.append(gold_cache[text])

        combined_golds = []
        j = [-1, -1]
        for span in candidate_spans:
            i = fields['segment_ids'].array[span[0]]
            j[i] += 1
            combined_golds.append(golds[i][j[i]])

        gold_text_field = TextField(
            [Token(g) for g in combined_golds],
            token_indexers=self.entity_indexer
        )
        fields['gold_entities'] = DictField({self.id_type: gold_text_field})

        if self.use_nsp_label:
            fields['next_sentence_label'] = LabelField(nsp_label,
                                                       skip_indexing=True)

        del fields['lm_label_ids']

        return Instance(fields)

    def _create_batches(self, instances, shuffle: bool = True):
        if self.iterate_forever:
            num_epochs = 100000000
        else:
            num_epochs = 1

        documents = self._group_instances_into_documents(instances)
        document_keys = list(documents.keys())

        # set the cache on the entity candidate generator
        # AND return the gold cache
        gold_cache = self._set_entity_candidate_generator_cache(instances)

        for epoch_num in range(num_epochs):
            new_instances = []
            for document_key, document_instances in documents.items():
                for k in range(len(document_instances)):
                    if (k == len(document_instances) - 1 or
                            random.random() < 0.5):
                        for _ in range(10):
                            random_key = random.choice(document_keys)
                            if random_key != document_key:
                                break

                        random_doc = documents[random_key]
                        instance_b = random.choice(random_doc)
                        nsp_label = 1

                    else:
                        # actual next
                        instance_b = document_instances[k + 1]
                        nsp_label = 0

                    instance_a = document_instances[k]

                    new_instances.append(
                        self._combine_instances(
                            instance_a, instance_b, nsp_label, gold_cache)
                    )

            random.shuffle(new_instances)

            start = 0
            while start < len(new_instances):
                end = start + self.batch_size
                yield Batch(new_instances[start:end])
                start = end

    def __call__(self, *args, **kwargs):
        for batch in super().__call__(*args, **kwargs):
            if self.dataset_index is not None:
                batch['dataset_index'] = torch.tensor(self.dataset_index)
            yield batch
