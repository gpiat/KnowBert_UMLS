import numpy as np

from typing import Iterable

from kb.bert_tokenizer_and_candidate_generator import\
    TokenizerAndCandidateGenerator
from allennlp.data.fields import ArrayField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

LABEL_MAP = {
    'O': 0,
    'B-test': 1,
    'I-test': 2,
    'B-problem': 3,
    'I-problem': 4,
    'B-treatment': 5,
    'I-treatment': 6,
    '[PAD]': 7
}


@DatasetReader.register('iob2_ner_reader')
class IOB2NERReader(DatasetReader):

    def __init__(self,
                 tokenizer_and_candidate_generator:
                 TokenizerAndCandidateGenerator,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.tokenizer_and_candidate_generator =\
            tokenizer_and_candidate_generator
        self.tokenizer_and_candidate_generator.whitespace_tokenize = True

    def _read(self, file_path: str) -> Iterable[Instance]:
        for seq in iob2_seq_generator(file_path):
            foo = self.tokenizer_and_candidate_generator
            # no worries about the join, n2c2 is whitespace tokenized
            token_candidates = foo.tokenize_and_generate_candidates(
                ' '.join(seq['tokenized_text']))
            # token_candidates has fields:
            # 'tokens', 'segment_ids', 'candidates', 'offsets_a', 'offsets_b']

            # offsets_a contains the list of indices of the first wordpiece
            # every token has been turned into except the first which is always
            # at index 1 (with index 0 being the CLS token) + it includes the
            # final [SEP] token which we don't want.
            offsets = [1] + token_candidates['offsets_a'][:-1]
            # getting labels per wordpiece: iterating through labels as we find
            # token-starting wordpieces.
            label_it = iter(seq['labels'])
            # I would love to use a listcomp here but there's a really subtle
            # quirk of python that makes that not possible
            # stackoverflow.com/questions/13905741/
            wordpiece_labels = []
            for i in range(len(token_candidates['tokens'])):
                if i not in offsets:
                    wordpiece_labels.append('[PAD]')
                else:
                    wordpiece_labels.append(next(label_it))

            # convert_tokens_candidates_to_fields does some AllenNLP black
            # magic on tokens, segment_ids and candidates
            fields = foo.convert_tokens_candidates_to_fields(
                token_candidates)

            label_ids = [LABEL_MAP[label] for label in wordpiece_labels]
            fields['label_ids'] = ArrayField(np.array(label_ids), dtype=np.int)

            yield Instance(fields)


def iob2_seq_generator(fname: str, split_iob: bool = False):
    """ Args:
            fname: file name to read from
            split_iob: wither to split the I- and B- tags from the labels
        Yield:
            dict:
                {'gold_spans': [[start_mention_1, end_mention_1],
                               [start_mention_2, end_mention_2]...],
                 'labels': [label_1, label_2...],
                 'tokenized_text': ['token_1', 'token_2'...]}
                the end of the mention index is inclusive
    """
    def init_dict():
        """ Utility function for initializing the dictionary that
            represents each sequence.
        """
        return {
            'gold_spans': [],
            'labels': [],
            'tokenized_text': [],
        }

    def close_span(gold_annotations):
        """ This funciton automatically looks at the most recently created
            entity span and closes it if need be, assuming the most recently
            found token is the last token in the span.
        """
        # checking whether the span needs to be closed
        if len(gold_annotations['gold_spans']) > 0 and\
           len(gold_annotations['gold_spans'][-1]) == 1:
            gold_annotations['gold_spans'][-1].append(
                # The span is represented as [begin, end] indices, both
                # inclusive. The end of the span is thus the index of the
                # last token. Accounting for 0-based indexing, this is
                # current length - 1
                len(gold_annotations['tokenized_text']) - 1
            )

    def start_span(gold_annotations):
        """ Starts a new mention span assuming the most recently found
            token is the first.
        """
        gold_annotations['gold_spans'].append(
            [len(gold_annotations['tokenized_text']) - 1])

    with open(fname, 'r') as f:
        lines = f.readlines()

    gold_annotations = init_dict()
    # Depending on whether the token starts, continues or is outside of a
    # mention respectively, each line is formatted as:
    # token  <TAB>  B-C0XXXXXX
    # token  <TAB>  I-C0XXXXXX
    # token  <TAB>  O
    for line in lines:
        line = line.strip()
        # if line is blank, start new sequence.
        if line == '':
            new_dict = init_dict()
            # This condition quietly handles cases where the dataset
            # has two blank lines without returning an empty dict.
            if gold_annotations != new_dict:
                close_span(gold_annotations)
                yield gold_annotations
                gold_annotations = new_dict
            continue

        token, tag = line.split('\t')
        gold_annotations['tokenized_text'].append(token)
        if tag == 'O':
            # we've already taken care of the token so in this case we
            # just need to make sure the last span is closed, in case
            # the previous token was part of a mention
            close_span(gold_annotations)
            gold_annotations['labels'].append(tag)
        else:
            marker, label = tag.split('-')
            if marker == 'B':
                # If this is the beginning of a mention span, we need to
                # make sure the previous one is closed
                close_span(gold_annotations)
                start_span(gold_annotations)
            if split_iob:
                gold_annotations['labels'].append(label)
            else:
                gold_annotations['labels'].append(tag)
            # if this is inside a mention, we've already done everything
            # we need to do.
