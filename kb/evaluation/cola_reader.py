from typing import Iterable

from kb.bert_tokenizer_and_candidate_generator import\
    TokenizerAndCandidateGenerator
from allennlp.data.fields import LabelField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

@DatasetReader.register('cola_reader')
class COLAReader(DatasetReader):

    def __init__(self,
                 tokenizer_and_candidate_generator:
                 TokenizerAndCandidateGenerator,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.tokenizer_and_candidate_generator =\
            tokenizer_and_candidate_generator
        # should not be necessary as specified in JSONNET
        # self.tokenizer_and_candidate_generator.whitespace_tokenize = False

    def _read(self, file_path: str) -> Iterable[Instance]:
        for seq in cola_tsv_seq_generator(file_path):
            foo = self.tokenizer_and_candidate_generator
            # no worries about the join, n2c2 is whitespace tokenized
            token_candidates = foo.tokenize_and_generate_candidates(
                seq['text'])
            # token_candidates has fields:
            # ['tokens', 'segment_ids', 'candidates', 'offsets_a', 'offsets_b']

            # convert_tokens_candidates_to_fields does some AllenNLP black
            # magic on tokens, segment_ids and candidates
            fields = foo.convert_tokens_candidates_to_fields(
                token_candidates)
            fields['label_ids'] = LabelField(seq['label'], skip_indexing=True)
            yield Instance(fields)


def cola_tsv_seq_generator(fname: str):
    """ Args:
            fname: file name to read from
        Yield:
            dict:
                {'label': 1 or 0
                 'text': string}
    """
    with open(fname, 'r') as f:
        lines = f.readlines()

    # for some reason, the following list comprehension doesn't work:
    # dataset = [{'text': text, 'label': label} for line in lines for (_, label, _, text) in line.split('\t')]
    # it crashes, interpreting (_, label, _, text) as unpacking the first element returned by split() rather than the whole thing.
    # splitting the comprehension seems to work fine.
    dataset = [line.split('\t') for line in lines]
    dataset = [{'text': text, 'label': int(label)} for (_, label, _, text) in dataset]
    for instance in dataset:
        yield instance
