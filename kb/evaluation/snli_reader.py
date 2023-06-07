import json

from typing import Iterable

from kb.bert_tokenizer_and_candidate_generator import\
    TokenizerAndCandidateGenerator
from allennlp.data.fields import LabelField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

LABEL_MAP = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}


@DatasetReader.register('snli_reader')
class SNLIReader(DatasetReader):

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
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            dataset = [json.loads(line) for line in lines]

        foo = self.tokenizer_and_candidate_generator
        for document in dataset:
            # no worries about the join, n2c2 is whitespace tokenized
            token_candidates = foo.tokenize_and_generate_candidates(
                document['sentence1'], document['sentence2'])
            # token_candidates has fields:
            # ['tokens', 'segment_ids', 'candidates', 'offsets_a', 'offsets_b']

            # convert_tokens_candidates_to_fields does some AllenNLP black
            # magic on tokens, segment_ids and candidates
            fields = foo.convert_tokens_candidates_to_fields(
                token_candidates)
            fields['label_ids'] = LabelField(
                LABEL_MAP[document['gold_label']], skip_indexing=True)
            yield Instance(fields)
