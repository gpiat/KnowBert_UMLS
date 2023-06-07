from typing import Iterable

from kb.bert_tokenizer_and_candidate_generator import\
    TokenizerAndCandidateGenerator
from allennlp.data.fields import LabelField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

LABEL_MAP = {"false": 0,
             "CPR:3": 1,
             "CPR:4": 2,
             "CPR:5": 3,
             "CPR:6": 4,
             "CPR:9": 5}


@DatasetReader.register('chemprot_relex_reader')
class ChemprotRelationExtractionReader(DatasetReader):

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
        for seq in blue_tsv_seq_generator(file_path):
            foo = self.tokenizer_and_candidate_generator
            # no worries about the join, n2c2 is whitespace tokenized
            token_candidates = foo.tokenize_and_generate_candidates(
                seq['text'])
            # token_candidates has fields:
            # ['tokens', 'segment_ids', 'candidates', 'offsets_a', 'offsets_b']

            # BERT tokenizer splits the entities in the BLUE formatted ChemProt
            #   dataset's, marked as "@TYPE$", as ["@", "##TYPE", "##$"].
            # Occasionally, the entity marker will be preceded by a character
            #   such as '(', in which case the "@" will be "##@".
            # There are no occurrences of @ in the dataset that are not marked
            #   entities. Approx. 92% of sentences have 2 marked entities, and
            #   the remaining have only 1 (these are all "false" relations).
            # We can thus safely use the first and last occurrence of @ in the
            #   sequence as index_a and index_b, the representations of which
            #   are then combined in order to help with classification.
            entity_indices = [i + 1 for i, x in
                              enumerate(token_candidates['tokens'])
                              if x == "@" or x == "##@"]

            # convert_tokens_candidates_to_fields does some AllenNLP black
            # magic on tokens, segment_ids and candidates
            fields = foo.convert_tokens_candidates_to_fields(
                token_candidates)
            fields['label_ids'] = LabelField(
                LABEL_MAP[seq['label']], skip_indexing=True)
            # @s that do not mark an entity are rare to non-existent. We take
            # the first and last systematically and expect this won't be an issue.
            fields['index_a'] = LabelField(entity_indices[0],
                                           skip_indexing=True)
            fields['index_b'] = LabelField(entity_indices[-1],
                                           skip_indexing=True)
            yield Instance(fields)


def blue_tsv_seq_generator(fname: str):
    """ Args:
            fname: file name to read from
        Yield:
            dict:
                {'index': str of the form "doc_id.entity1_id.entity2_id"
                    (e.g. 10064839.T49.T56)
                 'label': relation type among possible classes in LABEL_MAP
                 'text': string of the form
                    "Lorem ipsum @CHEMICAL$ sit @GENE$ amet"
                 }
    """
    with open(fname, 'r') as f:
        lines = f.readlines()

    # removing headers
    del lines[0]

    # each line is formatted as:
    # index <TAB> sentence <TAB> label
    for line in lines:
        line = line.strip()
        index, sentence, label = line.split('\t')

        yield {
            'index': index,
            'label': label,
            'text': sentence,
        }
