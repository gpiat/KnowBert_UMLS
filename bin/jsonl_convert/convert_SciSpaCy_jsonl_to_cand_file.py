import argparse
import glob
import gzip
import json
import pickle

from hashlib import md5
from kb.umls_candgen_utils import construct_cui_counts
from kb.umls_candgen_utils import process_matcher_output
from kb.umls_candgen_utils import UMLSSpacyPreprocessor
from os import path


def rm_ext(fname):
    """ This function removes up to 2 extensions from a filename.
        Useful for gzipped or bzipped files.
    """
    return path.splitext(path.splitext(fname)[0])[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cui_count_file', nargs='?',
        default="/home/users/gpiat/Documents/projects/"
        "KnowBert-UMLS/umls_data/cui_counts.json")
    parser.add_argument(
        '--in_files', nargs='+')
    #    default='/media/pmc/jsonl_pregen_candidates/'
    #            'en_core_sci_lg-0.5-10-20220509-185326/data/*')
    parser.add_argument(
        '--target', nargs='?', default='cui')
    parser.add_argument(
        '--maxcand', nargs='?', default=30, type=int)

    args = vars(parser.parse_args())
    filenames = args['in_files']
    assert args['target'] in ['cui', 'semtypes'], (
        "Bad --target argument, should be 'cui' or 'semtypes'")

    tokenizer = UMLSSpacyPreprocessor(whitespace_tokenize_only=True)
    cui_counts = construct_cui_counts(args['cui_count_file'])

    print(f"filenames: {filenames}")
    for jsonl_file in filenames:
        candidates = {}
        output_file = rm_ext(jsonl_file) + '.cand'
        print(f"opening file {jsonl_file}")
        print(f"output file file {output_file}")
        with gzip.open(jsonl_file, 'r') as f:
            for line in f:
                canddict = json.loads(line)
                seqhash = md5(canddict['text'].strip().encode()).hexdigest()
                candidates[seqhash] =\
                    process_matcher_output(
                        canddict['candidates'],
                        canddict['text'],
                        tokenizer,
                        cui_counts=cui_counts,
                        smoothing=1,
                        max_number_candidates=args['maxcand'],
                        target=args['target'])
        print("writing to output file")
        with open(output_file, 'wb') as f:
            pickle.dump(candidates, f)
