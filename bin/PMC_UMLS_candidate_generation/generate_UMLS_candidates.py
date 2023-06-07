import argparse
import pickle
import time

from collections import OrderedDict
from hashlib import md5
from kb.umls_candidate_gen import UMLSCandidateMentionGenerator
from os import path
from os import system
# I know, os.system is evil.


def backup(filename):
    """ This function simply backs up a file by copying it and appending .bkp
        to the file name. This is useful if the process is killed while
        writing to the file.
    """
    system(f'cp {filename} {filename + ".bkp"}')


def save(content, filename):
    """ This writes content to a file whilst backing it up
    """
    if path.exists(filename):
        backup(filename)
    with open(filename, 'wb') as f2:
        pickle.dump(content, f2)


def init_candidates(filename):
    """ This will seek to load contents from a pre-existing file or a backup
        thereof. Rather than crashing, most common edge cases will return an
        empty dict.
    """
    backup = filename + ".bkp"
    if path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                contents = pickle.load(f)
        except EOFError:
            if path.exists(backup):
                with open(backup, 'rb') as f2:
                    contents = pickle.load(f2)
            else:
                contents = dict()
    elif path.exists(backup):
        with open(backup, 'rb') as f2:
            contents = pickle.load(f2)
    else:
        contents = dict()
    return contents


def load_sequences(in_file):
    """ This function loads the contents of the corpus file and processes
        it to extract the sequences.
    """
    with open(in_file, 'r') as f1:
        # load the entire file in one go
        sequences = [line.strip().split('\t')[1:] for line in f1.readlines()]
    # flatten the list
    sequences = [seq for line in sequences for seq in line]
    # We remove duplicate sequences without altering order.
    # Unfortunately there is no OrderedSet in collections.
    sequences = list(OrderedDict.fromkeys(sequences).keys())
    return sequences


def process_candidates(in_file, out_file, mention_generator,
                       filenum, log_file):
    start = time.time()
    candidates = init_candidates(out_file)
    sequences = load_sequences(in_file)
    # we get all the hashes for the sequences
    hashes = [md5(seq.encode()).hexdigest() for seq in sequences]
    # we find all the indices of hashes (and thus sequences,
    # as both lists are ordered) which are not yet known
    absent = [i for i, hash_ in enumerate(hashes)
              if hash_ not in candidates.keys()]
    # keeping only hashes and sequences which are not yet known
    hashes = [hashes[i] for i in absent]
    sequences = [sequences[i] for i in absent]
    # getting number of batches previously processed for logging purposes
    pickup = int(len(candidates) / 1000)
    for i, seq in enumerate(sequences):
        candidates[hashes[i]] = mention_generator.get_mentions_raw_text(
            seq, whitespace_tokenize=True)
        if i % 1000 == 0:
            with open(log_file, 'a') as f:
                print(f"{filenum}: processed {int(i/1000) + pickup}k"
                      f" sequences in {(time.time() - start)/60} minutes",
                      file=f)
            save(candidates, out_file)
    save(candidates, out_file)
    with open(log_file, 'a') as f:
        print(f"{filenum} is Done!",
              file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cui_count_file', nargs='?',
        default="/home/users/gpiat/Documents/projects/"
        "KnowBert-UMLS/umls_data/cui_counts.json")
    parser.add_argument(
        '--qUMLS_fp', nargs='?',
        default="/home/users/gpiat/Software/QuickUMLS/")
    parser.add_argument(
        '--in_dir', nargs='?',
        default='/scratch/gpiat/pmc/oa_bulk_bert_512/')
    parser.add_argument(
        '--out_dir', nargs='?',
        default='/scratch/gpiat/pmc/oa_bulk_bert_512/')
    parser.add_argument(
        '--log_file', nargs='?',
        default='candidate_generation_output.txt')
    parser.add_argument('--digits', type=int, nargs='?', default=3)
    parser.add_argument('filenum', type=int)

    args = vars(parser.parse_args())

    mention_generator = UMLSCandidateMentionGenerator(
        cui_count_file=args['cui_count_file'],
        qUMLS_fp=args['qUMLS_fp'],
        qUMLS_thresh=0.7, similarity_name="jaccard")
    in_file = path.join(args['in_dir'],
                        str(args['filenum']).zfill(args['digits']) + '.txt')
    out_file = path.join(args['out_dir'],
                         str(args['filenum']).zfill(args['digits']) + '.cand')

    process_candidates(in_file, out_file, mention_generator,
                       args['filenum'], args['log_file'])
