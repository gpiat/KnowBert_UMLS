import re

from math import ceil
from math import floor
from quickumls import QuickUMLS
from unidecode import unidecode

# QUICKUMLSMATCHER = QuickUMLS("/scratch_global/gpiat/QuickUMLS/")

def jaro(query, key):
    # If the s are equal
    if (query == key):
        return 1.0
  
    # Length of two s
    len1 = len(query)
    len2 = len(key)
  
    # Maximum distance upto which matching
    # is allowed
    max_dist = floor(max(len1, len2) / 2) - 1
  
    # Count of matches
    match = 0
  
    # Hash for matches
    hash_query = [0] * len(query)
    hash_key = [0] * len(key)
  
    # Traverse through the first
    for i in range(len1):
  
        # Check if there is any matches
        for j in range(max(0, i - max_dist), 
                       min(len2, i + max_dist + 1)):
              
            # If there is a match
            if (query[i] == key[j] and hash_key[j] == 0):
                hash_query[i] = 1
                hash_key[j] = 1
                match += 1
                break
  
    # If there is no match
    if (match == 0):
        return 0.0
  
    # Number of transpositions
    t = 0
    point = 0
  
    # Count number of occurances
    # where two characters match but
    # there is a third matched character
    # in between the indices
    for i in range(len1):
        if (hash_query[i]):
  
            # Find the next matched character
            # in second
            while (hash_key[point] == 0):
                point += 1
  
            if (query[i] != key[point]):
                point += 1
                t += 1
    t = t//2
  
    # Return the Jaro Similarity
    return (match/ len1 + match / len2 + 
            (match - t + 1) / match)/ 3.0


def jarowinkler_equivalent(query, key, threshold=0.95):
    """ Given two strings and an optional threshold (default = 0.9),
        returns whether the two strings are close enough by the JW
        metric to be considered equivalent.
        Implements pruning described by
        http://www.semantic-web-journal.net/system/files/swj1128.pdf
    """
    # winkler boost threshold
    WBT = 0.7
    # winkler weight factor
    WWF = 0.1
    jaro_upperbound = 2 / 3 + len(query) / (3 * len(key))

    if threshold <= WBT and jaro_upperbound > threshold:
        return jaro(query, key) > threshold

    # If the jaro Similarity is above the winkler boost threshold
    if jaro_upperbound > WBT:
        # Find the length of common prefix 
        prefix = 0; 
        for i in range(min(len(query), len(key), 4)):
            # If the characters match 
            if (query[i] == key[i]) :
                prefix += 1; 
            # Else break 
            else :
                break;
        weighted_prefix = WWF * prefix
        jarowinkler_upperbound = jaro_upperbound + weighted_prefix * (1 - jaro_upperbound)
        if jarowinkler_upperbound > threshold:
            jaro_sim = jaro(query, key)
            # Calculate jaro winkler Similarity 
            jaro_sim += weighted_prefix * (1 - jaro_sim);
            return jaro_sim > threshold
        else:
            return False
    else:
        return False


def gen_ngrams(s: str, n: int = 3):
    """ Args:
            s: the string to split into n-grams
            n: the number of characters in the n-grams
    """
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def tfidf_ngram_equivalent(query, key):
    pass


def get_candidates_from_str(
    concepts: list,
    search_term: str,
    eq_function=lambda q, k: q == k
):
    """ Args:
            - concepts: list of dictionaries, each of the form
                {"cui": "C0XXXXXX", "str": "common medical expression"}
            - search_term: a string to search for among the "str" fields
                of the candidates
            - eq_function: a function which returns True if its two string
                arguments are "equivalent", else False. Defaults to the
                == operator, but can be replaced with a more sophisticated
                similarity measure.
    """
    search_term = unidecode(search_term).lower()
    # TODO:
    # "Pseudomonas aeruginosa (Pa) infection" won't be picked up on despite
    # "Pseudomonas aeruginosa infection" being in UMLS strings. The search
    # may also have malformed hyphenated terms. We might thus want to find
    # intelligent variations on the search term or compute a diff and set
    # a similarity cutoff or some other form of fuzzy matching like
    # Sim-String instead of an ==.
    return [concept for concept in concepts
            if eq_function(search_term, concept[1])]


# def quick_umls_candidate_cuis(search_term):
#     return QUICKUMLSMATCHER.match(search_term)



def find_candidate_cuis(
    concepts: list,
    search_term: str,
    eq_function=lambda q, k: q == k
):
    """ Args:
            - concepts: list of tuples, each of the form
                ("C0XXXXXX", "common medical expression") where "C0XXXXXX"
                is the concept CUI corresponding to the string
            - search_term: a string to search for among the "str" fields
                of the candidates
            - eq_function: a function which returns True if its two string
                arguments are "equivalent", else False. Defaults to the
                == operator, but can be replaced with a more sophisticated
                similarity measure.
    """
    search_term = unidecode(search_term).lower()
    return set([concept[0] for concept in
                get_candidates_from_str(concepts, search_term, eq_function)])


# HEADERS_MRCONSO = [
#     "cui",
#     "lat",
#     "ts",
#     "lui",
#     "stt",
#     "sui",
#     "ispref",
#     "aui",
#     "saui",
#     "scui",
#     "sdui",
#     "sab",
#     "tty",
#     "code",
#     "str",
#     "srl",
#     "suppress",
#     "cvf",
# ]

# def umls_concept_loader(
#     mrconso_file: str = None,
#     language: str = "ENG",
# ):
#     with open(mrconso_file, "r", encoding="UTF-8") as input_file:
#         for line in input_file:
#             content = line.strip().split("|")
#             cui, lat, str_ = content[0], content[1], content[14]

#             if lat != language:
#                 continue

#             str_ = unidecode(str_).lower()

#             # I tried using a dict here, but a tuple is much more
#             # memory efficient, which is important when we're
#             # talking about about gigabytes worth of text. Using a
#             # dict here crashes my PC.
#             yield cui, str_
