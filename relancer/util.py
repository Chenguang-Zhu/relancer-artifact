import statistics
import jellyfish
import signal
from contextlib import contextmanager

def countCandidateOccurrence(old_api, cand, github_mapping):
    if old_api not in github_mapping or github_mapping[old_api] is None or cand not in github_mapping[old_api]:
        return 0, 0
    cand_occur = github_mapping[old_api][cand]
    total_occur = 0
    for cand in github_mapping[old_api]:
        total_occur += github_mapping[old_api][cand]
    occurrence_percentage = cand_occur / total_occur
    return cand_occur, occurrence_percentage

def computeOccurrenceScore(old_api, cand, github_mapping):
    _, occurrence_percentage = countCandidateOccurrence(old_api, cand, github_mapping)
    occurrence_score = occurrence_percentage
    return occurrence_score

def computeTokenWiseNormalizedEditDistance(old_api, new_api):
    old_api_tokens = old_api.split('.')
    new_api_tokens = new_api.split('.')
    old_tokens_distances_list = []
    for i, old_token in enumerate(old_api_tokens):
        min_distance = 10000
        for j, new_token in enumerate(new_api_tokens):
            token_distance = jellyfish.damerau_levenshtein_distance(old_token, new_token)
            token_distance = token_distance / max(len(old_token), len(new_token))
            if token_distance < min_distance:
                min_distance = token_distance
        old_tokens_distances_list.append(min_distance)
    score = statistics.mean(old_tokens_distances_list)
    return score

def computeTokenWiseEditDistanceSimilarity(old_api, new_api):
    return 1 - computeTokenWiseNormalizedEditDistance(old_api, new_api)

def computeTokenWiseLCSSimilarity(old_api, new_api):
    old_api_tokens = old_api.split('.')
    new_api_tokens = new_api.split('.')
    old_tokens_scores_list = []
    for old_token in old_api_tokens:
        max_similarity = 0
        for new_token in new_api_tokens:
            lcs_similarity = computeLCSSimilarity(old_token, new_token)
            if lcs_similarity > max_similarity:
                max_similarity = lcs_similarity
        old_tokens_scores_list.append(max_similarity)
    score = statistics.mean(old_tokens_scores_list)
    return score

def computeNaiveFQNEditDistanceSimilarity(old_api, new_api):
    fqn_distance = jellyfish.hamming_distance(old_api, new_api)
    normalized_fqn_distance = fqn_distance / max(len(old_api), len(new_api))
    return 1 - normalized_fqn_distance

def computeLCSSimilarity(old_api, new_api):
    similarity = 2 * len(lcs(old_api, new_api)) / (len(old_api) + len(new_api))
    return similarity

def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[None] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

        # Following code is used to print LCS
    index = L[m][n]
    lcs = [""] * (index + 1)
    lcs[index] = ""
    i = m
    j = n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs

def logFileHasError(log_file):
    with open(log_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        if 'Error: ' in l.strip() and ' Error: ' not in l.strip():
            if 'SyntaxError: ' in l.strip():
                return False
            else:
                return True
    else:
        return False

@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        #print('Time Out!')
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

def raise_timeout(signum, frame):
    raise TimeoutError

def checkIfStringInFile(s, f):
    with open(f, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        if s in l:
            return True
    return False