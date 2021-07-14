import collections

from util import countCandidateOccurrence
from util import computeOccurrenceScore
from util import computeTokenWiseNormalizedEditDistance
from util import computeTokenWiseEditDistanceSimilarity
from util import computeTokenWiseLCSSimilarity
from util import computeNaiveFQNEditDistanceSimilarity

# --- Deprecated ---
def rankCandidatesUsingCombinedScores(old_api, all_candidates, github_mapping, apidoc_mapping):
    all_candidates_with_final_scores = collections.OrderedDict({})
    scores_list = []
    for cand in all_candidates:
        if old_api not in github_mapping or github_mapping[old_api] is None or cand not in github_mapping[old_api]:
            g = 0
        else:
            #g = computeTokenWiseLCSSimilarity(old_api, cand)
            g = computeOccurrenceScore(old_api, cand, github_mapping)
        if old_api not in apidoc_mapping or apidoc_mapping[old_api] is None or cand not in apidoc_mapping[old_api]:
            d = 0
        else:
            d = computeTokenWiseEditDistanceSimilarity(old_api, cand)
            #d = computeTokenWiseLCSSimilarity(old_api, cand)
        g = round(g, 2)
        d = round(d, 2)
        occurrence, occurrence_percentage = countCandidateOccurrence(old_api, cand, github_mapping)
        occurrence_percentage = round(occurrence_percentage, 2)
        last_token_similarity = computeTokenWiseEditDistanceSimilarity(old_api.split('.')[-1], cand.split('.')[-1])
        last_token_similarity = round(last_token_similarity, 2)
        if occurrence_percentage >= 0.5 and occurrence >= 5:
            combined_score = 1.0
            scores_list.append((cand, combined_score, g, d, occurrence, occurrence_percentage, last_token_similarity))
            continue
        #last_token_similarity = computeTokenWiseLCSSimilarity(old_api.split('.')[-1], cand.split('.')[-1])
        if last_token_similarity < 0.5:
            combined_score = 0.0
            scores_list.append((cand, combined_score, g, d, occurrence, occurrence_percentage, last_token_similarity))
            continue
        if d == 1.0:
            combined_score = 1.0
            scores_list.append((cand, combined_score, g, d, occurrence, occurrence_percentage, last_token_similarity))
            continue
        if last_token_similarity == 1:
            combined_score = (g + 2 * d) / 3
            combined_score = round(combined_score, 2)
            scores_list.append((cand, combined_score, g, d, occurrence, occurrence_percentage, last_token_similarity))
            continue
        combined_score = (g + d) / 2
        combined_score = round(combined_score, 2)
        scores_list.append((cand, combined_score, g, d, occurrence, occurrence_percentage, last_token_similarity))
    scores_list = sorted(scores_list, key=lambda x: x[1], reverse=True)
    for s in scores_list:
        cand = s[0]; combined_score = s[1]; g = s[2]; d = s[3]; occurrence = s[4]; occurrence_percentage = s[5]; last_token_similarity = s[6]
        all_candidates_with_final_scores[cand] = collections.OrderedDict({})
        all_candidates_with_final_scores[cand]['combined'] = combined_score
        all_candidates_with_final_scores[cand]['github'] = g
        all_candidates_with_final_scores[cand]['apidoc'] = d
        all_candidates_with_final_scores[cand]['occurrence'] = occurrence
        all_candidates_with_final_scores[cand]['occurrence_percentage'] = occurrence_percentage
        all_candidates_with_final_scores[cand]['last_token_similarity'] = last_token_similarity
    return all_candidates_with_final_scores

def rankCandidatesUsingGitHubOccurrence(old_api, all_candidates, github_mapping):
    all_candidates_with_final_scores = collections.OrderedDict({})
    scores_list = []
    for cand in all_candidates:
        if old_api not in github_mapping or cand not in github_mapping[old_api]:
            g = 0
        else:
            #g = computeTokenWiseLCSSimilarity(old_api, cand)
            g = computeOccurrenceScore(old_api, cand, github_mapping)
        g = round(g, 2)
        scores_list.append((cand, g))
    scores_list = sorted(scores_list, key=lambda x: x[1], reverse=True)
    for s in scores_list:
        cand = s[0]; g = s[1]
        all_candidates_with_final_scores[cand] = collections.OrderedDict({})
        all_candidates_with_final_scores[cand]['github'] = g
    return all_candidates_with_final_scores

def rankCandidatesUsingTokenWiseEditDistanceSimilarity(old_api, all_candidates, apidoc_mapping):
    all_candidates_with_final_scores = collections.OrderedDict({})
    scores_list = []
    for cand in all_candidates:
        if old_api not in apidoc_mapping or cand not in apidoc_mapping[old_api]:
            d = 0
        else:
            d = computeTokenWiseEditDistanceSimilarity(old_api, cand)
            # d = computeTokenWiseLCSSimilarity(old_api, cand)
        d = round(d, 2)
        scores_list.append((cand, d))
    scores_list = sorted(scores_list, key=lambda x: x[1], reverse=True)
    for s in scores_list:
        cand = s[0]; d = s[1];
        all_candidates_with_final_scores[cand] = collections.OrderedDict({})
        all_candidates_with_final_scores[cand]['apidoc'] = d
    return all_candidates_with_final_scores

def rankCandidatesUsingGitHubNaiveFQNEditDistanceSimilarity(old_api, all_candidates, github_mapping):
    all_candidates_with_final_scores = collections.OrderedDict({})
    scores_list = []
    for cand in all_candidates:
        if old_api not in github_mapping or cand not in github_mapping[old_api]:
            g = 0
        else:
            g = computeNaiveFQNEditDistanceSimilarity(old_api, cand)
        g = round(g, 2)
        scores_list.append((cand, g))
    scores_list = sorted(scores_list, key=lambda x: x[1], reverse=True)
    for s in scores_list:
        cand = s[0]
        g = s[1]
        all_candidates_with_final_scores[cand] = collections.OrderedDict({})
        all_candidates_with_final_scores[cand]['github'] = g
    return all_candidates_with_final_scores

def rankCandidatesUsingAPIDocNaiveFQNEditDistanceSimilarity(old_api, all_candidates, apidoc_mapping):
    all_candidates_with_final_scores = collections.OrderedDict({})
    scores_list = []
    for cand in all_candidates:
        if old_api not in apidoc_mapping or cand not in apidoc_mapping[old_api]:
            d = 0
        else:
            d = computeNaiveFQNEditDistanceSimilarity(old_api, cand)
        d = round(d, 2)
        scores_list.append((cand, d))
    scores_list = sorted(scores_list, key=lambda x: x[1], reverse=True)
    for s in scores_list:
        cand = s[0]
        d = s[1]
        all_candidates_with_final_scores[cand] = collections.OrderedDict({})
        all_candidates_with_final_scores[cand]['apidoc'] = d
    return all_candidates_with_final_scores


# Tune this function to change how we select top candidate
def selectBestCandidate_Deprecated(old_api, newly_added_apis):
    print('Old API: ' + old_api)
    print('All candidates: ' + str(newly_added_apis))
    ranked_new_api_candidates = rankCandidates_Deprecated(old_api, newly_added_apis)
    top_candidate = ranked_new_api_candidates[0]
    print('Top candidate: ' + top_candidate)
    return top_candidate

def rankCandidates_Deprecated(old_api, new_api_candidates):  # TODO: we can tune this function
    scores_list = []
    for new_api in new_api_candidates:
        score = computeTokenWiseNormalizedEditDistance(old_api, new_api)
        scores_list.append((new_api, score))
    scores_list = sorted(scores_list, key=lambda x: (x[1], len(x[0])))
    #for s in scores_list:
    #    print(s)
    ranked_new_api_candidates = collections.OrderedDict({})
    for s in scores_list:
        new_api = s[0]
        ranked_new_api_candidates[new_api] = s[1]
    return ranked_new_api_candidates