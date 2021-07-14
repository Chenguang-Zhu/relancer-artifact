import collections
import jellyfish
import random
import json
from relancer import getGitHubFQNCandidates
from relancer import getAPIDocFQNCandidates
from relancer import getGitHubArgNameCandidates
from relancer import getAPIDocArgNameCandidates
from relancer import getGitHubArgValueCandidates
from relancer import getAPIDocArgValueCandidates
from relancer import genFQNRenamingSolutions

def generateSolutions_GithubOnly(strategy, error_api, line_no, repair_action, project, notebook):
    from relancer import rankFQNCandidates_ML
    if repair_action['type'] == 'fqn':
        github_fqn_map_for_this_api = getGitHubFQNCandidates(error_api)
        print('GitHub Candidates:')
        print(github_fqn_map_for_this_api)
        apidoc_fqn_map_for_this_api = collections.OrderedDict({error_api: collections.OrderedDict({})})
        all_fqn_candidates_with_final_scores = rankFQNCandidates_ML(error_api, github_fqn_map_for_this_api, apidoc_fqn_map_for_this_api, strategy, project, notebook)
        all_fqn_candidates_with_final_scores_github_only = collections.OrderedDict({})
        if github_fqn_map_for_this_api[error_api] is not None:
            for new_fqn in all_fqn_candidates_with_final_scores:
                if new_fqn in github_fqn_map_for_this_api[error_api]:
                    all_fqn_candidates_with_final_scores_github_only[new_fqn] = all_fqn_candidates_with_final_scores[new_fqn]
        all_fqn_renaming_solutions = genFQNRenamingSolutions(error_api, line_no, all_fqn_candidates_with_final_scores_github_only)
        return all_fqn_renaming_solutions
    elif repair_action['type'] == 'arg_name':
        key_params_names = repair_action['key_params']
        explict_used_param_names = repair_action['explicit_used_params']
        github_solution_cands = getGitHubArgNameCandidates(error_api, key_params_names, explict_used_param_names)
        print('GitHub Candidates:')
        print(github_solution_cands)
        all_solution_cands = github_solution_cands
        return all_solution_cands
    elif repair_action['type'] == 'arg_value':
        key_params_names = repair_action['key_params']
        github_solution_cands = getGitHubArgValueCandidates(error_api, key_params_names)
        print('GitHub Candidates:')
        print(github_solution_cands)
        all_solution_cands = github_solution_cands
        return all_solution_cands

def generateSolutions_APIDocOnly(strategy, error_api, line_no, repair_action, project, notebook):
    if repair_action['type'] == 'fqn':
        github_fqn_map_for_this_api = collections.OrderedDict({error_api: collections.OrderedDict({})})
        apidoc_fqn_map_for_this_api = getAPIDocFQNCandidates(error_api)
        print('APIDoc Candidates (only show top 10):')
        print(list(apidoc_fqn_map_for_this_api[error_api].keys())[:10])
        all_fqn_candidates_with_final_scores = rankFQNCandidates_CombinationBaseline(error_api, github_fqn_map_for_this_api, apidoc_fqn_map_for_this_api, strategy, project, notebook)
        all_fqn_candidates_with_final_scores_apidoc_only = collections.OrderedDict({})
        if apidoc_fqn_map_for_this_api[error_api] is not None:
            for new_fqn in all_fqn_candidates_with_final_scores:
                if new_fqn in apidoc_fqn_map_for_this_api[error_api]:
                    all_fqn_candidates_with_final_scores_apidoc_only[new_fqn] = all_fqn_candidates_with_final_scores[new_fqn]
        all_fqn_renaming_solutions = genFQNRenamingSolutions(error_api, line_no, all_fqn_candidates_with_final_scores)
        return all_fqn_renaming_solutions
    elif repair_action['type'] == 'arg_name':
        key_params_names = repair_action['key_params']
        explict_used_param_names = repair_action['explicit_used_params']
        apidoc_solution_cands = getAPIDocArgNameCandidates(error_api, key_params_names, explict_used_param_names)
        print('APIDoc Candidates:')
        print(apidoc_solution_cands)
        all_solution_cands = apidoc_solution_cands
        return all_solution_cands
    elif repair_action['type'] == 'arg_value':
        key_params_names = repair_action['key_params']
        apidoc_solution_cands = getAPIDocArgValueCandidates(error_api, key_params_names)
        print('APIDoc Candidates:')
        print(apidoc_solution_cands)
        all_solution_cands = apidoc_solution_cands
        return all_solution_cands

def generateSolutions_CombinationBaseline(strategy, error_api, line_no, repair_action, project, notebook):
    if repair_action['type'] == 'fqn':
        github_fqn_map_for_this_api = getGitHubFQNCandidates(error_api)
        print('GitHub Candidates:')
        print(github_fqn_map_for_this_api)
        apidoc_fqn_map_for_this_api = getAPIDocFQNCandidates(error_api)
        print('APIDoc Candidates (only show top 10):')
        print(list(apidoc_fqn_map_for_this_api[error_api].keys())[:10])
        all_fqn_candidates_with_final_scores = \
            rankFQNCandidates_CombinationBaseline(error_api, github_fqn_map_for_this_api, apidoc_fqn_map_for_this_api, strategy, project, notebook)
        all_fqn_renaming_solutions = genFQNRenamingSolutions(error_api, line_no, all_fqn_candidates_with_final_scores)
        return all_fqn_renaming_solutions
    elif repair_action['type'] == 'arg_name':
        key_params_names = repair_action['key_params']
        explict_used_param_names = repair_action['explicit_used_params']
        github_solution_cands = getGitHubArgNameCandidates(error_api, key_params_names, explict_used_param_names)
        print('GitHub Candidates:')
        print(github_solution_cands)
        apidoc_solution_cands = getAPIDocArgNameCandidates(error_api, key_params_names, explict_used_param_names)
        print('APIDoc Candidates:')
        print(apidoc_solution_cands)
        all_solution_cands = apidoc_solution_cands
        all_solution_cands += [gc for gc in github_solution_cands if gc not in all_solution_cands]
        print('All Solution Candidates:')
        print(all_solution_cands)
        return all_solution_cands
    elif repair_action['type'] == 'arg_value':
        key_params_names = repair_action['key_params']
        github_solution_cands = getGitHubArgValueCandidates(error_api, key_params_names)
        print('GitHub Candidates:')
        print(github_solution_cands)
        apidoc_solution_cands = getAPIDocArgValueCandidates(error_api, key_params_names)
        print('APIDoc Candidates:')
        print(apidoc_solution_cands)
        all_solution_cands = apidoc_solution_cands
        all_solution_cands += [gc for gc in github_solution_cands if gc not in all_solution_cands]
        print('All Solution Candidates:')
        print(all_solution_cands)
        return all_solution_cands

def rankFQNCandidates_CombinationBaseline(error_api, github_fqn_map_for_this_api, apidoc_fqn_map_for_this_api, strategy, project, notebook):
    if error_api in github_fqn_map_for_this_api and github_fqn_map_for_this_api[error_api]:
        github_candidates = list(github_fqn_map_for_this_api[error_api].keys())
    else:
        github_candidates = []
    if error_api in apidoc_fqn_map_for_this_api and apidoc_fqn_map_for_this_api[error_api]:
        apidoc_candidates = list(apidoc_fqn_map_for_this_api[error_api].keys())
    else:
        apidoc_candidates = []
    all_candidates = github_candidates + [ac for ac in apidoc_candidates if ac not in github_candidates]
    print('All Candidates: ')
    print(all_candidates)
    all_fqn_candidates_with_final_scores = rankCandidatesUsingEditDistanceSimilarity(error_api, all_candidates)
    print('All FQN Candidates With Final Scores: ')
    print(all_fqn_candidates_with_final_scores)
    return all_fqn_candidates_with_final_scores

def rankCandidatesUsingEditDistanceSimilarity(old_api, all_candidates):
    all_candidates_with_final_scores = collections.OrderedDict({})
    scores_list = []
    for cand in all_candidates:
        d = computeNaiveFQNEditDistanceSimilarity(old_api, cand)
        d = round(d, 2)
        scores_list.append((cand, d))
    scores_list = sorted(scores_list, key=lambda x: x[1], reverse=True)
    for s in scores_list:
        cand = s[0]; d = s[1];
        all_candidates_with_final_scores[cand] = collections.OrderedDict({})
        all_candidates_with_final_scores[cand]['apidoc'] = d
    return all_candidates_with_final_scores

def computeNaiveFQNEditDistanceSimilarity(old_api, new_api):
    fqn_distance = jellyfish.hamming_distance(old_api, new_api)
    normalized_fqn_distance = fqn_distance / max(len(old_api), len(new_api))
    return 1 - normalized_fqn_distance

def generateSolutions_RandomActionBaseline(strategy, error_api, line_no, repair_action, project, notebook):
    randomized_actions = ['fqn', 'arg_name', 'arg_value']
    random.shuffle(randomized_actions)
    print('Randomized Actions: ' + str(randomized_actions))
    true_action = repair_action['type']
    print('True Action: ' + true_action)
    all_solution_cands = []
    action_to_cands_map = collections.OrderedDict({})
    for ra in randomized_actions:
        if ra == 'fqn':
            github_fqn_map_for_this_api = getGitHubFQNCandidates(error_api)
            print('*** Random Action ' + ra + ' GitHub Candidates: ')
            print(github_fqn_map_for_this_api)
            apidoc_fqn_map_for_this_api = getAPIDocFQNCandidates(error_api)
            print('*** Random Action ' + ra + ' APIDoc Candidates: ')
            print(list(apidoc_fqn_map_for_this_api[error_api].keys())[:10])
            from relancer import rankFQNCandidates_ML
            all_fqn_candidates_with_final_scores = \
                rankFQNCandidates_ML(error_api, github_fqn_map_for_this_api, apidoc_fqn_map_for_this_api, strategy, project, notebook)
            all_fqn_renaming_solutions = genFQNRenamingSolutions(error_api, line_no, all_fqn_candidates_with_final_scores)
            all_solution_cands += all_fqn_renaming_solutions
            action_to_cands_map[ra] = all_fqn_renaming_solutions
        elif ra == 'arg_name':
            key_params_names = []
            explict_used_param_names = []
            github_solution_cands = getGitHubArgNameCandidates(error_api, key_params_names, explict_used_param_names)
            print('*** Random Action ' + ra + ' GitHub Candidates: ' + str(len(github_solution_cands)))
            print(github_solution_cands[:10])
            apidoc_solution_cands = getAPIDocArgNameCandidates(error_api, key_params_names, explict_used_param_names)
            print('*** Random Action ' + ra + ' APIDoc Candidates: ' + str(len(apidoc_solution_cands)))
            print(apidoc_solution_cands[:10])
            all_arg_name_solution_cands = apidoc_solution_cands
            all_arg_name_solution_cands += [gc for gc in github_solution_cands if gc not in all_solution_cands]
            #print('All Solution Candidates:')
            #print(all_arg_name_solution_cands)
            all_solution_cands += all_arg_name_solution_cands
            action_to_cands_map[ra] = all_arg_name_solution_cands
        elif ra == 'arg_value':
            key_params_names = []
            github_solution_cands = getGitHubArgValueCandidates(error_api, key_params_names)
            print('*** Random Action ' + ra + ' GitHub Candidates: ' + str(len(github_solution_cands)))
            print(github_solution_cands[:10])
            apidoc_solution_cands = getAPIDocArgValueCandidates(error_api, key_params_names)
            print('*** Random Action ' + ra + ' APIDoc Candidates: ' + str(len(apidoc_solution_cands)))
            print(apidoc_solution_cands[:10])
            all_arg_value_solution_cands = apidoc_solution_cands
            all_arg_value_solution_cands += [gc for gc in github_solution_cands if gc not in all_solution_cands]
            #print('All Solution Candidates:')
            #print(all_arg_value_solution_cands)
            all_solution_cands += all_arg_value_solution_cands
            action_to_cands_map[ra] = all_arg_value_solution_cands
    need_to_try_wrong_solutions = []
    for ra in randomized_actions:
        if ra != true_action:
            need_to_try_wrong_solutions += action_to_cands_map[ra]
        if ra == true_action:
            break
    print('Have to try at least ' + str(len(need_to_try_wrong_solutions)) + ' before the correct action queue!')
    if len(need_to_try_wrong_solutions) > 2000:
        return all_solution_cands[:10]
    return all_solution_cands

def generateSolutions_NaiveBaseline(strategy, error_api, line_no, repair_action, project, notebook):
    randomized_actions = ['fqn', 'arg_name', 'arg_value']
    random.shuffle(randomized_actions)
    print('Randomized Actions: ' + str(randomized_actions))
    true_action = repair_action['type']
    print('True Action: ' + true_action)
    all_solution_cands = []
    action_to_cands_map = collections.OrderedDict({})
    for ra in randomized_actions:
        if ra == 'fqn':
            github_fqn_map_for_this_api = getGitHubFQNCandidates(error_api)
            print('*** Random Action ' + ra + ' GitHub Candidates: ')
            print(github_fqn_map_for_this_api)
            apidoc_fqn_map_for_this_api = getAPIDocFQNCandidates(error_api)
            print('*** Random Action ' + ra + ' APIDoc Candidates: ')
            print(list(apidoc_fqn_map_for_this_api[error_api].keys())[:10])
            all_fqn_candidates_with_final_scores = \
                rankFQNCandidates_CombinationBaseline(error_api, github_fqn_map_for_this_api, apidoc_fqn_map_for_this_api, strategy, project, notebook)
            all_fqn_renaming_solutions = genFQNRenamingSolutions(error_api, line_no, all_fqn_candidates_with_final_scores)
            all_solution_cands += all_fqn_renaming_solutions
            action_to_cands_map[ra] = all_fqn_renaming_solutions
        elif ra == 'arg_name':
            key_params_names = []
            explict_used_param_names = []
            github_solution_cands = getGitHubArgNameCandidates(error_api, key_params_names, explict_used_param_names)
            print('*** Random Action ' + ra + ' GitHub Candidates: ' + str(len(github_solution_cands)))
            print(github_solution_cands[:10])
            apidoc_solution_cands = getAPIDocArgNameCandidates(error_api, key_params_names, explict_used_param_names)
            print('*** Random Action ' + ra + ' APIDoc Candidates: ' + str(len(apidoc_solution_cands)))
            print(apidoc_solution_cands[:10])
            all_arg_name_solution_cands = apidoc_solution_cands
            all_arg_name_solution_cands += [gc for gc in github_solution_cands if gc not in all_solution_cands]
            #print('All Solution Candidates:')
            #print(all_arg_name_solution_cands)
            all_solution_cands += all_arg_name_solution_cands
            action_to_cands_map[ra] = all_arg_name_solution_cands
        elif ra == 'arg_value':
            key_params_names = []
            github_solution_cands = getGitHubArgValueCandidates(error_api, key_params_names)
            print('*** Random Action ' + ra + ' GitHub Candidates: ' + str(len(github_solution_cands)))
            print(github_solution_cands[:10])
            apidoc_solution_cands = getAPIDocArgValueCandidates(error_api, key_params_names)
            print('*** Random Action ' + ra + ' APIDoc Candidates: ' + str(len(apidoc_solution_cands)))
            print(apidoc_solution_cands[:10])
            all_arg_value_solution_cands = apidoc_solution_cands
            all_arg_value_solution_cands += [gc for gc in github_solution_cands if gc not in all_solution_cands]
            #print('All Solution Candidates:')
            #print(all_arg_value_solution_cands)
            all_solution_cands += all_arg_value_solution_cands
            action_to_cands_map[ra] = all_arg_value_solution_cands
    need_to_try_wrong_solutions = []
    for ra in randomized_actions:
        if ra != true_action:
            need_to_try_wrong_solutions += action_to_cands_map[ra]
        if ra == true_action:
            break
    print('Have to try at least ' + str(len(need_to_try_wrong_solutions)) + ' before the correct action queue!')
    if len(need_to_try_wrong_solutions) > 2000:
        return all_solution_cands[:10]
    return all_solution_cands
