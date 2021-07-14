import os
import json
import time
import requests
import filecmp
import collections
import subprocess as sub

from macros import DEPRECATED_APIS_LIST_FILE
from macros import ACTION_KEYWORDS
from macros import MINING_GITHUB_RESULTS_DIR
from macros import MINING_GITHUB_DOWNLOADS_DIR
from macros import API_FQN_MAPPING_JSON_FILE
from macros import GITHUB_ONLY_API_FQN_MAPPING_JSON_FILE
from macros import APIDOC_ONLY_API_FQN_MAPPING_JSON_FILE
from macros import GITHUB_ONLY_STRATEGY_API_FQN_MAPPING_JSON_FILE
from macros import APIDOC_ONLY_STRATEGY_API_FQN_MAPPING_JSON_FILE
from macros import GITHUB_NAIVE_STRATEGY_API_FQN_MAPPING_JSON_FILE
from macros import APIDOC_NAIVE_STRATEGY_API_FQN_MAPPING_JSON_FILE
from macros import SKLEARN_API_DOC_FILE
from macros import PANDAS_API_DOC_FILE
from macros import TENSORFLOW_API_DOC_FILE
from macros import STATSMODELS_API_DOC_FILE
from macros import SEABORN_API_DOC_FILE
from macros import NETWORKX_API_DOC_FILE
from macros import KERAS_API_DOC_FILE
from macros import SCIPY_API_DOC_FILE
from macros import PLOTLY_API_DOC_FILE
from macros import NUMPY_API_DOC_FILE
from macros import IMBLEARN_API_DOC_FILE
from macros import MAPPING_JSON_FILE

from api_extractor import extractAPIUsage

from ranking import rankCandidatesUsingCombinedScores
from ranking import rankCandidatesUsingGitHubOccurrence
from ranking import rankCandidatesUsingTokenWiseEditDistanceSimilarity
from ranking import rankCandidatesUsingGitHubNaiveFQNEditDistanceSimilarity
from ranking import rankCandidatesUsingAPIDocNaiveFQNEditDistanceSimilarity
from ranking import rankCandidates_Deprecated
from ranking import selectBestCandidate_Deprecated

from features import generateFeaturesCSV

# entrance_1
def runBuildMapping(apidoc, github, strategy):
    if github:
        runBuildMappingUsingGitHub()
    if apidoc:
        runBuildMappingUsingAPIdoc()
    postProcessMapping(strategy)

def runBuildMappingUsingGitHub(deprecated_apis_list_file=DEPRECATED_APIS_LIST_FILE):
    deprecated_apis = []
    with open(deprecated_apis_list_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        if l.strip() not in deprecated_apis:
            deprecated_apis.append(l.strip())
    for api in deprecated_apis:
        print('=== Processing old API: ' + api)
        buildMappingForOneAPIUsingGitHub(api)

def buildMappingForOneAPIUsingGitHub(api):
    commits = findAllCommitsUpdatingAPI(api)[:50]
    #print(len(commits))
    for c in commits:
        print('--- Processing commit: ' + c['sha'])
        old_project_dir, new_project_dir = cloneOldAndNewVersionProject(c, api)
        if old_project_dir is None and new_project_dir is None:  # extreme case: the first commit of repo
            continue
        changed_files = extractChangedFiles(old_project_dir, new_project_dir)
        print('Changed Files: ' + str(changed_files))
        changed_api_usage_files = findChangedAPIUsageFiles(api, changed_files, old_project_dir)
        old_file_to_api_fqns_map = extractAPIFQNsInProject(changed_api_usage_files, old_project_dir)
        new_file_to_api_fqns_map = extractAPIFQNsInProject(changed_api_usage_files, new_project_dir)
        api_mapping = computeAPIFQNMapping(api, old_file_to_api_fqns_map, new_file_to_api_fqns_map)  # save to json

def findAllCommitsUpdatingAPI(api, mining_github_results_dir=MINING_GITHUB_RESULTS_DIR):
    commits_json_file = mining_github_results_dir + '/' + api + '/commits.json'
    with open(commits_json_file, 'r') as fr:
        commits = json.load(fr)
    return commits

def cloneOldAndNewVersionProject(commit, api, mining_github_downloads_dir=MINING_GITHUB_DOWNLOADS_DIR):
    new_sha = commit['sha']
    if not commit['parents']:  # extreme case: the first commit of repo
        return None, None
    old_sha = commit['parents'][0]['sha']
    project_url = 'https://github.com/' + commit['repository']['full_name']
    project_name = commit['repository']['name']
    #print('OLD SHA: ' + old_sha)
    #print('NEW SHA: ' + new_sha)
    #print(project_url, project_name)
    cwd = os.getcwd()
    for sha in [old_sha, new_sha]:
        download_dir = mining_github_downloads_dir + '/' + api + '/' + sha
        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)
        os.chdir(download_dir)
        if os.path.isdir(download_dir + '/' + project_name):
            pass
        else:
            sub.run('git clone ' + project_url, shell=True, stdout=open(os.devnull, 'w'), stderr=sub.STDOUT)
            if not os.path.isdir(download_dir + '/' + project_name):  # commit repo becomes private
                continue
        os.chdir(download_dir + '/' + project_name)
        #print(download_dir + '/' + project_name)
        sub.run('git checkout ' + sha, shell=True, stdout=open(os.devnull, 'w'), stderr=sub.STDOUT)
    os.chdir(cwd)
    old_project_dir = mining_github_downloads_dir + '/' + api + '/' + old_sha + '/' + project_name
    new_project_dir = mining_github_downloads_dir + '/' + api + '/' + new_sha + '/' + project_name
    return old_project_dir, new_project_dir

def extractChangedFiles(old_project_dir, new_project_dir):
    changed_files = []
    for dir_path, subpaths, files in os.walk(old_project_dir):
        for f in files:
            if f.endswith('.py'):
                relative_path_to_project_dir = (dir_path + '/' + f).split(old_project_dir)[-1]
                old_file = old_project_dir + relative_path_to_project_dir
                new_file = new_project_dir + relative_path_to_project_dir
                if os.path.isfile(new_file):
                    if not filecmp.cmp(old_file, new_file):  # diff
                        changed_files.append(relative_path_to_project_dir)
            elif f.endswith('.ipynb'):
                sub.run('jupyter nbconvert --output-dir=\"' + dir_path + '\"' +
                        ' --to python \"' + dir_path + '/' + f + '\"' +
                        ' --TemplateExporter.exclude_raw=True', shell=True, stdout=open(os.devnull, 'w'), stderr=sub.STDOUT)
                sub.run('jupyter nbconvert --output-dir=\"' + dir_path.replace(old_project_dir, new_project_dir) + '\"' +
                        ' --to python \"' + dir_path.replace(old_project_dir, new_project_dir) + '/' + f + '\"' +
                        ' --TemplateExporter.exclude_raw=True', shell=True, stdout=open(os.devnull, 'w'), stderr=sub.STDOUT)
                relative_path_to_project_dir = (dir_path + '/' + f.replace('.ipynb', '.py')).split(old_project_dir)[-1]
                old_file = old_project_dir + relative_path_to_project_dir
                new_file = new_project_dir + relative_path_to_project_dir
                if os.path.isfile(new_file):
                    if not filecmp.cmp(old_file, new_file):  # diff
                        changed_files.append(relative_path_to_project_dir)
    return changed_files

def extractAPIFQNsInProject(changed_api_usage_files, project_dir):
    print('Changed API Usage Files: ' + str(changed_api_usage_files))
    file_to_api_fqns_map = collections.OrderedDict({})
    for f in changed_api_usage_files:
        # extract api list
        api_fqns_list_in_file = getAPIFQNsListINAFile(project_dir + f)
        file_to_api_fqns_map[f] = api_fqns_list_in_file
    return file_to_api_fqns_map

def findChangedAPIUsageFiles(api, changed_files, old_project_dir):
    changed_api_usage_files = []
    for dir_path, subpaths, files in os.walk(old_project_dir):
        for f in files:
            if f.endswith('.py'):
                relative_path_to_project_dir = (dir_path + '/' + f).split(old_project_dir)[-1]
                if relative_path_to_project_dir not in changed_files:
                    continue
                if isFileUsingAPI(api, dir_path + '/' + f):
                    changed_api_usage_files.append(relative_path_to_project_dir)
    return changed_api_usage_files

def isFileUsingAPI(api, python_file):  # TODO: use advanced AST-based filtering
    with open(python_file, 'r') as fr:
        lines = fr.readlines()
    is_using_api = False
    for i, l in enumerate(lines):
        if api.split('.')[-1] in l:
            is_using_api = True
            break
    return is_using_api

def getAPIFQNsListINAFile(f):
    all_api_fqns_list = []
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, _, _ = extractAPIUsage(f)
    if not function_call_info_list and not attribute_ref_info_list and not api_related_var_info_list:  # the file has syntax error
        return []
    for call_info in function_call_info_list:
        if call_info['fqn'] not in all_api_fqns_list:
            all_api_fqns_list.append(call_info['fqn'])
    for ref_info in attribute_ref_info_list:
        if ref_info['fqn'] not in all_api_fqns_list:
            all_api_fqns_list.append(ref_info['fqn'])
    for var_info in api_related_var_info_list:
        if var_info['fqn'] not in all_api_fqns_list:
            all_api_fqns_list.append(var_info['fqn'])
    all_api_fqns_list = sorted(all_api_fqns_list)
    #for fqn in all_api_fqns_list:
    #    print(fqn)
    return all_api_fqns_list

def computeAPIFQNMapping(api, old_file_to_api_fqns_map, new_file_to_api_fqns_map,
                         only_record_top_candidate_per_commit=False,
                         api_fqn_mapping_json_file=GITHUB_ONLY_API_FQN_MAPPING_JSON_FILE):
    #print('Old APIs Map: ' + str(old_file_to_api_fqns_map))
    #print('New APIs Map: ' + str(new_file_to_api_fqns_map))
    if os.path.isfile(api_fqn_mapping_json_file):
        with open(api_fqn_mapping_json_file, 'r') as fr:
            old_to_new_fqn_mapping = json.load(fr, object_hook=collections.OrderedDict)
    else:
        old_to_new_fqn_mapping = collections.OrderedDict({})
    if api not in old_to_new_fqn_mapping:
        old_to_new_fqn_mapping[api] = collections.OrderedDict({})
    for old_file in old_file_to_api_fqns_map:
        if old_file not in new_file_to_api_fqns_map:
            continue
        old_file_apis = old_file_to_api_fqns_map[old_file]
        if api not in old_file_apis:
            continue
        new_file_apis = new_file_to_api_fqns_map[old_file]
        newly_added_apis = [n for n in new_file_apis if n not in old_file_apis]
        newly_added_apis = pruneNewlyAddedAPIsByASTAnalysis(api, newly_added_apis, old_file_apis, new_file_apis)
        if not newly_added_apis:  # no new api is used, old api may be replaced by custom function
            continue
        # for each commit, how many candidates do we want to add?
        if only_record_top_candidate_per_commit:
            top_candidate = selectBestCandidate_Deprecated(api, newly_added_apis)
            if top_candidate not in old_to_new_fqn_mapping[api]:
                old_to_new_fqn_mapping[api][top_candidate] = 0
            old_to_new_fqn_mapping[api][top_candidate] += 1
        else:
            print('Old API: ' + api)
            print('All candidates: ' + str(newly_added_apis))
            for na in newly_added_apis:
                if na not in old_to_new_fqn_mapping[api]:
                    old_to_new_fqn_mapping[api][na] = 0
                old_to_new_fqn_mapping[api][na] += 1
    #print(old_to_new_fqn_mapping)
    with open(api_fqn_mapping_json_file, 'w') as fw:
        json.dump(old_to_new_fqn_mapping, fw, indent=2)
    return old_to_new_fqn_mapping

def pruneNewlyAddedAPIsByASTAnalysis(old_api, newly_added_apis, old_file_apis, new_file_apis):
    pruned_newly_added_apis = []
    old_levels = getLevels(old_api, old_file_apis)
    print('Old levels: ' + str(old_levels))
    for new_api in newly_added_apis:
        should_prune = False
        new_levels = getLevels(new_api, new_file_apis)
        print('Candidate: ' + new_api + ', New levels: ' + str(new_levels))
        for lv in new_levels:
            if lv not in old_levels:
                should_prune = True
                break
        if should_prune:
            print('Candidate pruned: ' + new_api)
            continue
        if new_api not in pruned_newly_added_apis:
            pruned_newly_added_apis.append(new_api)
    return pruned_newly_added_apis

def getLevels(target_api, apis_in_file):
    levels = []
    for api in apis_in_file:
        if api.startswith(target_api):
            if api == target_api:
                level = 0
            else:
                level = api.split(target_api)[1].count('.')
            if level not in levels:
                levels.append(level)
    if len(levels) > 1 and 0 in levels:  # if have deeper usages (usually packages), do not count level 0
        levels.remove(0)
    return levels

def postProcessMapping(strategy,
                       deprecated_apis_list_file=DEPRECATED_APIS_LIST_FILE,
                       api_fqn_mapping_json_file=API_FQN_MAPPING_JSON_FILE,
                       github_only_api_fqn_mapping_file=GITHUB_ONLY_API_FQN_MAPPING_JSON_FILE,
                       apidoc_only_api_fqn_mapping_file=APIDOC_ONLY_API_FQN_MAPPING_JSON_FILE):  # sort candidates using ranking metrics
    with open(deprecated_apis_list_file, 'r') as fr:
        deprecated_apis = [l.strip() for l in fr.readlines()]
    with open(github_only_api_fqn_mapping_file, 'r') as fr:
        github_mapping = json.load(fr, object_hook=collections.OrderedDict)
    with open(apidoc_only_api_fqn_mapping_file, 'r') as fr:
        apidoc_mapping = json.load(fr, object_hook=collections.OrderedDict)
    final_mapping = collections.OrderedDict({})
    for old_api in deprecated_apis:
        github_candidates = [] if old_api not in github_mapping or github_mapping[old_api] is None else list(github_mapping[old_api].keys())
        apidoc_candidates = [] if old_api not in apidoc_mapping or apidoc_mapping[old_api] is None else list(apidoc_mapping[old_api].keys())
        if strategy == 'combined':
            all_candidates = github_candidates + [ac for ac in apidoc_candidates if ac not in github_candidates]
            all_candidates_with_final_scores = rankCandidatesUsingCombinedScores(old_api, all_candidates, github_mapping, apidoc_mapping)
        elif strategy == 'github_only':
            all_candidates = github_candidates
            all_candidates_with_final_scores = rankCandidatesUsingGitHubOccurrence(old_api, all_candidates, github_mapping)
        elif strategy == 'apidoc_only':
            all_candidates = apidoc_candidates
            all_candidates_with_final_scores = rankCandidatesUsingTokenWiseEditDistanceSimilarity(old_api, all_candidates, apidoc_mapping)
        elif strategy == 'github_naive':
            all_candidates = github_candidates
            all_candidates_with_final_scores = rankCandidatesUsingGitHubNaiveFQNEditDistanceSimilarity(old_api, all_candidates, github_mapping)
        elif strategy == 'apidoc_naive':
            all_candidates = apidoc_candidates
            all_candidates_with_final_scores = rankCandidatesUsingAPIDocNaiveFQNEditDistanceSimilarity(old_api, all_candidates, apidoc_mapping)
        final_mapping[old_api] = all_candidates_with_final_scores
    if strategy == 'github_only':
        api_fqn_mapping_json_file = GITHUB_ONLY_STRATEGY_API_FQN_MAPPING_JSON_FILE
    elif strategy == 'apidoc_only':
        api_fqn_mapping_json_file = APIDOC_ONLY_STRATEGY_API_FQN_MAPPING_JSON_FILE
    elif strategy == 'github_naive':
        api_fqn_mapping_json_file = GITHUB_NAIVE_STRATEGY_API_FQN_MAPPING_JSON_FILE
    elif strategy == 'apidoc_naive':
        api_fqn_mapping_json_file = APIDOC_NAIVE_STRATEGY_API_FQN_MAPPING_JSON_FILE
    with open(api_fqn_mapping_json_file, 'w') as fw:
        json.dump(final_mapping, fw, indent=2)
    if strategy == 'combined':
        generateFeaturesCSV()


# entrance_2
def runBuildMappingUsingAPIdoc(deprecated_apis_list_file=DEPRECATED_APIS_LIST_FILE):
    deprecated_apis = []
    with open(deprecated_apis_list_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        if l.strip() not in deprecated_apis:
            deprecated_apis.append(l.strip())
    for api in deprecated_apis:
        print('=== Processing old API: ' + api)
        buildMappingForOneAPIUsingAPIdoc(api)

def buildMappingForOneAPIUsingAPIdoc(old_api,
                                     old_to_new_api_fqn_mapping_file=APIDOC_ONLY_API_FQN_MAPPING_JSON_FILE,
                                     sklearn_api_doc_file=SKLEARN_API_DOC_FILE,
                                     pandas_api_doc_file=PANDAS_API_DOC_FILE,
                                     tensorflow_api_doc_file=TENSORFLOW_API_DOC_FILE,
                                     statsmodels_api_doc_file=STATSMODELS_API_DOC_FILE,
                                     seaborn_api_doc_file=SEABORN_API_DOC_FILE,
                                     networkx_api_doc_file=NETWORKX_API_DOC_FILE,
                                     scipy_api_doc_file=SCIPY_API_DOC_FILE,
                                     plotly_api_doc_file=PLOTLY_API_DOC_FILE,
                                     keras_api_doc_file=KERAS_API_DOC_FILE,
                                     numpy_api_doc_file=NUMPY_API_DOC_FILE,
                                     imblearn_api_doc_file=IMBLEARN_API_DOC_FILE):
    top_level_package_name = old_api.split('.')[0]
    if top_level_package_name == 'sklearn':
        api_list = readAPIList(sklearn_api_doc_file)
    elif top_level_package_name == 'pandas':
        api_list = readAPIList(pandas_api_doc_file)
    elif top_level_package_name == 'tensorflow':
        api_list = readAPIList(tensorflow_api_doc_file)
    elif top_level_package_name == 'statsmodels':
        api_list = readAPIList(statsmodels_api_doc_file)
    elif top_level_package_name == 'seaborn':
        api_list = readAPIList(seaborn_api_doc_file)
    elif top_level_package_name == 'networkx':
        api_list = readAPIList(networkx_api_doc_file)
    elif top_level_package_name == 'scipy':
        api_list = readAPIList(scipy_api_doc_file)
    elif top_level_package_name == 'plotly':
        api_list = readAPIList(plotly_api_doc_file)
    elif top_level_package_name == 'keras':
        api_list = readAPIList(keras_api_doc_file)
    elif top_level_package_name == 'numpy':
        api_list = readAPIList(numpy_api_doc_file)
    elif top_level_package_name == 'imblearn':
        api_list = readAPIList(imblearn_api_doc_file)
    api_dict = collections.OrderedDict({})
    for api in api_list:
        api_dict[api] = 'doc'
    ranked_new_api_candidates = rankCandidates_Deprecated(old_api, api_dict)
    top_five_candidates = list(ranked_new_api_candidates.keys())[:5]
    print('Old API: ' + old_api)
    print('Top five candidates: ' + str(top_five_candidates))
    #for c in top_five_candidates:
    #    print(c, ranked_new_api_candidates[c])
    if os.path.isfile(old_to_new_api_fqn_mapping_file):
        with open(old_to_new_api_fqn_mapping_file, 'r') as fr:
            old_to_new_api_fqn_map = json.load(fr, object_hook=collections.OrderedDict)
    else:
        old_to_new_api_fqn_map = collections.OrderedDict({})
    if old_api not in old_to_new_api_fqn_map:
        old_to_new_api_fqn_map[old_api] = collections.OrderedDict({})
    for cand in top_five_candidates:
        if cand not in old_to_new_api_fqn_map[old_api]:
            old_to_new_api_fqn_map[old_api][cand] = 'doc'
    with open(old_to_new_api_fqn_mapping_file, 'w') as fw:
        json.dump(old_to_new_api_fqn_map, fw, indent=2)

def readAPIList(api_list_file):
    with open(api_list_file, 'r') as fr:
        lines = fr.readlines()
    api_list = []
    for i, l in enumerate(lines):
        #if ' ' in l:
        #    continue
        api = l.strip().split('(')[0]
        if api not in api_list:
            api_list.append(api)
    return api_list


# entrance_4
def runValidateMapping(manual_ground_truth_mapping_file=MAPPING_JSON_FILE,
                       generated_mapping_file=API_FQN_MAPPING_JSON_FILE):
    with open(generated_mapping_file, 'r') as fr:
        generated_mapping = json.load(fr)
    with open(manual_ground_truth_mapping_file, 'r') as fr:
        ground_truth_mapping = json.load(fr)
    top_one_match_number = 0
    top_three_match_number = 0
    top_five_match_number = 0
    top_ten_match_number = 0
    all_match_number = 0
    for old_api in generated_mapping:
        print('Old API: ' + old_api)
        new_api_candidates = list(generated_mapping[old_api].keys())
        print('New API Candidates: ' + str(new_api_candidates))
        ground_truth_candidates = ground_truth_mapping[old_api]
        print('Ground truth: ' + str(ground_truth_candidates))
        if new_api_candidates[0] in ground_truth_candidates:
            top_one_match_number += 1
            print('Hit top 1')
        for cand in new_api_candidates[:3]:
            if cand in ground_truth_candidates:
                top_three_match_number += 1
                print('Hit top 3')
                break
        for cand in new_api_candidates[:5]:
            if cand in ground_truth_candidates:
                top_five_match_number += 1
                print('Hit top 5')
                break
        for cand in new_api_candidates[:10]:
            if cand in ground_truth_candidates:
                top_ten_match_number += 1
                print('Hit top 10')
                break
        for cand in new_api_candidates:
            if cand in ground_truth_candidates:
                all_match_number += 1
                print('Hit eventually')
                break
    print('Top-1: ' + str(top_one_match_number))
    print('Top-3: ' + str(top_three_match_number))
    print('Top-5: ' + str(top_five_match_number))
    print('Top-10: ' + str(top_ten_match_number))
    print('All: ' + str(all_match_number))
    return top_one_match_number, top_three_match_number, top_five_match_number, top_ten_match_number, all_match_number

# entrance_3
def runMineCommits():
    collectTopKCommitsForAllAPIs()

def collectTopKCommitsForAllAPIs(action_keywords=ACTION_KEYWORDS,
                                 api_list_file=DEPRECATED_APIS_LIST_FILE,
                                 _results_dir=MINING_GITHUB_RESULTS_DIR):
    with open(api_list_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        api = l.strip().split('(')[0]
        print ('Processing API: ' + api)
        api_keywords = [api.split('.')[0], api.split('.')[-1]]  # e.g., [sklearn, jaccard_similarity_score]
        api_commits = []
        for ak in action_keywords:
            keywords = [ak] + api_keywords
            commits = collectTopKCommits(keywords, 25)
            for c in commits:
                if not isCommitDuplicate(c, api_commits):
                    api_commits.append(c)
        output_dir = _results_dir + '/' + api
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if os.path.isfile(output_dir + '/commits.json'):
            with open(output_dir + '/commits.json', 'r') as fr:
                existing_commits = \
                    json.load(fr, object_pairs_hook=collections.OrderedDict)
        else:
            existing_commits = []
        for ac in api_commits:
            if ac not in existing_commits:
                existing_commits.append(ac)
        with open(output_dir + '/commits.json', 'w') as fw:
            json.dump(existing_commits, fw, indent=2)

def collectTopKCommits(keywords, K):
    # keywords
    keywords_chain = ''
    for kw in keywords:
        keywords_chain += kw + '+'
    keywords_chain = keywords_chain[:-1]
    # pages
    #num_of_pages, remain = divmod(k, 100)
    #if remain != 0:
    #    num_of_pages += 1
    # mining
    all_commits = []
    i = 0
    while len(all_commits) < K:
        page_no = str(i+1)
        query = "https://api.github.com/search/commits?q=" + keywords_chain + \
            "&per_page=100&page=" + page_no
        print (query)
        response = requests.get(query,
        headers={'Accept': 'application/vnd.github.cloak-preview',
                 'Authorization': 'token 9b1b1b941faaa8dbf163adfe68debf5a3d30577f'}).\
                content.decode("utf-8")
        try:
            commits_on_page = \
                json.loads(response, object_pairs_hook=collections.OrderedDict)['items']
        except:  # reach 10 page search limit of github api
            print ('REACH GITHUB API 10 PAGES SEARCHING LIMIT!')
            return all_commits
        #print (commits_on_page)
        for c in commits_on_page:
            if isCommitDuplicate(c, all_commits):
                continue
            repo_full_name = c['repository']['full_name']
            repo_req = "https://api.github.com/repos/" + repo_full_name
            repo = json.loads(requests.get(repo_req, headers={'Authorization': \
            'token 9b1b1b941faaa8dbf163adfe68debf5a3d30577f'}).content.decode("utf-8"),
            object_pairs_hook=collections.OrderedDict)
            #print (repo)
            # filter out private repos
            try:
                if repo['private']:
                    continue
            except:
                print (repo)
                exit(0)
            # filter out repos that are not Python or Jupyter
            if repo['language'] not in ['Jupyter Notebook', 'Python']:
                continue
            # filter out repos that have < 100 stars
            #print (repo_full_name + ': ' + str(repo['stargazers_count']))
            if repo['stargazers_count'] < 0:
                continue
            all_commits.append(c)
            print ('LEN: ' + str(len(all_commits)))
            if len(all_commits) >= K:
                return all_commits
            time.sleep(5)
        time.sleep(10)
        i += 1
    return all_commits

# Helper_1
def isCommitDuplicate(commit, all_commits):
    sha = commit['sha']
    for c in all_commits:
        if c['sha'] == sha:
            return True
    return False

# helper_1
def getOnDemandDeprecatedAPIList(deprecated_apis_list_file=DEPRECATED_APIS_LIST_FILE):
    deprecated_apis = []
    with open(deprecated_apis_list_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        api = l.strip()
        deprecated_apis.append(api)
    return deprecated_apis
