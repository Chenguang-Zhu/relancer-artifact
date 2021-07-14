import os
import json
import shutil
import collections

from macros import DEPRECATED_APIS_LIST_FILE
from macros import MINE_GITHUB_TRAINING_MAPPING_JSON_FILE
from macros import MINE_GITHUB_TRAINING_ARG_MAPPING_JSON_FILE
from macros import META_INFO_MAPPING_FILE

from mapping import pruneNewlyAddedAPIsByASTAnalysis
from mapping import findAllCommitsUpdatingAPI
from mapping import cloneOldAndNewVersionProject
from mapping import extractChangedFiles
from mapping import extractAPIFQNsInProject
from mapping import findChangedAPIUsageFiles
from ranking import selectBestCandidate_Deprecated
from api_extractor import extractAPIUsage

def runBuildMappingUsingGitHubForCreatingTrainingSet(deprecated_apis_list_file=DEPRECATED_APIS_LIST_FILE):
    deprecated_apis = []
    with open(deprecated_apis_list_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        if l.strip() not in deprecated_apis:
            deprecated_apis.append(l.strip())
    for api in deprecated_apis:
        print('=== Processing old API: ' + api)
        mineMappingsFromGitHubForOneLib(api)

def mineMappingsFromGitHubForOneLib(lib):
    commits = findAllCommitsUpdatingAPI(lib)[:50]  # TODO: we only analyze top 50 for now: too much disk space!
    # print(len(commits))
    for c in commits:
        print('--- Processing commit: ' + c['sha'])
        old_project_dir, new_project_dir = cloneOldAndNewVersionProject(c, lib)
        if old_project_dir is None and new_project_dir is None:  # extreme case: the first commit of repo
            continue
        changed_files = extractChangedFiles(old_project_dir, new_project_dir)
        print('Changed Files: ' + str(changed_files))
        changed_api_usage_files = findChangedAPIUsageFiles(lib, changed_files, old_project_dir)
        #old_file_to_api_fqns_map = extractAPIFQNsInProject(changed_api_usage_files, old_project_dir)
        #new_file_to_api_fqns_map = extractAPIFQNsInProject(changed_api_usage_files, new_project_dir)
        #api_mapping = computeAPIFQNMappingForAllAPIsInThisFile(old_file_to_api_fqns_map, new_file_to_api_fqns_map)  # save to json
        old_file_to_api_fqns_to_args_map = extractAPIFQNsAndArgsInProject(changed_api_usage_files, old_project_dir)
        new_file_to_api_fqns_to_args_map = extractAPIFQNsAndArgsInProject(changed_api_usage_files, new_project_dir)
        #fqn_mapping_in_this_commit = computeAPIFQNMappingUsingArgMatch(old_file_to_api_fqns_to_args_map, new_file_to_api_fqns_to_args_map)  # save to json
        #dumpMetaInfo(c, fqn_mapping_in_this_commit)
        #api_arg_mapping = computeAPIArgMappingForAllAPIsInThisFile(old_file_to_api_fqns_to_args_map, new_file_to_api_fqns_to_args_map)
        computeAPIArgMappingForOneAPIInThisFile(lib, old_file_to_api_fqns_to_args_map, new_file_to_api_fqns_to_args_map)
        if os.path.isdir(old_project_dir):
            shutil.rmtree(old_project_dir)
        if os.path.isdir(new_project_dir):
            shutil.rmtree(new_project_dir)

def extractAPIFQNsAndArgsInProject(changed_api_usage_files, project_dir):
    print('Changed API Usage Files: ' + str(changed_api_usage_files))
    file_to_api_fqns_map = collections.OrderedDict({})
    for f in changed_api_usage_files:
        # extract api list
        api_fqns_fqns_to_args_map = getAPIFQNsAndArgsINAFile(project_dir + f)
        file_to_api_fqns_map[f] = api_fqns_fqns_to_args_map
    return file_to_api_fqns_map

def getAPIFQNsAndArgsINAFile(f):
    all_api_fqns_to_args_map = collections.OrderedDict({})
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, _, _ = extractAPIUsage(f)
    if not function_call_info_list and not attribute_ref_info_list and not api_related_var_info_list:  # the file has syntax error
        return collections.OrderedDict({})
    for call_info in function_call_info_list:
        all_api_fqns_to_args_map[call_info['fqn']] = call_info['args']
    return all_api_fqns_to_args_map


def computeAPIFQNMappingForAllAPIsInThisFile(old_file_to_api_fqns_map, new_file_to_api_fqns_map,
                                             only_record_top_candidate_per_commit=False,
                                             api_fqn_mapping_json_file=MINE_GITHUB_TRAINING_MAPPING_JSON_FILE):
    #print('Old APIs Map: ' + str(old_file_to_api_fqns_map))
    #print('New APIs Map: ' + str(new_file_to_api_fqns_map))
    if os.path.isfile(api_fqn_mapping_json_file):
        with open(api_fqn_mapping_json_file, 'r') as fr:
            old_to_new_fqn_mapping = json.load(fr, object_hook=collections.OrderedDict)
    else:
        old_to_new_fqn_mapping = collections.OrderedDict({})
    for old_file in old_file_to_api_fqns_map:
        if old_file not in new_file_to_api_fqns_map:
            continue
        old_file_apis = old_file_to_api_fqns_map[old_file]
        new_file_apis = new_file_to_api_fqns_map[old_file]
        newly_added_apis = [n for n in new_file_apis if n not in old_file_apis]
        newly_removed_apis = [n for n in old_file_apis if n not in new_file_apis]
        if not newly_added_apis:  # no new api is used, old api may be replaced by custom function
            continue
        if not newly_removed_apis:
            continue
        for api in newly_removed_apis:
            if api not in old_to_new_fqn_mapping:
                old_to_new_fqn_mapping[api] = collections.OrderedDict({})
            candidate_apis = pruneNewlyAddedAPIsByASTAnalysis(api, newly_added_apis, old_file_apis, new_file_apis)
            # for each commit, how many candidates do we want to add?
            if only_record_top_candidate_per_commit:
                top_candidate = selectBestCandidate_Deprecated(api, candidate_apis)
                if top_candidate not in old_to_new_fqn_mapping[api]:
                    old_to_new_fqn_mapping[api][top_candidate] = 0
                old_to_new_fqn_mapping[api][top_candidate] += 1
            else:
                print('Old API: ' + api)
                print('All candidates: ' + str(candidate_apis))
                for na in candidate_apis:
                    if na not in old_to_new_fqn_mapping[api]:
                        old_to_new_fqn_mapping[api][na] = 0
                    old_to_new_fqn_mapping[api][na] += 1
    #print(old_to_new_fqn_mapping)
    with open(api_fqn_mapping_json_file, 'w') as fw:
        json.dump(old_to_new_fqn_mapping, fw, indent=2)
    return old_to_new_fqn_mapping

def computeAPIFQNMappingUsingArgMatch(old_file_to_api_fqns_to_args_map, new_file_to_api_fqns_to_args_map,
                                      only_record_top_candidate_per_commit=False,
                                      api_fqn_mapping_json_file=MINE_GITHUB_TRAINING_MAPPING_JSON_FILE):
    if os.path.isfile(api_fqn_mapping_json_file):
        with open(api_fqn_mapping_json_file, 'r') as fr:
            old_to_new_fqn_mapping = json.load(fr, object_hook=collections.OrderedDict)
    else:
        old_to_new_fqn_mapping = collections.OrderedDict({})
    fqn_mapping_in_this_commit = collections.OrderedDict({})
    for old_file in old_file_to_api_fqns_to_args_map:
        if old_file not in new_file_to_api_fqns_to_args_map:
            continue
        old_file_apis_and_args = old_file_to_api_fqns_to_args_map[old_file]
        new_file_apis_and_args = new_file_to_api_fqns_to_args_map[old_file]
        newly_added_apis = [n for n in new_file_apis_and_args if n not in old_file_apis_and_args]
        newly_removed_apis = [n for n in old_file_apis_and_args if n not in new_file_apis_and_args]
        if not newly_added_apis:  # no new api is used, old api may be replaced by custom function
            continue
        if not newly_removed_apis:
            continue
        for api in newly_removed_apis:
            print('Old API: ' + api)
            candidate_apis = pruneNewlyAddedAPIsByArgMatching(api, newly_added_apis, old_file_apis_and_args, new_file_apis_and_args)
            if not candidate_apis:
                continue
            if api not in old_to_new_fqn_mapping:
                old_to_new_fqn_mapping[api] = collections.OrderedDict({})
            if api not in fqn_mapping_in_this_commit:
                fqn_mapping_in_this_commit[api] = []
            # for each commit, how many candidates do we want to add?
            if only_record_top_candidate_per_commit:
                top_candidate = selectBestCandidate_Deprecated(api, candidate_apis)
                if top_candidate not in old_to_new_fqn_mapping[api]:
                    old_to_new_fqn_mapping[api][top_candidate] = 0
                old_to_new_fqn_mapping[api][top_candidate] += 1
                if top_candidate not in fqn_mapping_in_this_commit[api]:
                    fqn_mapping_in_this_commit[api].append(top_candidate)
            else:
                print('Old API: ' + api)
                print('All candidates: ' + str(candidate_apis))
                for na in candidate_apis:
                    if na not in old_to_new_fqn_mapping[api]:
                        old_to_new_fqn_mapping[api][na] = 0
                    old_to_new_fqn_mapping[api][na] += 1
                    if na not in fqn_mapping_in_this_commit[api]:
                        fqn_mapping_in_this_commit[api].append(na)
    #print(old_to_new_fqn_mapping)
    with open(api_fqn_mapping_json_file, 'w') as fw:
        json.dump(old_to_new_fqn_mapping, fw, indent=2)
    return fqn_mapping_in_this_commit

def pruneNewlyAddedAPIsByArgMatching(api, newly_added_apis, old_file_apis_and_args, new_file_apis_and_args):
    old_args = old_file_apis_and_args[api]
    pruned_cands = []
    for cand in newly_added_apis:
        new_args = new_file_apis_and_args[cand]
        if old_args == new_args:
            if len(old_args) != 0 and len(new_args) != 0:
                pruned_cands.append(cand)
        else:
            print('Candidate pruned: ' + cand)
    return pruned_cands

def computeAPIArgMappingForAllAPIsInThisFile(old_file_to_api_fqns_to_args_map, new_file_to_api_fqns_to_args_map,
                                             only_record_top_candidate_per_commit=False,
                                             api_args_mapping_json_file=MINE_GITHUB_TRAINING_ARG_MAPPING_JSON_FILE):
    #print('Old APIs Map: ' + str(old_file_to_api_fqns_map))
    #print('New APIs Map: ' + str(new_file_to_api_fqns_map))
    if os.path.isfile(api_args_mapping_json_file):
        with open(api_args_mapping_json_file, 'r') as fr:
            old_to_new_args_mapping = json.load(fr, object_hook=collections.OrderedDict)
    else:
        old_to_new_args_mapping = collections.OrderedDict({})
    for old_file in old_file_to_api_fqns_to_args_map:
        if old_file not in new_file_to_api_fqns_to_args_map:
            continue
        old_file_apis_to_args_map = old_file_to_api_fqns_to_args_map[old_file]
        new_file_apis_to_args_map = new_file_to_api_fqns_to_args_map[old_file]
        shared_apis = [n for n in list(new_file_apis_to_args_map.keys()) if n in list(old_file_apis_to_args_map.keys())]
        if not shared_apis:
            continue
        for api in shared_apis:
            old_args = old_file_apis_to_args_map[api]
            new_args = new_file_apis_to_args_map[api]
            if old_args == new_args:
                continue
            print('Shared API: ' + api)
            print('Old Args: ' + str(old_args))
            print('New Args: ' + str(new_args))
            if api not in old_to_new_args_mapping:
                old_to_new_args_mapping[api] = []
            arg_change_dict = collections.OrderedDict({})
            arg_change_dict['old_args'] = old_args
            arg_change_dict['new_args'] = new_args
            if arg_change_dict not in old_to_new_args_mapping[api]:
                old_to_new_args_mapping[api].append(arg_change_dict)
    #print(old_to_new_fqn_mapping)
    with open(api_args_mapping_json_file, 'w') as fw:
        json.dump(old_to_new_args_mapping, fw, indent=2)
    return old_to_new_args_mapping

from macros import MAPPING_JSON_FILE
def computeAPIArgMappingForOneAPIInThisFile(old_api, old_file_to_api_fqns_to_args_map, new_file_to_api_fqns_to_args_map,
                                            only_record_top_candidate_per_commit=False,
                                            mapping_json_file=MAPPING_JSON_FILE,
                                            api_args_mapping_json_file=MINE_GITHUB_TRAINING_ARG_MAPPING_JSON_FILE):
    #print('Old APIs Map: ' + str(old_file_to_api_fqns_map))
    #print('New APIs Map: ' + str(new_file_to_api_fqns_map))
    with open(mapping_json_file, 'r') as fr:
        ground_truth_api_fqn_mapping = json.load(fr, object_hook=collections.OrderedDict)
    if os.path.isfile(api_args_mapping_json_file):
        with open(api_args_mapping_json_file, 'r') as fr:
            old_to_new_args_mapping = json.load(fr, object_hook=collections.OrderedDict)
    else:
        old_to_new_args_mapping = collections.OrderedDict({})
    for old_file in old_file_to_api_fqns_to_args_map:
        if old_file not in new_file_to_api_fqns_to_args_map:
            continue
        old_file_apis_to_args_map = old_file_to_api_fqns_to_args_map[old_file]
        new_file_apis_to_args_map = new_file_to_api_fqns_to_args_map[old_file]
        if old_api not in old_file_apis_to_args_map:
            continue
        old_args = old_file_apis_to_args_map[old_api]
        new_api_candidates = ground_truth_api_fqn_mapping[old_api]
        new_api_candidates.append(old_api)  # Special: same FQN, diff args
        for new_api in new_api_candidates:
            if new_api not in new_file_apis_to_args_map:
                continue
            new_args = new_file_apis_to_args_map[new_api]
            if old_args == new_args:
                continue
            print('API FQN Pair: ' + old_api + ' -> ' + new_api)
            print('Old Args: ' + str(old_args))
            print('New Args: ' + str(new_args))
            if old_api not in old_to_new_args_mapping:
                old_to_new_args_mapping[old_api] = []
            arg_change_dict = collections.OrderedDict({})
            arg_change_dict['new_fqn'] = new_api
            arg_change_dict['old_args'] = old_args
            arg_change_dict['new_args'] = new_args
            if arg_change_dict not in old_to_new_args_mapping[old_api]:
                old_to_new_args_mapping[old_api].append(arg_change_dict)
    #print(old_to_new_fqn_mapping)
    with open(api_args_mapping_json_file, 'w') as fw:
        json.dump(old_to_new_args_mapping, fw, indent=2)
    return old_to_new_args_mapping

def dumpMetaInfo(c, fqn_mapping_in_this_commit, mapping_meta_info_file=META_INFO_MAPPING_FILE):
    if os.path.isfile(mapping_meta_info_file):
        with open(mapping_meta_info_file, 'r') as fr:
            meta_info = json.load(fr, object_hook=collections.OrderedDict)
    else:
        meta_info = collections.OrderedDict({})
    for old_api in fqn_mapping_in_this_commit:
        if old_api not in meta_info:
            meta_info[old_api] = []
        existing_cands = [ni["new_api"] for ni in meta_info[old_api]]
        new_cands = fqn_mapping_in_this_commit[old_api]
        for cand in new_cands:
            if cand not in existing_cands:
                meta_info[old_api].append(
                    collections.OrderedDict({"new_api": cand, "commits": [c['sha']], "messages": [c['commit']['message']]}))
            else:
                i = existing_cands.index(cand)
                if c['sha'] not in meta_info[old_api][i]["commits"]:
                    meta_info[old_api][i]["commits"].append(c['sha'])
                if c['commit']['message'] not in meta_info[old_api][i]["messages"]:
                    meta_info[old_api][i]["messages"].append(c['commit']['message'])
    with open(mapping_meta_info_file, 'w') as fw:
        json.dump(meta_info, fw, indent=2)


from macros import MANUAL_KNOWLEDGE_FILE
from macros import MANUAL_TRAINING_MAPPING_JSON_FILE
from macros import MANUAL_TRAINING_ARG_MAPPING_JSON_FILE
def runBuildMappingUsingManualArrowFile(manual_knowledge_file=MANUAL_KNOWLEDGE_FILE,
                                        manual_training_name_mapping_json_file=MANUAL_TRAINING_MAPPING_JSON_FILE,
                                        manual_training_arg_mapping_json_file=MANUAL_TRAINING_ARG_MAPPING_JSON_FILE):
    with open(manual_knowledge_file, 'r') as fr:
        lines = fr.readlines()
    name_mapping = collections.OrderedDict({})
    args_mapping = collections.OrderedDict({})
    for i, l in enumerate(lines):
        if '-> None' in l:  # api removed
            continue
        old_api_fqn = l.strip().split(' -> ')[0].split()[-1].split(',')[0]
        new_api_fqn = l.strip().split(' -> ')[1].split(',')[0]
        if old_api_fqn != new_api_fqn:  # api name changed
            if old_api_fqn not in name_mapping:
                name_mapping[old_api_fqn] = collections.OrderedDict({})
            if new_api_fqn not in name_mapping[old_api_fqn]:
                name_mapping[old_api_fqn][new_api_fqn] = 1
            with open(manual_training_name_mapping_json_file, 'w') as fw:
                json.dump(name_mapping, fw, indent=2)
        if ',' not in l:
            continue
        print(l.strip())
        api_fqn = l.strip().split(' -> ')[-1].split(',')[0]
        if api_fqn not in args_mapping:
            args_mapping[api_fqn] = []
        old_part = l.strip().split(' -> ')[0].split()[-1]
        new_part = l.strip().split(' -> ')[1]
        arg_change_dict = collections.OrderedDict({})
        arg_change_dict['old_args'] = collections.OrderedDict({})
        arg_change_dict['new_args'] = collections.OrderedDict({})
        if ',' in old_part:
            if '=' in old_part:  # arg value update
                old_arg_name = old_part.split(',')[-1].split('=')[0]
                old_arg_value = old_part.split(',')[-1].split('=')[1].replace('\'', '').replace('\"', '')
            else:  # arg rename
                old_arg_name = old_part.split(',')[-1]
                old_arg_value = "*"
            arg_change_dict['old_args'][old_arg_name] = old_arg_value
        if '=' in new_part:  # arg value update
            new_arg_name = new_part.split(',')[-1].split('=')[0]
            new_arg_value = new_part.split(',')[-1].split('=')[1].replace('\'', '').replace('\"', '')
            arg_change_dict['new_args'][new_arg_name] = new_arg_value
        else:  # arg rename or delete
            new_arg_name = new_part.split(',')[-1]
            if new_arg_name == 'None':
                arg_change_dict['new_args'] = collections.OrderedDict({})
            else:
                new_arg_value = "*"
                arg_change_dict['new_args'][new_arg_name] = new_arg_value
        if arg_change_dict not in args_mapping[api_fqn]:
            args_mapping[api_fqn].append(arg_change_dict)
        with open(manual_training_arg_mapping_json_file, 'w') as fw:
            json.dump(args_mapping, fw, indent=2)
