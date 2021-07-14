import os
import shutil
import csv
import json
import collections
import subprocess as sub

from macros import MUTATION_EXP_SUBJECTS_FILE
from macros import MUTATION_EXP_CONVERTED_NOTEBOOKS_DIR
from macros import NAME_MUTATION_EXP_MUTATED_NOTEBOOKS_DIR
from macros import NAME_MUTATION_EXP_LOGS_DIR
from macros import NAME_MUTATION_EXP_PATCHES_DIR
from macros import ARG_MUTATION_EXP_MUTATED_NOTEBOOKS_DIR
from macros import ARG_MUTATION_EXP_LOGS_DIR
from macros import ARG_MUTATION_EXP_PATCHES_DIR
from macros import JUPYTER_EXP_CONVERTED_NOTEBOOKS_DIR

from macros import MANUAL_TRAINING_MAPPING_JSON_FILE
from macros import MANUAL_TRAINING_ARG_MAPPING_JSON_FILE
from macros import MANUAL_TRAINING_NAME_MUTATION_TEMPLATE_JSON_FILE
from macros import MANUAL_TRAINING_ARG_MUTATION_TEMPLATE_JSON_FILE
from macros import MANUAL_NAME_MUTATION_RESULTS_INFO_JSON_FILE
from macros import MANUAL_ARG_MUTATION_RESULTS_INFO_JSON_FILE

MINE_GITHUB_TRAINING_MAPPING_JSON_FILE = MANUAL_TRAINING_MAPPING_JSON_FILE
MINE_GITHUB_TRAINING_ARG_MAPPING_JSON_FILE = MANUAL_TRAINING_ARG_MAPPING_JSON_FILE
TRAINING_NAME_MUTATION_TEMPLATE_JSON_FILE = MANUAL_TRAINING_NAME_MUTATION_TEMPLATE_JSON_FILE
TRAINING_ARG_MUTATION_TEMPLATE_JSON_FILE = MANUAL_TRAINING_ARG_MUTATION_TEMPLATE_JSON_FILE
NAME_MUTATION_RESULTS_INFO_JSON_FILE = MANUAL_NAME_MUTATION_RESULTS_INFO_JSON_FILE
ARG_MUTATION_RESULTS_INFO_JSON_FILE = MANUAL_ARG_MUTATION_RESULTS_INFO_JSON_FILE

from macros import TIME_OUT_THRESHOLD
from macros import DEBUG_INFO_FILE
from macros import KAGGLE_API_USAGE_INFO_JSON_FILE
from macros import KAGGLE_API_USAGE_20_LIBS_INFO_JSON_FILE

from api_extractor import extractAPIUsage
from ast_migration import runASTMigration
from api_doc_and_err_msgs_migration import applySolution

from ranking import rankCandidates_Deprecated
from util import logFileHasError
from util import timeout
from util import checkIfStringInFile

def getSubjects(mutation_exp_subjects_json_file=MUTATION_EXP_SUBJECTS_FILE):
    with open(mutation_exp_subjects_json_file) as fr:
        subjects_dict = json.load(fr, object_hook=collections.OrderedDict)
    return subjects_dict['success_cases']

def runNameMutationUntilKErrorsPerAPI(K=10, timeout_threshold=TIME_OUT_THRESHOLD, logs_dir=NAME_MUTATION_EXP_LOGS_DIR,
                                      name_mutation_templates_file=TRAINING_NAME_MUTATION_TEMPLATE_JSON_FILE,
                                      mutation_results_info_json_file=NAME_MUTATION_RESULTS_INFO_JSON_FILE,
                                      jupyter_converted_notebooks_dir=JUPYTER_EXP_CONVERTED_NOTEBOOKS_DIR):
    if os.path.isfile(mutation_results_info_json_file):
        with open(mutation_results_info_json_file, 'r') as fr:
            api_errors_map = json.load(fr, object_hook=collections.OrderedDict)
    else:
        api_errors_map = collections.OrderedDict({})
    subjects = getSubjects()
    with open(name_mutation_templates_file, 'r') as fr:
        name_mutation_templates = json.load(fr, object_hook=collections.OrderedDict)
    for s in subjects:
        project = s.split('/')[0]
        notebook = s.split('/')[1]
        print('=== Processing Notebook: ' + project + '/' + notebook)
        for api in name_mutation_templates:
            if api not in api_errors_map:
                api_errors_map[api] = collections.OrderedDict({})
                api_errors_map[api]['num'] = 0
                api_errors_map[api]['error_cases'] = []
                api_errors_map[api]['tried_cases'] = []
            # check if reach K
            if api_errors_map[api]['num'] >= K:
                print('API ' + api + ' already has ' + str(K) + ' instances!')
                continue
            old_file = jupyter_converted_notebooks_dir + '/' + project + '/' + notebook + '.py'
            # Simply check if the notebook potentially has this api
            if not checkIfStringInFile(api.split('.')[-1], old_file):
                print('API ' + api + ' is not in this notebook!')
                continue
            before_mutation_log_file = logs_dir + '/' + api + '/' + project + '/' + notebook + '.old.log'
            after_mutation_log_file = logs_dir + '/' + api + '/' + project + '/' + notebook + '.new.log'
            if os.path.isfile(before_mutation_log_file):
                print('API ' + api + ' has run before!')
                if project + '/' + notebook not in api_errors_map[api]['tried_cases']:
                    api_errors_map[api]['tried_cases'].append(project + '/' + notebook)
                continue
            print('--- Try Mutating API: ' + api)
            changed = False
            with timeout(timeout_threshold):
                changed = applyOneNameMutationOnOneNotebook(api, project, notebook)
            if not os.path.isfile(before_mutation_log_file) and not os.path.isfile(after_mutation_log_file):  # patch is empty
                continue
            if not os.path.isfile(after_mutation_log_file):  # timeout
                continue
            if logFileHasError(before_mutation_log_file):
                continue
            if logFileHasError(after_mutation_log_file):
                if project + '/' + notebook not in api_errors_map[api]['error_cases']:
                    api_errors_map[api]['error_cases'].append(project + '/' + notebook)
                    api_errors_map[api]['num'] += 1
            if changed:
                if project + '/' + notebook not in api_errors_map[api]['tried_cases']:
                    api_errors_map[api]['tried_cases'].append(project + '/' + notebook)
            with open(mutation_results_info_json_file, 'w') as fw:
                json.dump(api_errors_map, fw, indent=2)

def runArgMutationUntilKErrorsPerAPI(K=10, timeout_threshold=TIME_OUT_THRESHOLD, logs_dir=ARG_MUTATION_EXP_LOGS_DIR,
                                      arg_mutation_templates_file=TRAINING_ARG_MUTATION_TEMPLATE_JSON_FILE,
                                      mutation_results_info_json_file=ARG_MUTATION_RESULTS_INFO_JSON_FILE,
                                      jupyter_converted_notebooks_dir=JUPYTER_EXP_CONVERTED_NOTEBOOKS_DIR):
    if os.path.isfile(mutation_results_info_json_file):
        with open(mutation_results_info_json_file, 'r') as fr:
            api_errors_map = json.load(fr, object_hook=collections.OrderedDict)
    else:
        api_errors_map = collections.OrderedDict({})
    subjects = getSubjects()
    with open(arg_mutation_templates_file, 'r') as fr:
        arg_mutation_templates = json.load(fr, object_hook=collections.OrderedDict)
    for s in subjects:
        project = s.split('/')[0]
        notebook = s.split('/')[1]
        print('=== Processing Notebook: ' + project + '/' + notebook)
        for api in arg_mutation_templates:
            if api not in api_errors_map:
                api_errors_map[api] = collections.OrderedDict({})
                api_errors_map[api]['num'] = 0
                api_errors_map[api]['error_cases'] = []
                api_errors_map[api]['tried_cases'] = []
            # check if reach K
            value_error_apis = ["sklearn.svm.SVC", "sklearn.svm.SVR", "sklearn.neighbors.NearestCentroid"]
            if api_errors_map[api]['num'] >= 50 and api in value_error_apis:
                print('API ' + api + ' already has ' + str(K) + ' instances!')
                continue
            if api_errors_map[api]['num'] >= K and api not in value_error_apis:
                print('API ' + api + ' already has ' + str(K) + ' instances!')
                continue
            old_file = jupyter_converted_notebooks_dir + '/' + project + '/' + notebook + '.py'
            # Simply check if the notebook potentially has this api
            if not checkIfStringInFile(api.split('.')[-1], old_file):
                print('API ' + api + ' is not in this notebook!')
                continue
            before_mutation_log_file = logs_dir + '/' + api + '/' + project + '/' + notebook + '.old.log'
            after_mutation_log_file = logs_dir + '/' + api + '/' + project + '/' + notebook + '.new.log'
            if os.path.isfile(before_mutation_log_file):
                print('API ' + api + ' has run before!')
                if project + '/' + notebook not in api_errors_map[api]['tried_cases']:
                    api_errors_map[api]['tried_cases'].append(project + '/' + notebook)
                continue
            print('--- Try Mutating API: ' + api)
            with timeout(timeout_threshold):
                changed = applyOneArgMutationOnOneNotebook(api, project, notebook)
            if not os.path.isfile(before_mutation_log_file) and not os.path.isfile(after_mutation_log_file):  # patch is empty
                continue
            if not os.path.isfile(after_mutation_log_file):  # timeout
                continue
            if logFileHasError(before_mutation_log_file):
                continue
            if logFileHasError(after_mutation_log_file):
                if project + '/' + notebook not in api_errors_map[api]['error_cases']:
                    api_errors_map[api]['error_cases'].append(project + '/' + notebook)
                    api_errors_map[api]['num'] += 1
            if changed:
                if project + '/' + notebook not in api_errors_map[api]['tried_cases']:
                    api_errors_map[api]['tried_cases'].append(project + '/' + notebook)
            with open(mutation_results_info_json_file, 'w') as fw:
                json.dump(api_errors_map, fw, indent=2)

def getNameMutationTemplateForAPI(api, name_mutation_templates_file=TRAINING_NAME_MUTATION_TEMPLATE_JSON_FILE):
    with open(name_mutation_templates_file, 'r') as fr:
        name_mutation_templates = json.load(fr, object_hook=collections.OrderedDict)
    return name_mutation_templates[api]

def getArgMutationTemplateForAPI(api, arg_mutation_templates_file=TRAINING_ARG_MUTATION_TEMPLATE_JSON_FILE):
    with open(arg_mutation_templates_file, 'r') as fr:
        arg_mutation_templates = json.load(fr, object_hook=collections.OrderedDict)
    return arg_mutation_templates[api]

def genNameMutationTemplatesFromTrainingData(training_mapping_file=MINE_GITHUB_TRAINING_MAPPING_JSON_FILE,
                                             name_mutation_templates_file=TRAINING_NAME_MUTATION_TEMPLATE_JSON_FILE):
    with open(training_mapping_file, 'r') as fr:
        training_migration_map = json.load(fr, object_hook=collections.OrderedDict)
    name_mutation_templates = collections.OrderedDict({})
    for old_api in training_migration_map:
        new_api_candidates = training_migration_map[old_api]
        for cand in new_api_candidates:
            if cand not in name_mutation_templates:
                name_mutation_templates[cand] = []
            if old_api not in name_mutation_templates[cand]:
                name_mutation_templates[cand].append(old_api)
    for fqn in name_mutation_templates:
        cands = name_mutation_templates[fqn]
        cands = list(rankCandidates_Deprecated(fqn, cands).keys())
        name_mutation_templates[fqn] = cands
    with open(name_mutation_templates_file, 'w') as fw:
        json.dump(name_mutation_templates, fw, indent=2)
    return name_mutation_templates

def genArgMutationTemplatesFromTrainingData(training_args_mapping_file=MINE_GITHUB_TRAINING_ARG_MAPPING_JSON_FILE,
                                            args_mutation_templates_file=TRAINING_ARG_MUTATION_TEMPLATE_JSON_FILE):
    with open(training_args_mapping_file, 'r') as fr:
        training_args_migration_map = json.load(fr, object_hook=collections.OrderedDict)
    args_mutation_templates = collections.OrderedDict({})
    for api in training_args_migration_map:
        if api not in args_mutation_templates:
            args_mutation_templates[api] = collections.OrderedDict({})
        args_pairs = training_args_migration_map[api]
        for pair in args_pairs:
            is_rename, arg, action_dict = findArgRename(pair)
            if is_rename:
                if arg not in args_mutation_templates[api]:
                    args_mutation_templates[api][arg] = []
                if action_dict not in args_mutation_templates[api][arg]:
                    args_mutation_templates[api][arg].append(action_dict)
                continue
            for arg in pair['old_args']:
                if arg not in pair['new_args']:
                    action_dict = collections.OrderedDict({})
                    action_dict['ACTION'] = 'ADD'
                    action_dict['VALUE'] = pair['old_args'][arg].strip('\'').strip('\"')
                    if action_dict['VALUE'] == 'COMPLEX_VALUE':  # add complex_value is not useful
                        continue
                    if arg not in args_mutation_templates[api]:
                        args_mutation_templates[api][arg] = []
                    if action_dict not in args_mutation_templates[api][arg]:
                        args_mutation_templates[api][arg].append(action_dict)
                else:
                    old_value = pair['old_args'][arg]
                    new_value = pair['new_args'][arg]
                    if old_value != new_value:
                        action_dict = collections.OrderedDict({})
                        action_dict['ACTION'] = 'UPD'
                        action_dict['FROM_VALUE'] = pair['new_args'][arg].strip('\'').strip('\"')
                        action_dict['TO_VALUE'] = pair['old_args'][arg].strip('\'').strip('\"')
                        if arg not in args_mutation_templates[api]:
                            args_mutation_templates[api][arg] = []
                        if action_dict not in args_mutation_templates[api][arg]:
                            args_mutation_templates[api][arg].append(action_dict)
            for arg in pair['new_args']:
                if arg not in pair['old_args']:
                    action_dict = collections.OrderedDict({})
                    action_dict['ACTION'] = 'DEL'
                    action_dict['VALUE'] = pair['new_args'][arg].strip('\'').strip('\"')
                    if arg not in args_mutation_templates[api]:
                        args_mutation_templates[api][arg] = []
                    if action_dict not in args_mutation_templates[api][arg]:
                        args_mutation_templates[api][arg].append(action_dict)
    with open(args_mutation_templates_file, 'w') as fw:
        json.dump(args_mutation_templates, fw, indent=2)

def findArgRename(pair):
    deleted_args = collections.OrderedDict({})
    added_args = collections.OrderedDict({})
    for arg in pair['old_args']:
        if arg not in pair['new_args']:
            deleted_args[arg] = ""
    for arg in pair['new_args']:
        if arg not in pair['old_args']:
            added_args[arg] = ""
    for da in deleted_args:
        da_value = pair['old_args'][da]
        for aa in added_args:
            aa_value = pair['new_args'][aa]
            if aa_value == da_value and aa_value != 'COMPLEX_VALUE':
                action_dict = collections.OrderedDict({})
                action_dict['ACTION'] = 'REN'
                action_dict['FROM_NAME'] = aa
                action_dict['TO_NAME'] = da
                return True, aa, action_dict
    return False, None, None

def replaceDatasetRelativePaths(old_file):
    with open(old_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        if '../../input/' in l:
            lines[i] = l.replace('../../input', '../../../input')
    with open(old_file, 'w') as fw:
        fw.write(''.join(lines))

def applyOneNameMutationOnOneNotebook(api, project, notebook,
                                      converted_notebooks_dir=MUTATION_EXP_CONVERTED_NOTEBOOKS_DIR,
                                      mutated_notebooks_dir=NAME_MUTATION_EXP_MUTATED_NOTEBOOKS_DIR,
                                      mutation_patches_dir=NAME_MUTATION_EXP_PATCHES_DIR,
                                      logs_dir=NAME_MUTATION_EXP_LOGS_DIR,
                                      jupyter_converted_notebooks_dir=JUPYTER_EXP_CONVERTED_NOTEBOOKS_DIR):
    print('--- Mutate API Name ' + api + ': ' + project + '/' + notebook)
    template = getNameMutationTemplateForAPI(api)
    old_file = converted_notebooks_dir + '/' + api + '/' + project + '/' + notebook + '.py'
    if not os.path.isdir(converted_notebooks_dir + '/' + api + '/' + project):
        os.makedirs(converted_notebooks_dir + '/' + api + '/' + project)
    if not os.path.isfile(old_file):
        shutil.copyfile(jupyter_converted_notebooks_dir + '/' + project + '/' + notebook + '.py', old_file)
        replaceDatasetRelativePaths(old_file)
    new_file = mutated_notebooks_dir + '/' + api + '/' + project + '/' + notebook + '.py'
    if not os.path.isdir(mutated_notebooks_dir + '/' + api + '/' + project):
        os.makedirs(mutated_notebooks_dir + '/' + api + '/' + project)
    api_mapping = collections.OrderedDict({api: template})
    shutil.copyfile(old_file, new_file)
    runASTMigration(project, notebook, api_mapping, new_file)
    cwd = os.getcwd()
    # save patches
    patch_output_dir = mutation_patches_dir + '/' + api + '/' + project
    if not os.path.isdir(patch_output_dir):
        os.makedirs(patch_output_dir)
    sub.run('diff -u ' + old_file + ' ' + new_file, shell=True,
            stdout=open(patch_output_dir + '/' + notebook + '.patch', 'w'),
            stderr=sub.STDOUT)
    # if patch is empty, do not need to run
    if os.stat(patch_output_dir + '/' + notebook + '.patch').st_size == 0:
        return False
    # run original
    os.chdir(converted_notebooks_dir + '/' + api + '/' + project)
    output_dir = logs_dir + '/' + api + '/' + project
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    buggy_log_file = output_dir + '/' + notebook + '.old.log'
    sub.run('python ' + notebook + '.py', shell=True,
            stdout=open(buggy_log_file, 'w'), stderr=sub.STDOUT)
    # run mutant
    os.chdir(mutated_notebooks_dir + '/' + api + '/' + project)
    output_dir = logs_dir + '/' + api + '/' + project
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fixed_log_file = output_dir + '/' + notebook + '.new.log'
    sub.run('python ' + notebook + '.py', shell=True,
            stdout=open(fixed_log_file, 'w'), stderr=sub.STDOUT)
    os.chdir(cwd)
    return True

def applyOneArgMutationOnOneNotebook(api, project, notebook,
                                     converted_notebooks_dir=MUTATION_EXP_CONVERTED_NOTEBOOKS_DIR,
                                     mutated_notebooks_dir=ARG_MUTATION_EXP_MUTATED_NOTEBOOKS_DIR,
                                     mutation_patches_dir=ARG_MUTATION_EXP_PATCHES_DIR,
                                     logs_dir=ARG_MUTATION_EXP_LOGS_DIR,
                                     jupyter_converted_notebooks_dir=JUPYTER_EXP_CONVERTED_NOTEBOOKS_DIR):
    print('--- Mutate API Arg ' + api + ': ' + project + '/' + notebook)
    template = getArgMutationTemplateForAPI(api)
    if len(template) == 0:
        return False
    old_file = converted_notebooks_dir + '/' + api + '/' + project + '/' + notebook + '.py'
    if not os.path.isdir(converted_notebooks_dir + '/' + api + '/' + project):
        os.makedirs(converted_notebooks_dir + '/' + api + '/' + project)
    if not os.path.isfile(old_file):
        shutil.copyfile(jupyter_converted_notebooks_dir + '/' + project + '/' + notebook + '.py', old_file)
        replaceDatasetRelativePaths(old_file)
    new_file = mutated_notebooks_dir + '/' + api + '/' + project + '/' + notebook + '.py'
    if not os.path.isdir(mutated_notebooks_dir + '/' + api + '/' + project):
        os.makedirs(mutated_notebooks_dir + '/' + api + '/' + project)
    shutil.copyfile(old_file, new_file)
    all_target_line_nos = collectAllTargetLineNos(api, new_file)
    mutatable_args = template
    target_arg = list(mutatable_args.keys())[0]  # Choose the first arg for now
    mutant = mutatable_args[target_arg][0]
    action = mutant['ACTION']
    if action == 'DEL':
        value = mutant['VALUE']
        solutions = []
        for line_no in all_target_line_nos:
            solution = (line_no, target_arg, 'delete', value)
            if solution not in solutions:
                solutions.append(solution)
    elif action == 'UPD':
        to_value = mutant['TO_VALUE']
        solutions = []
        for line_no in all_target_line_nos:
            solution = (line_no, target_arg, 'update', to_value)
            if solution not in solutions:
                solutions.append(solution)
    elif action == 'ADD':
        value = mutant['VALUE']
        solutions = []
        for line_no in all_target_line_nos:
            solution = (line_no, target_arg, 'add', value)
            if solution not in solutions:
                solutions.append(solution)
    elif action == 'REN':
        to_name = mutant['TO_NAME']
        solutions = []
        for line_no in all_target_line_nos:
            solution = (line_no, target_arg, 'ren', to_name)
            if solution not in solutions:
                solutions.append(solution)
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = extractAPIUsage(new_file)
    for sol in solutions:
        applySolution(sol, api, project, notebook,
                      new_file, new_file,
                      function_call_info_list,
                      attribute_ref_info_list,
                      api_related_var_info_list,
                      all_imported_names_map,
                      all_import_names_line_no_map)
    cwd = os.getcwd()
    # save patches
    patch_output_dir = mutation_patches_dir + '/' + api + '/' + project
    if not os.path.isdir(patch_output_dir):
        os.makedirs(patch_output_dir)
    sub.run('diff -u ' + old_file + ' ' + new_file, shell=True,
            stdout=open(patch_output_dir + '/' + notebook + '.patch', 'w'),
            stderr=sub.STDOUT)
    # if patch is empty, do not need to run
    if os.stat(patch_output_dir + '/' + notebook + '.patch').st_size == 0:
        return False
    # run original
    os.chdir(converted_notebooks_dir + '/' + api + '/' + project)
    output_dir = logs_dir + '/' + api + '/' + project
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    buggy_log_file = output_dir + '/' + notebook + '.old.log'
    sub.run('python ' + notebook + '.py', shell=True,
            stdout=open(buggy_log_file, 'w'), stderr=sub.STDOUT)
    # run mutant
    os.chdir(mutated_notebooks_dir + '/' + api + '/' + project)
    output_dir = logs_dir + '/' + api + '/' + project
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fixed_log_file = output_dir + '/' + notebook + '.new.log'
    sub.run('python ' + notebook + '.py', shell=True,
            stdout=open(fixed_log_file, 'w'), stderr=sub.STDOUT)
    os.chdir(cwd)
    return True

def collectAllTargetLineNos(api, new_file):
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, _, _ = extractAPIUsage(new_file)
    all_target_line_nos = []
    # func call
    if function_call_info_list is None:
        return all_target_line_nos
    for call_info in function_call_info_list:
        if call_info['fqn'] == api:
            target_line_no = call_info['line_no']
            if target_line_no not in all_target_line_nos:
                all_target_line_nos.append(target_line_no)
    # attr ref
    for attr_ref_info in attribute_ref_info_list:
        if attr_ref_info['fqn'] == api:
            target_line_no = attr_ref_info['line_no']
            if target_line_no not in all_target_line_nos:
                all_target_line_nos.append(target_line_no)
    # var
    for var_info in api_related_var_info_list:
        if var_info['fqn'] == api:
            target_line_no = var_info['line_no']
            if target_line_no not in all_target_line_nos:
                all_target_line_nos.append(target_line_no)
    return all_target_line_nos

def postProcessDebugInfo(debug_info_file=DEBUG_INFO_FILE, name_mutation_results_file=NAME_MUTATION_RESULTS_INFO_JSON_FILE):
    with open(name_mutation_results_file, 'r') as fr:
        name_mutation_results = json.load(fr, object_hook=collections.OrderedDict)
    for old_api in name_mutation_results:
        if 'tried_cases' not in list(name_mutation_results[old_api].keys()):
            name_mutation_results[old_api]['tried_cases'] = []
        for case in name_mutation_results[old_api]['error_cases']:
            if case not in name_mutation_results[old_api]['tried_cases']:
                name_mutation_results[old_api]['tried_cases'].append(case)
    with open(debug_info_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        if l.startswith('Migrate: ') and ' -> ' in l:
            old_api = l.split(':')[-1].split('->')[0].strip()
            j = i
            while not lines[j].startswith('=== Processing Notebook: '):
                j -= 1
            notebook = lines[j].strip().split(': ')[-1]
            if notebook not in name_mutation_results[old_api]['tried_cases']:
                name_mutation_results[old_api]['tried_cases'].append(notebook)
        elif 'has run before' in l:
            old_api = l.split()[1]
            j = i
            while not lines[j].startswith('=== Processing Notebook: '):
                j -= 1
            notebook = lines[j].strip().split(': ')[-1]
            if notebook not in name_mutation_results[old_api]['tried_cases']:
                name_mutation_results[old_api]['tried_cases'].append(notebook)
    with open(name_mutation_results_file, 'w') as fw:
        json.dump(name_mutation_results, fw, indent=2)

def getMostFrequentUsedAPIsInKagglePassingNotebooks(kaggle_api_usage_info_file=KAGGLE_API_USAGE_INFO_JSON_FILE,
                                                    kaggle_api_usage_20_libs_info_file=KAGGLE_API_USAGE_20_LIBS_INFO_JSON_FILE,
                                                    jupyter_converted_notebooks_dir=JUPYTER_EXP_CONVERTED_NOTEBOOKS_DIR):
    target_libs = ['sklearn', 'pandas_profiling', 'missingno', 'catboost', 'efficientnet', 'scikitplot', 'folium', 'fbprophet',
                   'eli5', 'mlxtend', 'plotly', 'bayes_opt', 'skimage', 'yellowbrick', 'graphviz', 'bs4', 'cufflinks', 'pandas']
    subjects = getSubjects()
    kaggle_used_apis = collections.OrderedDict({})
    for s in subjects:
        project = s.split('/')[0]
        notebook = s.split('/')[1]
        already_counted_api = []
        print('=== Processing Notebook: ' + project + '/' + notebook)
        old_file = jupyter_converted_notebooks_dir + '/' + project + '/' + notebook + '.py'
        function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = extractAPIUsage(old_file)
        if function_call_info_list is None:
            continue
        for f in function_call_info_list + attribute_ref_info_list + api_related_var_info_list:
            fqn = f['fqn']
            if '.' not in fqn:  # not api, but lib name
                continue
            if fqn.split('.')[0] not in target_libs:
                continue
            if fqn in already_counted_api:
                continue
            #print(fqn)
            if fqn not in kaggle_used_apis:
                kaggle_used_apis[fqn] = 1
                already_counted_api.append(fqn)
            else:
                kaggle_used_apis[fqn] += 1
                already_counted_api.append(fqn)
        # sort the map
        keys = list(kaggle_used_apis.keys())
        sorted_keys = sorted(keys, key=lambda x: kaggle_used_apis[x], reverse=True)
        sorted_map = collections.OrderedDict({})
        for k in sorted_keys:
            sorted_map[k] = kaggle_used_apis[k]
        # save the map
        with open(kaggle_api_usage_20_libs_info_file, 'w') as fw:
            json.dump(sorted_map, fw, indent=2)

def showNameMutationResults(name_mutation_results_json_file=NAME_MUTATION_RESULTS_INFO_JSON_FILE,
                            logs_dir=NAME_MUTATION_EXP_LOGS_DIR):
    with open(name_mutation_results_json_file, 'r') as fr:
        name_mutation_results = json.load(fr, object_hook=collections.OrderedDict)
    num_of_error_notebooks = 0
    num_of_error_apis = 0
    num_of_module_not_found_error_notebooks = 0
    num_of_attribute_error_notebooks = 0
    num_of_import_error_notebooks = 0
    num_of_type_error_notebooks = 0
    num_of_other_errors_notebooks = 0
    for api in name_mutation_results:
        num = name_mutation_results[api]['num']
        print('API: ' + api + ': ' + str(num))
        if not num == len(name_mutation_results[api]['error_cases']):
            print(api + 'Error Num Mismatch!')
            exit(0)
        if not num == 0:
            num_of_error_apis += 1
        for err in name_mutation_results[api]['error_cases']:
            num_of_error_notebooks += 1
            project = err.split('/')[0]
            notebook = err.split('/')[1]
            new_log_file = logs_dir + '/' + api + '/' + project + '/' + notebook + '.new.log'
            with open(new_log_file, 'r') as fr:
                lines = fr.readlines()
            for i,l in enumerate(lines):
                if 'Error: ' in l and ' Error: ' not in l:
                    print(err + ': ' + l.strip())
                    if 'ImportError: ' in l:
                        num_of_import_error_notebooks += 1
                    elif 'ModuleNotFoundError: ' in l:
                        num_of_module_not_found_error_notebooks += 1
                    elif 'AttributeError: ' in l:
                        num_of_attribute_error_notebooks += 1
                    #elif 'TypeError: ' in l:
                    #    num_of_type_error_notebooks += 1
                    else:
                        num_of_other_errors_notebooks += 1
                    break
    print('Num of Error APIs: ' + str(num_of_error_apis))
    print('Num of Error Notebooks: ' + str(num_of_error_notebooks))
    print('- ImportError: ' + str(num_of_import_error_notebooks))
    print('- ModuleNotFoundError: ' + str(num_of_module_not_found_error_notebooks))
    print('- AttributeError: ' + str(num_of_attribute_error_notebooks))
    #print('- TypeError: ' + str(num_of_type_error_notebooks))
    print('- Other Errors: ' + str(num_of_other_errors_notebooks))

def showArgMutationResults(arg_mutation_results_json_file=ARG_MUTATION_RESULTS_INFO_JSON_FILE,
                           logs_dir=ARG_MUTATION_EXP_LOGS_DIR):
    with open(arg_mutation_results_json_file, 'r') as fr:
        arg_mutation_results = json.load(fr, object_hook=collections.OrderedDict)
    num_of_error_notebooks = 0
    num_of_error_apis = 0
    num_of_value_error_notebooks = 0
    num_of_type_error_notebooks = 0
    num_of_other_errors_notebooks = 0
    for api in arg_mutation_results:
        num = arg_mutation_results[api]['num']
        print('API: ' + api + ': ' + str(num))
        if not num == len(arg_mutation_results[api]['error_cases']):
            print(api + 'Error Num Mismatch!')
            exit(0)
        if not num == 0:
            num_of_error_apis += 1
        for err in arg_mutation_results[api]['error_cases']:
            num_of_error_notebooks += 1
            project = err.split('/')[0]
            notebook = err.split('/')[1]
            new_log_file = logs_dir + '/' + api + '/' + project + '/' + notebook + '.new.log'
            with open(new_log_file, 'r') as fr:
                lines = fr.readlines()
            for i,l in enumerate(lines):
                if 'Error: ' in l and ' Error: ' not in l:
                    print(err + ': ' + l.strip())
                    if 'TypeError: ' in l:
                        num_of_type_error_notebooks += 1
                    elif 'ValueError: ' in l:
                        num_of_value_error_notebooks += 1
                    else:
                        num_of_other_errors_notebooks += 1
                    break
    print('Num of Error APIs: ' + str(num_of_error_apis))
    print('Num of Error Notebooks: ' + str(num_of_error_notebooks))
    print('- TypeError: ' + str(num_of_type_error_notebooks))
    print('- ValueError: ' + str(num_of_value_error_notebooks))
    #print('- TypeError: ' + str(num_of_type_error_notebooks))
    print('- Other Errors: ' + str(num_of_other_errors_notebooks))


from macros import ERROR_MSG_TO_REPAIR_ACTION_TRAIN_CSV_FILE
from macros import ERROR_MSG_TO_REPAIR_ACTION_TEST_CSV_FILE
def genErrMsgToRepairActionCSV(name_mutation_results_json_file=NAME_MUTATION_RESULTS_INFO_JSON_FILE,
                               name_mutation_logs_dir=NAME_MUTATION_EXP_LOGS_DIR,
                               arg_mutation_results_json_file=ARG_MUTATION_RESULTS_INFO_JSON_FILE,
                               arg_mutation_logs_dir=ARG_MUTATION_EXP_LOGS_DIR,
                               error_msg_to_repair_action_csv_file=ERROR_MSG_TO_REPAIR_ACTION_TRAIN_CSV_FILE):
    error_msg_to_repair_action_list = []
    with open(name_mutation_results_json_file, 'r') as fr:
        name_mutation_results = json.load(fr, object_pairs_hook=collections.OrderedDict)
    line_no = 2
    for api in name_mutation_results:
        if len(name_mutation_results[api]['error_cases']) != 0:
            print(api + ': lines ' + str(line_no) + ' ~ ' + str(len(name_mutation_results[api]['error_cases']) + line_no - 1))
            line_no += len(name_mutation_results[api]['error_cases'])
        for err in name_mutation_results[api]['error_cases']:
            project = err.split('/')[0]
            notebook = err.split('/')[1]
            new_log_file = name_mutation_logs_dir + '/' + api + '/' + project + '/' + notebook + '.new.log'
            with open(new_log_file, 'r') as fr:
                lines = fr.readlines()
            error_line_found = False
            for i, l in enumerate(lines):
                if 'Error: ' in l and ' Error: ' not in l:
                    error_line_found = True
                    err_msg = l.strip()
                    error_msg_to_repair_action_list.append((err_msg, 'FQN_RENAME'))
                    break
            if not error_line_found:
                print('Something wrong!')
                exit(0)
    with open(arg_mutation_results_json_file, 'r') as fr:
        arg_mutation_results = json.load(fr, object_hook=collections.OrderedDict)
    for api in arg_mutation_results:
        if len(arg_mutation_results[api]['error_cases']) != 0:
            print(api + ': lines ' + str(line_no) + ' ~ ' + str(len(arg_mutation_results[api]['error_cases']) + line_no - 1))
            line_no += len(arg_mutation_results[api]['error_cases'])
        for err in arg_mutation_results[api]['error_cases']:
            project = err.split('/')[0]
            notebook = err.split('/')[1]
            new_log_file = arg_mutation_logs_dir + '/' + api + '/' + project + '/' + notebook + '.new.log'
            with open(new_log_file, 'r') as fr:
                lines = fr.readlines()
            for i,l in enumerate(lines):
                if 'TypeError: ' in l:
                    err_msg = l.strip()
                    error_msg_to_repair_action_list.append((err_msg, 'ARG_RENAME_DELETE'))
                    break
                elif 'ValueError: ' in l:
                    err_msg = l.strip()
                    error_msg_to_repair_action_list.append((err_msg, 'ARG_VALUE_UPDATE'))
                    break
    with open(error_msg_to_repair_action_csv_file, 'w') as fw:
        for ea in error_msg_to_repair_action_list:
            fw.write(ea[0] + ',' + ea[1] + '\n')
    postProcessCSVFiles()


def postProcessCSVFiles(train_csv_file=ERROR_MSG_TO_REPAIR_ACTION_TRAIN_CSV_FILE,
                        test_csv_file=ERROR_MSG_TO_REPAIR_ACTION_TEST_CSV_FILE):
    for csv_file in [train_csv_file, test_csv_file]:
        with open(csv_file, 'r') as fr:
            lines = fr.readlines()
        with open(csv_file, 'w') as fw:
            writer = csv.writer(fw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['error_message', 'repair_action'])
            for i, l in enumerate(lines):
                err_msg = ','.join(lines[i].strip().split(',')[:-1])
                repair_action = lines[i].strip().split(',')[-1]
                writer.writerow([err_msg, repair_action])

def printNameMappingWeNeedMine(name_mutation_results_json_file=NAME_MUTATION_RESULTS_INFO_JSON_FILE,
                               name_mutation_logs_dir=NAME_MUTATION_EXP_LOGS_DIR,
                               manual_training_mapping_json_file=MANUAL_TRAINING_MAPPING_JSON_FILE):
    with open(manual_training_mapping_json_file, 'r') as fr:
        name_mapping = json.load(fr, object_hook=collections.OrderedDict)
    with open(name_mutation_results_json_file, 'r') as fr:
        name_mutation_results = json.load(fr, object_hook=collections.OrderedDict)
    for api in name_mutation_results:
        is_error = False
        for err in name_mutation_results[api]['error_cases']:
            project = err.split('/')[0]
            notebook = err.split('/')[1]
            new_log_file = name_mutation_logs_dir + '/' + api + '/' + project + '/' + notebook + '.new.log'
            with open(new_log_file, 'r') as fr:
                lines = fr.readlines()
            for i, l in enumerate(lines):
                if 'ImportError: ' in l:
                    is_error = True
                    break
                elif 'ModuleNotFoundError: ' in l:
                    is_error = True
                    break
                elif 'AttributeError: ' in l:
                    is_error = True
                    break
        if is_error:
            for old_api in name_mapping:
                if api in name_mapping[old_api]:
                    #print(old_api + ' -> ' + api)
                    print(old_api)
