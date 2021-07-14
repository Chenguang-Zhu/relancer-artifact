import os
import json
import re
import csv
import collections
import time
import shutil
import itertools
import subprocess as sub

from macros import ASE_RELANCER_FQN_MAPPING_JSON
from macros import ASE_ORIGINAL_NOTEBOOKS_DIR
from macros import ASE_FIXED_NOTEBOOKS_DIR
from macros import ASE_EXECUTION_LOGS_DIR
from macros import ASE_PATCHES_DIR

from api_extractor import extractAPIUsage

def runRelancer_OneCase(project, notebook, strategy):
    print('--- Running ' + project + '/' + notebook + ', strategy: ' + strategy)
    project_dir = ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project
    if not os.path.isdir(project_dir):
        os.makedirs(project_dir)
    changeDatasetPath(ASE_ORIGINAL_NOTEBOOKS_DIR + '/' + project + '/' + notebook + '.py')  # change dataset path
    shutil.copyfile(ASE_ORIGINAL_NOTEBOOKS_DIR + '/' + project + '/' + notebook + '.py',
                    project_dir + '/' + notebook + '.py')
    # clean any previous patch
    if os.path.isfile(ASE_PATCHES_DIR + '/' + strategy + '/' + project + '/' + notebook + '.patch'):
        os.remove(ASE_PATCHES_DIR + '/' + strategy + '/' + project + '/' + notebook + '.patch')
    runIterativeRepair_OneCase(project, notebook, strategy)

def runIterativeRepair_OneCase(project, notebook, strategy):
    repair_log_file = ASE_EXECUTION_LOGS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.repair.log'
    start_time = time.time()
    notebook_file = ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.py'
    patch_file = ASE_PATCHES_DIR + '/' + strategy + '/' + project + '/' + notebook + '.patch'
    prev_seen_err_msgs = []
    prev_tried_solutions = []
    remaining_solutions = []
    iter_no = 0
    previous_line_no_under_fix = 0
    previous_param_under_fix = None
    previous_repair_action_type = None
    num_of_errors_fixed_so_far = 0
    relative_offset = 0
    checkpoint_notebook_file = '/tmp/checkpoint.py'
    shutil.copyfile(notebook_file, checkpoint_notebook_file)
    checkpoint_patch_file = '/tmp/checkpoint.patch'
    with open(checkpoint_patch_file, 'w') as fw:
        fw.write('')
    checkpoint_error_api = None
    checkpoint_repair_action = None
    checkpoint_line_no = None
    while True:
        runCurrentNotebook(project, notebook, strategy)
        err_msg = extractErrorMsgFromLogFile(project, notebook, strategy)
        print('[ERR MSG]: ' + str(err_msg))
        if err_msg['text'] == '':
            print('[INFO] This case is fully fixed!')
            break
        prev_seen_err_msgs.append(err_msg)
        error_api, line_no = identifyErrorAPIFromErrMsg(err_msg, project, notebook, strategy)
        print('Error inducing API fqn: ' + error_api + ', line no: ' + str(line_no))
        repair_action = identifyRepairActionFromErrMsg_ML(err_msg, error_api, line_no, project, notebook, strategy)
        # if strategy == 'relancer' and not validateErrorAPI(iter_no, error_api, project, notebook):
        #    exit(0)
        # if strategy == 'relancer' and not validateRepairAction(iter_no, repair_action, project, notebook):
        #    exit(0)
        print('Repair Action: ' + repair_action['type'])
        if repair_action['type'] in ['fqn']:
            print('Line No: ' + str(line_no))
            print('Prev Line No: ' + str(previous_line_no_under_fix))
            if previous_line_no_under_fix + relative_offset == line_no:  # prev error still not fixed
                print('The previous issue is NOT fixed yet!')
                shutil.copyfile(checkpoint_notebook_file, notebook_file)  # recover
                if os.path.isfile(checkpoint_patch_file):
                    shutil.copyfile(checkpoint_patch_file, patch_file)
                error_api = checkpoint_error_api
                repair_action = checkpoint_repair_action
                line_no = checkpoint_line_no
            else:
                if len(prev_seen_err_msgs) > 1:
                    print('The previous issue is fixed!')
                #if previous_line_no_under_fix != line_no and prev_tried_solutions:  # prev error fixed, but a new error appears
                num_of_errors_fixed_so_far += 1
                shutil.copyfile(notebook_file, checkpoint_notebook_file)
                if os.path.isfile(patch_file):
                    shutil.copyfile(patch_file, checkpoint_patch_file)
                checkpoint_error_api = error_api
                checkpoint_repair_action = repair_action
                checkpoint_line_no = line_no
                remaining_solutions = []
                prev_tried_solutions = []
                iter_no += 1
                previous_line_no_under_fix = line_no
        elif repair_action['type'] in ['arg_name', 'arg_value']:
            print('Key Params: ' + str(repair_action['key_params']))
            print('Prev Params: ' + str(previous_param_under_fix))
            print('Prev Repair Action: ' + str(previous_repair_action_type))
            print('Line No: ' + str(line_no))
            print('Prev Line No: ' + str(previous_line_no_under_fix))
            if previous_line_no_under_fix + relative_offset == line_no and \
                repair_action['type'] == previous_repair_action_type and \
                set(repair_action['key_params']).issubset(set(previous_param_under_fix)) and \
                len(previous_param_under_fix) - len(repair_action['key_params']) <= 1:  # prev error still not fixed
                print('The previous issue is NOT fixed yet!')
                shutil.copyfile(checkpoint_notebook_file, notebook_file)  # recover
                if os.path.isfile(checkpoint_patch_file):
                    shutil.copyfile(checkpoint_patch_file, patch_file)  # recover
                error_api = checkpoint_error_api
                repair_action = checkpoint_repair_action
                line_no = checkpoint_line_no
            else:
                if len(prev_seen_err_msgs) > 1:
                    print('The previous issue is fixed!')
                #if repair_action['key_params'] != previous_param_under_fix:
                num_of_errors_fixed_so_far += 1
                shutil.copyfile(notebook_file, checkpoint_notebook_file)
                if os.path.isfile(patch_file):
                    shutil.copyfile(patch_file, checkpoint_patch_file)
                checkpoint_error_api = error_api
                checkpoint_repair_action = repair_action
                checkpoint_line_no = line_no
                remaining_solutions = []
                prev_tried_solutions = []
                iter_no += 1
                previous_line_no_under_fix = line_no
                previous_param_under_fix = repair_action['key_params']
                previous_repair_action_type = repair_action['type']
        else:  # unknown repair action
            print('Unknown error occurs!')
            if previous_line_no_under_fix + relative_offset == line_no:  # unknown error occurs in the same line, prev patch is invalid
                print('The previous issue is NOT fixed yet!')
                shutil.copyfile(checkpoint_notebook_file, notebook_file)  # recover
                error_api = checkpoint_error_api
                repair_action = checkpoint_repair_action
                line_no = checkpoint_line_no
            else:
                print('[INFO] This case cannot be fixed!')
                break  # cannot fix
        if not remaining_solutions and not prev_tried_solutions:  # the first run of each new error in the notebook
            if strategy == 'relancer':
                solutions = generateSolutions(strategy, error_api, line_no, repair_action, project, notebook)
            elif strategy == 'text':
                from baseline import generateSolutions_CombinationBaseline
                solutions = generateSolutions_CombinationBaseline(strategy, error_api, line_no, repair_action, project, notebook)
            elif strategy == 'random':
                from baseline import generateSolutions_RandomActionBaseline
                solutions = generateSolutions_RandomActionBaseline(strategy, error_api, line_no, repair_action, project, notebook)
            elif strategy == 'naive':
                from baseline import generateSolutions_NaiveBaseline
                solutions = generateSolutions_NaiveBaseline(strategy, error_api, line_no, repair_action, project, notebook)
            elif strategy == 'github':
                from baseline import generateSolutions_GithubOnly
                solutions = generateSolutions_GithubOnly(strategy, error_api, line_no, repair_action, project, notebook)
            elif strategy == 'apidoc':
                from baseline import generateSolutions_APIDocOnly
                solutions = generateSolutions_APIDocOnly(strategy, error_api, line_no, repair_action, project, notebook)
            remaining_solutions = solutions
        if not remaining_solutions:  # all solutions have been tried
            print('[INFO] This case cannot be fixed!')
            break  # cannot fix
        solution = remaining_solutions.pop(0)
        # --- logging -----------------
        print('Solution: ' + str(solution))
        with open(repair_log_file, 'a+') as fw:
            fw.write('[Try Solution]: ' + str(solution))
        # -----------------------------
        if solution in prev_tried_solutions:
            print('Solution has been tried before!')
            continue
        runFix(solution, repair_action, line_no, project, notebook, strategy)
        relative_offset = savePatch(strategy, project, notebook)
        print('Relative offset of this fix: ' + str(relative_offset))
        prev_tried_solutions.append(solution)
    end_time = time.time()
    duration = end_time - start_time
    with open(repair_log_file, 'a+') as fw:
        fw.write('[REPAIR EXEC TIME]: ' + str(duration))

def changeDatasetPath(notebook_file):
    with open(notebook_file, 'r') as fr:
        lines = fr.readlines()
    for i in range(len(lines)):
        if '\"../../input' in lines[i] or '\'../../input' in lines[i]:
            lines[i] = lines[i].replace('../../input', '../../../input')
    with open(notebook_file, 'w') as fw:
        fw.write(''.join(lines))

def runCurrentNotebook(project, notebook, strategy):
    cwd = os.getcwd()
    # run fixed
    os.chdir(ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project)
    output_dir = ASE_EXECUTION_LOGS_DIR + '/' + strategy + '/' + project
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    exec_log_file = output_dir + '/' + notebook + '.exec.log'
    repair_log_file = output_dir + '/' + notebook + '.repair.log'
    sub.run('python ' + notebook + '.py', shell=True, stdout=open(exec_log_file, 'w'), stderr=sub.STDOUT)
    with open(exec_log_file, 'r') as fr:
        exec_lines = fr.readlines()
    with open(repair_log_file, 'a+') as fw:  # append exec log to repair log
        fw.write(''.join(exec_lines))
    os.chdir(cwd)

def extractErrorMsgFromLogFile(project, notebook, strategy):
    log_file = ASE_EXECUTION_LOGS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.exec.log'
    with open(log_file, 'r') as fr:
        lines = fr.readlines()
    err_msg = collections.OrderedDict({})
    err_msg['project'] = project
    err_msg['notebook'] = notebook
    with open(log_file, 'r') as fr:
        lines = fr.readlines()
    # Check if it has any error
    error_exist = False
    for i, l in enumerate(lines):
        if 'Traceback (most recent call last):' in l and notebook + '.py' in lines[i+1]:
            error_exist = True
            for j in range(i, len(lines)):
                if 'Error: ' in lines[j].strip():
                    break
            if j != len(lines) - 1:
                err_msg['text'] = ''.join(lines[i+1:j+1])
            else:
                err_msg['text'] = ''.join(lines[i+1:])
            break
    if not error_exist:
        for i, l in enumerate(lines):
            if 'File \"' + notebook + '.py\", line ' in lines[i]:
                error_exist = True
                for j in range(i, len(lines)):
                    if 'Error: ' in lines[j].strip():
                        break
                if j != len(lines) - 1:
                    err_msg['text'] = ''.join(lines[i:j + 1])
                else:
                    err_msg['text'] = ''.join(lines[i:])
                break
    if not error_exist:
        err_msg['text'] = ''
    return err_msg

def identifyErrorAPIFromErrMsg(err_msg, project, notebook, strategy):
    notebook_file = ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.py'
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = extractAPIUsage(notebook_file)
    if function_call_info_list is None and attribute_ref_info_list is None and api_related_var_info_list is None and \
            all_imported_names_map is None and all_import_names_line_no_map is None:
        return 'UNKNOWN', 0
    #print(all_imported_names_map)
    #print(function_call_info_list)
    #print(attribute_ref_info_list)
    #print(api_related_var_info_list)
    #print(all_import_names_line_no_map)
    text = err_msg['text']
    lines = text.split('\n')
    error_api = 'UNKNOWN'
    for i, l in enumerate(lines):
        if l.strip().startswith('File ') and ', line ' in l and notebook in l:
            print(l)
            if 'import ' in lines[i+1]:  # import line
                if 'from ' in lines[i+1]:
                    from_part = lines[i+1].strip().split('from ')[-1].split(' import ')[0].strip()
                    import_part = lines[i+1].strip().split('import ')[-1].split(' as ')[0].split('.')[-1].split('#')[0].strip()
                    error_line = [l.strip() for l in lines if 'Error: ' in l and ' Error: ' not in l][0]
                    #print(error_line)
                    if '\'' not in error_line or error_line.split('\'')[1] in from_part:
                        err_api_short_name = from_part
                    else:
                        if ',' in import_part:  # multiple imports on one line
                            imported_modules = [m.strip() for m in import_part.split(',')]
                            for m in imported_modules:
                                if error_line.split('\'')[1] == m:
                                    err_api_short_name = m
                        else:
                            err_api_short_name = import_part
                    #for keyword in error_line.strip().split('\''):
                    #    if keyword != '' and keyword.count(' ') == 0 and err_api_short_name.startswith(keyword):
                    #        err_api_short_name = keyword
                else:
                    err_api_short_name = lines[i+1].strip().split('import ')[-1].split(' as ')[0].split('.')[-1]
            else:  # regular line
                err_api_short_name = lines[i+1].split('print(')[-1].split('(')[0].split('=')[-1].strip().split('.')[-1]
                try:
                    error_line = [l.strip() for l in lines if 'Error: ' in l and ' Error: ' not in l][0]
                    if err_api_short_name not in error_line:
                        err_api_short_name_cands = [token.split('(')[0] for token in lines[i+1].split('.')]
                        for cand in err_api_short_name_cands:
                            if '\'' + cand + '\'' in error_line:
                                err_api_short_name = cand
                    if err_api_short_name not in error_line:
                        err_api_short_name_cands = [token for token in lines[i + 1].split('(')]
                        for cand in err_api_short_name_cands:
                            if '.' in cand:
                                cand = cand.split('.')[-1]
                            if cand in [all_imported_names_map[i][0] for i in all_imported_names_map]:
                                err_api_short_name = cand
                except (IndexError, TypeError) as e:
                    pass
            line_no = int(l.strip().split(', line ')[1].split(',')[0])
            if ' ' in err_api_short_name:
                err_api_short_name = err_api_short_name.split()[-1]
            if err_api_short_name in ['fit', 'fit_transform']:
                err_api_short_name, line_no = findConstructorNameAndLineNo(lines[i+1].strip(), notebook_file)
            print('Error inducing API short name: ' + err_api_short_name + ', line no: ' + str(line_no))
            #print(function_call_info_list)
            for imported_fqn in all_import_names_line_no_map:
                import_line_nos = all_import_names_line_no_map[imported_fqn]
                if line_no in import_line_nos:
                    if imported_fqn.endswith(err_api_short_name):
                        error_api = imported_fqn
                        return error_api, line_no
            for call_info in function_call_info_list:
                if call_info['line_no'] == line_no:
                    if call_info['fqn'].endswith(err_api_short_name):
                        error_api = call_info['fqn']
                        return error_api, line_no
            for attr_ref in attribute_ref_info_list:
                if attr_ref['line_no'] == line_no:
                    if attr_ref['fqn'].endswith(err_api_short_name):
                        error_api = attr_ref['fqn']
                        return error_api, line_no
            for var_ref in api_related_var_info_list:
                if var_ref['line_no'] == line_no:
                    if var_ref['fqn'].endswith(err_api_short_name):
                        error_api = var_ref['fqn']
                        return error_api, line_no
    return error_api, line_no

def findConstructorNameAndLineNo(fit_line, notebook_file):
    var_name = fit_line.split('.')[0].split('=')[-1].strip()
    #print(var_name)
    with open(notebook_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        if '=' in lines[i] and var_name in lines[i].strip().split('=')[0] and var_name not in lines[i].strip().split('=')[-1]:
            print(lines[i])
            err_api_short_name = lines[i].strip().split('=')[1].split('.')[-1].split('(')[0].strip()
            line_no = i+1
    return err_api_short_name, line_no

def identifyRepairActionFromErrMsg_ML(err_msg, error_api, line_no, project, notebook, strategy):
    from macros import R_REPAIR_ACTION_MODEL_PREDICTION_CSV
    repair_action = collections.OrderedDict({})
    fp = open(R_REPAIR_ACTION_MODEL_PREDICTION_CSV, 'r')
    csv_reader = csv.reader(fp, delimiter=",", quotechar='"')
    repair_action_type = 'unknown'
    from error_msg_parsing import extractInfoLineFromErrMsg
    error_line = extractInfoLineFromErrMsg(err_msg['text'])
    print("*** " + error_line)
    for row in csv_reader:
        if error_line.strip() == row[0].strip():
            repair_action_type = row[-1]
            break
    if repair_action_type == 'FQN_RENAME':
        repair_action_type = 'fqn'
    elif repair_action_type == 'ARG_VALUE_UPDATE':
        repair_action_type = 'arg_value'
        try:
            text = err_msg['text']
            #error_line = [l.strip() for l in text.split('\n') if 'Error: ' in l and ' Error: ' not in l][0]
            # print(error_type)
        except IndexError:
            pass
        repair_action['key_params'] = identifyKeyParamsForArgValueChange(error_line, error_api)
    elif repair_action_type.startswith('ARG_RENAME'):
        repair_action_type = 'arg_name'
        try:
            text = err_msg['text']
            #error_line = [l.strip() for l in text.split('\n') if 'Error: ' in l and ' Error: ' not in l][0]
            # print(error_type)
        except IndexError:
            pass
        repair_action['key_params'], repair_action['explicit_used_params'] = identifyKeyParamsForArgNameChange(
            error_line, error_api, line_no, project, notebook, strategy)
    repair_action['type'] = repair_action_type
    # print(repair_action)
    return repair_action

def identifyKeyParamsForArgNameChange(error_line, error_api, line_no, project, notebook, strategy):
    args_to_fix = []
    notebook_file = ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.py'
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = \
        extractAPIUsage(notebook_file)
    explicit_used_args = []
    for func_call in function_call_info_list:
        if func_call['fqn'] == error_api and func_call['line_no'] == line_no:
            #print(func_call['args'])
            for arg in func_call['args']:
                if arg not in explicit_used_args:
                    explicit_used_args.append(arg)
                if ' ' + arg + ' ' in error_line or '\'' + arg + '\'' in error_line:
                    if arg not in args_to_fix:
                        args_to_fix.append(arg)
            if args_to_fix:
                return args_to_fix, explicit_used_args
    for attr_ref in attribute_ref_info_list:
        if attr_ref['fqn'] == error_api and attr_ref['line_no'] == line_no:
            if 'args' not in attr_ref:
                break
            #print(func_call['args'])
            for arg in attr_ref['args']:
                if arg not in explicit_used_args:
                    explicit_used_args.append(arg)
                if ' ' + arg + ' ' in error_line or '\'' + arg + '\'' in error_line:
                    if arg not in args_to_fix:
                        args_to_fix.append(arg)
            if args_to_fix:
                return args_to_fix, explicit_used_args
    for var_ref in api_related_var_info_list:
        if var_ref['fqn'] == error_api and var_ref['line_no'] == line_no:
            if 'args' not in var_ref:
                continue
            #print(func_call['args'])
            for arg in var_ref['args']:
                if arg not in explicit_used_args:
                    explicit_used_args.append(arg)
                if ' ' + arg + ' ' in error_line or '\'' + arg + '\'' in error_line:
                    if arg not in args_to_fix:
                        args_to_fix.append(arg)
            if args_to_fix:
                return args_to_fix, explicit_used_args
    if not args_to_fix:  # back to enumeration
        for func_call in function_call_info_list:
            if func_call['fqn'] == error_api and func_call['line_no'] == line_no:
                for arg in func_call['args']:
                    args_to_fix.append(arg)
    return args_to_fix, explicit_used_args

def identifyKeyParamsForArgValueChange(error_line, error_api):
    from macros import ASE_GITHUB_KNOWLEDGE_JSON
    from macros import ASE_APIDOC_KNOWLEDGE_JSON
    with open(ASE_GITHUB_KNOWLEDGE_JSON, 'r') as fr:
        github_knowledge = json.load(fr)
    with open(ASE_APIDOC_KNOWLEDGE_JSON, 'r') as fr:
        apidoc_knowledge = json.load(fr)
    all_known_params = []
    args_to_fix = []
    for k in apidoc_knowledge:
        if k['API'] == error_api:
            all_known_params += k['params']
            break
    for k in github_knowledge:
        if k['API'] == error_api:
            for p in k['params']:
                if p not in all_known_params:
                    all_known_params.append(p)
            break
    print('All known params: ' + str(all_known_params))
    all_known_params_names = [p['name'] for p in all_known_params]
    print('All known params\' names: ' + str(all_known_params_names))
    error_line_tokens = [word.strip().strip('\'').strip('\"').strip('.').strip(',').strip(':').lower() for word in error_line.strip().split()]
    print('Error line tokens: ' + str(error_line_tokens))
    for token in error_line_tokens:
        if token in all_known_params_names:
            if token not in args_to_fix:
                args_to_fix.append(token)
        elif token + 's' in all_known_params_names:
            if token + 's' not in args_to_fix:
                args_to_fix.append(token + 's')
    #print('Args to fix: ' + str(args_to_fix))
    if not args_to_fix:
        for p in all_known_params:
            if isinstance(p['values'], list):
                args_to_fix.append(p['name'])
    return args_to_fix

def generateSolutions(strategy, error_api, line_no, repair_action, project, notebook):
    if repair_action['type'] == 'fqn':
        github_fqn_map_for_this_api = getGitHubFQNCandidates(error_api)
        print('GitHub Candidates:')
        print(github_fqn_map_for_this_api)
        apidoc_fqn_map_for_this_api = getAPIDocFQNCandidates(error_api)
        print('APIDoc Candidates (only show top 10):')
        print(list(apidoc_fqn_map_for_this_api[error_api].keys())[:10])
        all_fqn_candidates_with_final_scores = rankFQNCandidates_ML(error_api, github_fqn_map_for_this_api, apidoc_fqn_map_for_this_api, strategy, project, notebook)
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

def getGitHubFQNCandidates(error_api):
    from macros import ASE_RELANCER_GITHUB_FQN_MAPPING_FILE
    if not os.path.isfile(ASE_RELANCER_GITHUB_FQN_MAPPING_FILE):
        from mapping import runBuildMappingUsingGitHub
        runBuildMappingUsingGitHub()
    with open(ASE_RELANCER_GITHUB_FQN_MAPPING_FILE, 'r') as fr:
        github_fqn_map = json.load(fr, object_pairs_hook=collections.OrderedDict)
    github_fqn_candidates_with_occurrences = collections.OrderedDict({})
    for old_api in github_fqn_map:
        if error_api == old_api:
            github_fqn_candidates_with_occurrences = github_fqn_map[old_api]
            break
    return collections.OrderedDict({error_api: github_fqn_candidates_with_occurrences})

def getAPIDocFQNCandidates(error_api):
    from macros import ASE_RELANCER_APIDOC_FQN_MAPPING_FILE
    if not os.path.isfile(ASE_RELANCER_APIDOC_FQN_MAPPING_FILE):
        from mapping import runBuildMappingUsingAPIdoc
        runBuildMappingUsingAPIdoc()
    with open(ASE_RELANCER_APIDOC_FQN_MAPPING_FILE, 'r') as fr:
        apidoc_fqn_map = json.load(fr, object_pairs_hook=collections.OrderedDict)
    apidoc_fqn_candidates = collections.OrderedDict({})
    for old_api in apidoc_fqn_map:
        if error_api == old_api:
            apidoc_fqn_candidates = apidoc_fqn_map[old_api]
            break
    return collections.OrderedDict({error_api: apidoc_fqn_candidates})

def rankFQNCandidates_ML(error_api, github_fqn_map_for_this_api, apidoc_fqn_map_for_this_api, strategy, project, notebook):
    from macros import ASE_CANDIDATE_RANKING_MODEL_OUTPUT_CSV_FILE
    if os.path.isfile(ASE_RELANCER_FQN_MAPPING_JSON):
        with open(ASE_RELANCER_FQN_MAPPING_JSON, 'r') as fr:
            mapping = json.load(fr, object_pairs_hook=collections.OrderedDict)
            return mapping[error_api] if error_api in mapping else collections.OrderedDict({})
    with open(ASE_CANDIDATE_RANKING_MODEL_OUTPUT_CSV_FILE, 'r') as fr:
        lines = fr.readlines()[1:]
    prob_dict = collections.OrderedDict({})
    for i, l in enumerate(lines):
        old_api = l.strip().split(',')[0]
        if old_api not in prob_dict:
            prob_dict[old_api] = []
        cand_api = l.strip().split(',')[1]
        prob = l.strip().split(',')[-1]
        cands = [x[0] for x in prob_dict[old_api]]
        if cand_api not in cands:
            prob_dict[old_api].append((cand_api, prob))
    # sort by prob
    for old_api in prob_dict:
        prob_dict[old_api] = sorted(prob_dict[old_api], key=lambda x: float(x[1]), reverse=True)
    mapping = collections.OrderedDict({})
    for old_api in prob_dict:
        if old_api not in mapping:
            mapping[old_api] = collections.OrderedDict({})
            for item in prob_dict[old_api]:
                if item[0] not in mapping[old_api]:
                    mapping[old_api][item[0]] = collections.OrderedDict({})
                mapping[old_api][item[0]]['score'] = float(item[1])
    with open(ASE_RELANCER_FQN_MAPPING_JSON, 'w') as fw:
        json.dump(mapping, fw, indent=2)
    all_fqn_candidates_with_final_scores = mapping[error_api] if error_api in mapping else collections.OrderedDict({})
    #if not all_fqn_candidates_with_final_scores:
    #    all_fqn_candidates_with_final_scores = rankFQNCandidates(error_api, github_fqn_map_for_this_api, apidoc_fqn_map_for_this_api, strategy, project, notebook)
    return all_fqn_candidates_with_final_scores

def genFQNRenamingSolutions(error_api, line_no, all_fqn_candidates_with_final_scores):
    # line_no, old_fqn, new_fqn
    all_fqn_renaming_solutions = []
    for fqn_candidate in all_fqn_candidates_with_final_scores:
        solution = collections.OrderedDict({})
        solution['action'] = 'fqn'
        solution['old_fqn'] = error_api
        solution['new_fqn'] = fqn_candidate
        solution['line_no'] = line_no
        if solution not in all_fqn_renaming_solutions:
            all_fqn_renaming_solutions.append(solution)
    if len(all_fqn_renaming_solutions) == 0 and '.' + error_api.split('.')[-1] + '.' + error_api.split('.')[-1] in error_api:
        solution = collections.OrderedDict({})
        solution['action'] = 'fqn'
        solution['old_fqn'] = error_api
        solution['new_fqn'] = error_api.replace('.' + error_api.split('.')[-1] + '.' + error_api.split('.')[-1], '.' + error_api.split('.')[-1])
        solution['line_no'] = line_no
        all_fqn_renaming_solutions.append(solution)
    return all_fqn_renaming_solutions

def getGitHubArgNameCandidates(error_api, key_params_names, explict_used_param_names):
    from macros import ASE_GITHUB_KNOWLEDGE_JSON
    with open(ASE_GITHUB_KNOWLEDGE_JSON, 'r') as fr:
        github_arg_knowledge = json.load(fr)
    solution_cands = []
    if error_api not in [k['API'] for k in github_arg_knowledge]:
        return solution_cands
    for k in github_arg_knowledge:
        if k['API'] == error_api:
            params = k['params']
    if not params:  # knowledge base does not have this api
        return solution_cands
    for kpn in key_params_names:
        # delete arg
        solution = collections.OrderedDict({})
        solution['API'] = error_api
        solution['action'] = 'arg_delete'
        solution['from_name'] = kpn
        solution['to_name'] = '*'
        if solution not in solution_cands:
            solution_cands.append(solution)
        # rename arg
        for p in params:
            if p['name'] != kpn and p['name'] not in explict_used_param_names:
                solution = collections.OrderedDict({})
                solution['API'] = error_api
                solution['action'] = 'arg_rename'
                solution['from_name'] = kpn
                solution['to_name'] = p['name']
                if solution not in solution_cands:
                    solution_cands.append(solution)
    if not key_params_names:  # no key arg identified, back to enumeration
        for from_p in params:
            for to_p in params:
                if from_p['name'] != to_p['name']:
                    solution = collections.OrderedDict({})
                    solution['API'] = error_api
                    solution['action'] = 'arg_rename'
                    solution['from_name'] = from_p['name']
                    solution['to_name'] = to_p['name']
                    if solution not in solution_cands:
                        solution_cands.append(solution)
            solution = collections.OrderedDict({})
            solution['API'] = error_api
            solution['action'] = 'arg_delete'
            solution['from_name'] = from_p['name']
            solution['to_name'] = '*'
            if solution not in solution_cands:
                solution_cands.append(solution)
        # subsets: change multiple args at the same time
        param_names = [p['name'] for p in params]
        for num_of_elements in range(2, len(param_names) + 1):
            for from_subset in set(itertools.combinations(set(param_names), num_of_elements)):
                from_params = ''
                for from_p in from_subset:
                    from_params += from_p + '&'
                from_params = from_params[:-1]
                for to_permutation in itertools.permutations(set(param_names), num_of_elements):
                    to_params = ''
                    for to_p in to_permutation:
                        to_params += to_p + '&'
                    to_params = to_params[:-1]
                    if not isChangingAllParams(from_params, to_params):
                        continue
                    solution = collections.OrderedDict({})
                    solution['API'] = error_api
                    solution['action'] = 'arg_rename'
                    solution['from_name'] = from_params
                    solution['to_name'] = to_params
                    if solution not in solution_cands:
                        solution_cands.append(solution)
                    if len(solution_cands) > 5000:
                        return solution_cands
    return solution_cands

def getAPIDocArgNameCandidates(error_api, key_params_names, explict_used_param_names):
    from macros import ASE_APIDOC_KNOWLEDGE_JSON
    with open(ASE_APIDOC_KNOWLEDGE_JSON, 'r') as fr:
        apidoc_arg_knowledge = json.load(fr)
    solution_cands = []
    params = []
    for k in apidoc_arg_knowledge:
        if k['API'] == error_api:
            params = k['params']
    if not params:  # knowledge base does not have this api
        return solution_cands
    for kpn in key_params_names:
        # delete arg
        solution = collections.OrderedDict({})
        solution['API'] = error_api
        solution['action'] = 'arg_delete'
        solution['from_name'] = kpn
        solution['to_name'] = '*'
        if solution not in solution_cands:
            solution_cands.append(solution)
        # rename arg
        for p in params:
            if p['name'] != kpn and p['name'] not in explict_used_param_names:
                solution = collections.OrderedDict({})
                solution['API'] = error_api
                solution['action'] = 'arg_rename'
                solution['from_name'] = kpn
                solution['to_name'] = p['name']
                if solution not in solution_cands:
                    solution_cands.append(solution)
    if not key_params_names:  # no key arg identified, back to enumeration
        for from_p in params:
            for to_p in params:
                if from_p['name'] != to_p['name']:
                    solution = collections.OrderedDict({})
                    solution['API'] = error_api
                    solution['action'] = 'arg_rename'
                    solution['from_name'] = from_p['name']
                    solution['to_name'] = to_p['name']
                    if solution not in solution_cands:
                        solution_cands.append(solution)
            solution = collections.OrderedDict({})
            solution['API'] = error_api
            solution['action'] = 'arg_delete'
            solution['from_name'] = from_p['name']
            solution['to_name'] = '*'
            if solution not in solution_cands:
                solution_cands.append(solution)
        # subsets: change multiple args at the same time
        param_names = [p['name'] for p in params]
        for num_of_elements in range(2, len(param_names) + 1):
            for from_subset in set(itertools.combinations(set(param_names), num_of_elements)):
                from_params = ''
                for from_p in from_subset:
                    from_params += from_p + '&'
                from_params = from_params[:-1]
                for to_permutation in itertools.permutations(set(param_names), num_of_elements):
                    to_params = ''
                    for to_p in to_permutation:
                        to_params += to_p + '&'
                    to_params = to_params[:-1]
                    if not isChangingAllParams(from_params, to_params):
                        continue
                    solution = collections.OrderedDict({})
                    solution['API'] = error_api
                    solution['action'] = 'arg_rename'
                    solution['from_name'] = from_params
                    solution['to_name'] = to_params
                    if solution not in solution_cands:
                        solution_cands.append(solution)
                    if len(solution_cands) > 5000:
                        return solution_cands
    return solution_cands

def getGitHubArgValueCandidates(error_api, key_params_names):
    from macros import ASE_GITHUB_KNOWLEDGE_JSON
    with open(ASE_GITHUB_KNOWLEDGE_JSON, 'r') as fr:
        github_arg_knowledge = json.load(fr)
    solution_cands = []
    if error_api not in [k['API'] for k in github_arg_knowledge]:
        return solution_cands
    for k in github_arg_knowledge:
        if k['API'] == error_api:
            params = k['params']
    if not params:  # knowledge base does not have this api
        return solution_cands
    for kpn in key_params_names:
        for p in params:
            if p['name'] == kpn:
                value_ranges = p['values']
                if isinstance(value_ranges, list):  # only handle categorical values
                    for value in value_ranges:
                        if value == 'None':
                            continue
                        solution = collections.OrderedDict({})
                        solution['API'] = error_api
                        solution['action'] = 'arg_value_update'
                        solution['arg_name'] = kpn
                        solution['target_value'] = value
                        if solution not in solution_cands:
                            solution_cands.append(solution)
    if not key_params_names:  # no key arg identified, back to enumeration
        for p in params:
            value_ranges = p['values']
            if isinstance(value_ranges, list):
                for value in value_ranges:
                    if value == 'None':
                        continue
                    solution = collections.OrderedDict({})
                    solution['API'] = error_api
                    solution['action'] = 'arg_value_update'
                    solution['arg_name'] = p['name']
                    solution['target_value'] = value
                    if solution not in solution_cands:
                        solution_cands.append(solution)
        # subsets: change multiple args at the same time
        param_names = [p['name'] for p in params]
        for num_of_elements in range(2, len(param_names) + 1):
            for subset in set(itertools.combinations(set(param_names), num_of_elements)):
                target_params = ''
                for p in subset:
                    target_params += p + '&'
                target_params = target_params[:-1]
                target_values = ''
                for p in subset:
                    value_ranges = getParamValueRange(p, params)
                    if isinstance(value_ranges, list):
                        for value in value_ranges:
                            target_values += value + '&'
                            if target_values.count('&') == target_params.count('&') + 1:
                                target_values = target_values[:-1]
                                solution = collections.OrderedDict({})
                                solution['API'] = error_api
                                solution['action'] = 'arg_value_update'
                                solution['arg_name'] = target_params
                                solution['target_value'] = target_values
                                if solution not in solution_cands:
                                    solution_cands.append(solution)
                                target_values = ''
                                if len(solution_cands) > 5000:
                                    return solution_cands
    return solution_cands

def getAPIDocArgValueCandidates(error_api, key_params_names):
    from macros import ASE_APIDOC_KNOWLEDGE_JSON
    with open(ASE_APIDOC_KNOWLEDGE_JSON, 'r') as fr:
        github_arg_knowledge = json.load(fr)
    solution_cands = []
    if error_api not in [k['API'] for k in github_arg_knowledge]:
        return solution_cands
    for k in github_arg_knowledge:
        if k['API'] == error_api:
            params = k['params']
    if not params:  # knowledge base does not have this api
        return solution_cands
    for kpn in key_params_names:
        for p in params:
            if p['name'] == kpn:
                value_ranges = p['values']
                if isinstance(value_ranges, list):  # only handle categorical values
                    for value in value_ranges:
                        if value == 'None':
                            continue
                        solution = collections.OrderedDict({})
                        solution['API'] = error_api
                        solution['action'] = 'arg_value_update'
                        solution['arg_name'] = kpn
                        solution['target_value'] = value
                        if solution not in solution_cands:
                            solution_cands.append(solution)
    if not key_params_names:  # no key arg identified, back to enumeration
        for p in params:
            value_ranges = p['values']
            if isinstance(value_ranges, list):
                for value in value_ranges:
                    if value == 'None':
                        continue
                    solution = collections.OrderedDict({})
                    solution['API'] = error_api
                    solution['action'] = 'arg_value_update'
                    solution['arg_name'] = p['name']
                    solution['target_value'] = value
                    if solution not in solution_cands:
                        solution_cands.append(solution)
        # subsets: change multiple args at the same time
        param_names = [p['name'] for p in params]
        for num_of_elements in range(2, len(param_names) + 1):
            for subset in set(itertools.combinations(set(param_names), num_of_elements)):
                target_params = ''
                for p in subset:
                    target_params += p + '&'
                target_params = target_params[:-1]
                target_values = ''
                for p in subset:
                    value_ranges = getParamValueRange(p, params)
                    if isinstance(value_ranges, list):
                        for value in value_ranges:
                            target_values += value + '&'
                            if target_values.count('&') == target_params.count('&') + 1:
                                target_values = target_values[:-1]
                                solution = collections.OrderedDict({})
                                solution['API'] = error_api
                                solution['action'] = 'arg_value_update'
                                solution['arg_name'] = target_params
                                solution['target_value'] = target_values
                                if solution not in solution_cands:
                                    solution_cands.append(solution)
                                target_values = ''
                                if len(solution_cands) > 5000:
                                    return solution_cands
    return solution_cands

def runFix(solution, repair_action, line_no, project, notebook, strategy):
    print('Run Fix')
    if solution['action'] == 'fqn':
        old_api = solution['old_fqn']
        new_api = solution['new_fqn']
        runFixAPIFQN(old_api, new_api, strategy, project, notebook)
    elif solution['action'] in ['arg_rename', 'arg_delete']:
        api = solution['API']
        action = solution['action']
        from_name = solution['from_name']
        to_name = solution['to_name']
        runFixArgName(api, action, line_no, from_name, to_name, strategy, project, notebook)
    elif solution['action'] == 'arg_value_update':
        api = solution['API']
        action = solution['action']
        arg_name = solution['arg_name']
        target_value = solution['target_value']
        runFixArgValue(api, action, line_no, arg_name, target_value, strategy, project, notebook)

def runFixAPIFQN(old_api, new_api, strategy, project, notebook):
    notebook_file = ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.py'
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = \
        extractAPIUsage(notebook_file)
    new_top_level_pkgs = []
    from ast_migration import migrateOneAPIOneCand_AST
    new_top_level_pkgs = migrateOneAPIOneCand_AST(old_api, new_api, project, notebook, notebook_file,
                                                  function_call_info_list,
                                                  attribute_ref_info_list,
                                                  api_related_var_info_list,
                                                  all_imported_names_map,
                                                  all_import_names_line_no_map,
                                                  new_top_level_pkgs)
    with open(notebook_file, 'r') as fr:
        lines = fr.readlines()
    for pkg in new_top_level_pkgs:
        if pkg.startswith('[SPECIAL] '):
            lines.insert(0, pkg.split('[SPECIAL] ')[1] + '\n')
        else:
            lines.insert(0, 'import ' + pkg + '\n')
    with open(notebook_file, 'w') as fw:
        fw.write(''.join(lines))

def runFixArgName(api, action, line_no, from_name, to_name, strategy, project, notebook):
    notebook_file = ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.py'
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = \
        extractAPIUsage(notebook_file)
    edited = False
    # func call
    for call_info in function_call_info_list:
        if call_info['fqn'] == api:
            target_line_no = call_info['line_no']
            orig_func_str = call_info['func_str']
            print('CALL', target_line_no, orig_func_str)
            if line_no != target_line_no:
                continue
            print(' -- should fix')
            applyArgNameEdit(api, action, line_no, from_name, to_name, strategy, project, notebook)
            edited = True
    # attr ref
    for attr_ref_info in attribute_ref_info_list:
        if attr_ref_info['fqn'] == api:
            target_line_no = attr_ref_info['line_no']
            orig_attr_ref_str = attr_ref_info['attr_str']
            print('ATTR', target_line_no, orig_attr_ref_str)
            if line_no != target_line_no:
                continue
            print(' -- should fix')
            applyArgNameEdit(api, action, line_no, from_name, to_name, strategy, project, notebook)
            edited = True
    # var
    for var_info in api_related_var_info_list:
        if var_info['fqn'] == api:
            target_line_no = var_info['line_no']
            orig_var_str = var_info['var_str']
            print('VAR', target_line_no, orig_var_str)
            if line_no != target_line_no:
                continue
            print(' -- should fix')
            applyArgNameEdit(api, action, line_no, from_name, to_name, strategy, project, notebook)
            edited = True
    if edited:
        print('Fix arg of ' + api)

def applyArgNameEdit(api, action, line_no, from_name, to_name, strategy, project, notebook):
    notebook_file = ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.py'
    with open(notebook_file, 'r') as fr:
        lines = fr.readlines()
    if '&' in from_name:
        from_name = from_name.split('&')[0]
    if '&' in to_name:
        to_name = to_name.split('&')[0]
    if action == 'arg_delete':
        pattern = from_name + r'[\s]?=[\s]?[^(,|\))]*(,|\))'
        if not re.search(pattern, lines[line_no - 1]):
            return
        matched_str = re.search(pattern, lines[line_no - 1])[0]
        if matched_str.endswith(','):
            lines[line_no - 1] = re.sub(pattern, '', lines[line_no - 1], 1)  # e.g., x=1, ax=2
        else:
            lines[line_no - 1] = re.sub(pattern, ')', lines[line_no - 1], 1)
        pattern = r',\s*\)'
        lines[line_no - 1] = re.sub(pattern, ')', lines[line_no - 1])
        pattern = r'\([^\[]*\]'
        lines[line_no - 1] = re.sub(pattern, '(', lines[line_no - 1])
        #if '(,' in lines[line_no - 1]:
        #    lines[line_no - 1] = lines[line_no - 1].replace('(,', '(')
    elif action == 'arg_rename':
        if from_name not in lines[line_no - 1]:  # Do not rename if the arg is not there
            return
        if '(' not in lines[line_no - 1] or ')' not in lines[line_no - 1]:  # no (), do not rename
            return
        pattern = from_name + "\s*=\s*"
        lines[line_no - 1] = re.sub(pattern, to_name + '=', lines[line_no - 1])
    with open(notebook_file, 'w') as fw:
        fw.write(''.join(lines))

def runFixArgValue(api, action, line_no, arg_name, target_value, strategy, project, notebook):
    notebook_file = ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.py'
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = \
        extractAPIUsage(notebook_file)
    edited = False
    # func call
    for call_info in function_call_info_list:
        if call_info['fqn'] == api:
            target_line_no = call_info['line_no']
            orig_func_str = call_info['func_str']
            print('CALL', target_line_no, orig_func_str)
            if line_no != target_line_no:
                continue
            print(' -- should fix')
            applyArgValueEdit(api, action, line_no, arg_name, target_value, strategy, project, notebook)
            edited = True
    # attr ref
    for attr_ref_info in attribute_ref_info_list:
        if attr_ref_info['fqn'] == api:
            target_line_no = attr_ref_info['line_no']
            orig_attr_ref_str = attr_ref_info['attr_str']
            print('ATTR', target_line_no, orig_attr_ref_str)
            if line_no != target_line_no:
                continue
            print(' -- should fix')
            applyArgValueEdit(api, action, line_no, arg_name, target_value, strategy, project, notebook)
            edited = True
    # var
    for var_info in api_related_var_info_list:
        if var_info['fqn'] == api:
            target_line_no = var_info['line_no']
            orig_var_str = var_info['var_str']
            print('VAR', target_line_no, orig_var_str)
            if line_no != target_line_no:
                continue
            print(' -- should fix')
            applyArgValueEdit(api, action, line_no, arg_name, target_value, strategy, project, notebook)
            edited = True
    if edited:
        print('Fix arg of ' + api)

def applyArgValueEdit(api, action, line_no, arg_name, target_value, strategy, project, notebook):
    notebook_file = ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.py'
    with open(notebook_file, 'r') as fr:
        lines = fr.readlines()
    short_func_name = api.split('.')[-1]
    if '&' in arg_name:
        arg_name = arg_name.split('&')[0]
    if '&' in target_value:
        target_value = target_value.split('&')[0]
    if action == 'arg_value_update':
        if target_value not in ['None', 'np.nan', 'True', 'False']:
            target_value = '\"' + str(target_value) + '\"'
        if arg_name in lines[line_no - 1] and lines[line_no - 1].split(arg_name)[-1].split('=')[0].strip() == '':  # explicit
            pattern = arg_name + r'=[^(,|\))]*(,|\))'
            if not re.search(pattern, lines[line_no - 1]):
                return
            matched_str = re.search(pattern, lines[line_no - 1])[0]
            if matched_str.endswith(','):
                lines[line_no - 1] = re.sub(pattern, arg_name + '=' + target_value + ',', lines[line_no - 1])
            else:
                lines[line_no - 1] = re.sub(pattern, arg_name + '=' + target_value + ')', lines[line_no - 1])
            if "(," in lines[line_no - 1]:
                lines[line_no - 1] = lines[line_no - 1].replace("(,", "(")
            #lines[target_line_no - 1] = lines[target_line_no - 1].split(arg_name + '=')[0] + \
            #                            arg_name + '=' + target_value + ',' + \
            #                           ','.join(lines[target_line_no - 1].split(arg_name + '=')[1].split(',')[:1])
        else:  # implicit
            if arg_name.startswith("IMPLICIT_"):
                pass
            else:
                try:
                    lines[line_no - 1] = lines[line_no - 1].split(short_func_name + '(')[0] + short_func_name + '(' + \
                                        lines[line_no - 1].split(short_func_name + '(')[1].split(')')[0] + \
                                        ',' + arg_name + '=' + target_value + ')' + \
                                        ')'.join(lines[line_no - 1].split(short_func_name + '(')[1].split(')')[1:])
                    if "(," in lines[line_no - 1]:
                        lines[line_no - 1] = lines[line_no - 1].replace("(,", "(")
                    # only one implicit arg, we know we need to replace it
                    if lines[line_no - 1].split(short_func_name + '(')[-1].split(',' + arg_name + '=')[0].count('=') == 0 and \
                            lines[line_no - 1].split(short_func_name + '(')[-1].split(',' + arg_name + '=')[0].count(',') == 0:
                        lines[line_no - 1] = lines[line_no - 1].split(short_func_name + '(')[0] + short_func_name + '(' + \
                                             arg_name + '=' + lines[line_no - 1].split(arg_name + '=')[-1]
                    #if lines[line_no - 1].split(short_func_name + '(')[-1].split(')')[0].count(',') == 1:
                    #    lines[line_no - 1] = lines[line_no - 1].split(short_func_name + '(')[0] + short_func_name + '(' + \
                    #                         lines[line_no - 1].split(short_func_name + '(')[-1].split(')')[0].split(',')[-1]
                except IndexError:
                    pass
    with open(notebook_file, 'w') as fw:
        fw.write(''.join(lines))

def savePatch(strategy, project, notebook):
    output_dir = ASE_PATCHES_DIR + '/' + strategy + '/' + project
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    patch_file = ASE_PATCHES_DIR + '/' + strategy + '/' + project + '/' + notebook + '.patch'
    original_notebook_file = ASE_ORIGINAL_NOTEBOOKS_DIR + '/' + project + '/' + notebook + '.py'
    updated_notebook_file = ASE_FIXED_NOTEBOOKS_DIR + '/' + strategy + '/' + project + '/' + notebook + '.py'
    if not os.path.isfile(patch_file):
        prev_abs_offset = 0
    else:
        prev_abs_offset = countPatchAddedLines(patch_file) - countPatchDeletedLines(patch_file)
    print('Prev Abs Offset: ' + str(prev_abs_offset))
    sub.run('diff -u ' + original_notebook_file + ' ' + updated_notebook_file, shell=True,
            stdout=open(patch_file, 'w'), stderr=sub.STDOUT)
    new_abs_offset = countPatchAddedLines(patch_file) - countPatchDeletedLines(patch_file)
    print('New Abs Offset: ' + str(new_abs_offset))
    relative_offset = new_abs_offset - prev_abs_offset
    return relative_offset

def countPatchAddedLines(patch_file):
    counter = 0
    with open(patch_file, 'r') as fr:
        for l in fr.readlines():
            if l.startswith('+') and not l.startswith('+++'):
                counter += 1
    return counter

def countPatchDeletedLines(patch_file):
    counter = 0
    with open(patch_file, 'r') as fr:
        for l in fr.readlines():
            if l.startswith('-') and not l.startswith('---'):
                counter += 1
    return counter

def isChangingAllParams(from_param_str, to_param_str):
    from_params = from_param_str.split('&')
    to_params = to_param_str.split('&')
    if len(from_params) != len(to_params):
        return False
    for i in range(len(from_params)):
        if from_params[i] == to_params[i]:
            return False
    return True

def getParamValueRange(p, params):
    for param in params:
        if param['name'] == p:
            values = param['values']
            if not isinstance(values, list):
                return []
            break
    valid_values = []
    for v in values:
        if v == 'None':
            continue
        #if v.startswith('<class '):
        #    continue
        if v not in valid_values:
            valid_values.append(v)
    return valid_values
