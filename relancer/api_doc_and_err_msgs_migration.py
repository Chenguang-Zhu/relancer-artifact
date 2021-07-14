import os
import json
import re
import collections
import time
import subprocess as sub

from typing import List

from macros import CONVERTED_NOTEBOOKS_DIR
from macros import FIXED_NOTEBOOKS_DIR
from macros import RESULTS_DIR
from macros import OUR_FIXED_NOTEBOOKS_DIR
from macros import OUR_PATCHES_DIR
from macros import OUR_RESULTS_DIR
from macros import ERROR_MSG_JSON_FILE
from macros import API_DOC_KNOWLEDGE_JSON_FILE

from api_extractor import extractAPIUsage

# entrance_3: fix all errors in a notebook, for each error, fix -> validate
def runAPIDocAndErrMsgsMigrationPipelineIteration(project, notebook, api_mapping, old_file, new_file,
                                                  converted_notebooks_dir=CONVERTED_NOTEBOOKS_DIR,
                                                  our_fixed_notebooks_dir=OUR_FIXED_NOTEBOOKS_DIR,
                                                  our_patches_dir=OUR_PATCHES_DIR,
                                                  our_results_dir=OUR_RESULTS_DIR,
                                                  api_doc_knowledge_json_file=API_DOC_KNOWLEDGE_JSON_FILE,
                                                  fixed_notebooks_dir=FIXED_NOTEBOOKS_DIR,
                                                  results_dir=RESULTS_DIR):
    start_time = time.time()
    file_to_fix = fixed_notebooks_dir + '/' + project + '/' + notebook + '.py'
    old_err_msg = analyzeErrorMsgFromLogFile(results_dir + '/' + project + '/' + notebook + '.new.log', project, notebook,
                                             new_file, api_doc_knowledge_json_file)
    all_seen_err_msgs = [old_err_msg]
    runFix(old_err_msg, project, notebook, api_mapping, old_file, file_to_fix, all_seen_err_msgs)

    end_time = time.time()
    duration = end_time - start_time
    time_log_file = our_results_dir + '/' + project + '/' + notebook + '.time'
    with open(time_log_file, 'w') as fw:
        fw.write('[REPAIR EXEC TIME]: ' + str(duration))

def runFix(old_err_msg, project, notebook, api_mapping, old_file, file_to_fix, all_seen_err_msgs,
           our_results_dir=OUR_RESULTS_DIR, our_fixed_notebooks_dir=OUR_FIXED_NOTEBOOKS_DIR,
           api_doc_knowledge_json_file=API_DOC_KNOWLEDGE_JSON_FILE):
    runAPIDocAndErrMsgsMigrationPipeline(project, notebook, api_mapping, old_err_msg, old_file, file_to_fix)
    # get the list of new log files after applying current fix
    all_new_log_files = findAllNewLogFilesAfterApplyingFix(file_to_fix, project, notebook, our_results_dir)
    for log_file in all_new_log_files:
        print("Log file: " + log_file)
        err_msg = analyzeErrorMsgFromLogFile(log_file, project, notebook, file_to_fix, api_doc_knowledge_json_file)
        print("Error Msg: " + str(err_msg))
        # successfully fixed
        if err_msg == "FIXED":
            print('===== Already fixed!')
            continue
        # have explore this err msg before
        if err_msg in all_seen_err_msgs:
            print('===== Have seen this error before, will not proceed!')
            continue
        # the old err is fixed, but there is a new err
        if err_msg != old_err_msg:
            if err_msg not in all_seen_err_msgs:
                all_seen_err_msgs.append(err_msg)
            if err_msg['line_no'] != old_err_msg['line_no']:  # another API call has error
                print('===== Need further fix: Line ' + str(err_msg['line_no']))
                file_to_fix = our_fixed_notebooks_dir + '/' + project + '/' + log_file.split('/')[-1].split('.new.log')[0] + '.py'
                runFix(err_msg, project, notebook, api_mapping, old_file, file_to_fix, all_seen_err_msgs)
            else:  # the current API call is still not fully fixed, could be due to a different arg
                print('===== Need further fix: Same line, different arg')
                file_to_fix = our_fixed_notebooks_dir + '/' + project + '/' + log_file.split('/')[-1].split('.new.log')[0] + '.py'
                runFix(err_msg, project, notebook, api_mapping, old_file, file_to_fix, all_seen_err_msgs)
        # the old err is not fixed, we cannot solve it
        else:
            print('===== Cannot fix this API call, will not proceed!')
            continue

def findAllNewLogFilesAfterApplyingFix(file_to_fix, project, notebook, our_results_dir):
    all_new_log_files = []
    logs_dir = our_results_dir + '/' + project
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)
    for f in os.listdir(logs_dir):
        if file_to_fix.split('/')[-1].split('.py')[0] + '-' in f.split('.new.log')[0]:
            if logs_dir + '/' + f not in all_new_log_files:
                all_new_log_files.append(logs_dir + '/' + f)
    all_new_log_files = sorted(all_new_log_files)
    return all_new_log_files

# entrance_4
def runAPIDocAndErrMsgsMigrationPipeline(project, notebook, api_mapping, err_msg, old_file, new_file,
                                         converted_notebooks_dir=CONVERTED_NOTEBOOKS_DIR,
                                         our_fixed_notebooks_dir=OUR_FIXED_NOTEBOOKS_DIR,
                                         our_patches_dir=OUR_PATCHES_DIR,
                                         our_results_dir=OUR_RESULTS_DIR):
    # generate multiple fixed versions
    arg_fixed_files = runAPIDocAndErrMsgsMigration(project, notebook, new_file, err_msg)
    # validate multiple fixed versions
    for i, arg_fixed_file in enumerate(arg_fixed_files):
        cwd = os.getcwd()
        # save patches
        patch_output_dir = our_patches_dir + '/' + project
        if not os.path.isdir(patch_output_dir):
            os.makedirs(patch_output_dir)
        sub.run('diff -u ' + old_file + ' ' + arg_fixed_file, shell=True,
                stdout=open(patch_output_dir + '/' + arg_fixed_file.split('/')[-1].split('.py')[0] + '.patch', 'w'),
                stderr=sub.STDOUT)
        # run fixed
        os.chdir(our_fixed_notebooks_dir + '/' + project)
        output_dir = our_results_dir + '/' + project
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        fixed_log_file = output_dir + '/' + arg_fixed_file.split('/')[-1].split('.py')[0] + '.new.log'
        sub.run('python ' + arg_fixed_file.split('/')[-1], shell=True,
                stdout=open(fixed_log_file, 'w'), stderr=sub.STDOUT)
        os.chdir(cwd)

def runAPIDocAndErrMsgsMigration(project, notebook, new_file, error_info,
                                 our_fixed_notebooks_dir=OUR_FIXED_NOTEBOOKS_DIR,
                                 api_doc_knowledge_json_file=API_DOC_KNOWLEDGE_JSON_FILE):
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = extractAPIUsage(new_file)
    #error_info = analyzeErrorMsgFromLogFile(project, notebook, err_msg, api_doc_knowledge_json_file)
    api = error_info['API']
    print('--- Error info:')
    print(error_info)
    solutions = computeSolutions(error_info, api_doc_knowledge_json_file)
    print('--- Solutions:')
    print(solutions)
    solutions = exploreCombinations(solutions)
    print('--- Solution Combinations:')
    print(solutions)
    print('--- ' + str(len(solutions)) + ' solutions to try in total')
    output_files = []
    for i, sol in enumerate(solutions):
        output_dir = our_fixed_notebooks_dir + '/' + project
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_file = output_dir + '/' + new_file.split('/')[-1].split('.py')[0] + '-' + str(i) + '.py'
        applySolution(sol, api, project, notebook, new_file, output_file,
                      function_call_info_list,
                      attribute_ref_info_list,
                      api_related_var_info_list,
                      all_imported_names_map,
                      all_import_names_line_no_map)
        output_files.append(output_file)
    return output_files

# --- Deprecated ---
def analyzeErrorMsgFromJSON(project, notebook, results_dir=RESULTS_DIR, error_msg_json_file=ERROR_MSG_JSON_FILE):
    log_file = results_dir + '/' + project + '/' + notebook + '.new.log'
    with open(error_msg_json_file, 'r') as fr:
        error_msgs = json.load(fr)
        for err_msg in error_msgs:
            if err_msg['project'] == project and err_msg['notebook'] == notebook:
                return err_msg
    return None

def analyzeErrorMsgFromLogFile(log_file, project, notebook, python_file, api_doc_knowledge_json_file):
    function_call_info_list, _, _, _, _ = extractAPIUsage(python_file)
    err_msg = collections.OrderedDict({})
    err_msg['project'] = project
    err_msg['notebook'] = notebook
    with open(log_file, 'r') as fr:
        lines = fr.readlines()
    # Check if it is fixed
    error_exist = False
    for i, l in enumerate(lines):
        if 'Error: ' in l:
            error_exist = True
    if not error_exist:
        return "FIXED"
    with open(api_doc_knowledge_json_file, 'r') as fr:
        api_doc_knowledge = json.load(fr)
    for i, l in enumerate(lines):
        if l.strip().startswith('File ') and ', line ' in l and ', in ' in l and notebook in l:
            err_api_short_name = lines[i+1].split('print(')[-1].split('(')[0].split('=')[-1].strip().split('.')[-1]
            if ' ' in err_api_short_name:
                err_api_short_name = err_api_short_name.split()[-1]
            print(err_api_short_name)
            for k in api_doc_knowledge:
                if k['API'].endswith('.' + err_api_short_name):
                    api = k['API']
    err_msg['API'] = api
    # Collect error line no
    for i, l in enumerate(lines):
        # the first error trace in the stack
        if l.strip().startswith('File ') and ', line ' in l and ', in ' in l and notebook in l:
            line_no = int(l.split(', line ')[1].split(',')[0])
            err_msg['line_no'] = line_no
    # Collect error args info
    for i, l in enumerate(lines):
        # error msg line
        if 'Error: ' in l:
            if 'got an unexpected keyword argument ' in l:  # delete/rename
                key_params = []
                param = collections.OrderedDict({})
                param_name = l.strip().split("\'")[1]
                todo_action = ['delete', 'rename']
                param['name'] = param_name
                param['todo_action'] = todo_action
                key_params.append(param)
            else:  # update
                key_params = []
                param_names_list = getParamNamesListFromAPIDoc(api, api_doc_knowledge_json_file)
                for word in l.strip().split(': ')[1].split():
                    word = word.strip('\'').strip('\"').strip(',').strip('.').lower()
                    if word in param_names_list:
                        param = collections.OrderedDict({})
                        param['name'] = word
                        param['todo_action'] = 'update'
                        key_params.append(param)
    err_msg['key_params'] = key_params
    return err_msg

def getParamNamesListFromAPIDoc(api, api_doc_knowledge_json_file):
    param_names_list = []
    with open(api_doc_knowledge_json_file, 'r') as fr:
        api_doc_knowledge = json.load(fr)
    for k in api_doc_knowledge:
        if k['API'] == api:
            for p in k['params']:
                if p['name'] not in param_names_list:
                    param_names_list.append(p['name'].lower())
    return param_names_list

def extractAPIDocKnowledge(api, api_doc_knowledge_json_file):
    with open(api_doc_knowledge_json_file, 'r') as fr:
        api_doc_knowledge_list = json.load(fr)
        for api_doc in api_doc_knowledge_list:
            if api_doc['API'] == api:
                return api_doc
    return None

def computeSolutions(error_info, api_doc_knowledge_json_file):
    api = error_info['API']
    line_no = error_info['line_no']
    key_params = error_info['key_params']
    solutions = []
    for param in key_params:
        name = param['name']
        todo_action = param['todo_action']
        if todo_action == 'update':
            values = getParamValuesScope(api, name, api_doc_knowledge_json_file)
            solutions.append((line_no, name, 'update', values))
        if isinstance(todo_action, List) and todo_action == ['delete', 'rename']:
            param_exist = checkIfParamExist(api, name, api_doc_knowledge_json_file)
            if param_exist:
                continue
            else:
                solutions.append((line_no, name, 'rename'))
                solutions.append((line_no, name, 'delete'))
    return solutions

def exploreCombinations(solutions):
    combinations = []
    for sol in solutions:
        if sol[2] == 'update':
            for v in sol[3]:
                combinations.append((sol[0], sol[1], sol[2], v))
        elif sol[2] == 'delete':
            combinations.append(sol)
        elif sol[2] == 'rename':
            continue
        else:
            continue
    return combinations

def getParamValuesScope(api, name, api_doc_knowledge_json_file):
    api_doc = extractAPIDocKnowledge(api, api_doc_knowledge_json_file)
    for p in api_doc['params']:
        if p['name'] == name:
            values = p['values']
            if isinstance(values, List):
                return values
    return None

def checkIfParamExist(api, name, api_doc_knowledge_json_file):
    api_doc = extractAPIDocKnowledge(api, api_doc_knowledge_json_file)
    for p in api_doc['params']:
        if p['name'] == name:
            return True
    return False

# helper_4
def applySolution(solution, api, project, notebook, new_file, output_file,
                  function_call_info_list,
                  attribute_ref_info_list,
                  api_related_var_info_list,
                  all_imported_names_map,
                  all_import_names_line_no_map):
    with open(new_file, 'r') as fr:
        lines = fr.readlines()
    edited = False
    err_line_no = int(solution[0])
    # func call
    for call_info in function_call_info_list:
        if call_info['fqn'] == api:
            target_line_no = call_info['line_no']
            orig_func_str = call_info['func_str']
            print('CALL', target_line_no, orig_func_str)
            if err_line_no != target_line_no:
                continue
            print(' -- should fix')
            applyEdit(api, solution, lines, target_line_no)
            edited = True
    # attr ref
    for attr_ref_info in attribute_ref_info_list:
        if attr_ref_info['fqn'] == api:
            target_line_no = attr_ref_info['line_no']
            orig_attr_ref_str = attr_ref_info['attr_str']
            print('ATTR', target_line_no, orig_attr_ref_str)
            if err_line_no != target_line_no:
                continue
            print(' -- should fix')
            applyEdit(api, solution, lines, target_line_no)
            edited = True
    # var
    for var_info in api_related_var_info_list:
        if var_info['fqn'] == api:
            target_line_no = var_info['line_no']
            orig_var_str = var_info['var_str']
            print('VAR', target_line_no, orig_var_str)
            if err_line_no != target_line_no:
                continue
            print(' -- should fix')
            applyEdit(api, solution, lines, target_line_no)
            edited = True
    if edited:
        print('Fix arg of ' + api)
    with open(output_file, 'w') as fw:
        fw.write(''.join(lines))


def applyEdit(api, solution, lines, target_line_no):
    short_func_name = api.split('.')[-1]
    if solution[2] == 'update':
        arg_name = solution[1]
        target_value = solution[3]
        if target_value == 'None':
            pass
        elif target_value == 'np.nan':
            pass
        else:
            target_value = '\"' + str(target_value) + '\"'
        if arg_name in lines[target_line_no - 1]:  # explicit
            pattern = arg_name + r'=[^(,|\))]*(,|\))'
            if not re.search(pattern, lines[target_line_no - 1]):
                return
            matched_str = re.search(pattern, lines[target_line_no - 1])[0]
            if matched_str.endswith(','):
                lines[target_line_no - 1] = re.sub(pattern, arg_name + '=' + target_value + ',', lines[target_line_no - 1])
            else:
                lines[target_line_no - 1] = re.sub(pattern, arg_name + '=' + target_value + ')', lines[target_line_no - 1])
            if "(," in lines[target_line_no - 1]:
                lines[target_line_no - 1] = lines[target_line_no - 1].replace("(,", "(")
        else:  # implicit
            if arg_name.startswith("IMPLICIT_"):
                pass
            else:
                try:
                    lines[target_line_no - 1] = lines[target_line_no - 1].split(short_func_name + '(')[0] + short_func_name + '(' + \
                                        lines[target_line_no - 1].split(short_func_name + '(')[1].split(')')[0] + \
                                        ',' + arg_name + '=' + target_value + ')' + \
                                        ')'.join(lines[target_line_no - 1].split(short_func_name + '(')[1].split(')')[1:])
                    if "(," in lines[target_line_no - 1]:
                        lines[target_line_no - 1] = lines[target_line_no - 1].replace("(,", "(")
                except IndexError:
                    pass
    elif solution[2] == 'delete':
        if len(solution) == 4:  # only delete arg of certain value
            value = solution[3]
            if value not in lines[target_line_no - 1]:
                return
        arg_name = solution[1]
        pattern = arg_name + r'[\s]?=[\s]?[^(,|\))]*(,|\))'
        if not re.search(pattern, lines[target_line_no - 1]):
            return
        matched_str = re.search(pattern, lines[target_line_no - 1])[0]
        if matched_str.endswith(','):
            lines[target_line_no - 1] = re.sub(pattern, '', lines[target_line_no - 1], 1)
        else:
            lines[target_line_no - 1] = re.sub(pattern, ')', lines[target_line_no - 1], 1)
        pattern = r',\s*\)'
        lines[target_line_no - 1] = re.sub(pattern, ')', lines[target_line_no - 1])
    elif solution[2] == 'add':
        arg_name = solution[1]
        if arg_name in lines[target_line_no - 1]:  # Do not add if the arg is already there
            return
        target_value = solution[3]
        if target_value == 'None':
            pass
        elif target_value == 'np.nan':
            pass
        else:
            target_value = '\"' + str(target_value) + '\"'
        api_short_name = api.split('.')[-1]
        if '(' not in lines[target_line_no - 1] or ')' not in lines[target_line_no - 1]:  # no (), do not add
            return
        if lines[target_line_no - 1].split(api_short_name)[-1].split('(')[-1].split(')')[0].strip() != '':
            lines[target_line_no - 1] = lines[target_line_no - 1].split(api_short_name)[0] + \
                                        api_short_name + \
                                        lines[target_line_no - 1].split(api_short_name)[-1].split(')')[0] + \
                                        ',' + arg_name + '=' + target_value + ')' + \
                                        ')'.join(lines[target_line_no - 1].split(api_short_name)[-1].split(')')[1:])
        else:
            lines[target_line_no - 1] = lines[target_line_no - 1].split(api_short_name)[0] + \
                                        api_short_name + \
                                        lines[target_line_no - 1].split(api_short_name)[-1].split(')')[0] + \
                                        arg_name + '=' + target_value + ')' + \
                                        ')'.join(lines[target_line_no - 1].split(api_short_name)[-1].split(')')[1:])
        if ',,' in lines[target_line_no - 1]:
            lines[target_line_no - 1] = lines[target_line_no - 1].replace(',,', ',')
    elif solution[2] == 'ren':
        arg_name = solution[1]
        if arg_name not in lines[target_line_no - 1]:  # Do not rename if the arg is not there
            return
        target_name = solution[3]
        if '(' not in lines[target_line_no - 1] or ')' not in lines[target_line_no - 1]:  # no (), do not rename
            return
        pattern = arg_name + "\s*=\s*"
        lines[target_line_no - 1] = re.sub(pattern, target_name + '=', lines[target_line_no - 1])
