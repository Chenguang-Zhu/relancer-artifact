import collections
import json
import os

from macros import SCRIPT_DIR

from api_extractor import extractAPIUsage

ALL_KNOWN_DEPRECATED_APIS_LIST_FILE = SCRIPT_DIR + '/../systematic-study/UNDER-APPROXIMATION-all-known-deprecation-apis.txt'
ALL_ERROR_CASES_LIST_FILE = SCRIPT_DIR + '/../systematic-study/1800-error-cases.txt'
ORIGINAL_NOTEBOOKS_DIR = SCRIPT_DIR + '/../../jupyter/_converted_notebooks'
DEPRECATION_API_USAGE_JSON_FILE = SCRIPT_DIR + '/../systematic-study/UNDER-APPROXIMATION-all-known-deprecation-apis-usage.json'

def countDeprecatedAPIUsagesInAllErrorCases():
    with open(ALL_KNOWN_DEPRECATED_APIS_LIST_FILE, 'r') as fr:
        lines = fr.readlines()
    if os.path.isfile(DEPRECATION_API_USAGE_JSON_FILE):
        with open(DEPRECATION_API_USAGE_JSON_FILE, 'r') as fr:
            known_apis_usages = json.load(fr, object_pairs_hook=collections.OrderedDict)
    else:
        known_apis_usages = collections.OrderedDict({})
    for i, l in enumerate(lines):
        api = l.strip()
        if api not in known_apis_usages:
            known_apis_usages[api] = []
    with open(ALL_ERROR_CASES_LIST_FILE, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        case = l.strip()
        print('--- Processing: ' + case)
        project = case.split('/')[0]
        notebook = case.split('/')[1]
        notebook_file = ORIGINAL_NOTEBOOKS_DIR + '/' + project + '/' + notebook + '.py'
        function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = \
            extractAPIUsage(notebook_file)
        if function_call_info_list is None and attribute_ref_info_list is None and api_related_var_info_list is None:  # synatx error
            for api in known_apis_usages:
                if searchByText(notebook_file, api):
                    if case not in known_apis_usages[api]:
                        known_apis_usages[api].append(case)
            continue
        for func_call in function_call_info_list:
            if func_call['fqn'] in known_apis_usages:
                if case not in known_apis_usages[func_call['fqn']]:
                    known_apis_usages[func_call['fqn']].append(case)
        for attr_ref in attribute_ref_info_list:
            if attr_ref['fqn'] in known_apis_usages:
                if case not in known_apis_usages[attr_ref['fqn']]:
                    known_apis_usages[attr_ref['fqn']].append(case)
        for var_ref in api_related_var_info_list:
            if var_ref['fqn'] in known_apis_usages:
                if case not in known_apis_usages[var_ref['fqn']]:
                    known_apis_usages[var_ref['fqn']].append(case)
    with open(DEPRECATION_API_USAGE_JSON_FILE, 'w') as fw:
        json.dump(known_apis_usages, fw, indent=2)

def searchByText(notebook_file, api):
    with open(notebook_file, 'r') as fr:
        python_lines = fr.readlines()
        for i, l in enumerate(python_lines):
            if not l.strip().startswith('#'):
                #if '.' + api.split('.')[-1] in l:
                #    return True
                if ' import ' + api.split('.')[-1] in l:
                    return True
                if api.split('.')[-1] + '(' in l:
                    return True
    return False
