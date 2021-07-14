import collections
import json

from macros import MODULE_DISTRIBUTION_DATASET_FILE
from macros import MODULE_DISTRIBUTION_CONVERTED_NOTEBOOKS_DIR
from macros import JUPYTER_EXP_CONVERTED_NOTEBOOKS_DIR

from api_extractor import extractAPIUsage

from mutation import getSubjects

from macros import ERROR_NOTEBOOKS_IMPORTED_MODULE_DISTRIBUTION_FILE
from macros import ERROR_NOTEBOOKS_USED_MODULE_DISTRIBUTION_FILE
def collectErrorNotebooksModuleDistribution(module_distribution_dataset_file=MODULE_DISTRIBUTION_DATASET_FILE,
                                            jupyter_exp_converted_notebooks_dir=JUPYTER_EXP_CONVERTED_NOTEBOOKS_DIR,
                                            error_notebooks_imported_module_distribution_json_file=ERROR_NOTEBOOKS_IMPORTED_MODULE_DISTRIBUTION_FILE,
                                            error_notebooks_used_module_distribution_json_file=ERROR_NOTEBOOKS_USED_MODULE_DISTRIBUTION_FILE):
    notebooks = []
    imported_modules = collections.OrderedDict({})
    used_modules = collections.OrderedDict({})
    with open(module_distribution_dataset_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        if l.strip() not in notebooks:
            notebooks.append(l.strip())
    for s in notebooks:
        project = s.split('/')[0]
        notebook = s.split('/')[1]
        already_counted_used = []
        already_counted_imported = []
        print('=== Processing Notebook: ' + project + '/' + notebook)
        old_file = jupyter_exp_converted_notebooks_dir + '/' + project + '/' + notebook + '.py'
        function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = extractAPIUsage(
            old_file)
        if function_call_info_list is None:
            continue
        imported_names = list(all_imported_names_map.keys())
        for i in imported_names:
            module = i.split('.')[0]
            if module in already_counted_imported:
                continue
            if module not in imported_modules:
                imported_modules[module] = 1
                already_counted_imported.append(module)
            else:
                imported_modules[module] += 1
                already_counted_imported.append(module)
        for f in function_call_info_list + attribute_ref_info_list + api_related_var_info_list:
            fqn = f['fqn']
            if '.' not in fqn:  # not api, but lib name
                continue
            # print(fqn)
            module = fqn.split('.')[0]
            if module in already_counted_used:
                continue
            if module not in used_modules:
                used_modules[module] = 1
                already_counted_used.append(module)
            else:
                used_modules[module] += 1
                already_counted_used.append(module)
        # sort the map
        keys = list(imported_modules.keys())
        sorted_keys = sorted(keys, key=lambda x: imported_modules[x], reverse=True)
        sorted_map = collections.OrderedDict({})
        for k in sorted_keys:
            sorted_map[k] = imported_modules[k]
        # save the map
        with open(error_notebooks_imported_module_distribution_json_file, 'w') as fw:
            json.dump(sorted_map, fw, indent=2)
        # sort the map
        keys = list(used_modules.keys())
        sorted_keys = sorted(keys, key=lambda x: used_modules[x], reverse=True)
        sorted_map = collections.OrderedDict({})
        for k in sorted_keys:
            sorted_map[k] = used_modules[k]
        # save the map
        with open(error_notebooks_used_module_distribution_json_file, 'w') as fw:
            json.dump(sorted_map, fw, indent=2)

from macros import PASSING_NOTEBOOKS_IMPORTED_MODULE_DISTRIBUTION_FILE
from macros import PASSING_NOTEBOOKS_USED_MODULE_DISTRIBUTION_FILE
def collectPassNotebooksModuleDistribution(jupyter_exp_converted_notebooks_dir=JUPYTER_EXP_CONVERTED_NOTEBOOKS_DIR,
                                               passing_notebooks_imported_module_distribution_json_file=PASSING_NOTEBOOKS_IMPORTED_MODULE_DISTRIBUTION_FILE,
                                               passing_notebooks_used_module_distribution_json_file=PASSING_NOTEBOOKS_USED_MODULE_DISTRIBUTION_FILE):
    subjects = getSubjects()
    imported_modules = collections.OrderedDict({})
    used_modules = collections.OrderedDict({})
    for s in subjects:
        project = s.split('/')[0]
        notebook = s.split('/')[1]
        already_counted_used = []
        already_counted_imported = []
        print('=== Processing Notebook: ' + project + '/' + notebook)
        old_file = jupyter_exp_converted_notebooks_dir + '/' + project + '/' + notebook + '.py'
        function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = extractAPIUsage(
            old_file)
        if function_call_info_list is None:
            continue
        imported_names = list(all_imported_names_map.keys())
        for i in imported_names:
            module = i.split('.')[0]
            if module in already_counted_imported:
                continue
            if module not in imported_modules:
                imported_modules[module] = 1
                already_counted_imported.append(module)
            else:
                imported_modules[module] += 1
                already_counted_imported.append(module)
        for f in function_call_info_list + attribute_ref_info_list + api_related_var_info_list:
            fqn = f['fqn']
            if '.' not in fqn:  # not api, but lib name
                continue
            # print(fqn)
            module = fqn.split('.')[0]
            if module in already_counted_used:
                continue
            if module not in used_modules:
                used_modules[module] = 1
                already_counted_used.append(module)
            else:
                used_modules[module] += 1
                already_counted_used.append(module)
        # sort the map
        keys = list(imported_modules.keys())
        sorted_keys = sorted(keys, key=lambda x: imported_modules[x], reverse=True)
        sorted_map = collections.OrderedDict({})
        for k in sorted_keys:
            sorted_map[k] = imported_modules[k]
        # save the map
        with open(passing_notebooks_imported_module_distribution_json_file, 'w') as fw:
            json.dump(sorted_map, fw, indent=2)
        # sort the map
        keys = list(used_modules.keys())
        sorted_keys = sorted(keys, key=lambda x: used_modules[x], reverse=True)
        sorted_map = collections.OrderedDict({})
        for k in sorted_keys:
            sorted_map[k] = used_modules[k]
        # save the map
        with open(passing_notebooks_used_module_distribution_json_file, 'w') as fw:
            json.dump(sorted_map, fw, indent=2)
