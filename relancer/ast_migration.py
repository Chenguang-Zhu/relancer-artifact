import re
import collections
from macros import CANNOT_USE_FQN_APIS

from api_extractor import extractAPIUsage

# entrance_4
def runASTMigration(project, notebook, api_mapping, new_file):
    function_call_info_list, attribute_ref_info_list, api_related_var_info_list, all_imported_names_map, all_import_names_line_no_map = extractAPIUsage(new_file)
    if function_call_info_list is None:  # syntax error
        return
    new_top_level_pkgs = []
    for old_api in api_mapping:
        new_api_candidates = api_mapping[old_api]
        if not new_api_candidates:
            continue
        new_top_level_pkgs = migrateOneAPI_AST(old_api, new_api_candidates, project, notebook, new_file,
                                               function_call_info_list,
                                               attribute_ref_info_list,
                                               api_related_var_info_list,
                                               all_imported_names_map,
                                               all_import_names_line_no_map,
                                               new_top_level_pkgs)
    with open(new_file, 'r') as fr:
        lines = fr.readlines()
    for pkg in new_top_level_pkgs:
        if pkg.startswith('[SPECIAL] '):
            lines.insert(0, pkg.split('[SPECIAL] ')[1] + '\n')
        else:
            lines.insert(0, 'import ' + pkg + '\n')
    with open(new_file, 'w') as fw:
        fw.write(''.join(lines))


# helper_4
def migrateOneAPI_AST(old_api, new_api_candidates, project, notebook, new_file,
                      function_call_info_list,
                      attribute_ref_info_list,
                      api_related_var_info_list,
                      all_imported_names_map,
                      all_import_names_line_no_map,
                      new_top_level_pkgs):
    if isinstance(new_api_candidates, collections.OrderedDict):  # auto mapping
        cand = list(new_api_candidates.keys())[0]
    elif isinstance(new_api_candidates, list):  # manual mapping
        cand = new_api_candidates[0]
    new_top_level_pkgs = migrateOneAPIOneCand_AST(old_api, cand, project, notebook, new_file,
                                                  function_call_info_list,
                                                  attribute_ref_info_list,
                                                  api_related_var_info_list,
                                                  all_imported_names_map,
                                                  all_import_names_line_no_map,
                                                  new_top_level_pkgs)
    return new_top_level_pkgs


# helper_4
def migrateOneAPIOneCand_AST(old_api, new_api, project, notebook, new_file,
                             function_call_info_list,
                             attribute_ref_info_list,
                             api_related_var_info_list,
                             all_imported_names_map,
                             all_import_names_line_no_map,
                             new_top_level_pkgs):
    with open(new_file, 'r') as fr:
        lines = fr.readlines()
    edited = False
    # func call
    for call_info in function_call_info_list:
        if call_info['fqn'] == old_api:
            target_line_no = call_info['line_no']
            orig_func_str = call_info['func_str']
            print('CALL', target_line_no, orig_func_str)
            if new_api in CANNOT_USE_FQN_APIS or checkOldAPIFromImportOrImport(old_api, all_import_names_line_no_map, lines) == 'F':
                from_part = '.'.join(new_api.split('.')[:-1])
                import_part = new_api.split('.')[-1]
                lines[target_line_no - 1] = lines[target_line_no - 1].replace(orig_func_str + '(', import_part + '(')
                pkg = '[SPECIAL] from ' + from_part + ' import ' + import_part
                if pkg not in new_top_level_pkgs:
                    new_top_level_pkgs.append(pkg)
            else:
                lines[target_line_no - 1] = lines[target_line_no - 1].replace(orig_func_str + '(', new_api + '(')
                pkg = new_api.split('.')[0]
                if pkg not in new_top_level_pkgs:
                    new_top_level_pkgs.append(pkg)
            edited = True

    # attr ref
    all_import_lines = []
    for im in all_import_names_line_no_map:
        all_import_lines += all_import_names_line_no_map[im]
    for attr_ref_info in attribute_ref_info_list:
        if attr_ref_info['fqn'] == old_api:
            target_line_no = attr_ref_info['line_no']
            orig_attr_ref_str = attr_ref_info['attr_str']
            print('ATTR', target_line_no, orig_attr_ref_str)
            if checkOldAPIFromImportOrImport(old_api, all_import_names_line_no_map, lines) == 'F':
                from_part = '.'.join(new_api.split('.')[:-1])
                import_part = new_api.split('.')[-1]
                if target_line_no in all_import_lines:
                    lines[target_line_no - 1] = lines[target_line_no - 1].replace(orig_attr_ref_str, new_api)
                else:
                    lines[target_line_no - 1] = lines[target_line_no - 1].replace(orig_attr_ref_str + '(', import_part + '(')
                pkg = '[SPECIAL] from ' + from_part + ' import ' + import_part
                if pkg not in new_top_level_pkgs:
                    new_top_level_pkgs.append(pkg)
            else:
                lines[target_line_no - 1] = lines[target_line_no - 1].replace(orig_attr_ref_str, new_api)
                pkg = new_api.split('.')[0]
                if pkg not in new_top_level_pkgs:
                    new_top_level_pkgs.append(pkg)
            edited = True
    # var
    all_import_lines = []
    for im in all_import_names_line_no_map:
        all_import_lines += all_import_names_line_no_map[im]
    for var_info in api_related_var_info_list:
        if var_info['fqn'] == old_api:
            target_line_no = var_info['line_no']
            orig_var_str = var_info['var_str']
            print('VAR', target_line_no, orig_var_str)
            if checkOldAPIFromImportOrImport(old_api, all_import_names_line_no_map, lines) == 'F':
                from_part = '.'.join(new_api.split('.')[:-1])
                import_part = new_api.split('.')[-1]
                if target_line_no in all_import_lines:
                    lines[target_line_no - 1] = lines[target_line_no - 1].replace(old_api, import_part)
                else:
                    lines[target_line_no - 1] = lines[target_line_no - 1].replace(orig_var_str, import_part)
                pkg = '[SPECIAL] from ' + from_part + ' import ' + import_part
                if pkg not in new_top_level_pkgs:
                    new_top_level_pkgs.append(pkg)
            else:
                if target_line_no in all_import_lines:
                    lines[target_line_no - 1] = lines[target_line_no - 1].replace(old_api, new_api)
                else:
                    lines[target_line_no - 1] = lines[target_line_no - 1].replace(orig_var_str, new_api)
                pkg = new_api.split('.')[0]
                if pkg not in new_top_level_pkgs:
                    new_top_level_pkgs.append(pkg)
            edited = True
    lines = removeOldImports(old_api, all_imported_names_map, all_import_names_line_no_map, lines)
    #print(lines)
    if edited:
        print('Migrate: ' + old_api + ' -> ' + new_api)
    with open(new_file, 'w') as fw:
        fw.write(''.join(lines))
    return new_top_level_pkgs


def removeOldImports(old_api, all_imported_names_map, all_import_names_line_no_map, lines):
    edited = False
    for import_name in all_import_names_line_no_map:
        if old_api == import_name:
            for line_no in all_import_names_line_no_map[import_name]:
                target_line_no = line_no
                print(target_line_no, lines[target_line_no - 1].strip())
                if ',' in lines[target_line_no - 1]:
                    lines[target_line_no - 1] = lines[target_line_no - 1].replace(old_api.split('.')[-1], '')
                    lines[target_line_no - 1] = re.sub(r",\s*,", ',', lines[target_line_no - 1])
                    if lines[target_line_no - 1].strip().endswith(','):
                        lines[target_line_no - 1] = lines[target_line_no - 1].strip().strip(',') + '\n'
                    if 'import ,' in lines[target_line_no - 1]:
                        lines[target_line_no - 1] = lines[target_line_no - 1].replace('import ,', 'import ')
                    if 'import  ,' in lines[target_line_no - 1]:
                        lines[target_line_no - 1] = lines[target_line_no - 1].replace('import  ,', 'import ')
                else:
                    lines[target_line_no - 1] = lines[target_line_no - 1].replace(lines[target_line_no - 1].strip(), 'pass')
                edited = True
    if not edited:  # Replace upper level
        if old_api in ['sklearn.learning_curve.learning_curve', 'sklearn.learning_curve.validation_curve'] and \
                'sklearn.learning_curve' in all_import_names_line_no_map:
            for line_no in all_import_names_line_no_map['sklearn.learning_curve']:
                target_line_no = line_no
                print(target_line_no, lines[target_line_no - 1].strip())
                if ',' in lines[target_line_no - 1]:
                    lines[target_line_no - 1] = lines[target_line_no - 1].replace(old_api.split('.')[-1], '')
                    lines[target_line_no - 1] = re.sub(r",\s*,", ',', lines[target_line_no - 1])
                else:
                    lines[target_line_no - 1] = lines[target_line_no - 1].replace(lines[target_line_no - 1].strip(), 'pass')
                edited = True
    return lines


def checkOldAPIFromImportOrImport(old_api, all_import_names_line_no_map, lines):
    old_api_short_name = old_api.split('.')[-1]
    if old_api_short_name == 'joblib':
        return 'I'
    for import_name in all_import_names_line_no_map:
        if not old_api == import_name:
            continue
        for line_no in all_import_names_line_no_map[import_name]:
            target_line_no = line_no
            if 'import ' in lines[target_line_no - 1] and old_api_short_name in lines[target_line_no - 1].split('import ')[-1]:
                if 'from ' in lines[target_line_no - 1].split('import ')[0]:
                    return 'F'
    return 'I'


