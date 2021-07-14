import collections
import libcst as cst

from collectors.function_calls_collector import FunctionCallsCollector
from collectors.attribute_refs_collector import AttributeRefsCollector
from collectors.names_collector import NamesCollector
from collectors.user_defined_functions_collector import UserDefinedFunctionsCollector

def parseProgramToAST(program):
    with open(program, 'r') as fr:
        program_str = fr.read()
    program_ast = cst.parse_module(program_str)
    #print(program_ast)
    return program_ast

def findAllImportNodes(scopes, ranges):
    import_nodes = []
    from_import_nodes = []
    for scope in scopes:
        for ass in scope.assignments:
            if isinstance(ass, cst.metadata.BuiltinAssignment):
                continue
            if isinstance(ass.node, cst.Import):
                # print(ass.node)
                import_nodes.append(ass.node)
            elif isinstance(ass.node, cst.ImportFrom):
                from_import_nodes.append(ass.node)
    import_nodes = sorted(import_nodes, key=lambda x: ranges[x].start.line)
    # print(import_nodes)
    from_import_nodes = sorted(from_import_nodes, key=lambda x: ranges[x].start.line)
    # print(from_import_nodes)
    return import_nodes, from_import_nodes

def collectImportNames(all_import_names_map, all_import_names_line_no_map, import_nodes, scopes, ranges):
    imported_names = []
    for node in import_nodes:
        #print(node)
        line_no = ranges[node].start.line
        # print(line_no)
        names = node.names
        for name in names:
            module_name = ''
            node = name.name
            while isinstance(node, cst.Attribute):
                module_name = node.attr.value + '.' + module_name
                node = node.value
            orig_name = node.value + '.' + module_name
            orig_name = orig_name[:-1]  # remove ending "."
            all_import_names_map[orig_name] = []
            all_import_names_map[orig_name].append(orig_name)
            if orig_name not in all_import_names_line_no_map:
                all_import_names_line_no_map[orig_name] = []
            if line_no not in all_import_names_line_no_map[orig_name]:
                all_import_names_line_no_map[orig_name].append(line_no)
            # e.g., import seaborn as sns
            if name.asname is not None:
                alias_name = name.asname.name.value
                if alias_name not in all_import_names_map[orig_name]:
                    all_import_names_map[orig_name].append(alias_name)
                imported_names.append(alias_name)
    return all_import_names_map, all_import_names_line_no_map, imported_names

def collectFromImportNames(all_import_names_map, all_import_names_line_no_map, from_import_nodes, scopes, ranges):
    from_imported_names = []
    for node in from_import_nodes:
        #print(node)
        line_no = ranges[node].start.line
        #print(line_no)
        module = node.module
        if module is None:  # e.g., from . import abc
            continue
        module_name = ''
        while not isinstance(module, str):
            if isinstance(module, cst.Attribute):
                module_name = module.attr.value + '.' + module_name
            elif isinstance(module, cst.Name):
                module_name = module.value + '.' + module_name
            else:
                print(module)
                print('STRANGE FROM IMPORT!')
                exit(0)
            module = module.value
        names = node.names
        for name in names:
            orig_name = name.name.value
            full_name = module_name + orig_name
            all_import_names_map[full_name] = []
            all_import_names_map[full_name].append(orig_name)
            if full_name not in all_import_names_line_no_map:
                all_import_names_line_no_map[full_name] = []
            if line_no not in all_import_names_line_no_map[full_name]:
                all_import_names_line_no_map[full_name].append(line_no)
            # e.g., import seaborn as sns
            if name.asname is not None:
                alias_name = name.asname.name.value
                if alias_name not in all_import_names_map[full_name]:
                    all_import_names_map[full_name].append(alias_name)
                from_imported_names.append(alias_name)
    return all_import_names_map, all_import_names_line_no_map, from_imported_names

# return: map: fqn -> all names and alias
def collectAllImportNames(scopes, ranges):
    all_import_names_map = collections.OrderedDict({})
    all_import_names_line_no_map = collections.OrderedDict({})
    import_nodes, from_import_nodes = findAllImportNodes(scopes, ranges)
    all_import_names_map, all_import_names_line_no_map, _ = collectImportNames(all_import_names_map, all_import_names_line_no_map, import_nodes, scopes, ranges)
    all_import_names_map, all_import_names_line_no_map, _ = collectFromImportNames(all_import_names_map, all_import_names_line_no_map, from_import_nodes, scopes, ranges)
    return all_import_names_map, all_import_names_line_no_map

def getFullyQualifiedName(name, all_imported_names_map):
    for fqn in all_imported_names_map:
        if name in all_imported_names_map[fqn]:
            return fqn
    return name

def getFullyQualifiedFunctionCalls(all_imported_names_map, all_function_calls):
    fqn_function_calls = []
    for item in all_function_calls:
        function_call = item['func_str']
        fqn_list = []
        pkg_fqn = getFullyQualifiedName(function_call.split('.')[0], all_imported_names_map)
        fqn_function_call = pkg_fqn
        if len(function_call.split('.')) > 1:
            fqn_function_call += '.'
            fqn_function_call += '.'.join(function_call.split('.')[1:])
        item['fqn'] = fqn_function_call
        fqn_function_calls.append(item)
    return fqn_function_calls

def excludeUserDefinedFunctionCalls(all_function_calls, wrapper):
    collector = UserDefinedFunctionsCollector()
    wrapper.visit(collector)
    user_defined_functions = collector.user_defined_funcs
    third_party_function_calls = []
    for item in all_function_calls:
        func = item['func_str']
        if func not in user_defined_functions:
            third_party_function_calls.append(item)
    return third_party_function_calls

def excludeDuplicateFunctionCalls(all_function_calls):
    unique_function_calls = []
    for item in all_function_calls:
        if item not in unique_function_calls:
            unique_function_calls.append(item)
    return unique_function_calls

def formatFunctionCalls(all_imported_names_map, fqn_function_calls):
    formatted_function_calls = []
    for item in fqn_function_calls:
        fqn_function_call = item['fqn']
        if fqn_function_call in all_imported_names_map:
            formatted_function_calls.append(item)
            continue
        formatted_function_call = ''
        short_func_name = fqn_function_call.split('.')[-1]
        module_name_str = '.'.join(fqn_function_call.split('.')[:-1])
        is_in_imported_module = False
        for imported_name in all_imported_names_map:
            if module_name_str.startswith(imported_name):
                is_in_imported_module = True
        if is_in_imported_module:
            formatted_function_call += module_name_str
        else:
            formatted_function_call += 'UNKNOWN'
        formatted_function_call += '.' + short_func_name
        item['fqn'] = formatted_function_call
        formatted_function_calls.append(item)
    return formatted_function_calls

def excludeUnknownFunctionCalls(formatted_function_calls):
    known_function_calls = []
    for item in formatted_function_calls:
        fqn = item['fqn']
        if fqn.startswith('UNKNOWN.'):
            continue
        if 'compat.v1.v1.' in fqn:  # workaround
            item['fqn'] = fqn.replace('.v1.v1', '.v1')
        known_function_calls.append(item)
    return known_function_calls

def getFullyQualifiedAttributeRefs(all_imported_names_map, all_attribute_refs):
    fqn_attribute_refs = []
    for item in all_attribute_refs:
        attr_ref = item['attr_str']
        fqn_list = []
        pkg_fqn = getFullyQualifiedName(attr_ref.split('.')[0], all_imported_names_map)
        fqn_attr_ref = pkg_fqn
        if len(attr_ref.split('.')) > 1:
            fqn_attr_ref += '.'
            fqn_attr_ref += '.'.join(attr_ref.split('.')[1:])
        item['fqn'] = fqn_attr_ref
        fqn_attribute_refs.append(item)
    return fqn_attribute_refs

def formatAttributeRefs(all_imported_names_map, fqn_attribute_refs):
    formatted_attribute_refs = []
    for item in fqn_attribute_refs:
        fqn_attr_ref = item['fqn']
        if fqn_attr_ref in all_imported_names_map:
            formatted_attribute_refs.append(item)
            continue
        formatted_attr_ref = ''
        short_attr_name = fqn_attr_ref.split('.')[-1]
        module_name_str = '.'.join(fqn_attr_ref.split('.')[:-1])
        is_in_imported_module = False
        for imported_name in all_imported_names_map:
            if module_name_str.startswith(imported_name):
                is_in_imported_module = True
            if fqn_attr_ref.startswith(imported_name):
                is_in_imported_module = True
            if imported_name.startswith(fqn_attr_ref):
                is_in_imported_module = True
        if is_in_imported_module:
            formatted_attr_ref += module_name_str
        else:
            formatted_attr_ref += 'UNKNOWN'
        formatted_attr_ref += '.' + short_attr_name
        item['fqn'] = formatted_attr_ref
        formatted_attribute_refs.append(item)
    return formatted_attribute_refs

def excludeUnknownAttributeRefs(formatted_attribute_refs):
    known_attribute_refs = []
    for item in formatted_attribute_refs:
        fqn = item['fqn']
        if fqn.startswith('UNKNOWN.'):
            continue
        known_attribute_refs.append(item)
    return known_attribute_refs

def getFullyQualifiedVars(all_imported_names_map, all_api_related_vars):
    fqn_vars = []
    for item in all_api_related_vars:
        var = item['var_str']
        pkg_fqn = getFullyQualifiedName(var.split('.')[0], all_imported_names_map)
        fqn_var = pkg_fqn
        if len(var.split('.')) > 1:
            fqn_var += '.'
            fqn_var += '.'.join(var.split('.')[1:])
        item['fqn'] = fqn_var
        fqn_vars.append(item)
    return fqn_vars


def extractAPIUsage(program_file):
    try:
        program_ast = parseProgramToAST(program_file)
    except cst._exceptions.ParserSyntaxError:
        print('SLICE HAS SYNTAX ERROR!')
        return None, None, None, None, None
    wrapper = cst.metadata.MetadataWrapper(program_ast)
    function_call_collector = FunctionCallsCollector()
    wrapper.visit(function_call_collector)
    attribute_ref_collector = AttributeRefsCollector(function_call_collector.function_calls)
    wrapper.visit(attribute_ref_collector)
    scopes = set(wrapper.resolve(cst.metadata.ScopeProvider).values())
    ranges = wrapper.resolve(cst.metadata.PositionProvider)
    all_imported_names_map, all_import_names_line_no_map = collectAllImportNames(scopes, ranges)
    #print('ALL IMPORTED NAMES AND ALIAS:')
    #print(all_imported_names_map)
    names_collector = NamesCollector(all_imported_names_map, function_call_collector.direct_imported_function_names)
    wrapper.visit(names_collector)
    #print('ALL IMPORTED NAMES LINE NOS:')
    #print(all_import_names_line_no_map)
    all_function_calls = function_call_collector.function_call_info_list
    all_function_calls = excludeUserDefinedFunctionCalls(all_function_calls, wrapper)
    #print('ALL FUNCTION CALLS:')
    #print(all_function_calls)
    fqn_function_calls = getFullyQualifiedFunctionCalls(all_imported_names_map, all_function_calls)
    #print('FQN FUNCTION CALLS:')
    #print(fqn_function_calls)
    formatted_function_calls = formatFunctionCalls(all_imported_names_map, fqn_function_calls)
    #print('FORMATTED FUNCTION CALLS:')
    #print(formatted_function_calls)
    api_calls = excludeUnknownFunctionCalls(formatted_function_calls)
    #print('API CALLS:')
    #print(api_calls)

    all_attribute_refs = attribute_ref_collector.attribute_refs_info_list
    #print('ALL ATTRIBUTE REFS:')
    #print(all_attribute_refs)
    fqn_attribute_refs = getFullyQualifiedAttributeRefs(all_imported_names_map, all_attribute_refs)
    #print('FQN ATTRIBUTE REFS:')
    #print(fqn_attribute_refs)
    formatted_attribute_refs = formatAttributeRefs(all_imported_names_map, fqn_attribute_refs)
    #print('FORMATTED ATTRIBUTE REFS:')
    #print(formatted_attribute_refs)
    api_refs = excludeUnknownAttributeRefs(formatted_attribute_refs)
    #print('API ATTRIBUTE REFS:')
    #print(api_refs)

    api_related_vars = names_collector.api_related_vars
    #print('API RELATED VARS:')
    #print(api_related_vars)
    fqn_api_related_vars = getFullyQualifiedVars(all_imported_names_map, api_related_vars)
    #print('FQN VARS:')
    #print(fqn_api_related_vars)

    return api_calls, api_refs, api_related_vars, all_imported_names_map, all_import_names_line_no_map

