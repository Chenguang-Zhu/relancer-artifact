import libcst as cst
import collections
from typing import List, Tuple, Dict, Optional

class FunctionCallsCollector(cst.CSTVisitor):

    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self):
        self.function_calls: List = []
        self.function_short_names: List = []
        self.function_call_info_list = []
        self.direct_imported_function_names = []

    def visit_Call(self, node: "Call") -> Optional[bool]:
        if isinstance(node.func, cst.Attribute):  # c.m(...)
            function_call = self.getCallFullyQualifiedName(node.func)
            line_no = self.get_metadata(cst.metadata.PositionProvider, node).start.line
            self.function_calls.append(function_call)
            self.function_short_names.append(function_call.split('.')[-1])
            function_call_info = collections.OrderedDict({})
            function_call_info['func_str'] = function_call
            function_call_info['line_no'] = line_no
            args = collections.OrderedDict({})
            for i, a in enumerate(node.args):
                if a.keyword is None:
                    arg_name = "IMPLICIT_" + str(i)
                else:
                    arg_name = a.keyword.value
                if isinstance(a.value, cst.SimpleString) or isinstance(a.value, cst.Integer) or \
                        isinstance(a.value, cst.Float):
                    arg_value = a.value.value
                else:
                    arg_value = str(type(a.value))
                args[arg_name] = arg_value
            function_call_info['args'] = args
            self.function_call_info_list.append(function_call_info)
        elif isinstance(node.func, cst.Name):   # m(...)
            function_name = node.func.value
            function_call = function_name
            line_no = self.get_metadata(cst.metadata.PositionProvider, node).start.line
            self.function_calls.append(function_call)
            self.function_short_names.append(function_name)
            function_call_info = collections.OrderedDict({})
            function_call_info['func_str'] = function_call
            function_call_info['line_no'] = line_no
            args = collections.OrderedDict({})
            for i, a in enumerate(node.args):
                if a.keyword is None:
                    arg_name = "IMPLICIT_" + str(i)
                else:
                    arg_name = a.keyword.value
                if isinstance(a.value, cst.SimpleString) or isinstance(a.value, cst.Integer) or \
                    isinstance(a.value, cst.Float):
                    arg_value = a.value.value
                else:
                    arg_value = str(type(a.value))
                args[arg_name] = arg_value
            function_call_info['args'] = args
            self.function_call_info_list.append(function_call_info)
            self.direct_imported_function_names.append(function_name)

    def getCallFullyQualifiedName(self, func_node):
        if isinstance(func_node, cst.Name):  # recursive call may trigger this condition
            return func_node.value
        try:
            func_node.value
        except AttributeError:
            print()
            print('UNKNOWN FUNC TYPE!')
            print(func_node)
            return "XXX"
        if isinstance(func_node.value, cst.Name):
            module_name = func_node.value.value
            function_name = func_node.attr.value
            function_call = module_name + '.' + function_name
            return function_call
        elif isinstance(func_node.value, cst.Subscript):
            module_name = "UNKNOWN_TYPE"
            try:
                function_name = func_node.attr.value
            except AttributeError:
                print()
                print('UNKNOWN FUNC TYPE!')
                print(func_node)
                return "XXX"
            function_call = module_name + '.' + function_name
            return function_call
        elif isinstance(func_node.value, cst.Call):
            function_name = func_node.attr.value
            return self.getCallFullyQualifiedName(func_node.value.func) + '.' + function_name
        elif isinstance(func_node.value, cst.Attribute):
            function_name = func_node.attr.value
            if isinstance(func_node.value.value, cst.Name):
                return func_node.value.value.value + '.' + func_node.value.attr.value + '.' + function_name
            elif isinstance(func_node.value.value, cst.Call):
                #print(func_node.value)
                attr_name = func_node.value.attr.value
                return self.getCallFullyQualifiedName(func_node.value.value.func) + \
                       '.' + attr_name + '.' + function_name
            elif isinstance(func_node.value.value, cst.Subscript):
                attr_name = func_node.value.attr.value
                #print(func_node.value)
                return "UNKNOWN_TYPE" + '.' + attr_name + '.' + function_name
            elif isinstance(func_node.value.value, cst.Attribute):
                #print(func_node)
                return self.getAttributeFullyQualifiedName(func_node.value) + '.' + function_name
            else:
                print('UNKNOWN VALUE TYPE!')
                print(func_node.value.value)
                return "XXX"
        elif isinstance(func_node.value, cst.SimpleString):  # e.g., "".join()
            return "str." + func_node.attr.value
        elif isinstance(func_node.value, cst.ConcatenatedString):  # e.g., ("" + "").join()
            return "str." + func_node.attr.value
        elif isinstance(func_node.value, cst.FormattedString):
            return "str." + func_node.attr.value
        elif isinstance(func_node.value, cst.Comparison):  # e.g., (z > 3).xxx()
            return "bool." + func_node.attr.value
        elif isinstance(func_node.value, cst.BinaryOperation):  # e.g., (a - b).min()
            return "bin_op." + func_node.attr.value
        elif isinstance(func_node.value, cst.BooleanOperation):
            return "bool_op." + func_node.attr.value
        elif isinstance(func_node.value, cst.UnaryOperation):
            return "una_op." + func_node.attr.value
        elif isinstance(func_node.value, cst.ListComp):
            return "list_comp." + func_node.attr.value
        elif isinstance(func_node.value, cst.DictComp):
            return "dict_comp." + func_node.attr.value
        elif isinstance(func_node.value, cst.Dict):
            return "dict." + func_node.attr.value
        elif isinstance(func_node.value, cst.List):
            return "list." + func_node.attr.value
        elif isinstance(func_node.value, cst.Tuple):
            return "tuple." + func_node.attr.value
        elif isinstance(func_node.value, cst.Set):
            return "set." + func_node.attr.value
        elif isinstance(func_node.value, cst.Lambda):
            return "lambda." + func_node.attr.value
        elif isinstance(func_node.value, cst.Float):
            return "float." + func_node.attr.value
        elif isinstance(func_node.value, cst.Await):
            return "await." + func_node.attr.value
        elif isinstance(func_node.value, cst.GeneratorExp):
            return "generatorexp." + func_node.attr.value
        elif isinstance(func_node.value, cst.Call):
            return "call." + func_node.attr.value
        elif isinstance(func_node.value, cst.IfExp):
            return "ifexp." + func_node.attr.value
        elif isinstance(func_node.value, cst.Subscript):
            return "subscript." + func_node.attr.value
        elif isinstance(func_node.value, cst.SetComp):
            return "setcomp." + func_node.attr.value
        else:
            print()
            print('UNKNOWN FUNC TYPE!')
            print(func_node)
            return "XXX"

    def getAttributeFullyQualifiedName(self, func_node):
        function_name = func_node.attr.value
        if isinstance(func_node.value.value, cst.Name):
            attr_name = func_node.value.attr.value
            return func_node.value.value.value + '.' + attr_name + '.' + function_name
        elif isinstance(func_node.value.value, cst.Call):
            attr_name = func_node.value.attr.value
            return self.getCallFullyQualifiedName(func_node.value.value.func) + \
                   '.' + attr_name + '.' + function_name
        elif isinstance(func_node.value.value, cst.Subscript):
            attr_name = func_node.value.attr.value
            return "UNKNOWN_TYPE" + '.' + attr_name + '.' + function_name
        elif isinstance(func_node.value.value, cst.Attribute):
            attr_name = func_node.value.attr.value
            return self.getAttributeFullyQualifiedName(func_node.value) + '.' + attr_name + '.' + function_name
        else:
            print('UNKNOWN VALUE TYPE!')
            print(func_node.value.value)
            return "XXX"
