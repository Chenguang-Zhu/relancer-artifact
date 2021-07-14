import libcst as cst
import collections
from typing import List, Tuple, Dict, Optional

class AttributeRefsCollector(cst.CSTVisitor):

    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, function_calls):
        self.attribute_refs: List = []
        self.attribute_refs_info_list = []
        self.function_calls = function_calls

    def visit_Attribute(self, node: "Attribute") -> Optional[bool]:
        line_no = self.get_metadata(cst.metadata.PositionProvider, node).start.line
        attr_ref_info = collections.OrderedDict({})
        attr_ref_info['attr_str'] = self.getAttrRefFullyQualifiedName(node)
        attr_ref_info['line_no'] = line_no
        # do not count function call attrs
        if attr_ref_info['attr_str'] in self.function_calls:
            return
        self.attribute_refs.append(attr_ref_info['attr_str'])
        self.attribute_refs_info_list.append(attr_ref_info)

    def getAttrRefFullyQualifiedName(self, attr_node):
        if isinstance(attr_node, str):  # NOT SURE WHY THIS HAPPENS
            return "str"
        if isinstance(attr_node.value, cst.Name):
            return attr_node.value.value + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Call):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Subscript):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.SimpleString):
            return 'str' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.BinaryOperation):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.UnaryOperation):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.BooleanOperation):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Comparison):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.ConcatenatedString):
            return 'str' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.FormattedString):
            return 'str' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.ListComp):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.DictComp):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Set):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.List):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Tuple):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Dict):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Lambda):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Float):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Await):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.GeneratorExp):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Call):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.IfExp):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.Subscript):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        elif isinstance(attr_node.value, cst.SetComp):
            return 'UNKNOWN_TYPE' + '.' + attr_node.attr.value
        else:
            return self.getAttrRefFullyQualifiedName(attr_node.value) + '.' + attr_node.attr.value

