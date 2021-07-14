import libcst as cst
import collections
from typing import List, Tuple, Dict, Optional

class NamesCollector(cst.CSTVisitor):

    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, all_imported_names_to_alias_map, direct_imported_function_names):
        self.api_related_vars: List = []
        self.all_imported_names_to_alias_map = all_imported_names_to_alias_map
        self.direct_imported_function_names = direct_imported_function_names

    def visit_Name(self, node: "Name") -> Optional[bool]:
        #print(node)
        line_no = self.get_metadata(cst.metadata.PositionProvider, node).start.line
        for im in self.all_imported_names_to_alias_map:
            aliases = self.all_imported_names_to_alias_map[im]
            for alias in aliases:
                if node.value == alias:
                    var_info = collections.OrderedDict({})
                    var_info['var_str'] = node.value
                    var_info['line_no'] = line_no
                    if var_info['var_str'] in self.direct_imported_function_names:  # do not re-count function names
                        continue
                    if var_info not in self.api_related_vars:
                        self.api_related_vars.append(var_info)


