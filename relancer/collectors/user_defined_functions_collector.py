import libcst as cst
import collections
from typing import List, Tuple, Dict, Optional

class UserDefinedFunctionsCollector(cst.CSTVisitor):

    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self):
        self.user_defined_funcs: List = []

    def visit_FunctionDef(self, node: "FunctionDef") -> Optional[bool]:
        func_name = node.name.value
        if func_name not in self.user_defined_funcs:
            self.user_defined_funcs.append(func_name)
