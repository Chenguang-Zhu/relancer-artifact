import libcst as cst
from typing import List, Tuple, Dict, Optional

class UserDefinedClassesCollector(cst.CSTVisitor):

    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self):
        self.user_defined_classes: List = []

    def visit_ClassDef(self, node: "ClassDef") -> Optional[bool]:
        class_name = node.name.value
        if class_name not in self.user_defined_classes:
            self.user_defined_classes.append(class_name)
