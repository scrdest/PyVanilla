import os
import sys
import glob
import pytest


parent_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(parent_dir)
if project_dir not in sys.path: sys.path.append(project_dir)

testdirs = [
    p for p
    in glob.iglob('**/*.py', recursive=True)
    if not any(
        (sp.startswith('_') for sp in os.path.split(p))
    )
]

pytest.main(testdirs)
