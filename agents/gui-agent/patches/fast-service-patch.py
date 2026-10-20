"""Patch fast-service's Python 3.12+ f-string usage for compatibility with
python:3.11 (used in os_world because desktop-env's transitive deps
require <3.12)."""

import re
from pathlib import Path

target = Path("src/fast_service/benchmark/intermediate/stateful.py")
text = target.read_text()
old = r'f"load requests from: \n{' + "'\\n'" + r'.join(function_dict.values())}"'
new = "'load requests from: \\n' + '\\n'.join(function_dict.values())"
if old in text:
    text = text.replace(old, new)
    target.write_text(text)
    print(f"patched {target}")
else:
    print(f"no patch needed (offending line not found in {target})")

# Validate the file parses cleanly.
import ast

ast.parse(text)
print("parse OK")
