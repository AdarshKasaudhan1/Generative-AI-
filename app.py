import os
import sys

# Set working directory to "DSA class test" subfolder
base_dir = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.path.join(base_dir, "DSA class test")

if os.path.exists(sub_dir):
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    os.chdir(sub_dir)
    with open(os.path.join(sub_dir, "app.py"), "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, os.path.join(sub_dir, "app.py"), "exec"), globals())
else:
    raise FileNotFoundError(f"Could not locate 'DSA class test' directory at {sub_dir}")
