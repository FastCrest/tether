"""Patch FluxVLA's setup.py to remove CUDA extension builds."""
from pathlib import Path

setup_py = Path("/opt/FluxVLA/setup.py")
lines = setup_py.read_text().splitlines()

out = []
skip = False
for line in lines:
    if "ext_modules=[" in line:
        out.append("    ext_modules=[],")
        skip = True
        continue
    if skip:
        if line.strip() == "],":
            skip = False
        continue
    if "cmdclass=" in line and "BuildExtension" in line:
        out.append("    cmdclass={},")
        continue
    out.append(line)

setup_py.write_text("\n".join(out) + "\n")
print("Patched setup.py: removed ext_modules + cmdclass")
