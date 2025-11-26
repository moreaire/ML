# Local Scripts and Experiments

This directory is for **quick experiments and personal utilities only**. For real projects, use `~/workspace/git/`.

## Purpose

This folder is your scratchpad for:
- ✅ Quick test scripts
- ✅ One-off analyses
- ✅ Personal utilities
- ✅ Trying out new ideas
- ✅ Temporary code

**NOT for:**
- ❌ Full projects (use git/)
- ❌ Shared code (use git/)
- ❌ Long-term work (use git/)

## Philosophy

**If it's worth keeping, it belongs in git/**

When a script or idea grows beyond a quick experiment:
1. Create a new project in `git/`
2. Move the code there
3. Add proper structure (README, .gitignore, etc.)
4. Commit to version control

## Example Usage

```bash
cd ~/workspace/local

# Quick DICOM test
python test_dicom_read.py

# Try out a new library
pip install some-new-package
python experiment.py

# One-off data conversion
python convert_old_format.py
```

## Organization Tips

Keep it loose but somewhat organized:

```
local/
├── quick-tests/
│   ├── test_pylinac.py
│   └── dicom_header_check.py
├── utilities/
│   ├── file_renamer.py
│   └── data_converter.py
└── experiments/
    ├── try_new_algorithm.py
    └── plot_test.py
```

## Cleanup Regularly

Review this folder monthly:
- Delete obsolete experiments
- Move valuable code to git projects
- Archive old scripts you might need later

```bash
# Archive old experiments
mkdir ~/workspace/local/archive-2025
mv old_* ~/workspace/local/archive-2025/
```

## When to Promote to git/

Promote local scripts to git/ when:
1. **Reusability** - You'll use it again
2. **Sharing** - Others might need it
3. **Complexity** - It's growing beyond a single file
4. **Value** - It solves a real problem

### Promotion Process

```bash
# 1. Create project in git/
cd ~/workspace/git
mkdir useful-tool
cd useful-tool

# 2. Move script
cp ~/workspace/local/my_useful_script.py ./

# 3. Add structure
cp ../.gitignore .
touch README.md requirements.txt

# 4. Initialize git
git init
git add .
git commit -m "Initial commit"
git remote add origin https://git.kronisto.net/user/useful-tool.git
git push -u origin main

# 5. Clean up local
rm ~/workspace/local/my_useful_script.py
```

## Examples

### Quick DICOM Test
```python
# ~/workspace/local/dicom_test.py
import pydicom

ds = pydicom.dcmread('test.dcm')
print(ds.PatientName)
print(ds.StudyDate)
```

### Simple Utility
```python
# ~/workspace/local/rename_files.py
import os

for file in os.listdir('.'):
    if file.endswith('.old'):
        new_name = file.replace('.old', '.bak')
        os.rename(file, new_name)
```

### Experiment with New Library
```python
# ~/workspace/local/try_streamlit.py
import streamlit as st

st.title('Quick Test')
st.write('Testing streamlit!')
value = st.slider('Pick a value', 0, 100)
st.write(f'You picked: {value}')
```

## Tips

1. **Don't overcomplicate** - No need for git, tests, or documentation here
2. **Keep filenames descriptive** - `test_pylinac_starshot.py` not `test1.py`
3. **Comment your intent** - Future you will thank you
4. **Clean up regularly** - Don't let it become a junk drawer
5. **Promote good code** - Move valuable scripts to git/

## Difference from git/

| Aspect | local/ | git/ |
|--------|--------|------|
| **Purpose** | Quick experiments | Real projects |
| **Version Control** | No | Yes (required) |
| **Structure** | Loose | Organized |
| **Sharing** | Personal only | Team accessible |
| **Documentation** | Minimal/none | README required |
| **Lifespan** | Temporary | Long-term |
| **Data** | Small test files | Proper data/ folders |

## Remember

**local/ is your playground. git/ is your workshop.**

Experiment freely here, but when something works, give it a proper home in git/!
