# Git Repositories - Project-Centric Development

This directory is for cloning and creating git repositories. **Each project should be self-contained** with its own data, notebooks, and code.

## 🎯 Key Concept: One Folder Per Project

**Open individual project folders in VSCode**, not the entire workspace. This keeps your environment clean and focused, even with 20+ projects.

```bash
# Clone a project
cd ~/workspace/git
git clone https://git.kronisto.net/user/dose-analysis.git

# Open ONLY that project in VSCode
# File → Open Folder → ~/workspace/git/dose-analysis
```

## 📁 Recommended Project Structures

### Structure 1: Analysis Project

Perfect for data analysis, research studies, or QA reports:

```
dose-analysis/
├── README.md                # Project overview and documentation
├── .gitignore              # Copy from ~/workspace/git/.gitignore
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment (optional)
├── data/                    # Data files (gitignored)
│   ├── raw/                # Original data
│   ├── processed/          # Cleaned data
│   └── results/            # Analysis outputs
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_analysis.ipynb
│   └── 03_visualization.ipynb
├── src/                     # Python modules
│   ├── __init__.py
│   ├── data_processing.py
│   └── analysis.py
└── figures/                 # Generated plots (optional)
```

**Usage:**
```bash
cd ~/workspace/git/dose-analysis
conda create -n dose-analysis python=3.12
conda activate dose-analysis
pip install -r requirements.txt
jupyter lab
```

### Structure 2: Application/Tool Project

For developing tools, scripts, or applications:

```
qa-automation/
├── README.md
├── .gitignore
├── requirements.txt
├── setup.py                 # For installable packages
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_qa_checks.py
│   └── test_integration.py
├── src/qa_automation/       # Main package
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── qa_checks.py
│   └── reports.py
├── data/                    # Test data only
│   └── test_samples/
├── docs/                    # Documentation
│   ├── usage.md
│   └── api.md
└── examples/                # Example usage scripts
    └── run_qa_check.py
```

**Usage:**
```bash
cd ~/workspace/git/qa-automation
pip install -e .              # Editable install
pytest tests/                 # Run tests
python examples/run_qa_check.py
```

### Structure 3: Research Paper/Study

For academic work with code, data, and manuscript:

```
linac-calibration-study/
├── README.md
├── .gitignore
├── requirements.txt
├── data/                    # Experimental data (gitignored)
│   ├── raw/
│   ├── processed/
│   └── metadata.csv
├── analysis/                # Analysis scripts
│   ├── 01_data_cleaning.py
│   ├── 02_statistical_analysis.py
│   └── 03_generate_figures.py
├── notebooks/               # Exploratory analysis
│   └── exploration.ipynb
├── figures/                 # Publication figures
│   ├── figure_1.png
│   └── figure_2.png
├── manuscript/              # Paper drafts (LaTeX or Word)
│   ├── paper.tex
│   └── references.bib
└── results/                 # Statistical outputs
    └── summary_stats.csv
```

### Structure 4: Machine Learning Project

For ML/AI projects:

```
ct-segmentation-ml/
├── README.md
├── .gitignore
├── requirements.txt
├── data/                    # Datasets (gitignored)
│   ├── raw/
│   ├── train/
│   ├── val/
│   └── test/
├── notebooks/               # Experimentation
│   ├── eda.ipynb           # Exploratory data analysis
│   └── model_experiments.ipynb
├── src/                     # Source code
│   ├── data/               # Data loading/preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/             # Model definitions
│   │   ├── __init__.py
│   │   └── unet.py
│   ├── training/           # Training scripts
│   │   ├── __init__.py
│   │   └── train.py
│   └── utils/              # Helper functions
│       └── metrics.py
├── configs/                 # Configuration files
│   └── training_config.yaml
├── models/                  # Saved models (gitignored)
│   └── best_model.pth
└── experiments/             # Experiment tracking
    └── experiment_log.md
```

## 🔧 Setting Up a New Project

### Option 1: Clone Existing Repository

```bash
cd ~/workspace/git
git clone https://git.kronisto.net/username/project-name.git
cd project-name

# Set up environment
conda create -n project-name python=3.12
conda activate project-name
pip install -r requirements.txt
```

### Option 2: Create New Repository

```bash
# Create directory
cd ~/workspace/git
mkdir my-new-project
cd my-new-project

# Initialize git
git init

# Copy .gitignore template
cp ../.gitignore .

# Create basic structure
mkdir -p data/{raw,processed} notebooks src
touch README.md requirements.txt

# Create initial commit
git add .
git commit -m "Initial project structure"

# Push to Gitea
git remote add origin https://git.kronisto.net/username/my-new-project.git
git push -u origin main
```

## 📝 Essential Files for Every Project

### 1. README.md

```markdown
# Project Name

## Overview
Brief description of the project

## Setup
\`\`\`bash
conda create -n project-name python=3.12
conda activate project-name
pip install -r requirements.txt
\`\`\`

## Usage
How to run the project

## Data
Description of data (sources, format, location)

## Results
Summary of findings or outputs
```

### 2. .gitignore

Copy the template from `~/workspace/git/.gitignore`:

```bash
cp ~/workspace/git/.gitignore ./
```

**Always customize for your project!** Add project-specific patterns.

### 3. requirements.txt

```bash
# Generate from current environment
pip freeze > requirements.txt

# Or create manually
pydicom>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
```

### 4. environment.yml (Optional, for Conda)

```yaml
name: project-name
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - numpy
  - pandas
  - matplotlib
  - pip:
    - pydicom
    - pylinac
```

## 🔒 Security Best Practices

### 1. Use .gitignore Properly

```gitignore
# Data directories
data/
Data/
*/data/

# DICOM files
*.dcm
*.DCM

# Jupyter checkpoints
.ipynb_checkpoints/

# Python
__pycache__/
*.py[cod]

# Environment files
.env
*.env

# Model files (can be large)
models/*.pth
models/*.h5
*.pkl
```

### 2. Never Commit PHI

```bash
# WRONG - Don't do this!
git add data/patient_scans/

# RIGHT - Add only code
git add src/ notebooks/ README.md requirements.txt
```

### 3. Check Before Committing

```bash
# See what you're about to commit
git status
git diff --staged

# Review files
git ls-files

# If you accidentally staged sensitive files
git reset HEAD sensitive_file.dcm
```

## 🌐 Git Workflow

### Daily Development

```bash
# Start work
cd ~/workspace/git/my-project
git pull                      # Get latest changes

# Make changes, then commit
git status                    # Check what changed
git add modified_file.py      # Stage specific files
git commit -m "Add new feature"

# Push to remote
git push
```

### Branching for Features

```bash
# Create feature branch
git checkout -b feature/new-analysis

# Work and commit
git add .
git commit -m "Implement new analysis method"

# Push branch
git push -u origin feature/new-analysis

# Merge via pull request on Gitea web interface
```

### Collaborating

```bash
# Get teammate's changes
git pull

# If conflicts occur
git status                    # See conflicting files
# Edit files to resolve conflicts
git add resolved_file.py
git commit -m "Resolve merge conflicts"
git push
```

## 🎯 Working with Multiple Projects

### Switching Between Projects

```bash
# List all projects
ls ~/workspace/git/

# Switch to project 1
cd ~/workspace/git/dose-analysis
conda activate dose-analysis

# Work on project 1...

# Switch to project 2
cd ~/workspace/git/qa-automation
conda activate qa-automation

# Work on project 2...
```

### Opening Projects in VSCode

**Method 1: From Terminal**
```bash
cd ~/workspace/git/my-project
code .
```

**Method 2: From VSCode**
- File → Open Folder
- Navigate to ~/workspace/git/my-project
- Click "Open"

**Pro Tip:** VSCode remembers recently opened folders - use File → Open Recent

## 📊 Example: Complete Project Setup

```bash
# 1. Clone project
cd ~/workspace/git
git clone https://git.kronisto.net/physics/linac-qa.git
cd linac-qa

# 2. Create conda environment
conda create -n linac-qa python=3.12
conda activate linac-qa

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open in VSCode
# File → Open Folder → ~/workspace/git/linac-qa

# 5. Start working!
# - Notebooks appear in sidebar
# - Data is in data/ folder
# - All project files visible
# - Clean, focused workspace
```

## ❓ Common Questions

**Q: Should I commit my data folder?**
A: **No!** Data should be in .gitignore. Commit only code and documentation.

**Q: Can I have notebooks in multiple projects?**
A: Yes! Each project has its own notebooks/ folder. They stay organized with that project's code and data.

**Q: How do I share data with teammates?**
A: Use shared network drives, not git. Document the data location in your README.md.

**Q: What if my project grows large?**
A: That's fine! Each project folder is independent. You can have 50+ projects in git/ without issues.

**Q: Can I use GitHub for some projects and Gitea for others?**
A: Yes! Just remember: **Gitea for medical data, GitHub for public code only.**

## 💡 Tips

1. **One project = One folder = One git repository**
2. **Open projects individually** in VSCode, not the whole workspace
3. **Copy .gitignore** to every new project
4. **Use descriptive names** for projects (e.g., `imrt-qa-tool` not `project1`)
5. **Keep README.md updated** so others (and future you) understand the project
6. **Commit often** with clear messages
7. **Use branches** for experimental features

---

**Remember:** Organized projects today save hours of confusion tomorrow!
