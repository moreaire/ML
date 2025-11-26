# Medical Physics Development Environment

Welcome to your code-server development environment! This workspace is pre-configured for medical physics research, development, and education.

## 🎯 Project-Centric Workflow

**Important:** Don't work directly in the workspace folder! Instead:

1. **Clone or create projects** in `git/` or `local/`
2. **Open individual project folders** in VSCode (File → Open Folder)
3. **Work within that project** - keep data, notebooks, code together

This keeps your environment organized as you work on 10, 20, or more projects!

### Example Workflow

```bash
# Clone a project
cd ~/workspace/git
git clone https://git.kronisto.net/user/dose-analysis.git

# Open JUST that project folder in VSCode
# File → Open Folder → ~/workspace/git/dose-analysis

# Now you see only that project's files
# - Not mixed with other projects
# - Clean, focused workspace
```

## 📁 Workspace Structure

```
workspace/
├── README.md    # This file - general guidelines
├── git/         # Clone repositories here
│   ├── project-1/
│   ├── project-2/
│   └── ...
└── local/       # Quick experiments and personal scripts
    ├── test-script.py
    └── ...
```

**Each project should contain its own:**
- `notebooks/` - Jupyter notebooks for that project
- `data/` - Data files for that project (gitignored)
- `src/` or scripts - Source code
- `.gitignore` - Customized for the project
- `README.md` - Project documentation

See `git/README.md` for detailed project structure examples.

## 🔐 Git: GitHub vs Gitea (Local)

### Gitea (Recommended for PHI/Patient Data)
**URL:** https://git.kronisto.net/

**Pros:**
- ✅ **On-premises** - Data never leaves the facility
- ✅ **HIPAA-friendly** - Safer for potential PHI
- ✅ **Fast** - Local network speeds
- ✅ **No file size limits** - Store large datasets
- ✅ **Complete control** - We manage access and security

**Cons:**
- ❌ Only accessible within Kronisto subnet
- ❌ No external collaboration without VPN
- ❌ Requires local account (contact Ara Alexandrian for access)

### GitHub (Public Cloud)
**URL:** https://github.com/

**Pros:**
- ✅ Accessible anywhere
- ✅ Easy external collaboration
- ✅ Rich ecosystem (Actions, Issues, etc.)
- ✅ Free for public/private repos

**Cons:**
- ❌ **NEVER use for PHI/patient data**
- ❌ File size limits (100MB per file, 1GB per repo recommended)
- ❌ Data stored on external servers
- ❌ Potential privacy/security risks

### ⚠️ **CRITICAL SECURITY GUIDELINE**

**Use Gitea for:**
- Any code that processes patient data
- Projects with DICOM files or PHI
- Internal tools and workflows
- Large medical imaging datasets

**Use GitHub for:**
- Public research code (anonymized)
- Educational materials
- Open-source contributions
- Personal projects (non-medical)

## 🔒 Security & Privacy Guidelines

### 1. Project-Level .gitignore

Each project should have its own `.gitignore`. A template is provided in `git/.gitignore` - copy it when starting new projects:

```bash
cp ~/workspace/git/.gitignore ~/workspace/git/my-new-project/
```

### 2. Never Commit Patient Data

```bash
# BAD - Don't do this!
git add patient_scans/*.dcm

# GOOD - Use sample/anonymized data
git add test_data/anonymized_sample.dcm
```

### 3. Check Before Pushing

```bash
git status              # See what will be committed
git diff --staged       # Review actual changes
git log --oneline -5    # Check recent commits
```

### 4. Sensitive Configurations

```bash
# Use environment variables, not hardcoded values
# Create a .env file (in .gitignore)
echo "API_KEY=your_key_here" > .env
```

## 🌐 Container Networking (container-net)

Your code-server runs in an isolated Docker network:
- **Network:** 172.20.0.0/16
- **Your container:** 172.20.0.XX
- **Nginx Proxy:** 172.20.0.3
- **Gitea:** git.kronisto.net → 172.20.0.6

### Running Web Applications

If you develop web apps (Flask, Streamlit, Dash, etc.), access them via code-server's built-in proxy:

```
https://code-USERNAME.kronisto.net/proxy/PORT/
```

**Example:**
```bash
# Start Flask app
python app.py --host 0.0.0.0 --port 5000

# Access at:
# https://code-USERNAME.kronisto.net/proxy/5000/
```

For production apps, contact admin to set up dedicated Nginx proxy.

## 🐍 Python & Conda

### Create Project Environment

```bash
# Navigate to your project
cd ~/workspace/git/my-project

# Create environment
conda create -n myproject python=3.12
conda activate myproject
pip install -r requirements.txt
```

### Jupyter Notebooks

**Recommended:** Use VSCode's built-in Jupyter support
- Just open any `.ipynb` file in your project
- Notebooks stay organized with your project

**Alternative:** Jupyter Lab
```bash
cd ~/workspace/git/my-project
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
# Access via: https://code-USERNAME.kronisto.net/proxy/8888/
```

## 🛠️ Common Commands

### Git Workflow
```bash
# Clone from Gitea
cd ~/workspace/git
git clone https://git.kronisto.net/username/repo.git
cd repo

# Daily workflow
git pull                    # Get latest changes
git status                  # Check what changed
git add file.py             # Stage specific file
git commit -m "Description" # Commit changes
git push                    # Upload to remote
```

### Conda Management
```bash
conda env list              # List environments
conda activate envname      # Switch environment
conda deactivate            # Exit to base
conda list                  # Show installed packages
```

### File Management
```bash
# Navigate
cd ~/workspace/git          # Go to repositories
cd ~/workspace/local        # Go to local scripts

# Search
grep -r "search term" .     # Search in files
find . -name "*.py"         # Find Python files

# Disk usage
du -sh *                    # Size of directories
```

## 📚 Pre-installed Packages

### Scientific Stack
- NumPy, SciPy, Pandas
- Matplotlib, Plotly, Seaborn
- scikit-learn, scikit-image

### Medical Physics
- pydicom - DICOM handling
- SimpleITK - Medical image processing
- pylinac - Radiation therapy QA
- dicompyler-core - Treatment planning

### Development Tools
- Jupyter Lab
- black, flake8, pylint
- Git, curl, wget

## ❓ Getting Help

- **Project Organization:** See `git/README.md`
- **Git Access:** Contact Ara Alexandrian (ara@kronisto.net)
- **Container Issues:** Contact system administrator
- **Medical Physics Questions:** Check team documentation

## 💡 Tips for Success

1. **One project, one folder** - Keep related files together
2. **Open specific folders** - Don't work from workspace root
3. **Use .gitignore** - Protect PHI and large files
4. **Check before committing** - Review changes carefully
5. **Gitea for medical data** - GitHub for public code only

---

**Remember:** Always be mindful of patient privacy and data security. When in doubt about what data is safe to commit, ask!
