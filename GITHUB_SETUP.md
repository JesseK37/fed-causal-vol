# Getting this project onto GitHub — step-by-step

This guide is written for someone who has never put a project on GitHub
before.  It covers everything from creating an account to making your
first commit.

---

## Step 1 — Create a GitHub account

Go to https://github.com and sign up.  Use a professional username
(e.g. your name or initials + surname).  This username appears on
every project link you share with employers.

---

## Step 2 — Install Git on your machine

**Mac:** Git usually comes pre-installed.  Check by running:
```bash
git --version
```
If not installed, run `xcode-select --install`.

**Windows:** Download from https://git-scm.com/download/win and
install with default options.

**Linux:**
```bash
sudo apt install git
```

---

## Step 3 — Configure Git with your identity

Run these two commands once (they label your commits):
```bash
git config --global user.name "Your Full Name"
git config --global user.email "your@email.com"
```

---

## Step 4 — Create a new repository on GitHub

1. Log in to GitHub
2. Click the **+** icon (top right) → **New repository**
3. Name it: `fed-causal-vol`
4. Set to **Public** (so employers can see it)
5. Do NOT tick "Add a README" — we already have one
6. Click **Create repository**

GitHub will show you a page with setup commands.  Keep it open.

---

## Step 5 — Initialise Git in your project folder

Open a terminal, navigate to the project folder, and run:

```bash
cd path/to/fed_causal_vol

# Initialise a git repository
git init

# Tell git which branch to use (modern convention)
git branch -M main

# Connect to GitHub (paste your actual repo URL from Step 4)
git remote add origin https://github.com/YOUR_USERNAME/fed-causal-vol.git
```

---

## Step 6 — Make your first commit

A commit is a snapshot of your project at a point in time.
Think of it as saving a named version.

```bash
# See what files git notices
git status

# Stage all files (add them to the next commit)
git add .

# Create the commit with a message
git commit -m "Initial commit: project scaffold and notebooks"

# Push to GitHub
git push -u origin main
```

After this, refresh your GitHub repo page — you should see all the
files.

---

## Step 7 — The daily workflow (after your first commit)

Every time you make meaningful progress, commit it:

```bash
# Check what changed
git status
git diff

# Stage changed files
git add notebooks/01_data_ingestion.py   # stage a specific file
# OR
git add .                                # stage everything

# Commit
git commit -m "Add data ingestion notebook - FRED and yfinance fetch"

# Push to GitHub
git push
```

**Good commit message habits:**
- Use the present tense: "Add X", "Fix Y", "Refactor Z"
- Be specific enough that future-you knows what changed
- Commit often — small commits are better than large ones

---

## Step 8 — What NOT to commit

The `.gitignore` file already handles this, but understand why:

- **API keys** — Never commit `.env`.  If you accidentally do,
  rotate your key immediately (treat it as compromised).
- **Data files** — Large CSVs and databases are not suited to Git.
  The `.gitignore` excludes them; your README explains how to
  reproduce the data.
- **Notebook checkpoints** — `.ipynb_checkpoints/` is noise.

---

## Step 9 — Making your profile look professional

On your GitHub profile page:
1. Add a bio mentioning your background
2. Pin this repository (and others as you build them)
3. Add a profile README (optional but impactful — GitHub shows it
   at the top of your profile if you create a repo named
   `YOUR_USERNAME/YOUR_USERNAME`)

---

## Useful Git commands for reference

```bash
git log --oneline         # See your commit history
git diff HEAD~1           # See what changed since last commit
git checkout -b new-branch  # Create a new branch for experiments
git stash                 # Temporarily shelve uncommitted changes
```

---

That is genuinely all you need for a solo portfolio project.
Branches, pull requests, and collaboration workflows come later —
for now, commit → push → repeat.
