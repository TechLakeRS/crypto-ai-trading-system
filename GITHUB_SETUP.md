# GitHub Setup Instructions

Since the GitHub CLI is not installed, follow these steps to push your code to GitHub:

## Option 1: Install GitHub CLI (Recommended)

```bash
# On macOS with Homebrew
brew install gh

# After installation, authenticate:
gh auth login

# Then create and push the repository:
gh repo create crypto-ai-trading-system --public --description "Multi-AI Cryptocurrency Trading System" --source=. --remote=origin --push
```

## Option 2: Manual Setup via GitHub Website

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `crypto-ai-trading-system`
   - Description: `Multi-AI Cryptocurrency Trading System with sentiment analysis and technical indicators`
   - Set to **Public** (or Private if preferred)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. **After creating the repository, GitHub will show you commands. Use these:**

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/crypto-ai-trading-system.git

# Or if using SSH:
git remote add origin git@github.com:YOUR_USERNAME/crypto-ai-trading-system.git

# Push the code
git branch -M main
git push -u origin main
```

## Option 3: Using Personal Access Token

If you get authentication errors:

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` scope
3. Use the token as your password when prompted

```bash
# When pushing, use:
git push -u origin main
# Username: YOUR_GITHUB_USERNAME
# Password: YOUR_PERSONAL_ACCESS_TOKEN
```

## After Pushing

Your repository will be available at:
`https://github.com/YOUR_USERNAME/crypto-ai-trading-system`

## Repository Structure

```
crypto-ai-trading-system/
├── CTanalysis/          # Original strategy markdown files
├── src/                 # Source code
│   ├── config/         # System configuration
│   ├── data_collection/ # Exchange and on-chain data
│   ├── sentiment/      # Multi-AI sentiment analysis
│   └── technical/      # Technical indicators
├── README.md           # Project documentation
├── TODO.md            # Remaining tasks
├── requirements.txt   # Python dependencies
└── .gitignore        # Git ignore rules
```

## Next Steps After GitHub Setup

1. **Set up GitHub Secrets** for API keys:
   - Go to Settings → Secrets and variables → Actions
   - Add secrets for exchange APIs, AI model keys, etc.

2. **Enable GitHub Actions** for CI/CD (optional)

3. **Set up branch protection** for main branch (optional)

4. **Add collaborators** if working with a team

5. **Create issues** from TODO.md items for tracking

## Quick Git Commands Reference

```bash
# Check current status
git status

# Add new changes
git add .

# Commit changes
git commit -m "Your commit message"

# Push changes
git push

# Pull latest changes
git pull

# Create new branch
git checkout -b feature/branch-name

# Switch branches
git checkout main
```

## Troubleshooting

If you encounter issues:

1. **Authentication failed:**
   - Use personal access token instead of password
   - Or set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

2. **Remote already exists:**
   ```bash
   git remote remove origin
   git remote add origin YOUR_REPO_URL
   ```

3. **Branch name issues:**
   ```bash
   git branch -M main
   ```

---

Remember to never commit sensitive information like API keys. Use environment variables and .env files (which are gitignored).