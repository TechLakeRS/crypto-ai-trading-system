#!/bin/bash

# GitHub Push Script
# Replace YOUR_USERNAME with your actual GitHub username

echo "Setting up GitHub remote and pushing..."

# Option 1: HTTPS (easier, works everywhere)
# Uncomment the line below and replace YOUR_USERNAME
# git remote add origin https://github.com/YOUR_USERNAME/crypto-ai-trading-system.git

# Option 2: SSH (more secure, need SSH keys set up)
# Uncomment the line below and replace YOUR_USERNAME
# git remote add origin git@github.com:YOUR_USERNAME/crypto-ai-trading-system.git

# After uncommenting and editing one of the above, run:
git branch -M main
git push -u origin main

echo "Done! Your repository should now be on GitHub."
echo "Visit: https://github.com/YOUR_USERNAME/crypto-ai-trading-system"