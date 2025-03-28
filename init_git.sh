#!/bin/bash

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Pedestrian Detection Project"

# Instructions for adding remote repo
echo "Repository initialized!"
echo "To connect to GitHub, run:"
echo "git remote add origin https://github.com/yourusername/pedestrian-detection.git"
echo "git branch -M main"
echo "git push -u origin main" 