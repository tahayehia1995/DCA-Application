# Script to push DCA Application to GitHub
# Run this AFTER creating the GitHub repository

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubUsername,
    
    [Parameter(Mandatory=$false)]
    [string]$RepositoryName = "DCA-Application"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pushing DCA Application to GitHub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path ".git")) {
    Write-Host "Error: Not a git repository. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

# Check if remote already exists
$remoteExists = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Remote 'origin' already exists: $remoteExists" -ForegroundColor Yellow
    $overwrite = Read-Host "Do you want to update it? (y/n)"
    if ($overwrite -eq "y" -or $overwrite -eq "Y") {
        git remote set-url origin "https://github.com/$GitHubUsername/$RepositoryName.git"
    } else {
        Write-Host "Keeping existing remote." -ForegroundColor Yellow
    }
} else {
    Write-Host "Adding remote repository..." -ForegroundColor Green
    git remote add origin "https://github.com/$GitHubUsername/$RepositoryName.git"
}

# Rename branch to main if needed
$currentBranch = git branch --show-current
if ($currentBranch -ne "main") {
    Write-Host "Renaming branch from '$currentBranch' to 'main'..." -ForegroundColor Green
    git branch -M main
}

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Green
Write-Host "Note: You may be prompted for GitHub credentials." -ForegroundColor Yellow
Write-Host "Use a Personal Access Token (PAT) instead of password." -ForegroundColor Yellow
Write-Host "Get one at: https://github.com/settings/tokens" -ForegroundColor Yellow
Write-Host ""

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Repository URL: https://github.com/$GitHubUsername/$RepositoryName" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Go to https://share.streamlit.io/" -ForegroundColor White
    Write-Host "2. Sign in with GitHub" -ForegroundColor White
    Write-Host "3. Click 'New app'" -ForegroundColor White
    Write-Host "4. Select your repository: $GitHubUsername/$RepositoryName" -ForegroundColor White
    Write-Host "5. Main file path: streamlit_app/app.py" -ForegroundColor White
    Write-Host "6. Click 'Deploy'" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Push failed. Please check the error above." -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "- Repository doesn't exist on GitHub yet" -ForegroundColor White
    Write-Host "- Authentication failed (use PAT instead of password)" -ForegroundColor White
    Write-Host "- Network connectivity issues" -ForegroundColor White
    Write-Host ""
}

