# Deployment Guide - Streamlit Cloud

This guide will help you deploy your DCA Application to Streamlit Cloud.

## Prerequisites

1. **GitHub Account**: You need a GitHub account
2. **Streamlit Cloud Account**: Sign up at https://streamlit.io/cloud (free tier available)

## Step 1: Create GitHub Repository

Since GitHub CLI is not installed, follow these steps:

### Option A: Using GitHub Website (Recommended)

1. Go to https://github.com/new
2. Repository name: `DCA-Application` (or your preferred name)
3. Description: "Decline Curve Analysis Application with Streamlit Interface"
4. Choose **Public** (required for free Streamlit Cloud) or **Private** (if you have Streamlit Cloud Pro)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click **Create repository**

### Option B: Using GitHub Desktop

1. Open GitHub Desktop
2. File → New Repository
3. Name: `DCA-Application`
4. Choose the local path: `C:\Users\taha.yehia\OneDrive - Texas A&M University\1. PhD Work\4. Codes\DCA-Application`
5. Click **Create Repository**
6. Click **Publish repository** to GitHub

## Step 2: Push Code to GitHub

After creating the repository on GitHub, run these commands:

```powershell
cd "C:\Users\taha.yehia\OneDrive - Texas A&M University\1. PhD Work\4. Codes\DCA-Application"

# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/DCA-Application.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Note**: You'll be prompted for your GitHub credentials. Use a Personal Access Token (PAT) instead of password:
- Go to https://github.com/settings/tokens
- Generate new token (classic)
- Select scopes: `repo`
- Copy the token and use it as password when prompted

## Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click **Sign in** and authorize with your GitHub account
3. Click **New app**
4. Select your repository: `YOUR_USERNAME/DCA-Application`
5. **Main file path**: `streamlit_app/app.py`
6. **Python version**: Select 3.11 (or 3.10 if 3.11 is not available)
7. Click **Deploy**

## Step 4: Configure Streamlit Cloud (if needed)

After deployment, you can configure:
- **App URL**: Your app will be available at `https://YOUR_APP_NAME.streamlit.app`
- **Settings**: Click the ⋮ menu → Settings
  - **Python version**: 3.11 recommended
  - **Secrets**: Add any environment variables if needed

## Important Notes

### File Structure for Streamlit Cloud

Your project structure is already correct:
- ✅ `streamlit_app/app.py` - Main entry point
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System packages (empty, which is fine)

### Python Version

- Streamlit Cloud supports Python 3.8-3.11
- **Important**: PyCaret requires Python 3.8-3.11 (not 3.12+)
- Select Python 3.11 in Streamlit Cloud settings

### Dependencies

All dependencies are listed in `requirements.txt`:
- Streamlit will automatically install them during deployment
- First deployment may take 5-10 minutes

### Data Files

- Test data in `Test_Data/` folder will be included in the repository
- Users can upload their own CSV files through the Streamlit interface
- No additional configuration needed

## Troubleshooting

### Deployment Fails

1. **Check Python version**: Must be 3.8-3.11
2. **Check requirements.txt**: Ensure all packages are valid
3. **Check logs**: Click "Manage app" → "Logs" to see error messages

### App Crashes

1. **Check Streamlit logs**: Look for Python errors
2. **Verify imports**: Ensure all modules are in the repository
3. **Check file paths**: All paths should be relative

### Missing Dependencies

1. **Add to requirements.txt**: Add any missing packages
2. **Redeploy**: Streamlit Cloud will reinstall dependencies

## Updating Your App

After making changes:

```powershell
cd "C:\Users\taha.yehia\OneDrive - Texas A&M University\1. PhD Work\4. Codes\DCA-Application"

# Make your changes, then:
git add .
git commit -m "Your commit message"
git push origin main
```

Streamlit Cloud will automatically redeploy your app when you push changes to the main branch.

## Security Notes

- **Public Repository**: Your code will be visible to everyone
- **Private Repository**: Requires Streamlit Cloud Pro (paid)
- **Data**: Users upload data through the interface - it's not stored permanently
- **Secrets**: Use Streamlit Secrets for any API keys or sensitive data

## Support

- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-cloud
- Streamlit Community: https://discuss.streamlit.io/

