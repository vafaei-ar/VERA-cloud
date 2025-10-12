# Azure App Service Deployment Guide

## Current Issue
The Azure App Service `vera-cloud-app` appears to be missing or misconfigured.

## Solution Steps

### 1. Check Azure Portal
1. Go to [Azure Portal](https://portal.azure.com)
2. Search for "App Services" in the search bar
3. Look for `vera-cloud-app` in your subscription

### 2. If App Service doesn't exist, create it:
1. Go to "App Services" in Azure Portal
2. Click "Create" → "Web App"
3. Use these settings:
   - **Name**: `vera-cloud-app`
   - **Resource Group**: `vera-cloud-rg` (or create new)
   - **Runtime**: Python 3.11
   - **Region**: East US 2
   - **Pricing Plan**: Free F1 (or Basic B1 for production)

### 3. Configure App Service Settings
Once created, go to Configuration → Application settings and add:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_SPEECH_KEY`
- `AZURE_SPEECH_REGION`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `REDIS_CONNECTION_STRING`

### 4. Enable GitHub Deployment
1. Go to Deployment Center
2. Select "GitHub" as source
3. Connect your repository
4. Select branch: `main`
5. Enable continuous deployment

### 5. Alternative: Manual Deployment
If GitHub Actions fails, you can deploy manually:
1. Install Azure CLI
2. Run: `az webapp deployment source config-zip --resource-group vera-cloud-rg --name vera-cloud-app --src vera-cloud-deployment.zip`

## Current URL
The app should be accessible at:
`https://vera-cloud-app.azurewebsites.net/`

(Note: The slot name `dbhrdyfbg8cyhfam` in the error suggests there might be a deployment slot issue)
