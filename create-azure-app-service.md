# Create Azure App Service - Step by Step Guide

## Current Issue
The Azure App Service `vera-cloud-app` with slot `dbhrdyfbg8cyhfam` is not found (404 error).

## Solution: Create New Azure App Service

### Step 1: Access Azure Portal
1. Go to [Azure Portal](https://portal.azure.com)
2. Sign in with your Azure account

### Step 2: Create Resource Group (if needed)
1. Search for "Resource groups" in the top search bar
2. Click "Create" if `vera-cloud-rg` doesn't exist
3. Name: `vera-cloud-rg`
4. Region: `East US 2`
5. Click "Review + Create" → "Create"

### Step 3: Create App Service
1. Search for "App Services" in the top search bar
2. Click "Create" → "Web App"
3. Fill in the details:
   - **Subscription**: Your subscription
   - **Resource Group**: `vera-cloud-rg`
   - **Name**: `vera-cloud-app`
   - **Publish**: Code
   - **Runtime stack**: Python 3.11
   - **Operating System**: Linux
   - **Region**: East US 2
   - **Pricing Plan**: Free F1 (or Basic B1 for production)
4. Click "Review + Create" → "Create"

### Step 4: Configure App Service Settings
Once created, go to your App Service:

1. **Go to Configuration → Application settings**
2. **Add these environment variables**:
   ```
   AZURE_OPENAI_ENDPOINT = your_openai_endpoint
   AZURE_OPENAI_API_KEY = your_openai_key
   AZURE_SPEECH_KEY = your_speech_key
   AZURE_SPEECH_REGION = eastus2
   AZURE_SEARCH_ENDPOINT = your_search_endpoint
   AZURE_SEARCH_API_KEY = your_search_key
   REDIS_CONNECTION_STRING = your_redis_connection
   ```

### Step 5: Configure Deployment
1. **Go to Deployment Center**
2. **Select "GitHub" as source**
3. **Connect your GitHub account**
4. **Select repository**: `vafaei-ar/VERA-cloud`
5. **Select branch**: `main`
6. **Enable continuous deployment**

### Step 6: Test the Application
The app should be accessible at:
`https://vera-cloud-app.azurewebsites.net/`

## Alternative: Manual Deployment
If GitHub Actions fails, you can deploy manually:

1. **Install Azure CLI**:
   ```bash
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

2. **Login to Azure**:
   ```bash
   az login
   ```

3. **Deploy the zip file**:
   ```bash
   az webapp deployment source config-zip \
     --resource-group vera-cloud-rg \
     --name vera-cloud-app \
     --src vera-cloud-deployment.zip
   ```

## Troubleshooting
- If you get permission errors, make sure you have Contributor access to the resource group
- If the app still shows 503/404 errors, check the App Service logs in the Azure Portal
- Make sure all environment variables are set correctly
