# Azure App Service Logs - Troubleshooting Guide

## How to Access Azure App Service Logs

### Method 1: Azure Portal (Easiest)
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your App Service: `vera-cloud-app`
3. Go to **Monitoring** → **Log stream** (for real-time logs)
4. Or go to **Monitoring** → **Diagnostic settings** → **Logs** (for historical logs)

### Method 2: Kudu Console
1. Go to: `https://vera-cloud-app.scm.azurewebsites.net/`
2. Navigate to **Debug console** → **CMD**
3. Go to `LogFiles` folder to see application logs
4. Check `LogFiles/Application` for Python application logs

### Method 3: Azure CLI (Command Line)
```bash
# Get recent logs
az webapp log tail --name vera-cloud-app --resource-group vera-cloud-rg

# Download logs
az webapp log download --name vera-cloud-app --resource-group vera-cloud-rg --log-file logs.zip
```

### Method 4: Application Insights (if enabled)
1. Go to Azure Portal
2. Search for "Application Insights"
3. Find your App Service's Application Insights
4. Go to **Logs** to query application logs

## What to Look For in Logs

### Common Error Patterns:
1. **ModuleNotFoundError**: Missing Python packages
2. **ImportError**: Import issues with Azure services
3. **ConnectionError**: Can't connect to Azure services
4. **ConfigurationError**: Missing environment variables
5. **Port binding errors**: Application can't start on port 8000

### Key Log Locations:
- **Application logs**: `LogFiles/Application/`
- **Web server logs**: `LogFiles/http/`
- **Deployment logs**: `LogFiles/Git/`
- **Python logs**: `LogFiles/Application/`

## Quick Diagnostic Commands

### Check if App Service is running:
```bash
az webapp show --name vera-cloud-app --resource-group vera-cloud-rg --query "state"
```

### Check app settings:
```bash
az webapp config appsettings list --name vera-cloud-app --resource-group vera-cloud-rg
```

### Check deployment status:
```bash
az webapp deployment list --name vera-cloud-app --resource-group vera-cloud-rg
```

## Common Issues and Solutions

### 1. Application Not Starting
- Check startup command in Configuration
- Verify Python version compatibility
- Check for missing dependencies

### 2. Environment Variables Missing
- Verify all required Azure service credentials are set
- Check for typos in variable names

### 3. Port Binding Issues
- Ensure app is binding to 0.0.0.0:8000
- Check if port 8000 is available

### 4. Import Errors
- Verify all packages in requirements.txt are installed
- Check Python path configuration

