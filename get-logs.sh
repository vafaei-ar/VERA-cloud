#!/bin/bash

# Azure App Service Log Retrieval Script
# This script helps you get logs from your Azure App Service

echo "Getting Azure App Service logs for VERA Cloud..."

# Set variables
RESOURCE_GROUP="vera-cloud-rg"
APP_NAME="vera-cloud-app"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "Azure CLI is not installed. Please install it first:"
    echo "curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo "Please login to Azure first:"
    echo "az login"
    exit 1
fi

echo "Checking App Service status..."
az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "state" -o tsv

echo ""
echo "Getting recent logs (last 100 lines)..."
echo "=========================================="
az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP --lines 100

echo ""
echo "Downloading full logs..."
az webapp log download --name $APP_NAME --resource-group $RESOURCE_GROUP --log-file vera-cloud-logs.zip

echo ""
echo "Checking app settings..."
echo "=========================================="
az webapp config appsettings list --name $APP_NAME --resource-group $RESOURCE_GROUP --query "[?name=='AZURE_OPENAI_ENDPOINT' || name=='AZURE_SPEECH_KEY' || name=='REDIS_CONNECTION_STRING'].{Name:name, Value:value}" -o table

echo ""
echo "Checking deployment status..."
echo "=========================================="
az webapp deployment list --name $APP_NAME --resource-group $RESOURCE_GROUP --query "[0].{Status:status, Message:message, Author:author}" -o table

echo ""
echo "Logs downloaded to: vera-cloud-logs.zip"
echo "You can extract and examine the logs to identify the issue."

