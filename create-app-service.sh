#!/bin/bash

# Azure App Service Creation Script
# This script creates the Azure App Service for VERA Cloud

echo "Creating Azure App Service for VERA Cloud..."

# Set variables
RESOURCE_GROUP="vera-cloud-rg"
APP_NAME="vera-cloud-app"
LOCATION="eastus2"
RUNTIME="PYTHON|3.11"

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

# Create resource group if it doesn't exist
echo "Creating resource group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create App Service plan (Free tier)
echo "Creating App Service plan..."
az appservice plan create \
    --name "${APP_NAME}-plan" \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku FREE

# Create Web App
echo "Creating Web App: $APP_NAME"
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan "${APP_NAME}-plan" \
    --name $APP_NAME \
    --runtime $RUNTIME

# Configure app settings (you'll need to add your actual values)
echo "Configuring app settings..."
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --settings \
        AZURE_OPENAI_ENDPOINT="" \
        AZURE_OPENAI_API_KEY="" \
        AZURE_SPEECH_KEY="" \
        AZURE_SPEECH_REGION="eastus2" \
        AZURE_SEARCH_ENDPOINT="" \
        AZURE_SEARCH_API_KEY="" \
        REDIS_CONNECTION_STRING=""

echo "App Service created successfully!"
echo "URL: https://$APP_NAME.azurewebsites.net/"
echo ""
echo "Next steps:"
echo "1. Add your actual Azure service credentials in the App Service settings"
echo "2. Configure GitHub deployment in the Azure Portal"
echo "3. Test the application"
