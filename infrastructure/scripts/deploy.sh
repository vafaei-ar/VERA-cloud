#!/bin/bash
# VERA Cloud - Azure Deployment Script
# Deploys VERA to Azure Container Apps

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP="vera-cloud-rg"
LOCATION="eastus"
ENVIRONMENT="dev"
APP_NAME="vera"
CONTAINER_REGISTRY="veraacr.azurecr.io"
IMAGE_TAG="latest"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists az; then
        print_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    if ! command_exists bicep; then
        print_error "Bicep CLI is not installed. Please install it first."
        exit 1
    fi
    
    print_success "All prerequisites are installed"
}

# Function to login to Azure
azure_login() {
    print_status "Logging in to Azure..."
    
    if ! az account show >/dev/null 2>&1; then
        az login
    fi
    
    print_success "Logged in to Azure"
}

# Function to create resource group
create_resource_group() {
    print_status "Creating resource group: $RESOURCE_GROUP"
    
    if az group show --name "$RESOURCE_GROUP" >/dev/null 2>&1; then
        print_warning "Resource group $RESOURCE_GROUP already exists"
    else
        az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
        print_success "Resource group created"
    fi
}

# Function to create Azure services
create_azure_services() {
    print_status "Creating Azure services..."
    
    # Azure OpenAI
    print_status "Creating Azure OpenAI service..."
    az cognitiveservices account create \
        --name "${APP_NAME}-openai" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --kind OpenAI \
        --sku S0 \
        --custom-domain "${APP_NAME}-openai" \
        --yes
    
    # Azure Speech Service
    print_status "Creating Azure Speech Service..."
    az cognitiveservices account create \
        --name "${APP_NAME}-speech" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --kind SpeechServices \
        --sku S0 \
        --custom-domain "${APP_NAME}-speech" \
        --yes
    
    # Azure AI Search
    print_status "Creating Azure AI Search service..."
    az search service create \
        --name "${APP_NAME}-search" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --sku standard \
        --partition-count 1 \
        --replica-count 1
    
    # Azure Cache for Redis
    print_status "Creating Azure Cache for Redis..."
    az redis create \
        --name "${APP_NAME}-redis" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --sku Standard \
        --vm-size c1 \
        --enable-non-ssl-port false
    
    print_success "Azure services created"
}

# Function to get service endpoints and keys
get_service_config() {
    print_status "Getting service configuration..."
    
    # Get Azure OpenAI configuration
    OPENAI_ENDPOINT=$(az cognitiveservices account show \
        --name "${APP_NAME}-openai" \
        --resource-group "$RESOURCE_GROUP" \
        --query "properties.endpoint" \
        --output tsv)
    
    OPENAI_KEY=$(az cognitiveservices account keys list \
        --name "${APP_NAME}-openai" \
        --resource-group "$RESOURCE_GROUP" \
        --query "key1" \
        --output tsv)
    
    # Get Azure Speech configuration
    SPEECH_KEY=$(az cognitiveservices account keys list \
        --name "${APP_NAME}-speech" \
        --resource-group "$RESOURCE_GROUP" \
        --query "key1" \
        --output tsv)
    
    # Get Azure Search configuration
    SEARCH_ENDPOINT=$(az search service show \
        --name "${APP_NAME}-search" \
        --resource-group "$RESOURCE_GROUP" \
        --query "hostName" \
        --output tsv)
    SEARCH_ENDPOINT="https://${SEARCH_ENDPOINT}"
    
    SEARCH_KEY=$(az search admin-key show \
        --name "${APP_NAME}-search" \
        --resource-group "$RESOURCE_GROUP" \
        --query "primaryKey" \
        --output tsv)
    
    # Get Redis configuration
    REDIS_CONNECTION_STRING=$(az redis list-keys \
        --name "${APP_NAME}-redis" \
        --resource-group "$RESOURCE_GROUP" \
        --query "primaryKey" \
        --output tsv)
    REDIS_HOSTNAME=$(az redis show \
        --name "${APP_NAME}-redis" \
        --resource-group "$RESOURCE_GROUP" \
        --query "hostName" \
        --output tsv)
    REDIS_CONNECTION_STRING="rediss://:${REDIS_CONNECTION_STRING}@${REDIS_HOSTNAME}:6380"
    
    print_success "Service configuration retrieved"
}

# Function to build and push Docker image
build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    # Build image
    docker build -t "$CONTAINER_REGISTRY/vera:$IMAGE_TAG" -f docker/Dockerfile .
    
    # Login to container registry
    az acr login --name "${APP_NAME}acr"
    
    # Push image
    docker push "$CONTAINER_REGISTRY/vera:$IMAGE_TAG"
    
    print_success "Docker image built and pushed"
}

# Function to deploy with Bicep
deploy_bicep() {
    print_status "Deploying infrastructure with Bicep..."
    
    az deployment group create \
        --resource-group "$RESOURCE_GROUP" \
        --template-file infrastructure/bicep/main.bicep \
        --parameters \
            environment="$ENVIRONMENT" \
            appName="$APP_NAME" \
            azureOpenAIEndpoint="$OPENAI_ENDPOINT" \
            azureOpenAIKey="$OPENAI_KEY" \
            azureSpeechKey="$SPEECH_KEY" \
            azureSpeechRegion="$LOCATION" \
            azureSearchEndpoint="$SEARCH_ENDPOINT" \
            azureSearchKey="$SEARCH_KEY" \
            redisConnectionString="$REDIS_CONNECTION_STRING" \
            containerRegistryLoginServer="$CONTAINER_REGISTRY" \
            containerImageTag="$IMAGE_TAG"
    
    print_success "Infrastructure deployed"
}

# Function to get deployment outputs
get_deployment_outputs() {
    print_status "Getting deployment outputs..."
    
    CONTAINER_APP_URL=$(az deployment group show \
        --resource-group "$RESOURCE_GROUP" \
        --name "main" \
        --query "properties.outputs.containerAppUrl.value" \
        --output tsv)
    
    print_success "Deployment completed!"
    print_success "VERA Cloud is available at: $CONTAINER_APP_URL"
}

# Function to run health check
health_check() {
    print_status "Running health check..."
    
    if [ -n "$CONTAINER_APP_URL" ]; then
        sleep 30  # Wait for container to start
        
        if curl -f "$CONTAINER_APP_URL/health" >/dev/null 2>&1; then
            print_success "Health check passed"
        else
            print_warning "Health check failed - service may still be starting"
        fi
    fi
}

# Main deployment function
main() {
    print_status "Starting VERA Cloud deployment..."
    
    check_prerequisites
    azure_login
    create_resource_group
    create_azure_services
    get_service_config
    build_and_push_image
    deploy_bicep
    get_deployment_outputs
    health_check
    
    print_success "VERA Cloud deployment completed successfully!"
    print_status "You can now access VERA at: $CONTAINER_APP_URL"
}

# Run main function
main "$@"
