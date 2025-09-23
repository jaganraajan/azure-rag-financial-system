#!/bin/bash

# Azure RAG Financial System Deployment Script
# This script deploys the Azure RAG Financial System to Azure

set -e

# Configuration
RESOURCE_GROUP_NAME="rg-azure-rag-financial"
LOCATION="East US"
DEPLOYMENT_NAME="azure-rag-deployment-$(date +%Y%m%d%H%M%S)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first:"
    echo "https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if user is logged in
if ! az account show &> /dev/null; then
    print_warning "You are not logged in to Azure CLI"
    print_status "Logging in to Azure..."
    az login
fi

# Get current subscription
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
print_status "Using subscription: $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"

# Create resource group if it doesn't exist
print_status "Checking if resource group '$RESOURCE_GROUP_NAME' exists..."
if ! az group show --name $RESOURCE_GROUP_NAME &> /dev/null; then
    print_status "Creating resource group '$RESOURCE_GROUP_NAME' in '$LOCATION'..."
    az group create --name $RESOURCE_GROUP_NAME --location "$LOCATION"
    print_success "Resource group created successfully"
else
    print_success "Resource group already exists"
fi

# Deploy ARM template
print_status "Starting deployment '$DEPLOYMENT_NAME'..."
print_status "This may take 10-15 minutes..."

DEPLOYMENT_RESULT=$(az deployment group create \
    --resource-group $RESOURCE_GROUP_NAME \
    --name $DEPLOYMENT_NAME \
    --template-file azuredeploy.json \
    --parameters azuredeploy.parameters.json \
    --query 'properties.outputs' \
    --output json)

if [ $? -eq 0 ]; then
    print_success "Deployment completed successfully!"
    
    # Extract outputs
    WEB_APP_URL=$(echo $DEPLOYMENT_RESULT | jq -r '.webAppUrl.value')
    SEARCH_SERVICE=$(echo $DEPLOYMENT_RESULT | jq -r '.searchServiceName.value')
    OPENAI_ENDPOINT=$(echo $DEPLOYMENT_RESULT | jq -r '.openAIEndpoint.value')
    STORAGE_ACCOUNT=$(echo $DEPLOYMENT_RESULT | jq -r '.storageAccountName.value')
    KEY_VAULT=$(echo $DEPLOYMENT_RESULT | jq -r '.keyVaultName.value')
    
    print_success "Deployment outputs:"
    echo "  Web App URL: $WEB_APP_URL"
    echo "  Search Service: $SEARCH_SERVICE"
    echo "  OpenAI Endpoint: $OPENAI_ENDPOINT"
    echo "  Storage Account: $STORAGE_ACCOUNT"
    echo "  Key Vault: $KEY_VAULT"
    
    # Get service keys
    print_status "Retrieving service keys..."
    
    SEARCH_KEY=$(az search admin-key show --resource-group $RESOURCE_GROUP_NAME --service-name $SEARCH_SERVICE --query primaryKey -o tsv)
    OPENAI_KEY=$(az cognitiveservices account keys list --resource-group $RESOURCE_GROUP_NAME --name ${OPENAI_ENDPOINT##*//} --name ${OPENAI_ENDPOINT%%.openai*} --query key1 -o tsv)
    
    # Store secrets in Key Vault
    print_status "Storing secrets in Key Vault..."
    az keyvault secret set --vault-name $KEY_VAULT --name "search-admin-key" --value "$SEARCH_KEY" > /dev/null
    az keyvault secret set --vault-name $KEY_VAULT --name "openai-api-key" --value "$OPENAI_KEY" > /dev/null
    
    print_success "Secrets stored in Key Vault"
    
    # Configure app settings with secrets
    print_status "Configuring application settings..."
    
    APP_NAME=$(echo $DEPLOYMENT_RESULT | jq -r '.webAppUrl.value' | sed 's|https://||' | sed 's|\.azurewebsites\.net||')
    
    az webapp config appsettings set \
        --resource-group $RESOURCE_GROUP_NAME \
        --name $APP_NAME \
        --settings \
            AZURE_SEARCH_ADMIN_KEY="$SEARCH_KEY" \
            AZURE_OPENAI_API_KEY="$OPENAI_KEY" \
        > /dev/null
    
    print_success "Application settings configured"
    
    # Create demo filings
    print_status "Setting up demo data..."
    echo "To set up demo data, run the following command locally:"
    echo "python main.py rag --process --input-dir demo_filings"
    
    print_success "Deployment completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Deploy OpenAI models (gpt-4 and text-embedding-ada-002) in Azure OpenAI Studio"
    echo "2. Run 'python main.py rag --process' to index demo documents"
    echo "3. Visit $WEB_APP_URL to use the application"
    echo ""
    echo "Environment variables for local development:"
    echo "export AZURE_SEARCH_SERVICE_NAME='$SEARCH_SERVICE'"
    echo "export AZURE_SEARCH_ADMIN_KEY='$SEARCH_KEY'"
    echo "export AZURE_OPENAI_ENDPOINT='$OPENAI_ENDPOINT'"
    echo "export AZURE_OPENAI_API_KEY='$OPENAI_KEY'"
    
else
    print_error "Deployment failed!"
    exit 1
fi