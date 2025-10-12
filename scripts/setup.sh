#!/bin/bash
# VERA Cloud - Setup Script
# Sets up the development environment for VERA Cloud

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
VENV_NAME="vera-cloud"
REQUIREMENTS_FILE="requirements.txt"

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
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.11+ first."
        exit 1
    fi
    
    if ! command_exists pip; then
        print_error "pip is not installed. Please install pip first."
        exit 1
    fi
    
    if ! command_exists git; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    print_success "All prerequisites are installed"
}

# Function to create virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment $VENV_NAME already exists"
    else
        python3 -m venv "$VENV_NAME"
        print_success "Virtual environment created"
    fi
}

# Function to activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [ -f "$VENV_NAME/bin/activate" ]; then
        source "$VENV_NAME/bin/activate"
        print_success "Virtual environment activated"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
}

# Function to install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "$REQUIREMENTS_FILE" ]; then
        pip install -r "$REQUIREMENTS_FILE"
        print_success "Python dependencies installed"
    else
        print_error "Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p data/sessions
    mkdir -p frontend/static
    
    print_success "Directories created"
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment file..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# VERA Cloud Environment Configuration
# Copy this file and update with your Azure service details

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_openai_endpoint_here
AZURE_OPENAI_API_KEY=your_openai_key_here

# Azure Speech Service
AZURE_SPEECH_KEY=your_speech_key_here
AZURE_SPEECH_REGION=eastus

# Azure AI Search
AZURE_SEARCH_ENDPOINT=your_search_endpoint_here
AZURE_SEARCH_API_KEY=your_search_key_here

# Redis Cache
REDIS_CONNECTION_STRING=your_redis_connection_string_here

# Application Insights
APPLICATION_INSIGHTS_CONNECTION_STRING=your_app_insights_connection_string_here

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string_here

# Environment
ENVIRONMENT=dev
LOG_LEVEL=INFO
EOF
        print_success "Environment file created (.env)"
        print_warning "Please update .env with your Azure service details"
    else
        print_warning "Environment file already exists"
    fi
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    if command_exists pytest; then
        python -m pytest tests/ -v
        print_success "Tests completed"
    else
        print_warning "pytest not found, skipping tests"
    fi
}

# Function to show next steps
show_next_steps() {
    print_success "Setup completed successfully!"
    echo
    print_status "Next steps:"
    echo "1. Update .env file with your Azure service details"
    echo "2. Run the application: python -m uvicorn api.main:app --reload"
    echo "3. Open http://localhost:8000 in your browser"
    echo
    print_status "For production deployment:"
    echo "1. Run: ./infrastructure/scripts/deploy.sh"
    echo "2. Follow the deployment prompts"
    echo
    print_status "For development:"
    echo "1. Activate virtual environment: source $VENV_NAME/bin/activate"
    echo "2. Start the application: python -m uvicorn api.main:app --reload"
}

# Main setup function
main() {
    print_status "Starting VERA Cloud setup..."
    
    check_prerequisites
    create_venv
    activate_venv
    install_dependencies
    create_directories
    create_env_file
    run_tests
    show_next_steps
    
    print_success "VERA Cloud setup completed!"
}

# Run main function
main "$@"
