#!/usr/bin/env python3
"""
Azure App Service startup script for VERA Cloud
"""
import os
import sys
import subprocess

if __name__ == "__main__":
    # Change to the app directory
    os.chdir('/home/site/wwwroot')
    
    # Set environment variables for Azure App Service
    os.environ.setdefault('AZURE_OPENAI_ENDPOINT', '')
    os.environ.setdefault('AZURE_OPENAI_API_KEY', '')
    os.environ.setdefault('AZURE_SPEECH_KEY', '')
    os.environ.setdefault('AZURE_SPEECH_REGION', '')
    os.environ.setdefault('AZURE_SEARCH_ENDPOINT', '')
    os.environ.setdefault('AZURE_SEARCH_API_KEY', '')
    os.environ.setdefault('REDIS_CONNECTION_STRING', '')
    
    # Start the FastAPI application
    cmd = [
        sys.executable, '-m', 'uvicorn', 
        'api.main:app', 
        '--host', '0.0.0.0', 
        '--port', '8000'
    ]
    
    subprocess.run(cmd)

