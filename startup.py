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
    
    # Start the FastAPI application
    cmd = [
        sys.executable, '-m', 'uvicorn', 
        'api.main:app', 
        '--host', '0.0.0.0', 
        '--port', '8000'
    ]
    
    subprocess.run(cmd)

