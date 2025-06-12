#!/usr/bin/env python3
"""
Setup script for Facial Vision project
Handles virtual environment creation and dependency installation
"""

import os
import sys
import subprocess
import platform

def main():
    print("Facial Vision Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Instructions for virtual environment
    print("\nVirtual Environment Setup:")
    print("-" * 30)
    
    if platform.system() == "Linux":
        print("\n1. First, ensure python3-venv is installed:")
        print("   sudo apt-get update")
        print("   sudo apt-get install python3-venv")
        print("\n2. Create virtual environment:")
        print("   python3 -m venv venv")
        print("\n3. Activate virtual environment:")
        print("   source venv/bin/activate")
    elif platform.system() == "Windows":
        print("\n1. Create virtual environment:")
        print("   python -m venv venv")
        print("\n2. Activate virtual environment:")
        print("   venv\\Scripts\\activate")
    else:  # macOS
        print("\n1. Create virtual environment:")
        print("   python3 -m venv venv")
        print("\n2. Activate virtual environment:")
        print("   source venv/bin/activate")
    
    print("\n4. Install dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\nNote: Some dependencies may require additional system packages:")
    print("- For face_recognition: cmake, dlib dependencies")
    print("- For opencv-python: libgl1-mesa-glx, libglib2.0-0")
    
    if platform.system() == "Linux":
        print("\nInstall system dependencies (Ubuntu/Debian):")
        print("sudo apt-get install cmake libboost-all-dev")
        print("sudo apt-get install libgl1-mesa-glx libglib2.0-0")
    
    print("\n" + "=" * 50)
    print("After setup, you can run:")
    print("- python scripts/process_folder.py")
    print("- python scripts/process_video.py")

if __name__ == "__main__":
    main()