#!/bin/bash

# Facial Vision Dependency Installation Script
# This script installs all required system and Python dependencies

set -e  # Exit on error

echo "========================================="
echo "Facial Vision Dependency Installation"
echo "========================================="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if [ -f /etc/debian_version ]; then
        DISTRO="debian"
    elif [ -f /etc/redhat-release ]; then
        DISTRO="redhat"
    else
        DISTRO="unknown"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    OS="unknown"
fi

echo "Detected OS: $OS"

# Function to install system dependencies
install_system_deps() {
    echo -e "\n📦 Installing system dependencies..."
    
    if [ "$OS" == "linux" ] && [ "$DISTRO" == "debian" ]; then
        echo "Installing dependencies for Debian/Ubuntu..."
        
        # Check if running with sudo
        if [ "$EUID" -ne 0 ]; then 
            echo "Please run with sudo for system package installation:"
            echo "sudo $0"
            exit 1
        fi
        
        # Update package list
        apt-get update
        
        # Install Python development packages
        apt-get install -y python3-dev python3-pip python3-venv python3-setuptools
        
        # Install build tools
        apt-get install -y build-essential cmake
        
        # Install face_recognition dependencies
        apt-get install -y libboost-all-dev
        apt-get install -y libgtk-3-dev
        apt-get install -y libboost-python-dev
        
        # Install OpenCV dependencies
        apt-get install -y libopencv-dev
        apt-get install -y libgl1-mesa-glx libglib2.0-0
        apt-get install -y libsm6 libxext6 libxrender-dev libgomp1
        
        # Install additional libraries
        apt-get install -y libatlas-base-dev gfortran
        apt-get install -y libjpeg-dev libpng-dev
        
    elif [ "$OS" == "macos" ]; then
        echo "Installing dependencies for macOS..."
        
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Please install it first:"
            echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            exit 1
        fi
        
        # Install dependencies
        brew install cmake
        brew install boost
        brew install boost-python3
        brew install opencv
        brew install pkg-config
        
    else
        echo "⚠️  Unsupported OS. Please install dependencies manually."
        echo "Required packages:"
        echo "  - Python 3.8+ with development headers"
        echo "  - CMake"
        echo "  - Boost libraries"
        echo "  - OpenCV"
        echo "  - BLAS/LAPACK libraries"
        exit 1
    fi
    
    echo "✅ System dependencies installed successfully!"
}

# Function to setup Python virtual environment
setup_venv() {
    echo -e "\n🐍 Setting up Python virtual environment..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "Python version: $PYTHON_VERSION"
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✅ Virtual environment created"
    else
        echo "ℹ️  Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    echo "✅ Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    echo "✅ pip upgraded"
}

# Function to install Python packages
install_python_deps() {
    echo -e "\n📚 Installing Python packages..."
    
    # Install dlib first (face_recognition dependency)
    echo "Installing dlib (this may take a few minutes)..."
    pip install dlib
    
    # Install other packages
    echo "Installing remaining packages..."
    pip install -r requirements.txt
    
    echo "✅ Python packages installed successfully!"
}

# Function to download pre-trained models
download_models() {
    echo -e "\n🤖 Downloading pre-trained models..."
    
    # Create models directory
    mkdir -p data/models
    
    # Download dlib face detection model
    if [ ! -f "data/models/shape_predictor_68_face_landmarks.dat" ]; then
        echo "Downloading facial landmarks model..."
        wget -O data/models/shape_predictor_68_face_landmarks.dat.bz2 \
            http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        bunzip2 data/models/shape_predictor_68_face_landmarks.dat.bz2
    fi
    
    # Download dlib face recognition model
    if [ ! -f "data/models/dlib_face_recognition_resnet_model_v1.dat" ]; then
        echo "Downloading face recognition model..."
        wget -O data/models/dlib_face_recognition_resnet_model_v1.dat.bz2 \
            http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
        bunzip2 data/models/dlib_face_recognition_resnet_model_v1.dat.bz2
    fi
    
    echo "✅ Models downloaded successfully!"
}

# Main installation flow
main() {
    echo -e "\n🚀 Starting installation process..."
    
    # Check if running as root for system packages
    if [ "$1" == "--system" ]; then
        install_system_deps
    fi
    
    # Setup Python environment
    setup_venv
    
    # Install Python packages
    install_python_deps
    
    # Optional: Download models
    if [ "$1" == "--with-models" ] || [ "$2" == "--with-models" ]; then
        download_models
    fi
    
    echo -e "\n========================================="
    echo "✅ Installation completed successfully!"
    echo "========================================="
    echo -e "\nTo activate the virtual environment, run:"
    echo "  source venv/bin/activate"
    echo -e "\nTo test the installation, run:"
    echo "  python -c 'import face_recognition; print(\"face_recognition imported successfully!\")'"
    echo -e "\nFor system dependencies (requires sudo):"
    echo "  sudo $0 --system"
    echo -e "\nTo download pre-trained models:"
    echo "  $0 --with-models"
}

# Run main function
main "$@"