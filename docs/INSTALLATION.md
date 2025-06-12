# Installation Guide for Facial Vision

This guide provides step-by-step instructions for installing all dependencies required by the Facial Vision project.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- Administrative/sudo access for system packages

## Quick Installation

```bash
# Clone the repository (if not already done)
cd /home/adamoyomi/code/facial-vision

# Run the automated installation script
./scripts/install_dependencies.sh

# For system dependencies (requires sudo)
sudo ./scripts/install_dependencies.sh --system

# To also download pre-trained models
./scripts/install_dependencies.sh --with-models
```

## Manual Installation

### Step 1: System Dependencies

#### Ubuntu/Debian

```bash
# Update package list
sudo apt-get update

# Install Python development packages
sudo apt-get install -y python3-dev python3-pip python3-venv python3-setuptools

# Install build tools
sudo apt-get install -y build-essential cmake

# Install libraries for face_recognition
sudo apt-get install -y libboost-all-dev libboost-python-dev

# Install OpenCV dependencies
sudo apt-get install -y libopencv-dev libgl1-mesa-glx libglib2.0-0
sudo apt-get install -y libsm6 libxext6 libxrender-dev libgomp1

# Install additional libraries
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y libjpeg-dev libpng-dev

# Install tools for downloading models
sudo apt-get install -y wget bzip2
```

#### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake
brew install boost boost-python3
brew install opencv
brew install pkg-config
brew install wget
```

#### Windows (WSL2)

For Windows users, we recommend using WSL2 (Windows Subsystem for Linux) and following the Ubuntu/Debian instructions above.

### Step 2: Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 3: Python Dependencies

```bash
# Install dlib first (this may take several minutes)
pip install dlib

# Install all other dependencies
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models (Optional)

```bash
# Create models directory
mkdir -p data/models

# Download facial landmarks model
wget -O data/models/shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 data/models/shape_predictor_68_face_landmarks.dat.bz2

# Download face recognition model
wget -O data/models/dlib_face_recognition_resnet_model_v1.dat.bz2 \
    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 data/models/dlib_face_recognition_resnet_model_v1.dat.bz2
```

## Troubleshooting

### Common Issues

1. **"error: Microsoft Visual C++ 14.0 is required" (Windows)**
   - Install Visual Studio Build Tools from Microsoft

2. **"CMake must be installed" error**
   - Ensure CMake is installed: `sudo apt-get install cmake`

3. **"No module named 'pip'"**
   - Install pip: `curl https://bootstrap.pypa.io/get-pip.py | python3`

4. **dlib installation fails**
   - Ensure all build dependencies are installed
   - Try installing dlib separately: `pip install dlib --verbose`

5. **ImportError with OpenCV**
   - Install missing libraries: `sudo apt-get install libgl1-mesa-glx`

### Verifying Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Test imports
python -c "import face_recognition; print('✓ face_recognition')"
python -c "import cv2; print('✓ OpenCV')"
python -c "import numpy; print('✓ NumPy')"
python -c "import PIL; print('✓ Pillow')"

# Run the example script
python scripts/example_face_detection.py
```

## Dependencies Overview

### Core Dependencies

- **face_recognition**: Main face detection and recognition library
- **opencv-python**: Computer vision library for image/video processing
- **numpy**: Numerical computing library
- **Pillow**: Image processing library

### Video Processing

- **opencv-python**: Video frame extraction and processing

### Data Handling

- **pyyaml**: Configuration file parsing
- **python-dateutil**: Date/time handling
- **exifread**: Extract metadata from images

### Integration

- **ipfshttpclient**: IPFS integration
- **requests**: HTTP requests for API integration

### Alternative Models

- **deepface**: Alternative face recognition models
- **mtcnn**: Alternative face detection model

## Next Steps

After installation:

1. Test the installation with example scripts
2. Add test images to `data/input/`
3. Configure settings in `config/default.yaml`
4. Run face detection on your images

For development setup, see `docs/architecture.md`.