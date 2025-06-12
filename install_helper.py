#!/usr/bin/env python3
"""
Installation helper for Facial Vision project
Python-based alternative to the bash installation script
"""

import subprocess
import sys
import os
import platform
from pathlib import Path


class DependencyInstaller:
    def __init__(self):
        self.os_type = platform.system().lower()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent
        
    def check_python_version(self):
        """Check if Python version is 3.8+"""
        print(f"Python version: {sys.version}")
        if self.python_version < (3, 8):
            print("❌ Error: Python 3.8 or higher is required")
            sys.exit(1)
        print("✅ Python version OK")
        
    def create_virtual_environment(self):
        """Create Python virtual environment"""
        venv_path = self.project_root / "venv"
        
        if venv_path.exists():
            print("ℹ️  Virtual environment already exists")
            return
            
        print("🐍 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            print("\nTry installing python3-venv:")
            print("  Ubuntu/Debian: sudo apt-get install python3-venv")
            print("  or create manually: python3 -m venv venv")
            sys.exit(1)
            
    def get_pip_command(self):
        """Get the pip command for the virtual environment"""
        if self.os_type == "windows":
            return str(self.project_root / "venv" / "Scripts" / "pip")
        else:
            return str(self.project_root / "venv" / "bin" / "pip")
            
    def get_python_command(self):
        """Get the python command for the virtual environment"""
        if self.os_type == "windows":
            return str(self.project_root / "venv" / "Scripts" / "python")
        else:
            return str(self.project_root / "venv" / "bin" / "python")
            
    def upgrade_pip(self):
        """Upgrade pip in virtual environment"""
        print("📦 Upgrading pip...")
        pip_cmd = self.get_pip_command()
        try:
            subprocess.run([pip_cmd, "install", "--upgrade", "pip", "setuptools", "wheel"], 
                         check=True)
            print("✅ pip upgraded")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to upgrade pip: {e}")
            
    def install_python_packages(self):
        """Install Python packages from requirements.txt"""
        print("\n📚 Installing Python packages...")
        pip_cmd = self.get_pip_command()
        
        # Install dlib first (often problematic)
        print("Installing dlib (this may take several minutes)...")
        try:
            subprocess.run([pip_cmd, "install", "dlib"], check=True)
            print("✅ dlib installed")
        except subprocess.CalledProcessError:
            print("⚠️  dlib installation failed. You may need to install system dependencies:")
            print("  Ubuntu/Debian: sudo apt-get install cmake libboost-all-dev")
            print("  macOS: brew install cmake boost")
            
        # Install remaining packages
        print("\nInstalling remaining packages...")
        requirements_file = self.project_root / "requirements.txt"
        try:
            subprocess.run([pip_cmd, "install", "-r", str(requirements_file)], 
                         check=True)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Some packages failed to install: {e}")
            print("\nYou may need to install system dependencies first.")
            
    def test_installation(self):
        """Test if key packages are installed correctly"""
        print("\n🧪 Testing installation...")
        python_cmd = self.get_python_command()
        
        packages = [
            ("face_recognition", "Face Recognition"),
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("PIL", "Pillow"),
            ("yaml", "PyYAML")
        ]
        
        all_ok = True
        for module, name in packages:
            try:
                subprocess.run([python_cmd, "-c", f"import {module}"], 
                             check=True, capture_output=True)
                print(f"✅ {name}")
            except subprocess.CalledProcessError:
                print(f"❌ {name}")
                all_ok = False
                
        return all_ok
        
    def print_system_dependencies(self):
        """Print system dependencies for manual installation"""
        print("\n📋 System Dependencies Required:")
        print("=" * 50)
        
        if self.os_type == "linux":
            print("Ubuntu/Debian:")
            print("  sudo apt-get update")
            print("  sudo apt-get install -y \\")
            print("    python3-dev python3-pip python3-venv \\")
            print("    build-essential cmake \\")
            print("    libboost-all-dev libboost-python-dev \\")
            print("    libopencv-dev libgl1-mesa-glx libglib2.0-0 \\")
            print("    libsm6 libxext6 libxrender-dev \\")
            print("    libatlas-base-dev gfortran")
            
        elif self.os_type == "darwin":
            print("macOS:")
            print("  brew install cmake boost boost-python3 opencv pkg-config")
            
        elif self.os_type == "windows":
            print("Windows:")
            print("  1. Install Visual Studio Build Tools")
            print("  2. Or use WSL2 and follow Linux instructions")
            
        print("=" * 50)
        
    def run(self):
        """Run the installation process"""
        print("🚀 Facial Vision Installation Helper")
        print("=" * 50)
        
        # Check Python version
        self.check_python_version()
        
        # Print system dependencies
        self.print_system_dependencies()
        
        # Create virtual environment
        print("\n" + "=" * 50)
        response = input("\nCreate virtual environment? (y/n): ").lower()
        if response == 'y':
            self.create_virtual_environment()
            self.upgrade_pip()
            
            # Install packages
            response = input("\nInstall Python packages? (y/n): ").lower()
            if response == 'y':
                self.install_python_packages()
                
                # Test installation
                if self.test_installation():
                    print("\n✅ Installation completed successfully!")
                else:
                    print("\n⚠️  Some packages are not installed correctly.")
                    print("Please check the error messages above.")
        
        # Print next steps
        print("\n📝 Next Steps:")
        print("=" * 50)
        print("1. Activate virtual environment:")
        if self.os_type == "windows":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("\n2. Test face detection:")
        print("   python scripts/example_face_detection.py")
        print("\n3. Add test images to data/input/")
        print("\n4. Configure settings in config/default.yaml")


if __name__ == "__main__":
    installer = DependencyInstaller()
    installer.run()