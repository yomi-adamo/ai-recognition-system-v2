#!/usr/bin/env python
"""
Simple test runner for facial-vision tests
"""
import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with coverage"""
    project_root = Path(__file__).parent.parent

    # Run pytest with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",  # Verbose output
        "--cov=src",  # Coverage for src directory
        "--cov-report=term-missing",  # Show missing lines
        "--cov-report=html",  # Generate HTML report
        "-x",  # Stop on first failure
        str(project_root / "tests"),
    ]

    print("Running tests with coverage...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode == 0:
        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
