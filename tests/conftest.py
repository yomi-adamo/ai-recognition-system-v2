"""
Pytest configuration and shared fixtures for facial-vision tests
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image"""
    # Create a simple test image
    image = Image.new("RGB", (640, 480), color="white")
    image_path = temp_dir / "test_image.jpg"
    image.save(image_path)
    return image_path


@pytest.fixture
def sample_image_array():
    """Create a sample image as numpy array"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_detection():
    """Sample face detection data"""
    return {
        "bounds": (100, 100, 200, 200),  # (top, right, bottom, left)
        "confidence": 0.98,
        "landmarks": {
            "left_eye": (120, 130),
            "right_eye": (180, 130),
            "nose": (150, 160),
            "mouth_left": (130, 180),
            "mouth_right": (170, 180),
        },
    }


@pytest.fixture
def sample_config():
    """Sample configuration dictionary"""
    return {
        "face_detection": {"model": "hog", "tolerance": 0.6, "min_face_size": 40},
        "clustering": {"algorithm": "hdbscan", "min_cluster_size": 2, "similarity_threshold": 0.6},
        "output": {"chip_size": [224, 224], "jpeg_quality": 85, "format": "json"},
        "paths": {"output": "data/output", "temp": "data/temp"},
    }


@pytest.fixture
def mock_face_detector():
    """Mock face detector for testing"""
    detector = Mock()
    detector.detect.return_value = [
        {"bounds": (100, 100, 200, 200), "confidence": 0.98, "encoding": np.random.rand(128)}
    ]
    return detector


@pytest.fixture
def mock_face_clusterer():
    """Mock face clusterer for testing"""
    clusterer = Mock()
    clusterer.extract_embeddings.return_value = [np.random.rand(128)]
    clusterer.assign_cluster.return_value = "person_1"
    return clusterer


@pytest.fixture
def mock_file_handler(temp_dir):
    """Mock file handler with temp directory"""
    from src.utils.file_handler import FileHandler

    handler = FileHandler(base_output_dir=temp_dir)
    return handler


@pytest.fixture
def sample_metadata():
    """Sample file metadata"""
    return {
        "name": "test_video.mp4",
        "path": "/path/to/test_video.mp4",
        "size": 1048576,
        "size_mb": 1.0,
        "created": "2024-01-15T10:00:00",
        "modified": "2024-01-15T10:00:00",
        "extension": ".mp4",
        "mime_type": "video/mp4",
        "hash": "abcdef123456",
    }


@pytest.fixture
def sample_cluster_data():
    """Sample clustering data"""
    return {
        "person_1": {
            "centroid": np.random.rand(128),
            "chip_count": 5,
            "last_seen": "2024-01-15T10:00:00",
            "representative_chips": ["chip_001.jpg", "chip_002.jpg"],
        },
        "person_2": {
            "centroid": np.random.rand(128),
            "chip_count": 3,
            "last_seen": "2024-01-15T10:05:00",
            "representative_chips": ["chip_003.jpg"],
        },
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Reset ConfigManager singleton
    from src.utils.config import ConfigManager

    ConfigManager._instance = None
    ConfigManager._config = None
    yield
    # Cleanup after test
    ConfigManager._instance = None
    ConfigManager._config = None


@pytest.fixture
def mock_blockchain_response():
    """Mock blockchain API response"""
    return {
        "id": "asset-uuid-123",
        "name": "test_video.mp4",
        "type": "video",
        "author": "test-device",
        "hash": "sha256-hash",
        "timestamp": "2024-01-15T10:00:00Z",
        "metadata": {"fileHash": "abcdef123456"},
    }


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_gpu: Tests that require GPU")


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if needed"""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if not cuda_available:
        skip_gpu = pytest.mark.skip(reason="GPU/CUDA not available")
        for item in items:
            if "requires_gpu" in item.keywords:
                item.add_marker(skip_gpu)
