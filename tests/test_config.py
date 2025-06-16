"""
Tests for configuration management
"""
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import ConfigManager, get_config, validate_config


class TestConfigManager:
    """Test ConfigManager functionality"""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "face_detection": {"model": "test_model", "tolerance": 0.5},
                "output": {"format": "json"},
                "paths": {"output": "test/output"},
                "test_section": {"nested": {"value": 42}},
            }
            yaml.dump(config_data, f)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_singleton_pattern(self):
        """Test that ConfigManager follows singleton pattern"""
        config1 = ConfigManager()
        config2 = ConfigManager()
        assert config1 is config2

    def test_get_config_convenience_function(self):
        """Test get_config() returns ConfigManager instance"""
        config = get_config()
        assert isinstance(config, ConfigManager)

    def test_get_nested_value(self):
        """Test getting nested configuration values"""
        config = get_config()
        # Should return default value if key doesn't exist
        assert config.get("nonexistent.key", "default") == "default"

    def test_environment_override(self):
        """Test environment variable overrides"""
        # Set environment variable
        os.environ["FACIAL_VISION_TEST_VALUE"] = "env_value"

        # Reset singleton to force reload
        ConfigManager._instance = None
        ConfigManager._config = None

        config = get_config()
        assert config.get("test.value") == "env_value"

        # Cleanup
        del os.environ["FACIAL_VISION_TEST_VALUE"]

    def test_get_clustering_config(self):
        """Test getting clustering configuration with defaults"""
        config = get_config()
        clustering = config.get_clustering_config()

        assert isinstance(clustering, dict)
        assert clustering["algorithm"] == "hdbscan"
        assert clustering["min_cluster_size"] == 2
        assert clustering["metric"] == "cosine"

    def test_get_maverix_config(self):
        """Test getting Maverix configuration"""
        config = get_config()
        maverix = config.get_maverix_config()

        assert isinstance(maverix, dict)
        assert maverix["base_url"] == "http://localhost:3000"
        assert maverix["retry_attempts"] == 3

    def test_get_paths_config(self):
        """Test getting paths configuration"""
        config = get_config()
        paths = config.get_paths_config()

        assert isinstance(paths, dict)
        assert all(isinstance(p, Path) for p in paths.values())
        assert "input" in paths
        assert "output" in paths
        assert "temp" in paths

    def test_validate_config_success(self):
        """Test configuration validation with valid config"""
        # Ensure required keys exist in default config
        assert validate_config() is True

    def test_validate_config_missing_keys(self):
        """Test configuration validation with missing keys"""
        # Reset singleton and create empty config
        ConfigManager._instance = None
        ConfigManager._config = {}

        config = get_config()
        config._config = {}  # Empty config

        assert validate_config() is False


class TestConfigurationDataclasses:
    """Test configuration dataclasses"""

    def test_clustering_config_dataclass(self):
        """Test ClusteringConfig dataclass"""
        from src.utils.config import ClusteringConfig

        config = ClusteringConfig()
        assert config.algorithm == "hdbscan"
        assert config.min_cluster_size == 2
        assert config.similarity_threshold == 0.6

        # Test with custom values
        custom_config = ClusteringConfig(
            algorithm="dbscan", min_cluster_size=5, similarity_threshold=0.8
        )
        assert custom_config.algorithm == "dbscan"
        assert custom_config.min_cluster_size == 5
        assert custom_config.similarity_threshold == 0.8

    def test_processing_config_dataclass(self):
        """Test ProcessingConfig dataclass"""
        from src.utils.config import ProcessingConfig

        config = ProcessingConfig()
        assert config.video_frame_skip == 30
        assert config.min_face_size == 40
        assert config.batch_size == 32
        assert config.enable_gpu is True
