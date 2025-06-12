import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Singleton configuration manager for the application"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.config_dir = Path(__file__).parent.parent.parent / "config"
            self.load_config()
    
    def load_config(self, config_file: str = "default.yaml") -> None:
        """Load configuration from YAML file"""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
        
        # Load environment overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides"""
        # Example: FACIAL_VISION_IPFS_API_ENDPOINT overrides ipfs.api_endpoint
        for key in os.environ:
            if key.startswith("FACIAL_VISION_"):
                config_key = key[14:].lower().replace("_", ".")
                value = os.environ[key]
                
                # Convert to appropriate type
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                
                self._set_nested(config_key, value)
                logger.debug(f"Applied environment override: {config_key} = {value}")
    
    def _set_nested(self, key: str, value: Any) -> None:
        """Set a nested configuration value using dot notation"""
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split(".")
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_face_detection_config(self) -> Dict[str, Any]:
        """Get face detection configuration"""
        return self.get("face_detection", {})
    
    def get_video_processing_config(self) -> Dict[str, Any]:
        """Get video processing configuration"""
        return self.get("video_processing", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.get("output", {})
    
    def get_ipfs_config(self) -> Dict[str, Any]:
        """Get IPFS configuration"""
        return self.get("ipfs", {})
    
    def get_blockchain_config(self) -> Dict[str, Any]:
        """Get blockchain configuration"""
        return self.get("blockchain", {})
    
    def reload(self, config_file: Optional[str] = None) -> None:
        """Reload configuration from file"""
        if config_file:
            self.load_config(config_file)
        else:
            self.load_config()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary"""
        return self._config.copy()
    
    def __str__(self) -> str:
        return f"ConfigManager({self.config_dir})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Convenience function
def get_config() -> ConfigManager:
    """Get the singleton ConfigManager instance"""
    return ConfigManager()