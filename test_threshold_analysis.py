#!/usr/bin/env python3

import sys
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors.image_processor import ImageProcessor
from src.core.chip_generator import ChipGenerator

def test_threshold(threshold_value):
    """Test clustering with a specific threshold"""
    print(f"\n=== Testing threshold {threshold_value} ===")
    
    # Create config with new threshold
    config_path = Path("config/default.yaml")
    backup_path = Path("config/default.yaml.backup")
    
    # Backup original config
    shutil.copy(config_path, backup_path)
    
    try:
        # Load and modify config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['clustering']['similarity_threshold'] = threshold_value
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Clear existing registry
        registry_path = Path("data/cluster_registry.json")
        if registry_path.exists():
            registry_path.unlink()
        
        # Initialize processors
        processor = ImageProcessor()
        chip_generator = ChipGenerator()
        
        # Process the test image
        image_path = "data/input/Photos/people-collage-design.jpg"
        
        # Create test output directory
        test_dir = Path(f"threshold_test_{threshold_value}")
        test_dir.mkdir(exist_ok=True)
        
        print(f"Processing {image_path}...")
        result = processor.process_image(image_path, str(test_dir))
        
        # Count clusters created
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
                cluster_count = len(registry.get('clusters', {}))
                
                # Copy registry to test directory
                shutil.copy(registry_path, test_dir / "cluster_registry.json")
        else:
            cluster_count = 0
            
        print(f"Threshold {threshold_value}: {cluster_count} clusters created")
        
        return {
            'threshold': threshold_value,
            'clusters': cluster_count,
            'test_dir': str(test_dir)
        }
        
    finally:
        # Restore original config
        shutil.copy(backup_path, config_path)
        backup_path.unlink()

def main():
    # Test different thresholds
    thresholds_to_test = [0.95, 0.96, 0.97, 0.98, 0.99]
    
    results = []
    
    for threshold in thresholds_to_test:
        result = test_threshold(threshold)
        if result:
            results.append(result)
    
    # Summary
    print("\n=== THRESHOLD ANALYSIS SUMMARY ===")
    print("Expected: ~24 separate clusters (one per face)")
    print()
    for result in results:
        print(f"Threshold {result['threshold']}: {result['clusters']} clusters")
    
    # Save results
    with open("threshold_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()