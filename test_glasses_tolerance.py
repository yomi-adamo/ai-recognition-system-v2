#!/usr/bin/env python3
"""
Test different similarity thresholds to handle glasses/accessories
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from test_clustering_accuracy import ClusteringAccuracyTester


def test_with_lower_threshold(threshold=0.85):
    """Test clustering with lower similarity threshold"""
    print(f"🔧 Testing with similarity_threshold: {threshold}")
    print("=" * 50)
    
    # Read current config
    config_path = Path("config/default.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Store original threshold
    original_threshold = config['clustering']['similarity_threshold']
    print(f"Original threshold: {original_threshold}")
    
    # Update threshold
    config['clustering']['similarity_threshold'] = threshold
    
    # Save temporary config
    temp_config_path = Path("config/glasses_test.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created temporary config: {temp_config_path}")
    print(f"New threshold: {threshold}")
    
    return temp_config_path, original_threshold


def reset_and_test_yomi4():
    """Reset clustering and test only yomi4 to see if it joins person_25"""
    print("\n🔄 Resetting clustering registry...")
    
    # Reset registry
    registry_path = Path("data/cluster_registry.json")
    if registry_path.exists():
        backup_path = Path("data/cluster_registry_before_glasses_test.json")
        registry_path.rename(backup_path)
        print(f"Backed up registry to: {backup_path}")
    
    print("\n📋 Test Plan:")
    print("1. Reset clustering registry")
    print("2. Apply lower similarity threshold")
    print("3. Re-process all yomi images")
    print("4. Check if yomi4 joins the same cluster")
    
    print("\n🚀 To run the test:")
    print("1. Copy the new config:")
    print("   cp config/glasses_test.yaml config/default.yaml")
    print("2. Run the batch test:")
    print("   python3 test_clustering_batch.py")
    print("3. Check if all yomi images are in the same cluster")


def suggest_thresholds():
    """Suggest different threshold values to try"""
    print("\n💡 Recommended Similarity Thresholds to Test:")
    print("=" * 50)
    
    thresholds = [
        (0.85, "More tolerant - good for glasses/accessories"),
        (0.80, "Even more tolerant - handles lighting changes"),
        (0.75, "Very tolerant - may group different people"),
        (0.70, "Most tolerant - risk of false positives")
    ]
    
    for threshold, description in thresholds:
        print(f"  {threshold}: {description}")
    
    print("\n🔧 To test a specific threshold:")
    print("   python3 test_glasses_tolerance.py --threshold 0.85")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test clustering with different similarity thresholds"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold to test (default: 0.85)"
    )
    parser.add_argument(
        "--suggestions-only",
        action="store_true",
        help="Only show threshold suggestions"
    )
    
    args = parser.parse_args()
    
    if args.suggestions_only:
        suggest_thresholds()
        return
    
    # Test with specified threshold
    temp_config, original = test_with_lower_threshold(args.threshold)
    
    reset_and_test_yomi4()
    
    print(f"\n📄 Next Steps:")
    print(f"1. Use the new config file: {temp_config}")
    print(f"2. Test with all yomi images")
    print(f"3. If results are good, update config/default.yaml")
    print(f"4. Original threshold was: {original}")


if __name__ == "__main__":
    main()