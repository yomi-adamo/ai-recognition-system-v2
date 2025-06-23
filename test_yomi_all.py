#!/usr/bin/env python3
"""
Test all yomi images with the new threshold to see clustering behavior
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from test_clustering_append import append_to_existing_test, analyze_combined_results


def main():
    """Test all yomi images"""
    test_dir = Path("clustering_tests_glasses/test_20250622_235509")
    
    yomi_images = [
        "data/input/Photos/yomi1.jpg",
        "data/input/Photos/yomi2.jpg", 
        "data/input/Photos/yomi3.jpg",
        "data/input/Photos/yomi4.jpg",
        "data/input/Photos/yomi5.jpg"
    ]
    
    print("🧪 Testing All Yomi Images with Lower Threshold")
    print("=" * 50)
    print(f"Similarity threshold: 0.85 (was 0.9625)")
    
    results = {}
    
    for image_path in yomi_images:
        image_path = Path(image_path)
        print(f"\n🖼️  Processing {image_path.name}")
        result = append_to_existing_test(image_path, test_dir)
        if result:
            results[image_path.name] = result
    
    # Final analysis
    print(f"\n{'='*50}")
    print("FINAL ANALYSIS")
    
    # Add group.jpg result
    group_result_path = test_dir / "raw_result.json"
    if group_result_path.exists():
        import json
        with open(group_result_path, 'r') as f:
            results['group.jpg'] = json.load(f)
    
    analyze_combined_results(test_dir, results)


if __name__ == "__main__":
    main()