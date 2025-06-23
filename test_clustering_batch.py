#!/usr/bin/env python3
"""
Test face clustering with multiple images in batch.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from test_clustering_append import append_to_existing_test, analyze_combined_results


def test_multiple_images(image_paths, test_dir):
    """Test clustering with multiple images"""
    print("🧪 Batch Face Clustering Test")
    print("=" * 50)
    
    test_dir = Path(test_dir)
    results = {}
    
    # Process each image
    for image_path in image_paths:
        image_path = Path(image_path)
        print(f"\n{'='*50}")
        print(f"Processing {image_path.name}")
        
        result = append_to_existing_test(image_path, test_dir)
        if result:
            results[image_path.name] = result
        else:
            print(f"❌ Failed to process {image_path.name}")
    
    # Collect all results including previous ones
    all_results = {}
    
    # Add previous results
    for result_file in test_dir.glob("*_result.json"):
        if result_file.stem != "raw_result":
            image_name = result_file.stem.replace("_result", "") + ".jpg"
            try:
                with open(result_file, 'r') as f:
                    all_results[image_name] = json.load(f)
            except:
                pass
    
    # Add original result if exists
    original_result_path = test_dir / "raw_result.json"
    if original_result_path.exists():
        try:
            with open(original_result_path, 'r') as f:
                all_results['people-collage-design.jpg'] = json.load(f)
        except:
            pass
    
    # Final analysis
    print(f"\n{'='*50}")
    analyze_combined_results(test_dir, all_results)
    
    # Summary
    print(f"\n📊 Batch Processing Summary")
    print(f"{'='*50}")
    print(f"Images processed: {len(results)}")
    print(f"Total images in test: {len(all_results)}")
    
    # Read final cluster state
    registry_path = Path("data/cluster_registry.json")
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            clusters = registry.get('clusters', {})
            
            # Find clusters with multiple faces
            multi_face_clusters = [
                (cid, cluster) for cid, cluster in clusters.items() 
                if cluster.get('chip_count', 0) > 1
            ]
            
            if multi_face_clusters:
                print(f"\n📋 Clusters with multiple faces:")
                for cluster_id, cluster_data in sorted(multi_face_clusters):
                    count = cluster_data.get('chip_count', 0)
                    print(f"  {cluster_id}: {count} faces")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test clustering with multiple images in batch"
    )
    parser.add_argument(
        "--test-dir",
        default="clustering_tests/test_20250622_231308",
        help="Existing test directory"
    )
    
    args = parser.parse_args()
    
    # Define images to process
    images = [
        "data/input/Photos/yomi1.jpg",
        "data/input/Photos/yomi2.jpg", 
        "data/input/Photos/yomi3.jpg",
        "data/input/Photos/yomi4.jpg"
    ]
    
    test_multiple_images(images, args.test_dir)


if __name__ == "__main__":
    main()