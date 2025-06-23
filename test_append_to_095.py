#!/usr/bin/env python3

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors.image_processor import ImageProcessor

def append_to_existing_test(image_path, existing_test_dir="threshold_test_0.95"):
    """Append new image processing to existing test directory"""
    
    test_dir = Path(existing_test_dir)
    if not test_dir.exists():
        print(f"❌ Test directory {test_dir} does not exist!")
        return None
    
    print(f"\n🔄 Processing {image_path} and appending to {test_dir}")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Process the image and append to existing test directory
    result = processor.process_image(image_path, str(test_dir))
    
    # Check registry to see cluster assignments
    registry_path = Path("data/cluster_registry.json")
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Copy updated registry to test directory
        shutil.copy(registry_path, test_dir / "cluster_registry.json")
        
        # Find which clusters this image's faces were assigned to
        cluster_assignments = result.get('cluster_assignments', [])
        
        print(f"✅ Processed {Path(image_path).name}")
        print(f"   Faces detected: {len(cluster_assignments)}")
        print(f"   Cluster assignments: {cluster_assignments}")
        
        return {
            'image': Path(image_path).name,
            'faces_detected': len(cluster_assignments),
            'cluster_assignments': cluster_assignments,
            'total_clusters': len(registry.get('clusters', {}))
        }
    
    return None

def main():
    # Test images to append
    test_images = [
        "data/input/Photos/group.jpg",
        "data/input/Photos/yomi1.jpg", 
        "data/input/Photos/yomi2.jpg",
        "data/input/Photos/yomi3.jpg",
        "data/input/Photos/yomi4.jpg",  # Should cluster with others despite glasses
        "data/input/Photos/yomi5.jpg"
    ]
    
    results = []
    
    print("🧪 Testing cluster assignments with 0.95 threshold")
    print("=" * 60)
    
    for image_path in test_images:
        if Path(image_path).exists():
            result = append_to_existing_test(image_path)
            if result:
                results.append(result)
        else:
            print(f"⚠️  Image not found: {image_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CLUSTERING ASSIGNMENT SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(f"{result['image']:<15} → {result['cluster_assignments']}")
    
    # Check yomi clustering specifically
    yomi_results = {r['image']: r['cluster_assignments'] for r in results if 'yomi' in r['image']}
    
    if yomi_results:
        print(f"\n🔍 YOMI CLUSTERING ANALYSIS:")
        print("-" * 40)
        
        # Find all unique clusters assigned to yomi images
        all_yomi_clusters = set()
        for assignments in yomi_results.values():
            all_yomi_clusters.update(assignments)
        
        print(f"Yomi images assigned to {len(all_yomi_clusters)} cluster(s): {sorted(all_yomi_clusters)}")
        
        # Check if all yomi images are in same cluster (should be)
        if len(all_yomi_clusters) == 1:
            print("✅ All yomi images correctly clustered together!")
        else:
            print("❌ Yomi images split across multiple clusters")
            for img, clusters in yomi_results.items():
                print(f"   {img}: {clusters}")
    
    # Save results
    with open("threshold_095_append_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to threshold_095_append_results.json")

if __name__ == "__main__":
    main()