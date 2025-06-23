#!/usr/bin/env python3
"""
Test face clustering by appending new images to existing clusters.
This helps verify if different faces are incorrectly grouped together.
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors.image_processor import ImageProcessor


def append_to_existing_test(image_path: Path, existing_test_dir: Path):
    """Process a new image using existing cluster registry"""
    print(f"\n🖼️  Processing additional image: {image_path}")
    print(f"📁 Using existing test directory: {existing_test_dir}")
    print("-" * 50)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    if not existing_test_dir.exists():
        print(f"❌ Test directory not found: {existing_test_dir}")
        return None
    
    # Read the current cluster registry state
    registry_path = Path("data/cluster_registry.json")
    print(f"\n📋 Current cluster registry state:")
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            total_clusters = len(registry.get('clusters', {}))
            total_chips = sum(
                cluster.get('chip_count', 0) 
                for cluster in registry.get('clusters', {}).values()
            )
            print(f"   Total clusters: {total_clusters}")
            print(f"   Total face chips: {total_chips}")
    
    try:
        # Process image with clustering (using existing registry)
        processor = ImageProcessor(enable_clustering=True)
        result = processor.process_image(
            image_path=image_path,
            output_dir=str(existing_test_dir / "chips"),
            save_chips=True
        )
        
        # Save results with image name prefix
        result_file = existing_test_dir / f"{image_path.stem}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n📄 Results saved to: {result_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_combined_results(test_dir: Path, image_results: dict):
    """Analyze clustering results across multiple images"""
    print("\n📊 Combined Clustering Analysis")
    print("-" * 50)
    
    # Read current registry state
    registry_path = Path("data/cluster_registry.json")
    if not registry_path.exists():
        print("❌ No cluster registry found")
        return
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    clusters = registry.get('clusters', {})
    
    # Analyze cluster growth
    cluster_sizes = defaultdict(int)
    cluster_images = defaultdict(set)
    
    # Go through all chips in the test directory
    chips_dir = test_dir / "chips"
    if chips_dir.exists():
        for cluster_dir in chips_dir.iterdir():
            if cluster_dir.is_dir():
                cluster_id = cluster_dir.name
                chip_count = len(list(cluster_dir.glob("*.jpg")))
                cluster_sizes[cluster_id] = chip_count
    
    # Analyze results from each image
    for image_name, result in image_results.items():
        if result:
            chips = result.get('metadata', {}).get('chips', [])
            for chip in chips:
                cluster_id = chip.get('clusterId', 'unknown')
                cluster_images[cluster_id].add(image_name)
    
    # Report findings
    print(f"\n📋 Cluster Summary:")
    print(f"Total clusters: {len(cluster_sizes)}")
    
    # Find clusters with faces from multiple images
    multi_image_clusters = [
        (cluster_id, images) 
        for cluster_id, images in cluster_images.items() 
        if len(images) > 1
    ]
    
    if multi_image_clusters:
        print(f"\n⚠️  Clusters containing faces from multiple images:")
        for cluster_id, images in multi_image_clusters:
            print(f"\n  {cluster_id}: {cluster_sizes[cluster_id]} faces from {len(images)} images")
            for img in images:
                print(f"    - {img}")
    else:
        print("\n✅ No clusters contain faces from multiple images")
    
    # Detailed cluster distribution
    print(f"\n📊 Detailed Cluster Distribution:")
    for cluster_id in sorted(cluster_sizes.keys()):
        size = cluster_sizes[cluster_id]
        images = cluster_images.get(cluster_id, set())
        print(f"  {cluster_id}: {size} faces from {images}")
    
    # Generate combined analysis report
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'total_clusters': len(cluster_sizes),
        'cluster_distribution': dict(cluster_sizes),
        'cluster_sources': {
            cluster_id: list(images) 
            for cluster_id, images in cluster_images.items()
        },
        'multi_image_clusters': [
            {
                'cluster_id': cluster_id,
                'face_count': cluster_sizes[cluster_id],
                'source_images': list(images)
            }
            for cluster_id, images in multi_image_clusters
        ],
        'potential_misclassifications': len(multi_image_clusters)
    }
    
    # Save combined analysis
    analysis_file = test_dir / "combined_clustering_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n📄 Combined analysis saved to: {analysis_file}")
    
    # Create summary
    if multi_image_clusters:
        print(f"\n⚠️  ATTENTION: Found {len(multi_image_clusters)} clusters with faces from different images!")
        print("These may represent misclassifications where different people were grouped together.")
    else:
        print(f"\n✅ Clustering appears to be working correctly - no cross-image groupings detected!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test clustering by appending new images to existing test session"
    )
    parser.add_argument(
        "--image",
        default="data/input/Photos/group.jpg",
        help="Path to additional test image"
    )
    parser.add_argument(
        "--test-dir",
        default="clustering_tests/test_20250622_231308",
        help="Existing test directory to append to"
    )
    
    args = parser.parse_args()
    
    print("🧪 Face Clustering Append Test")
    print("=" * 50)
    
    test_dir = Path(args.test_dir)
    image_path = Path(args.image)
    
    # Process the new image
    result = append_to_existing_test(image_path, test_dir)
    
    if result:
        # Collect all results
        image_results = {}
        
        # Add original result if exists
        original_result_path = test_dir / "raw_result.json"
        if original_result_path.exists():
            with open(original_result_path, 'r') as f:
                image_results['people-collage-design.jpg'] = json.load(f)
        
        # Add new result
        image_results[image_path.name] = result
        
        # Analyze combined results
        analyze_combined_results(test_dir, image_results)
        
        print("\n✅ Test completed!")
        print(f"📁 All results in: {test_dir}")
    else:
        print("\n❌ Test failed - please check the error messages above")


if __name__ == "__main__":
    main()