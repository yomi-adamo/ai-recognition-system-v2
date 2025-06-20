#!/usr/bin/env python3
"""
Debug clustering by processing images and showing cluster assignments.
This helps verify if people are correctly grouped together.
"""

import os
import sys
from pathlib import Path
import json
from collections import defaultdict

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from processors.image_processor import ImageProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


def debug_clustering(directory_path: str):
    """Process images and show clustering results"""
    print(f"\n🔍 Debugging Clustering for: {directory_path}")
    print("=" * 60)
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = []
    
    for file in Path(directory_path).iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(str(file))
    
    # Clear cluster registry to start fresh
    cluster_registry_path = "data/cluster_registry.json"
    if os.path.exists(cluster_registry_path):
        print(f"🗑️  Clearing existing cluster registry: {cluster_registry_path}")
        os.remove(cluster_registry_path)
    
    # Process each image and collect results
    all_results = []
    cluster_to_files = defaultdict(list)
    
    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {Path(image_path).name}")
        
        try:
            result = processor.process(image_path, output_dir="debug_output", save_chips=True)
            
            faces = result.get('metadata', {}).get('chips', [])
            if faces:
                print(f"  ✅ Found {len(faces)} face(s)")
                
                # Group by cluster
                image_clusters = defaultdict(int)
                for face in faces:
                    cluster_id = face.get('clusterId', 'unknown')
                    image_clusters[cluster_id] += 1
                    cluster_to_files[cluster_id].append({
                        'file': Path(image_path).name,
                        'confidence': face.get('confidence', 0.0)
                    })
                
                # Show clusters for this image
                for cluster_id, count in image_clusters.items():
                    print(f"  👤 Cluster {cluster_id}: {count} face(s)")
                
                all_results.append({
                    'file': Path(image_path).name,
                    'faces': faces,
                    'clusters': dict(image_clusters)
                })
            else:
                print(f"  ❌ No faces found")
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            print(f"  ❌ Error: {e}")
    
    # Print clustering summary
    print("\n" + "=" * 60)
    print("📊 CLUSTERING SUMMARY")
    print("=" * 60)
    print(f"Total unique clusters: {len(cluster_to_files)}")
    print(f"Average faces per cluster: {sum(len(files) for files in cluster_to_files.values()) / len(cluster_to_files):.2f}")
    
    print("\n📋 Cluster Details:")
    print("-" * 60)
    
    # Sort clusters by number of appearances
    sorted_clusters = sorted(cluster_to_files.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cluster_id, appearances in sorted_clusters:
        print(f"\n🔷 Cluster {cluster_id} ({len(appearances)} appearances):")
        
        # Group by file
        file_groups = defaultdict(list)
        for app in appearances:
            file_groups[app['file']].append(app['confidence'])
        
        for file, confidences in sorted(file_groups.items()):
            avg_confidence = sum(confidences) / len(confidences)
            print(f"   - {file}: {len(confidences)} face(s), avg confidence: {avg_confidence:.3f}")
    
    # Check for potential same-person clusters
    print("\n" + "=" * 60)
    print("🤔 POTENTIAL CLUSTERING ISSUES")
    print("=" * 60)
    
    # Look for files with "yomi" in the name - they should be the same person
    yomi_clusters = defaultdict(list)
    for cluster_id, appearances in cluster_to_files.items():
        for app in appearances:
            if 'yomi' in app['file'].lower():
                yomi_clusters[cluster_id].append(app['file'])
    
    if len(yomi_clusters) > 1:
        print("⚠️  'yomi' images are in different clusters:")
        for cluster_id, files in yomi_clusters.items():
            print(f"   - Cluster {cluster_id}: {', '.join(files)}")
    elif len(yomi_clusters) == 1:
        print("✅ All 'yomi' images are in the same cluster")
    
    # Save detailed results
    results_file = "clustering_debug_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total_images': len(image_files),
                'total_clusters': len(cluster_to_files),
                'avg_faces_per_cluster': sum(len(files) for files in cluster_to_files.values()) / len(cluster_to_files) if cluster_to_files else 0
            },
            'clusters': {k: v for k, v in sorted_clusters},
            'image_results': all_results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Show current clustering configuration
    import yaml
    config_path = "config/default.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        clustering_config = config.get('clustering', {})
        print(f"\n⚙️  Current Clustering Configuration:")
        print(f"   - Algorithm: {clustering_config.get('algorithm')}")
        print(f"   - Similarity Threshold: {clustering_config.get('similarity_threshold')}")
        print(f"   - Min Cluster Size: {clustering_config.get('min_cluster_size')}")
        print(f"   - Metric: {clustering_config.get('metric')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_clustering.py <directory_path>")
        print("Example: python debug_clustering.py data/input/Photos")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    if not os.path.isdir(directory_path):
        print(f"❌ Error: '{directory_path}' is not a valid directory")
        sys.exit(1)
    
    debug_clustering(directory_path)