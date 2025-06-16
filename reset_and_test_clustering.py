#!/usr/bin/env python3
"""
Reset clustering registry and test with fresh parameters
"""

import sys
import json
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors.video_processor import VideoProcessor
from src.processors.image_processor import ImageProcessor


def reset_clustering_registry():
    """Reset the clustering registry to start fresh"""
    print("🔄 Resetting clustering registry...")
    
    registry_path = Path("data/cluster_registry.json")
    
    if registry_path.exists():
        # Backup existing registry
        backup_path = registry_path.with_suffix('.json.backup')
        registry_path.rename(backup_path)
        print(f"   Backed up existing registry to: {backup_path}")
    
    # Create fresh empty registry
    fresh_registry = {
        "clusters": {},
        "metadata": {
            "last_updated": "",
            "total_clusters": 0,
            "total_chips": 0
        }
    }
    
    registry_path.parent.mkdir(exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(fresh_registry, f, indent=2)
    
    print(f"   Created fresh registry at: {registry_path}")
    print("✅ Registry reset complete")


def test_with_aggressive_clustering(test_file: str):
    """Test clustering with very aggressive separation parameters"""
    print(f"\n🧪 Testing aggressive clustering: {test_file}")
    print("-" * 50)
    
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"❌ File not found: {test_file}")
        return False
    
    try:
        # Reset registry first
        reset_clustering_registry()
        
        # Process with very strict clustering
        if test_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            processor = ImageProcessor(enable_clustering=True)
            result = processor.process_image(
                image_path=test_path,
                output_dir="aggressive_clustering_test",
                save_chips=True
            )
        else:
            processor = VideoProcessor(enable_clustering=True)
            result = processor.process_video(
                video_path=test_path,
                output_dir="aggressive_clustering_test",
                save_chips=True
            )
        
        # Analyze results
        stats = result.get('metadata', {}).get('processing_stats', {})
        faces_detected = stats.get('faces_detected', stats.get('total_face_detections', 0))
        clusters_assigned = stats.get('clusters_assigned', 0)
        
        print(f"📊 Results with aggressive clustering:")
        print(f"   Faces detected: {faces_detected}")
        print(f"   Clusters created: {clusters_assigned}")
        
        # Check cluster distribution
        chips = result.get('metadata', {}).get('chips', [])
        cluster_distribution = {}
        
        for chip in chips:
            cluster_id = chip.get('clusterId', 'unknown')
            cluster_distribution[cluster_id] = cluster_distribution.get(cluster_id, 0) + 1
        
        print(f"   Cluster distribution:")
        for cluster_id, count in sorted(cluster_distribution.items()):
            print(f"     {cluster_id}: {count} faces")
        
        # Success criteria
        if clusters_assigned > 1:
            print("✅ SUCCESS: Multiple clusters created!")
        elif clusters_assigned == 1 and faces_detected <= 3:
            print("⚠️  ACCEPTABLE: Single cluster (few faces)")
        else:
            print("❌ ISSUE: All faces in single cluster")
            
            # Suggest even more aggressive settings
            print("\n💡 Try even more aggressive settings:")
            print("   similarity_threshold: 0.2  # Very strict")
            print("   cluster_selection_epsilon: 0.01  # Very permissive")
            print("   algorithm: 'dbscan'  # Alternative algorithm")
            
        # Save results
        output_file = Path("aggressive_clustering_test") / f"{test_path.stem}_result.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"📄 Results saved to: {output_file}")
        
        return clusters_assigned > 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dbscan_alternative(test_file: str):
    """Test with DBSCAN as alternative clustering algorithm"""
    print(f"\n🔬 Testing DBSCAN alternative: {test_file}")
    print("-" * 50)
    
    # Temporarily modify config to use DBSCAN
    config_path = Path("config/default.yaml")
    
    # Read current config
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Create DBSCAN version
    dbscan_config = config_content.replace(
        'algorithm: "hdbscan"',
        'algorithm: "dbscan"'
    )
    
    # Add eps parameter after similarity_threshold
    dbscan_config = dbscan_config.replace(
        'similarity_threshold: 0.3  # Very strict for better separation',
        'similarity_threshold: 0.3  # Very strict for better separation\n  eps: 0.5  # DBSCAN distance threshold'
    )
    
    # Write temporary config
    temp_config_path = config_path.with_suffix('.dbscan.yaml')
    with open(temp_config_path, 'w') as f:
        f.write(dbscan_config)
    
    print(f"   Created DBSCAN config: {temp_config_path}")
    print("   To use this config permanently, copy it to config/default.yaml")
    
    # Test instructions
    print("\n📋 To test DBSCAN:")
    print(f"   1. cp {temp_config_path} config/default.yaml")
    print(f"   2. python reset_and_test_clustering.py {test_file}")
    print("   3. Check if clustering improves")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reset clustering and test with aggressive parameters")
    parser.add_argument("file", help="Test file (image or video)")
    parser.add_argument("--reset-only", action="store_true", help="Only reset registry")
    parser.add_argument("--dbscan", action="store_true", help="Generate DBSCAN config")
    
    args = parser.parse_args()
    
    print("🔧 Clustering Reset and Test Tool")
    print("=" * 50)
    
    if args.reset_only:
        reset_clustering_registry()
        return
    
    if args.dbscan:
        test_dbscan_alternative(args.file)
        return
    
    # Main test
    success = test_with_aggressive_clustering(args.file)
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Clustering working - multiple clusters created!")
    else:
        print("⚠️  Still having clustering issues")
        print("\n🔧 Next steps:")
        print("1. Try DBSCAN: python reset_and_test_clustering.py --dbscan your_file.mp4")
        print("2. Check if face embeddings are too similar")
        print("3. Consider using CNN model for better face embeddings")


if __name__ == "__main__":
    main()