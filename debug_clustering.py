#!/usr/bin/env python3
"""
Debug clustering issues - check why faces are all assigned to same cluster
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.face_clusterer import FaceClusterer, ClusterManager
from src.utils.config import get_config


def debug_clustering_config():
    """Debug clustering configuration"""
    print("🔍 Debugging Clustering Configuration")
    print("=" * 50)
    
    config = get_config()
    clustering_config = config.get_clustering_config()
    
    print("Current clustering config:")
    for key, value in clustering_config.items():
        print(f"  {key}: {value}")
    
    print("\n🧪 Testing ClusterManager with different parameters")
    
    # Test with more permissive settings
    try:
        # Original settings
        cluster_manager = ClusterManager(
            algorithm="hdbscan",
            min_cluster_size=2,
            cluster_selection_epsilon=0.4,
            similarity_threshold=0.6
        )
        print("✅ Original ClusterManager created successfully")
        
        # Test with more permissive settings
        permissive_manager = ClusterManager(
            algorithm="hdbscan",
            min_cluster_size=2,
            cluster_selection_epsilon=0.05,  # Very permissive
            similarity_threshold=0.3,        # Lower threshold (more strict)
            metric="cosine"
        )
        print("✅ Permissive ClusterManager created successfully")
        
        # Test with DBSCAN as alternative
        dbscan_manager = ClusterManager(
            algorithm="dbscan",
            min_cluster_size=2,
            eps=0.5,
            similarity_threshold=0.5
        )
        print("✅ DBSCAN ClusterManager created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating ClusterManager: {e}")
        return False


def test_dummy_embeddings():
    """Test clustering with dummy face embeddings"""
    print("\n🎭 Testing with Dummy Face Embeddings")
    print("=" * 50)
    
    try:
        cluster_manager = ClusterManager(
            algorithm="hdbscan",
            min_cluster_size=2,
            cluster_selection_epsilon=0.1,
            similarity_threshold=0.4
        )
        
        # Create some dummy embeddings representing different faces
        # In practice, these would be 128-dimensional from face_recognition
        embedding_dim = 128
        
        # Person 1 - multiple similar faces
        person1_base = np.random.rand(embedding_dim)
        person1_faces = [
            person1_base + np.random.normal(0, 0.05, embedding_dim),  # slight variation
            person1_base + np.random.normal(0, 0.05, embedding_dim),  # slight variation
            person1_base + np.random.normal(0, 0.05, embedding_dim),  # slight variation
        ]
        
        # Person 2 - multiple similar faces
        person2_base = np.random.rand(embedding_dim) + 2.0  # offset to make different
        person2_faces = [
            person2_base + np.random.normal(0, 0.05, embedding_dim),
            person2_base + np.random.normal(0, 0.05, embedding_dim),
            person2_base + np.random.normal(0, 0.05, embedding_dim),
        ]
        
        # Person 3 - single face (should be noise/outlier)
        person3_faces = [np.random.rand(embedding_dim) + 4.0]
        
        all_embeddings = person1_faces + person2_faces + person3_faces
        
        print(f"Created {len(all_embeddings)} dummy embeddings:")
        print(f"  Person 1: {len(person1_faces)} faces")
        print(f"  Person 2: {len(person2_faces)} faces") 
        print(f"  Person 3: {len(person3_faces)} faces")
        
        # Test clustering
        labels = cluster_manager.fit_predict(all_embeddings)
        
        print(f"\nClustering results:")
        print(f"  Labels: {labels}")
        print(f"  Unique clusters: {len(np.unique(labels[labels >= 0]))}")
        print(f"  Noise points (label -1): {np.sum(labels == -1)}")
        
        # Expected: 2 clusters (person 1 and person 2), 1 noise point (person 3)
        unique_clusters = len(np.unique(labels[labels >= 0]))
        if unique_clusters >= 2:
            print("✅ Clustering working correctly - found multiple clusters")
        else:
            print("⚠️  Clustering issue - only found 1 or 0 clusters")
            
    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")


def suggest_fixes():
    """Suggest configuration fixes"""
    print("\n💡 Suggested Fixes for Clustering Issues")
    print("=" * 50)
    
    print("1. Adjust clustering parameters in config/default.yaml:")
    print("   clustering:")
    print("     cluster_selection_epsilon: 0.1  # More permissive (was 0.4)")
    print("     similarity_threshold: 0.4       # More strict (was 0.6)")
    print("     min_cluster_size: 2             # Keep same")
    print()
    
    print("2. Try different algorithm:")
    print("   clustering:")
    print("     algorithm: 'dbscan'             # Instead of 'hdbscan'")
    print("     eps: 0.4                        # DBSCAN distance threshold")
    print()
    
    print("3. Check embedding quality:")
    print("   - Ensure face_recognition is extracting good embeddings")
    print("   - Verify faces are clearly visible and well-aligned")
    print("   - Consider using 'cnn' model instead of 'hog' for better embeddings")
    print()
    
    print("4. Enable debug logging:")
    print("   logging:")
    print("     level: 'DEBUG'                  # See detailed clustering info")


def main():
    """Run all debugging tests"""
    print("🐛 Facial Vision Clustering Debugger")
    print("=" * 60)
    
    config_ok = debug_clustering_config()
    if config_ok:
        test_dummy_embeddings()
    
    suggest_fixes()
    
    print("\n" + "=" * 60)
    print("🔧 Next steps:")
    print("1. Run this script to verify clustering components")
    print("2. Try the suggested configuration changes")
    print("3. Test with real images again")
    print("4. Check debug logs for clustering details")


if __name__ == "__main__":
    main()