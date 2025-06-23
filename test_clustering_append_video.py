#!/usr/bin/env python3
"""
Test face clustering by appending video files to existing clusters.
This helps verify if faces from videos are correctly clustered with existing image clusters.
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors.video_processor import VideoProcessor


def append_video_to_existing_test(video_path: Path, existing_test_dir: Path):
    """Process a new video using existing cluster registry"""
    print(f"\n🎬 Processing additional video: {video_path}")
    print(f"📁 Using existing test directory: {existing_test_dir}")
    print("-" * 50)
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return None
    
    if not existing_test_dir.exists():
        print(f"❌ Test directory not found: {existing_test_dir}")
        return None
    
    # Check video file extension
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    if video_path.suffix.lower() not in video_extensions:
        print(f"❌ Unsupported video format: {video_path.suffix}")
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
    else:
        print("❌ No cluster registry found!")
        return None
    
    try:
        # Process video with clustering (using existing registry)
        processor = VideoProcessor(enable_clustering=True)
        
        # Use the same chips directory as images (not separate video_chips)
        chips_output_dir = existing_test_dir / "chips"
        chips_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 Processing video frames...")
        print(f"📂 Video chips will be added to existing cluster folders: {chips_output_dir}")
        
        result = processor.process_video(
            video_path=str(video_path),
            output_dir=str(chips_output_dir)
        )
        
        # Save results with video name prefix
        result_file = existing_test_dir / f"{video_path.stem}_video_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n📄 Results saved to: {result_file}")
        
        # Extract cluster assignment info
        timeline = result.get('timeline', [])
        cluster_assignments = []
        faces_detected = 0
        
        for frame_data in timeline:
            frame_faces = frame_data.get('faces', [])
            faces_detected += len(frame_faces)
            for face in frame_faces:
                cluster_id = face.get('clusterId')
                if cluster_id and cluster_id not in cluster_assignments:
                    cluster_assignments.append(cluster_id)
        
        print(f"\n📊 Video Processing Summary:")
        print(f"   Total frames processed: {len(timeline)}")
        print(f"   Total faces detected: {faces_detected}")
        print(f"   Unique clusters assigned: {len(cluster_assignments)}")
        print(f"   Cluster IDs: {cluster_assignments}")
        
        return {
            'video': video_path.name,
            'frames_processed': len(timeline),
            'faces_detected': faces_detected,
            'unique_clusters': len(cluster_assignments),
            'cluster_assignments': cluster_assignments,
            'result': result
        }
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_video_clustering(test_dir: Path, video_results: dict):
    """Analyze video clustering results with existing image clusters"""
    print("\n📊 Video Clustering Analysis")
    print("-" * 50)
    
    # Read current registry state
    registry_path = Path("data/cluster_registry.json")
    if not registry_path.exists():
        print("❌ No cluster registry found")
        return
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    clusters = registry.get('clusters', {})
    
    # Analyze which clusters contain video faces
    video_clusters = set()
    for video_name, result in video_results.items():
        if result:
            video_clusters.update(result.get('cluster_assignments', []))
    
    print(f"\n📋 Video Cluster Summary:")
    print(f"Video faces assigned to {len(video_clusters)} cluster(s): {sorted(video_clusters)}")
    
    # Check for cross-media clustering (video faces in image clusters)
    cross_media_clusters = []
    for cluster_id in video_clusters:
        if cluster_id in clusters:
            chip_count = clusters[cluster_id].get('chip_count', 0)
            cross_media_clusters.append((cluster_id, chip_count))
    
    if cross_media_clusters:
        print(f"\n🔗 Cross-Media Clustering Detected:")
        print("The following clusters contain both image and video faces:")
        for cluster_id, total_chips in cross_media_clusters:
            print(f"   {cluster_id}: {total_chips} total chips (includes video faces)")
        print("\nThis indicates the system successfully matched video faces with existing image clusters!")
    else:
        print(f"\n🆕 All video faces created new clusters (no matches with existing images)")
    
    # Detailed breakdown
    print(f"\n📊 Detailed Video Results:")
    for video_name, result in video_results.items():
        if result:
            print(f"\n  {video_name}:")
            print(f"    Frames: {result['frames_processed']}")
            print(f"    Faces: {result['faces_detected']}")
            print(f"    Clusters: {result['cluster_assignments']}")
    
    # Generate video analysis report
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'total_video_clusters': len(video_clusters),
        'video_cluster_ids': sorted(video_clusters),
        'cross_media_clusters': [
            {
                'cluster_id': cluster_id,
                'total_chips': chip_count
            }
            for cluster_id, chip_count in cross_media_clusters
        ],
        'video_results': video_results
    }
    
    # Save video analysis
    analysis_file = test_dir / "video_clustering_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n📄 Video analysis saved to: {analysis_file}")
    
    # Summary
    if cross_media_clusters:
        print(f"\n✅ SUCCESS: Video faces matched with {len(cross_media_clusters)} existing cluster(s)!")
        print("The clustering system successfully recognized people from videos in existing image clusters.")
    else:
        print(f"\n📝 INFO: All video faces created new clusters.")
        print("This could mean the people in the video don't match any existing image clusters.")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test clustering by appending video files to existing test session"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file to test"
    )
    parser.add_argument(
        "--test-dir",
        default="threshold_test_0.95",
        help="Existing test directory to append to"
    )
    
    args = parser.parse_args()
    
    print("🧪 Video Face Clustering Append Test")
    print("=" * 50)
    
    test_dir = Path(args.test_dir)
    video_path = Path(args.video)
    
    # Process the video
    result = append_video_to_existing_test(video_path, test_dir)
    
    if result:
        # Collect video results
        video_results = {video_path.name: result}
        
        # Analyze video clustering
        analyze_video_clustering(test_dir, video_results)
        
        print("\n✅ Video test completed!")
        print(f"📁 All results in: {test_dir}")
        print(f"🎬 Video chips in: {test_dir}/chips/ (mixed with image chips)")
    else:
        print("\n❌ Video test failed - please check the error messages above")


if __name__ == "__main__":
    main()