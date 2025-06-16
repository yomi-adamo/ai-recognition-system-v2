#!/usr/bin/env python3
"""
Test GPS inclusion in chips and clustering functionality
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors.image_processor import ImageProcessor
from src.processors.video_processor import VideoProcessor


def test_gps_in_chips(test_file: str):
    """Test if GPS coordinates are properly included in face chips"""
    print(f"🌍 Testing GPS inclusion: {test_file}")
    print("-" * 40)
    
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"❌ File not found: {test_file}")
        return False
    
    try:
        # Determine processor type
        if test_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            processor = ImageProcessor(enable_clustering=True)
            result = processor.process_image(
                image_path=test_path,
                output_dir="gps_test_output",
                save_chips=True
            )
        else:
            processor = VideoProcessor(enable_clustering=True)
            result = processor.process_video(
                video_path=test_path,
                output_dir="gps_test_output", 
                save_chips=True
            )
        
        # Check for GPS in main metadata
        main_gps = result.get('metadata', {}).get('GPS')
        print(f"📍 Main GPS coordinates: {main_gps}")
        
        # Check GPS in each chip
        chips = result.get('metadata', {}).get('chips', [])
        print(f"💎 Total chips: {len(chips)}")
        
        gps_chips = 0
        for i, chip in enumerate(chips):
            chip_gps = chip.get('gps')
            if chip_gps:
                gps_chips += 1
                print(f"  Chip {i+1}: GPS = {chip_gps}")
            else:
                print(f"  Chip {i+1}: No GPS data")
        
        print(f"✅ Chips with GPS: {gps_chips}/{len(chips)}")
        
        # Check clustering
        stats = result.get('metadata', {}).get('processing_stats', {})
        faces_detected = stats.get('faces_detected', stats.get('total_face_detections', 0))
        clusters_assigned = stats.get('clusters_assigned', 0)
        
        print(f"👥 Faces detected: {faces_detected}")
        print(f"🔗 Clusters assigned: {clusters_assigned}")
        
        if clusters_assigned == 1 and faces_detected > 1:
            print("⚠️  CLUSTERING ISSUE: All faces assigned to same cluster")
            print("    This might indicate clustering parameters need adjustment")
        elif clusters_assigned > 1:
            print("✅ CLUSTERING OK: Multiple clusters detected")
        
        # Save detailed results for inspection
        output_file = Path("gps_test_output") / f"{test_path.stem}_detailed_result.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"📄 Detailed results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing file: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_clustering_config():
    """Check current clustering configuration"""
    print("🔧 Current Clustering Configuration")
    print("-" * 40)
    
    try:
        from src.utils.config import get_config
        config = get_config()
        clustering_config = config.get_clustering_config()
        
        print("Clustering settings:")
        for key, value in clustering_config.items():
            print(f"  {key}: {value}")
            
        # Check if settings look reasonable
        epsilon = clustering_config.get('cluster_selection_epsilon', 0.4)
        threshold = clustering_config.get('similarity_threshold', 0.6)
        
        if epsilon > 0.3:
            print("⚠️  cluster_selection_epsilon might be too high (> 0.3)")
        if threshold > 0.5:
            print("⚠️  similarity_threshold might be too high (> 0.5)")
            
        print("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error checking config: {e}")
        return False


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GPS and clustering functionality")
    parser.add_argument("--file", help="Test file (image or video)")
    parser.add_argument("--config-only", action="store_true", help="Only check configuration")
    
    args = parser.parse_args()
    
    print("🧪 GPS and Clustering Test")
    print("=" * 50)
    
    # Always check config first
    config_ok = check_clustering_config()
    print()
    
    if args.config_only:
        return
    
    if args.file:
        success = test_gps_in_chips(args.file)
        print()
        print("=" * 50)
        if success:
            print("✅ Test completed - check output files for details")
        else:
            print("❌ Test failed - see errors above")
    else:
        print("Usage examples:")
        print("  python test_gps_and_clustering.py --file /path/to/image.jpg")
        print("  python test_gps_and_clustering.py --file /path/to/video.mp4")
        print("  python test_gps_and_clustering.py --config-only")


if __name__ == "__main__":
    main()