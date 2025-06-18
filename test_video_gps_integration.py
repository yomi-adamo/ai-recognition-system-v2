#!/usr/bin/env python3
"""
Test script for video processing with frame-specific GPS integration
"""
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.processors.video_processor import VideoProcessor
from src.outputs.json_formatter import JSONFormatter
from src.utils.logger import get_facial_vision_logger

logger = get_facial_vision_logger(__name__)

def test_video_gps_integration(video_path: str, output_dir: str = "test_gps_output", max_duration: float = 30.0):
    """Test video processing with GPS integration"""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Testing video processing with GPS integration: {video_path}")
    print(f"Processing duration limited to: {max_duration} seconds")
    print("=" * 60)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Initialize video processor
    processor = VideoProcessor(enable_clustering=True)
    
    # Temporarily reduce frame interval for testing to limit processing time
    original_frame_interval = processor.frame_interval
    processor.frame_interval = 60  # Process every 60th frame instead of every 30th
    print(f"   - Adjusted frame interval to {processor.frame_interval} for faster testing")
    
    # Process video
    print("1. Processing video...")
    try:
        result = processor.process_video(
            video_path=video_path,
            output_dir=output_dir,
            save_chips=True,
            enable_clustering=True
        )
        
        print(f"   ✓ Video processed successfully!")
        print(f"   - Faces detected: {result['metadata']['processing_stats']['total_face_detections']}")
        print(f"   - Chips generated: {result['metadata']['processing_stats']['chips_generated']}")
        
        # Check for GPS in chips
        chips = result['metadata'].get('chips', [])
        gps_chips_count = 0
        
        print("\n2. Checking GPS in generated chips...")
        for i, chip in enumerate(chips):
            if 'gps' in chip:
                gps_chips_count += 1
                if i < 3:  # Show first 3 as examples
                    print(f"   ✓ Chip {i+1}: GPS = {chip['gps']}")
                    if 'videoTimestamp' in chip:
                        print(f"     - Video timestamp: {chip['videoTimestamp']}")
        
        if gps_chips_count > 0:
            print(f"   ✓ {gps_chips_count}/{len(chips)} chips have GPS coordinates")
        else:
            print("   ⚠ No chips have GPS coordinates")
        
        # Convert to blockchain format
        print("\n3. Converting to blockchain JSON format...")
        formatter = JSONFormatter()
        blockchain_json = formatter.create_blockchain_asset_json(result)
        
        # Save blockchain JSON
        json_output_path = output_dir / "blockchain_format_with_gps.json"
        with open(json_output_path, 'w') as f:
            json.dump(blockchain_json, f, indent=2)
        
        print(f"   ✓ Blockchain JSON saved to: {json_output_path}")
        
        # Check GPS in blockchain format
        if 'metadata' in blockchain_json and 'chips' in blockchain_json['metadata']:
            blockchain_chips = blockchain_json['metadata']['chips']
            gps_blockchain_chips = sum(1 for chip in blockchain_chips if 'gps' in chip)
            print(f"   ✓ {gps_blockchain_chips}/{len(blockchain_chips)} chips have GPS in blockchain format")
        
        # Show sample chip with GPS
        for chip in blockchain_chips:
            if 'gps' in chip:
                print(f"\n   Sample chip with GPS:")
                print(f"   - File: {chip.get('file', 'unknown')}")
                print(f"   - Cluster: {chip.get('clusterId', 'unknown')}")
                print(f"   - GPS: {chip['gps']}")
                if 'videoTimestamp' in chip:
                    print(f"   - Video timestamp: {chip['videoTimestamp']}")
                break
        
    except Exception as e:
        print(f"   ✗ Error processing video: {e}")
        logger.error(f"Video processing failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Video GPS integration test completed")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video_gps_integration.py <video_file> [output_dir]")
        print("\nExample:")
        print("  python test_video_gps_integration.py data/input/sample_video.mp4")
        print("  python test_video_gps_integration.py data/input/sample_video.mp4 my_output")
        sys.exit(1)
    
    video_file = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "test_gps_output"
    
    test_video_gps_integration(video_file, output_directory)