#!/usr/bin/env python3
"""
Test script for video GPS track extraction functionality
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.metadata_extractor import MetadataExtractor
from src.utils.logger import get_facial_vision_logger

logger = get_facial_vision_logger(__name__)

def test_gps_extraction(video_path: str):
    """Test GPS extraction from a video file"""
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Testing GPS extraction from: {video_path}")
    print("=" * 50)
    
    # Initialize metadata extractor
    extractor = MetadataExtractor()
    
    # Extract general metadata
    print("1. Extracting general video metadata...")
    metadata = extractor.extract_video_metadata(video_path)
    
    # Check for existing GPS in metadata
    existing_gps = metadata.get('gps')
    if existing_gps:
        print(f"   ✓ Found existing GPS in metadata: {existing_gps}")
    else:
        print("   ⚠ No existing GPS found in general metadata")
    
    # Test GPS track extraction
    print("\n2. Extracting GPS track...")
    gps_track = extractor.extract_gps_track_from_video(video_path)
    
    if gps_track:
        print(f"   ✓ GPS track extracted successfully!")
        print(f"   - Source: {gps_track.get('source', 'unknown')}")
        print(f"   - Type: {gps_track.get('type', 'track')}")
        print(f"   - Total points: {gps_track.get('total_points', 0)}")
        
        coordinates = gps_track.get('coordinates', [])
        if coordinates:
            print(f"   - First coordinate: {coordinates[0]}")
            if len(coordinates) > 1:
                print(f"   - Last coordinate: {coordinates[-1]}")
    else:
        print("   ✗ No GPS track found")
    
    # Test frame-specific GPS extraction
    print("\n3. Testing frame-specific GPS extraction...")
    test_timestamps = [0.0, 5.0, 10.0, 30.0]
    
    for timestamp in test_timestamps:
        gps_at_time = extractor.get_gps_at_timestamp(video_path, timestamp)
        if gps_at_time:
            print(f"   ✓ GPS at {timestamp}s: {gps_at_time}")
        else:
            print(f"   ⚠ No GPS found at {timestamp}s")
    
    print("\n" + "=" * 50)
    print("GPS extraction test completed")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_gps_extraction.py <video_file>")
        print("\nExample:")
        print("  python test_gps_extraction.py data/input/sample_video.mp4")
        sys.exit(1)
    
    test_gps_extraction(sys.argv[1])