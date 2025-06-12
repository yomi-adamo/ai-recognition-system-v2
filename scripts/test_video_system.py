#!/usr/bin/env python3
"""
Test the video processing system with sample video
"""

import sys
import numpy as np
from pathlib import Path
import cv2
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processors.video_processor import VideoProcessor
from src.utils.logger import setup_logger
from src.utils.config import get_config


def create_test_video_with_faces():
    """Create a simple test video with moving face-like patterns"""
    
    # Video parameters
    width, height = 640, 480
    fps = 24
    duration_seconds = 10
    total_frames = fps * duration_seconds
    
    # Create video writer
    video_path = Path("data/input/smriti.mp4")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    print(f"Creating test video: {video_path}")
    print(f"Duration: {duration_seconds}s, {total_frames} frames at {fps} FPS")
    
    for frame_num in range(total_frames):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray
        
        # Add some background pattern
        cv2.rectangle(frame, (0, 0), (width//2, height//2), (220, 220, 220), -1)
        cv2.rectangle(frame, (width//2, height//2), (width, height), (220, 220, 220), -1)
        
        # Calculate time-based positions for moving faces
        t = frame_num / fps
        
        # Face 1: Moving horizontally
        face1_x = int(100 + 200 * np.sin(t * 0.5))
        face1_y = 150
        draw_simple_face(frame, face1_x, face1_y, 40)
        
        # Face 2: Moving vertically (appears after 3 seconds)
        if t > 3.0:
            face2_x = 400
            face2_y = int(200 + 100 * np.cos(t * 0.8))
            draw_simple_face(frame, face2_x, face2_y, 35)
        
        # Face 3: Static face (appears after 6 seconds)
        if t > 6.0:
            face3_x = 500
            face3_y = 100
            draw_simple_face(frame, face3_x, face3_y, 45)
        
        # Scene change at 5 seconds (change background color)
        if t > 5.0:
            frame[:, :] = frame[:, :] * 0.7 + np.array([50, 100, 150]) * 0.3
        
        # Add frame number text
        cv2.putText(frame, f"Frame {frame_num:04d} | Time {t:.1f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        
        out.write(frame)
        
        # Progress
        if frame_num % 50 == 0:
            progress = frame_num / total_frames * 100
            print(f"Progress: {progress:.1f}%")
    
    out.release()
    print(f"✓ Test video created: {video_path}")
    return video_path


def draw_simple_face(frame, center_x, center_y, radius):
    """Draw a simple face pattern"""
    # Face outline
    cv2.circle(frame, (center_x, center_y), radius, (200, 180, 160), -1)
    cv2.circle(frame, (center_x, center_y), radius, (150, 130, 110), 2)
    
    # Eyes
    eye_offset = radius // 3
    eye_size = radius // 8
    cv2.circle(frame, (center_x - eye_offset, center_y - eye_offset//2), 
               eye_size, (50, 50, 50), -1)
    cv2.circle(frame, (center_x + eye_offset, center_y - eye_offset//2), 
               eye_size, (50, 50, 50), -1)
    
    # Nose
    nose_size = radius // 10
    cv2.circle(frame, (center_x, center_y), nose_size, (150, 120, 100), -1)
    
    # Mouth
    mouth_width = radius // 2
    mouth_height = radius // 6
    cv2.ellipse(frame, (center_x, center_y + eye_offset), 
               (mouth_width, mouth_height), 0, 0, 180, (100, 50, 50), -1)


def test_video_processing():
    """Test the video processing pipeline"""
    logger = setup_logger("test_video_system", level="INFO")
    
    print("Testing Video Processing System")
    print("=" * 50)
    
    # Create test video
    print("1. Creating test video...")
    try:
        video_path = create_test_video_with_faces()
    except Exception as e:
        print(f"   ✗ Failed to create test video: {e}")
        return False
    
    # Test video processor
    print("\n2. Testing VideoProcessor...")
    try:
        processor = VideoProcessor()
        
        # Set faster processing for testing
        processor.frame_interval = 12  # Process every 12th frame (2 FPS effective)
        processor.scene_change_threshold = 25.0
        processor.max_faces_per_frame = 10
        
        print(f"   Frame interval: {processor.frame_interval}")
        print(f"   Scene threshold: {processor.scene_change_threshold}%")
        
        # Process the video
        result = processor.process_video(video_path, save_chips=False)
        
        stats = result['processing_stats']
        print(f"   ✓ Video processed successfully")
        print(f"     Total frames: {stats['total_frames_in_video']}")
        print(f"     Frames processed: {stats['frames_processed']}")
        print(f"     Face detections: {stats['total_face_detections']}")
        print(f"     Unique faces: {stats['unique_faces']}")
        print(f"     Duration: {stats['duration_seconds']:.1f}s")
        
    except Exception as e:
        print(f"   ✗ Video processing failed: {e}")
        return False
    
    # Test JSON generation
    print("\n3. Testing JSON generation...")
    try:
        # Timeline JSON
        timeline = processor.create_timeline_json(result)
        print(f"   ✓ Timeline JSON: {len(timeline)} entries")
        
        # Unique faces JSON
        unique_faces = processor.get_unique_faces_json(result)
        print(f"   ✓ Unique faces JSON: {len(unique_faces)} entries")
        
        # Validate JSON structure
        if len(timeline) > 0:
            sample = timeline[0]
            required_fields = ["file", "type", "name", "metadata", "topics"]
            missing = [f for f in required_fields if f not in sample]
            
            if missing:
                print(f"   ✗ Missing JSON fields: {missing}")
                return False
            
            metadata = sample["metadata"]
            video_fields = ["timestamp", "video_timestamp", "frame_number", "confidence"]
            missing_meta = [f for f in video_fields if f not in metadata]
            
            if missing_meta:
                print(f"   ✗ Missing metadata fields: {missing_meta}")
                return False
            
            print(f"   ✓ JSON structure valid")
            print(f"     Sample entry: frame {metadata['frame_number']}, "
                  f"confidence {metadata['confidence']:.2%}")
        
    except Exception as e:
        print(f"   ✗ JSON generation failed: {e}")
        return False
    
    return True


def test_video_script():
    """Test the process_video.py script"""
    print("\n4. Testing process_video.py script...")
    
    video_path = Path("data/input/test_video.mp4")
    if not video_path.exists():
        print("   ✗ Test video not found")
        return False
    
    try:
        import subprocess
        
        # Test script execution
        result = subprocess.run([
            sys.executable, "scripts/process_video.py",
            str(video_path),
            "--frame-interval", "24",  # Process every 24th frame for speed
            "--json-only",
            "--unique-only",
            "--no-progress"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent, timeout=60)
        
        if result.returncode == 0:
            print("   ✓ process_video.py executed successfully")
            
            # Try to parse JSON output
            import json
            output_data = json.loads(result.stdout)
            print(f"   ✓ Valid JSON output with {len(output_data)} faces")
            
            if len(output_data) > 0:
                sample = output_data[0]
                print(f"     Sample: {sample['name']}, "
                      f"frame {sample['metadata']['frame_number']}")
        else:
            print(f"   ✗ process_video.py failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ✗ Script execution timed out")
        return False
    except Exception as e:
        print(f"   ✗ Script test failed: {e}")
        return False
    
    return True


def main():
    """Run all video tests"""
    print("Facial Vision Video Processing Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Test video processing
    if not test_video_processing():
        all_passed = False
    
    # Test script
    if not test_video_script():
        all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VIDEO TESTS PASSED! Video processing is ready to use.")
        print("\nNext steps:")
        print("1. Add real videos to data/input/")
        print("2. Run: python scripts/process_video.py data/input/your_video.mp4")
        print("3. Use --timeline for full detection timeline")
        print("4. Use --unique-only for deduplicated faces")
        print("\nExample commands:")
        print("  # Basic processing")
        print("  python scripts/process_video.py video.mp4")
        print()
        print("  # Fast processing (every 60th frame)")
        print("  python scripts/process_video.py video.mp4 --frame-interval 60")
        print()
        print("  # Timeline with all detections")
        print("  python scripts/process_video.py video.mp4 --timeline")
    else:
        print("❌ SOME VIDEO TESTS FAILED. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure OpenCV is properly installed")
        print("2. Check video codec support")
        print("3. Verify face_recognition dependencies")
    
    print("=" * 60)


if __name__ == "__main__":
    main()