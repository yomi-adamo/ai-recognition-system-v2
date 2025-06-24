#!/usr/bin/env python3
"""
Process video for face detection and generate JSON output with timeline
Usage: python scripts/process_video.py <video_path> [options]
"""

import argparse
import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processors.video_processor import VideoProcessor
from src.utils.file_handler import FileHandler
from src.utils.logger import setup_logger
from src.outputs.json_formatter import JSONFormatter


class ProgressBar:
    """Simple progress bar for video processing"""
    
    def __init__(self, total_length=50):
        self.total_length = total_length
        self.start_time = time.time()
    
    def update(self, progress):
        """Update progress bar (progress from 0.0 to 1.0)"""
        filled_length = int(self.total_length * progress)
        bar = '#' * filled_length + '-' * (self.total_length - filled_length)
        elapsed = time.time() - self.start_time
        
        if progress > 0:
            eta = elapsed / progress * (1 - progress)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        print(f'\rProgress: |{bar}| {progress:.1%} {eta_str}', end='', flush=True)
        
        if progress >= 1.0:
            print()  # New line when complete


def main():
    parser = argparse.ArgumentParser(description='Process video for face detection')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--output-dir', '-o', default='data/output',
                       help='Output directory (default: data/output)')
    parser.add_argument('--frame-interval', '-f', type=int, default=30,
                       help='Process every Nth frame (default: 30)')
    parser.add_argument('--scene-detection', '-s', type=float, default=30.0,
                       help='Scene change detection threshold 0-100 (default: 30.0)')
    parser.add_argument('--max-faces', '-m', type=int, default=20,
                       help='Maximum faces per frame (default: 20)')
    parser.add_argument('--no-save-chips', action='store_true',
                       help='Do not save face chips to disk')
    parser.add_argument('--unique-only', action='store_true',
                       help='Save only unique faces (deduplicated)')
    parser.add_argument('--timeline', action='store_true',
                       help='Generate timeline JSON with all detections')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--json-only', action='store_true',
                       help='Output JSON to stdout only')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    parser.add_argument('--recognize', action='store_true',
                       help='Enable face recognition')
    parser.add_argument('--no-recognize', action='store_true',
                       help='Disable face recognition')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("process_video", level=args.log_level,
                         console=not args.json_only)
    
    try:
        # Validate input
        video_path = Path(args.video_path)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            sys.exit(1)
        
        # Check video format
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        if video_path.suffix.lower() not in video_extensions:
            logger.error(f"Unsupported video format: {video_path.suffix}")
            sys.exit(1)
        
        # Initialize components
        # Determine recognition setting
        enable_recognition = True  # Default
        if args.no_recognize:
            enable_recognition = False
        elif args.recognize:
            enable_recognition = True
            
        processor = VideoProcessor(enable_clustering=enable_recognition)
        file_handler = FileHandler(args.output_dir)
        formatter = JSONFormatter()
        
        # Update processor configuration
        processor.frame_interval = args.frame_interval
        processor.scene_change_threshold = args.scene_detection
        processor.max_faces_per_frame = args.max_faces
        
        # Create output directory
        output_dir = file_handler.create_timestamped_output_dir("video_processing")
        
        if not args.json_only:
            print(f"Processing video: {video_path.name}")
            print(f"Frame interval: every {args.frame_interval} frames")
            print(f"Scene change threshold: {args.scene_detection}%")
            print(f"Max faces per frame: {args.max_faces}")
            print(f"Face recognition: {'enabled' if enable_recognition else 'disabled'}")
            print(f"Output directory: {output_dir}")
            print()
        
        # Setup progress bar
        progress_bar = None
        if not args.no_progress and not args.json_only:
            progress_bar = ProgressBar()
            
        def progress_callback(progress):
            if progress_bar:
                progress_bar.update(progress)
        
        # Process video
        save_chips = not args.no_save_chips
        start_time = time.time()
        
        result = processor.process_video(
            video_path,
            output_dir=output_dir if save_chips else None,
            save_chips=save_chips,
            progress_callback=progress_callback
        )
        
        processing_time = time.time() - start_time
        
        # Generate JSON outputs
        if args.unique_only:
            # Only unique faces
            face_results = processor.get_unique_faces_json(result)
            output_name = "unique_faces"
        else:
            # All detections or timeline
            if args.timeline:
                face_results = processor.create_timeline_json(result)
                output_name = "timeline"
            else:
                face_results = processor.get_unique_faces_json(result)
                output_name = "unique_faces"
        
        if not args.json_only:
            # Save results to files
            json_output_path = output_dir / f"{video_path.stem}_{output_name}.json"
            formatter.save_to_file(face_results, json_output_path)
            
            # Save timeline if requested (in addition to unique faces)
            if args.timeline and not args.unique_only:
                timeline_results = processor.create_timeline_json(result)
                timeline_path = output_dir / f"{video_path.stem}_full_timeline.json"
                formatter.save_to_file(timeline_results, timeline_path)
            
            # Create summary report
            stats = result.get('metadata', {}).get('processing_stats', {})
            stats['processing_time_seconds'] = round(processing_time, 2)
            stats['faces_per_second'] = round(stats.get('total_face_detections', 0) / processing_time, 2) if processing_time > 0 else 0
            
            summary = {
                'video_file': video_path.name,
                'processing_stats': stats,
                'configuration': {
                    'frame_interval': args.frame_interval,
                    'scene_change_threshold': args.scene_detection,
                    'max_faces_per_frame': args.max_faces
                },
                'output_files': {
                    'faces_json': str(json_output_path.relative_to(output_dir)),
                    'timeline_json': f"{video_path.stem}_full_timeline.json" if args.timeline else None
                }
            }
            
            summary_path = output_dir / "video_processing_summary.json"
            file_handler.save_json_output(summary, summary_path)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Video Processing Complete")
            print(f"{'='*60}")
            print(f"Video: {video_path.name}")
            print(f"Duration: {stats.get('duration_seconds', 0):.1f} seconds")
            print(f"Total frames: {stats.get('total_frames_in_video', 0):,}")
            print(f"Frames processed: {stats.get('frames_processed', 0):,}")
            print(f"Frames with faces: {stats.get('frames_with_faces', 0):,}")
            print(f"Total face detections: {stats.get('total_face_detections', 0):,}")
            print(f"Unique faces found: {stats.get('clusters_assigned', 0):,}")
            
            # Show clustering stats if enabled
            if stats.get('clustering_enabled', False):
                print(f"Clustering enabled: Yes")
                print(f"Clusters assigned: {stats.get('clusters_assigned', 0)}")
            
            print(f"Processing time: {processing_time:.1f} seconds")
            print(f"Speed: {stats.get('faces_per_second', 0):.1f} faces/second")
            print()
            print(f"Output directory: {output_dir}")
            print(f"Face results: {json_output_path}")
            
            if args.timeline:
                timeline_path = output_dir / f"{video_path.stem}_full_timeline.json"
                print(f"Timeline: {timeline_path}")
            
            print(f"Summary: {summary_path}")
            
            if save_chips:
                chip_count = len([f for f in output_dir.glob("*.jpg")])
                print(f"Face chips saved: {chip_count}")
        else:
            # JSON-only output
            print(json.dumps(face_results, indent=2))
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()