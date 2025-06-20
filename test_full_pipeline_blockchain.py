#!/usr/bin/env python3
"""
Test the complete facial-vision pipeline with blockchain integration.
This script processes an image/video through face detection, clustering,
and uploads both the original file and analysis results to the blockchain.
"""

import asyncio
import os
import sys
import json
import tempfile
import zipfile
from pathlib import Path
import argparse
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.blockchain_integration import BlockchainIntegration
from processors.image_processor import ImageProcessor
from processors.video_processor import VideoProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


async def process_and_upload_image(image_path: str) -> dict:
    """Process an image and upload to blockchain"""
    print(f"\n📸 Processing Image: {image_path}")
    print("=" * 60)
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Process the image with chip saving enabled
    print("🔍 Detecting faces...")
    result = processor.process(image_path, output_dir="test_output", save_chips=True)
    
    faces = result.get('metadata', {}).get('chips', [])
    if not faces:
        print("❌ No faces detected in image")
        return None
    
    print(f"✅ Detected {len(faces)} faces")
    
    # Print cluster information
    clusters = {}
    for face in faces:
        cluster_id = face.get('clusterId', 'unknown')
        clusters[cluster_id] = clusters.get(cluster_id, 0) + 1
    
    print(f"👥 Found {len(clusters)} unique person(s):")
    for cluster_id, count in clusters.items():
        print(f"   - {cluster_id}: {count} face(s)")
    
    # Upload to blockchain
    async with BlockchainIntegration() as blockchain:
        # Check if file already exists
        file_hash = blockchain._calculate_file_hash(image_path)
        existing = await blockchain.check_asset_exists_by_hash(file_hash)
        
        if existing:
            print(f"\n⚠️  Image already on blockchain: {existing['id']}")
            parent_id = existing['id']
        else:
            # Upload original image
            print(f"\n📤 Uploading original image to blockchain...")
            original_result = await blockchain.upload_original_asset(
                file_path=image_path,
                asset_type="image",
                name=f"Facial Vision - {Path(image_path).name}",
                author="facial-vision-system",
                metadata={
                    "processedBy": "facial-vision v1.0",
                    "faces_detected": len(faces),
                    "clusters_found": len(clusters),
                    **result.get('metadata', {})
                },
                topics=["facial_vision", "face_detection", "image"]
            )
            parent_id = original_result['assetId']
            print(f"✅ Original asset uploaded: {parent_id}")
        
        # Upload analysis results
        print(f"\n📤 Uploading face analysis results...")
        
        # Prepare chip metadata and create chip bundle
        chip_metadata = []
        chip_bundle_path = None
        
        # Check if we have actual chip files to upload
        chip_files_exist = any(face.get('file') and os.path.exists(face.get('file', '')) for face in faces)
        
        if chip_files_exist:
            import tempfile
            import zipfile
            
            # Create temporary zip file for chip bundle
            temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            chip_bundle_path = temp_zip.name
            temp_zip.close()
            
            print(f"📦 Creating chip bundle organized by clusters...")
            
            with zipfile.ZipFile(chip_bundle_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Create manifest
                manifest = {
                    "version": "1.0",
                    "source_file": image_path,
                    "created": datetime.utcnow().isoformat() + "Z",
                    "clusters": {}
                }
                
                # Group chips by cluster and add to zip
                cluster_counts = {}
                for face in faces:
                    cluster_id = face.get('clusterId', 'unknown')
                    chip_file = face.get('file', '')
                    
                    if chip_file and os.path.exists(chip_file):
                        # Count chips per cluster
                        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                        chip_count = cluster_counts[cluster_id]
                        
                        # Create cluster directory structure in zip
                        zip_path = f"{cluster_id}/chip_{chip_count:03d}.jpg"
                        
                        # Add chip to zip
                        zip_file.write(chip_file, zip_path)
                        
                        # Update manifest
                        if cluster_id not in manifest["clusters"]:
                            manifest["clusters"][cluster_id] = []
                        manifest["clusters"][cluster_id].append({
                            "file": zip_path,
                            "original_path": chip_file,
                            "confidence": face.get('confidence', 0.0),
                            "face_bounds": face.get('face_bounds', {})
                        })
                        
                        # Update chip metadata with zip path
                        chip_data = {
                            "file": zip_path,  # Path within the zip
                            "type": "face_chip",
                            "clusterId": cluster_id,
                            "timestamp": face.get('timestamp', ''),
                            "face_bounds": face.get('face_bounds', {}),
                            "confidence": face.get('confidence', 0.0)
                        }
                        chip_metadata.append(chip_data)
                
                # Add manifest to zip
                manifest_json = json.dumps(manifest, indent=2)
                zip_file.writestr("manifest.json", manifest_json)
            
            print(f"✅ Chip bundle created with {len(clusters)} clusters:")
            for cluster_id, count in cluster_counts.items():
                print(f"   - {cluster_id}: {count} chip(s)")
        
        else:
            # No chip files, create metadata-only entries
            for face in faces:
                chip_data = {
                    "file": face.get('file', ''),
                    "type": "face_chip",
                    "clusterId": face.get('clusterId', 'unknown'),
                    "timestamp": face.get('timestamp', ''),
                    "face_bounds": face.get('face_bounds', {}),
                    "confidence": face.get('confidence', 0.0)
                }
                chip_metadata.append(chip_data)
        
        # Prepare analysis metadata including GPS if available
        analysis_metadata = {
            "sourceFile": image_path,
            "processingTime": result.get('processing_time', 0),
            "algorithmVersion": "1.0",
            "clusters": clusters
        }
        
        # Add GPS data if available
        gps_data = result.get('metadata', {}).get('GPS')
        if gps_data:
            analysis_metadata["GPS"] = gps_data
            print(f"📍 Including GPS coordinates: {gps_data['lat']:.5f}, {gps_data['lon']:.5f}")
        
        # Add device info if available
        device_info = result.get('metadata', {}).get('device')
        if device_info:
            analysis_metadata["device"] = device_info
        
        analysis_result = await blockchain.upload_face_analysis_results(
            parent_asset_id=parent_id,
            face_chips_metadata=chip_metadata,
            analysis_metadata=analysis_metadata,
            chip_bundle_path=chip_bundle_path
        )
        
        # Clean up temporary chip bundle
        if chip_bundle_path and os.path.exists(chip_bundle_path):
            os.unlink(chip_bundle_path)
        
        if chip_bundle_path:
            print(f"✅ Analysis + chip bundle uploaded: {analysis_result['assetId']}")
            print(f"📁 IPFS CID: {analysis_result.get('ipfsCid', 'N/A')}")
        else:
            print(f"✅ Analysis uploaded: {analysis_result['assetId']}")
        
        # Get provenance chain
        print(f"\n🔗 Retrieving provenance chain...")
        provenance = await blockchain.get_asset_provenance(analysis_result['assetId'])
        if provenance:
            print(f"✅ Provenance chain verified")
            print(f"   - Root: {provenance.get('origin', {}).get('id', 'N/A')}")
            print(f"   - Derivations: {len(provenance.get('derivations', []))}")
        
        return {
            "original_asset_id": parent_id,
            "analysis_asset_id": analysis_result['assetId'],
            "faces_detected": len(faces),
            "clusters_found": len(clusters),
            "provenance": provenance
        }


async def process_and_upload_video(video_path: str) -> dict:
    """Process a video and upload to blockchain"""
    print(f"\n🎥 Processing Video: {video_path}")
    print("=" * 60)
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Process the video
    print("🔍 Processing video frames...")
    print("⏳ This may take a while for longer videos...")
    
    try:
        result = processor.process(video_path, output_dir="test_output", save_chips=True)
        
        faces = result.get('metadata', {}).get('chips', [])
        if not faces:
            print("❌ No faces detected in video")
            return None
        
        print(f"✅ Processed {result.get('metadata', {}).get('processing_stats', {}).get('frames_processed', 0)} frames")
        print(f"✅ Detected {len(faces)} total face appearances")
        
        # Get cluster information
        clusters = {}
        for face in faces:
            cluster_id = face.get('clusterId', 'unknown')
            clusters[cluster_id] = clusters.get(cluster_id, 0) + 1
        
        print(f"👥 Found {len(clusters)} unique person(s):")
        for cluster_id, count in clusters.items():
            print(f"   - {cluster_id}: {count} appearance(s)")
        
        # Upload to blockchain
        async with BlockchainIntegration() as blockchain:
            # Check if file already exists
            file_hash = blockchain._calculate_file_hash(video_path)
            existing = await blockchain.check_asset_exists_by_hash(file_hash)
            
            if existing:
                print(f"\n⚠️  Video already on blockchain: {existing['id']}")
                parent_id = existing['id']
            else:
                # Upload original video
                print(f"\n📤 Uploading original video to blockchain...")
                print("⏳ This may take a while for large videos...")
                
                original_result = await blockchain.upload_original_asset(
                    file_path=video_path,
                    asset_type="video",
                    name=f"Facial Vision - {Path(video_path).name}",
                    author="facial-vision-system",
                    metadata={
                        "processedBy": "facial-vision v1.0",
                        "frames_processed": result.get('metadata', {}).get('processing_stats', {}).get('frames_processed', 0),
                        "faces_detected": len(faces),
                        "clusters_found": len(clusters),
                        "duration": result.get('metadata', {}).get('duration', 'unknown'),
                        "fps": result.get('metadata', {}).get('fps', 0),
                        **result.get('metadata', {})
                    },
                    topics=["facial_vision", "face_detection", "video"]
                )
                parent_id = original_result['assetId']
                print(f"✅ Original asset uploaded: {parent_id}")
            
            # Upload analysis results
            print(f"\n📤 Uploading face analysis results...")
            
            # Create chip bundle with actual face chip files
            chip_bundle_path = None
            chip_metadata = []
            
            # Look for chip files in test_output directory
            output_dir = Path("test_output")
            if output_dir.exists():
                # Create temporary chip bundle zip
                chip_bundle_path = tempfile.mktemp(suffix=".zip")
                cluster_counts = {}
                
                with zipfile.ZipFile(chip_bundle_path, 'w') as zip_file:
                    # Prepare manifest data
                    manifest = {
                        "source_video": str(video_path),
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "total_faces": len(faces),
                        "clusters": {},
                        "chips": []
                    }
                    
                    for face in faces:
                        cluster_id = face.get('clusterId', 'unknown')
                        chip_filename = face.get('file', '')
                        
                        if chip_filename:
                            # Handle full path vs relative path
                            if chip_filename.startswith('test_output/'):
                                # Remove redundant test_output prefix if present
                                chip_filename = chip_filename[len('test_output/'):]
                            
                            # Look for the actual chip file - handle video nested structure bug
                            possible_paths = [
                                output_dir / chip_filename,  # Direct path like test_output/person_20/chip_001.jpg
                                output_dir / cluster_id / Path(chip_filename).name,  # In cluster dir
                                output_dir / cluster_id / chip_filename,  # Full path in cluster dir
                                # Handle video processor bug that creates nested dirs
                                output_dir / cluster_id / Path(chip_filename).name / cluster_id / Path(chip_filename).name,
                            ]
                            
                            chip_path = None
                            for path in possible_paths:
                                if path.exists() and path.is_file():
                                    chip_path = path
                                    break
                            
                            if chip_path:
                                # Add chip to zip with cluster directory structure
                                zip_path = f"{cluster_id}/{Path(chip_filename).name}"
                                zip_file.write(chip_path, zip_path)
                                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                        
                        # Create chip metadata
                        chip_data = {
                            "file": f"{cluster_id}/{Path(chip_filename).name}" if chip_filename else f"frame_{face.get('frame_number', 0):06d}_chip.jpg",
                            "type": "face_chip",
                            "clusterId": cluster_id,
                            "timestamp": face.get('timestamp', ''),
                            "frame_number": face.get('frame_number', 0),
                            "time_in_video": face.get('time_in_video', ''),
                            "face_bounds": face.get('face_bounds', {}),
                            "confidence": face.get('confidence', 0.0)
                        }
                        chip_metadata.append(chip_data)
                        manifest["chips"].append(chip_data)
                    
                    # Update manifest clusters
                    manifest["clusters"] = cluster_counts
                    
                    # Add manifest to zip
                    manifest_json = json.dumps(manifest, indent=2)
                    zip_file.writestr("manifest.json", manifest_json)
                
                if cluster_counts:
                    print(f"✅ Video chip bundle created with {len(cluster_counts)} clusters:")
                    for cluster_id, count in cluster_counts.items():
                        print(f"   - {cluster_id}: {count} chip(s)")
                else:
                    # No actual chip files found, remove empty zip
                    os.unlink(chip_bundle_path)
                    chip_bundle_path = None
                    print("⚠️  No chip files found in test_output directory")
            
            # If no chip bundle was created, prepare metadata-only entries
            if not chip_bundle_path:
                for face in faces:
                    chip_data = {
                        "file": face.get('file', ''),
                        "type": "face_chip",
                        "clusterId": face.get('clusterId', 'unknown'),
                        "timestamp": face.get('timestamp', ''),
                        "frame_number": face.get('frame_number', 0),
                        "time_in_video": face.get('time_in_video', ''),
                        "face_bounds": face.get('face_bounds', {}),
                        "confidence": face.get('confidence', 0.0)
                    }
                    chip_metadata.append(chip_data)
            
            # Prepare analysis metadata including GPS if available
            analysis_metadata = {
                "sourceFile": video_path,
                "processingTime": result.get('processing_time', 0),
                "algorithmVersion": "1.0",
                "frames_processed": result.get('metadata', {}).get('processing_stats', {}).get('frames_processed', 0),
                "clusters": clusters
            }
            
            # Add GPS data if available
            gps_data = result.get('metadata', {}).get('GPS')
            if gps_data:
                analysis_metadata["GPS"] = gps_data
                print(f"📍 Including GPS coordinates: {gps_data['lat']:.5f}, {gps_data['lon']:.5f}")
            
            # Add device info if available
            device_info = result.get('metadata', {}).get('device')
            if device_info:
                analysis_metadata["device"] = device_info
            
            analysis_result = await blockchain.upload_face_analysis_results(
                parent_asset_id=parent_id,
                face_chips_metadata=chip_metadata,
                analysis_metadata=analysis_metadata,
                chip_bundle_path=chip_bundle_path
            )
            
            # Clean up temporary chip bundle
            if chip_bundle_path and os.path.exists(chip_bundle_path):
                os.unlink(chip_bundle_path)
            
            if chip_bundle_path:
                print(f"✅ Analysis + chip bundle uploaded: {analysis_result['assetId']}")
                print(f"📁 IPFS CID: {analysis_result.get('ipfsCid', 'N/A')}")
            else:
                print(f"✅ Analysis uploaded: {analysis_result['assetId']}")
            
            # Get provenance chain
            print(f"\n🔗 Retrieving provenance chain...")
            provenance = await blockchain.get_asset_provenance(analysis_result['assetId'])
            if provenance:
                print(f"✅ Provenance chain verified")
                print(f"   - Root: {provenance.get('origin', {}).get('id', 'N/A')}")
                print(f"   - Derivations: {len(provenance.get('derivations', []))}")
            
            return {
                "original_asset_id": parent_id,
                "analysis_asset_id": analysis_result['assetId'],
                "faces_detected": len(faces),
                "clusters_found": len(clusters),
                "frames_processed": result.get('metadata', {}).get('processing_stats', {}).get('frames_processed', 0),
                "provenance": provenance
            }
            
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        logger.error(f"Video processing error: {e}", exc_info=True)
        return None


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process media files through facial-vision and upload to blockchain"
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        help="Path to image or video file to process"
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test with sample files from data/input"
    )
    
    args = parser.parse_args()
    
    print("\n🚀 Facial Vision + Blockchain Integration Test")
    print("=" * 60)
    
    if args.test_all:
        # Test with sample files
        test_files = [
            "data/input/Photos/group.jpg",
            "data/input/Photos/yomi1.jpg",
            "data/input/Videos/test_stroll.mp4"
        ]
        
        results = []
        for file_path in test_files:
            if os.path.exists(file_path):
                if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                    result = await process_and_upload_video(file_path)
                else:
                    result = await process_and_upload_image(file_path)
                
                if result:
                    results.append({
                        "file": file_path,
                        "result": result
                    })
            else:
                print(f"\n⚠️  File not found: {file_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Summary of Results:")
        print("=" * 60)
        for item in results:
            print(f"\n📄 {item['file']}:")
            print(f"   - Original Asset: {item['result']['original_asset_id']}")
            print(f"   - Analysis Asset: {item['result']['analysis_asset_id']}")
            print(f"   - Faces Detected: {item['result']['faces_detected']}")
            print(f"   - Unique Persons: {item['result']['clusters_found']}")
        
    elif args.file_path:
        # Process single file
        if not os.path.exists(args.file_path):
            print(f"❌ File not found: {args.file_path}")
            return
        
        if args.file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            await process_and_upload_video(args.file_path)
        else:
            await process_and_upload_image(args.file_path)
    
    else:
        # No arguments, show usage
        parser.print_help()
        print("\nExamples:")
        print("  python test_full_pipeline_blockchain.py data/input/Photos/group.jpg")
        print("  python test_full_pipeline_blockchain.py data/input/Videos/test_stroll.mp4")
        print("  python test_full_pipeline_blockchain.py --test-all")


if __name__ == "__main__":
    asyncio.run(main())