#!/usr/bin/env python3
"""
Test script for blockchain integration with Maverix.
This script tests the full workflow of uploading assets and face analysis results.
"""

import asyncio
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.blockchain_integration import BlockchainIntegration
from utils.logger import get_logger

logger = get_logger(__name__)


async def test_connection():
    """Test 1: Basic connection to Maverix"""
    print("\n=== Test 1: Testing Connection to Maverix ===")
    async with BlockchainIntegration() as blockchain:
        try:
            # Try to query assets
            async with blockchain.session.get(f"{blockchain.provenance_endpoint}/assets?limit=1") as response:
                if response.status == 200:
                    print("✅ Successfully connected to Maverix")
                    data = await response.json()
                    print(f"   Found {data.get('pagination', {}).get('total', 0)} existing assets")
                else:
                    print(f"❌ Connection failed with status: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\n⚠️  Make sure Maverix is running on http://localhost:3000")
            print("   Run: cd ~/code/maverix/maverix && docker-compose up")
            return False
    return True


async def test_asset_existence_check():
    """Test 2: Check if asset exists by hash and metadata"""
    print("\n=== Test 2: Testing Asset Existence Checks ===")
    
    async with BlockchainIntegration() as blockchain:
        # Test with a sample file
        test_file = "data/input/Photos/yomi1.jpg"
        if os.path.exists(test_file):
            file_hash = blockchain._calculate_file_hash(test_file)
            print(f"📄 File: {test_file}")
            print(f"   Hash: {file_hash}")
            
            # Check by hash
            exists_by_hash = await blockchain.check_asset_exists_by_hash(file_hash)
            if exists_by_hash:
                print(f"✅ Asset found by hash: {exists_by_hash.get('id')}")
            else:
                print("ℹ️  Asset not found by hash (expected for first run)")
            
            # Check by metadata
            exists_by_meta = await blockchain.check_asset_exists_by_metadata("yomi1.jpg", "test-camera")
            if exists_by_meta:
                print(f"✅ Asset found by metadata: {exists_by_meta.get('id')}")
            else:
                print("ℹ️  Asset not found by metadata (expected for first run)")
        else:
            print("⚠️  Test file not found. Skipping existence check.")


async def test_upload_image():
    """Test 3: Upload a single image as original asset"""
    print("\n=== Test 3: Testing Image Upload ===")
    
    test_image = "data/input/Photos/yomi1.jpg"
    if not os.path.exists(test_image):
        print(f"⚠️  Test image not found: {test_image}")
        return None
        
    async with BlockchainIntegration() as blockchain:
        try:
            # Check if already exists
            file_hash = blockchain._calculate_file_hash(test_image)
            existing = await blockchain.check_asset_exists_by_hash(file_hash)
            
            if existing:
                print(f"ℹ️  Image already uploaded with ID: {existing['id']}")
                return existing['id']
            
            # Upload new image
            print(f"📤 Uploading: {test_image}")
            result = await blockchain.upload_original_asset(
                file_path=test_image,
                asset_type="image",
                name="Test Portrait - Yomi",
                author="test-camera",
                metadata={
                    "description": "Test portrait for facial recognition",
                    "location": "Test Lab",
                    "deviceId": "CAM-001"
                },
                topics=["test", "portrait", "facial_vision"]
            )
            
            print(f"✅ Upload successful!")
            print(f"   Asset ID: {result['assetId']}")
            print(f"   IPFS CID: {result['ipfsCid']}")
            print(f"   Topics: {', '.join(result['topics'][:3])}...")
            
            return result['assetId']
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return None


async def test_upload_video():
    """Test 4: Upload a video as original asset"""
    print("\n=== Test 4: Testing Video Upload ===")
    
    test_video = "data/input/test_stroll.mp4"
    if not os.path.exists(test_video):
        print(f"⚠️  Test video not found: {test_video}")
        return None
        
    async with BlockchainIntegration() as blockchain:
        try:
            # Check if already exists
            file_hash = blockchain._calculate_file_hash(test_video)
            existing = await blockchain.check_asset_exists_by_hash(file_hash)
            
            if existing:
                print(f"ℹ️  Video already uploaded with ID: {existing['id']}")
                return existing['id']
            
            # Upload new video
            print(f"📤 Uploading: {test_video} (this may take a moment...)")
            result = await blockchain.upload_original_asset(
                file_path=test_video,
                asset_type="video",
                name="Test Stroll Video",
                author="security-cam-02",
                metadata={
                    "description": "Test video for face detection",
                    "duration": "10 seconds",
                    "fps": 30,
                    "resolution": "1920x1080",
                    "GPS": {"lat": 39.2557, "lon": -76.7112}
                },
                topics=["test", "video", "security_footage"]
            )
            
            print(f"✅ Upload successful!")
            print(f"   Asset ID: {result['assetId']}")
            print(f"   IPFS CID: {result['ipfsCid']}")
            
            return result['assetId']
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return None


async def test_face_analysis_upload(parent_asset_id: str):
    """Test 5: Upload face analysis results as derived asset"""
    print("\n=== Test 5: Testing Face Analysis Upload ===")
    
    if not parent_asset_id:
        print("⚠️  No parent asset ID provided. Skipping test.")
        return None
        
    async with BlockchainIntegration() as blockchain:
        try:
            # Simulate face analysis results
            face_chips_metadata = [
                {
                    "file": "person_1/chip_001.jpg",
                    "type": "image",
                    "name": "chip_001",
                    "timestamp": "2024-01-15T13:24:00Z",
                    "clusterId": "person_1",
                    "confidence": 0.98,
                    "face_bounds": {"x": 84, "y": 122, "w": 64, "h": 64},
                    "quality": {"blur": 0.2, "brightness": 0.75}
                },
                {
                    "file": "person_1/chip_002.jpg",
                    "type": "image", 
                    "name": "chip_002",
                    "timestamp": "2024-01-15T13:24:02Z",
                    "clusterId": "person_1",
                    "confidence": 0.95,
                    "face_bounds": {"x": 81, "y": 120, "w": 62, "h": 63}
                },
                {
                    "file": "person_2/chip_003.jpg",
                    "type": "image",
                    "name": "chip_003",
                    "timestamp": "2024-01-15T13:24:02Z",
                    "clusterId": "person_2",
                    "confidence": 0.99,
                    "face_bounds": {"x": 91, "y": 118, "w": 60, "h": 60}
                }
            ]
            
            analysis_metadata = {
                "sourceFile": "test_media.mp4",
                "processingDuration": 12.5,
                "algorithmVersion": "1.0",
                "detectionModel": "MTCNN",
                "clusteringMethod": "DBSCAN",
                "totalFramesProcessed": 300
            }
            
            # Create a mock chip bundle (in real scenario, this would contain actual chips)
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                with zipfile.ZipFile(tmp.name, 'w') as zf:
                    # Add a simple manifest
                    manifest = {
                        "version": "1.0",
                        "chips": face_chips_metadata,
                        "created": "2024-01-15T13:25:00Z"
                    }
                    zf.writestr("manifest.json", json.dumps(manifest, indent=2))
                    
                    # Add mock chip files
                    for chip in face_chips_metadata:
                        zf.writestr(chip["file"], b"mock image data")
                
                chip_bundle_path = tmp.name
            
            print(f"📤 Uploading face analysis for parent asset: {parent_asset_id}")
            result = await blockchain.upload_face_analysis_results(
                parent_asset_id=parent_asset_id,
                face_chips_metadata=face_chips_metadata,
                analysis_metadata=analysis_metadata,
                chip_bundle_path=chip_bundle_path,
                author="facial-vision-system"
            )
            
            print(f"✅ Analysis upload successful!")
            print(f"   Derived Asset ID: {result['assetId']}")
            print(f"   IPFS CID: {result['ipfsCid']}")
            print(f"   Total faces: {len(face_chips_metadata)}")
            print(f"   Clusters found: {len(set(c['clusterId'] for c in face_chips_metadata))}")
            
            # Clean up temp file
            os.unlink(chip_bundle_path)
            
            return result['assetId']
            
        except Exception as e:
            print(f"❌ Analysis upload failed: {e}")
            return None


async def test_get_provenance(asset_id: str):
    """Test 6: Get provenance chain for an asset"""
    print("\n=== Test 6: Testing Provenance Retrieval ===")
    
    if not asset_id:
        print("⚠️  No asset ID provided. Skipping test.")
        return
        
    async with BlockchainIntegration() as blockchain:
        try:
            print(f"📊 Getting provenance for asset: {asset_id}")
            provenance = await blockchain.get_asset_provenance(asset_id)
            
            if provenance:
                print("✅ Provenance retrieved successfully!")
                print(f"   Origin: {provenance.get('origin', {}).get('id')}")
                print(f"   Derivations: {len(provenance.get('derivations', []))}")
                
                # Print tree structure
                if provenance.get('derivations'):
                    print("\n   Provenance Tree:")
                    print(f"   └── {provenance['origin']['id']} (origin)")
                    for deriv in provenance['derivations']:
                        print(f"       └── {deriv['asset']['id']} (derived)")
            else:
                print("❌ Failed to retrieve provenance")
                
        except Exception as e:
            print(f"❌ Provenance retrieval failed: {e}")


async def test_download_asset(asset_id: str):
    """Test 7: Download an asset file from IPFS"""
    print("\n=== Test 7: Testing Asset Download ===")
    
    if not asset_id:
        print("⚠️  No asset ID provided. Skipping test.")
        return
        
    async with BlockchainIntegration() as blockchain:
        try:
            output_path = f"test_download_{asset_id[:8]}.bin"
            print(f"📥 Downloading asset: {asset_id}")
            print(f"   Output: {output_path}")
            
            success = await blockchain.download_asset_file(asset_id, output_path)
            
            if success:
                file_size = os.path.getsize(output_path)
                print(f"✅ Download successful!")
                print(f"   File size: {file_size:,} bytes")
                
                # Clean up
                os.unlink(output_path)
            else:
                print("❌ Download failed")
                
        except Exception as e:
            print(f"❌ Download failed: {e}")


async def run_all_tests():
    """Run all blockchain integration tests"""
    print("\n🚀 Starting Blockchain Integration Tests")
    print("=" * 50)
    
    # Test 1: Connection
    if not await test_connection():
        print("\n❌ Cannot proceed without Maverix connection.")
        return
    
    # Test 2: Existence checks
    await test_asset_existence_check()
    
    # Test 3: Upload image
    image_asset_id = await test_upload_image()
    
    # Test 4: Upload video
    video_asset_id = await test_upload_video()
    
    # Test 5: Upload face analysis (using video as parent)
    analysis_asset_id = None
    if video_asset_id:
        analysis_asset_id = await test_face_analysis_upload(video_asset_id)
    
    # Test 6: Get provenance
    if analysis_asset_id:
        await test_get_provenance(analysis_asset_id)
    elif video_asset_id:
        await test_get_provenance(video_asset_id)
    
    # Test 7: Download asset
    if image_asset_id:
        await test_download_asset(image_asset_id)
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\n📝 Summary:")
    print(f"   - Image Asset: {image_asset_id or 'Not created'}")
    print(f"   - Video Asset: {video_asset_id or 'Not created'}")
    print(f"   - Analysis Asset: {analysis_asset_id or 'Not created'}")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_all_tests())