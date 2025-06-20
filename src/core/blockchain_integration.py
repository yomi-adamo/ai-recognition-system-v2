import os
import json
import hashlib
import aiohttp
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime
import mimetypes

logger = logging.getLogger(__name__)


class BlockchainIntegration:
    """
    Handles integration with Maverix blockchain system via REST API.
    This class manages the registration of assets and their provenance tracking.
    """
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip('/')
        self.provenance_endpoint = f"{self.base_url}/provenance"
        self.session = None
        self._asset_cache = {}  # Simple cache for asset lookups
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    async def check_asset_exists_by_hash(self, file_hash: str) -> Optional[Dict]:
        """
        Check if an asset with the given file hash already exists.
        
        Args:
            file_hash: SHA-256 hash of the file
            
        Returns:
            Asset data if found, None otherwise
        """
        if file_hash in self._asset_cache:
            return self._asset_cache[file_hash]
            
        try:
            # Query assets and check metadata for matching hash
            async with self.session.get(
                f"{self.provenance_endpoint}/assets",
                params={"limit": 100}  # Adjust as needed
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    assets = data.get('assets', [])
                    
                    for asset in assets:
                        metadata = asset.get('metadata', {})
                        if metadata.get('fileHash') == file_hash:
                            self._asset_cache[file_hash] = asset
                            return asset
                            
        except Exception as e:
            logger.error(f"Error checking asset existence by hash: {e}")
            
        return None
        
    async def check_asset_exists_by_metadata(self, filename: str, author: str) -> Optional[Dict]:
        """
        Check if an asset with the given filename and author already exists.
        
        Args:
            filename: Name of the file
            author: Creator/author of the asset
            
        Returns:
            Asset data if found, None otherwise
        """
        cache_key = f"{filename}:{author}"
        if cache_key in self._asset_cache:
            return self._asset_cache[cache_key]
            
        try:
            # Query by author
            async with self.session.get(
                f"{self.provenance_endpoint}/assets",
                params={"author": author, "limit": 100}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    assets = data.get('assets', [])
                    
                    for asset in assets:
                        if asset.get('name') == filename:
                            self._asset_cache[cache_key] = asset
                            return asset
                            
        except Exception as e:
            logger.error(f"Error checking asset existence by metadata: {e}")
            
        return None
        
    async def upload_original_asset(
        self,
        file_path: str,
        asset_type: str,
        name: str,
        author: str,
        metadata: Dict[str, Any],
        topics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Upload an original asset (video/image) to the blockchain.
        
        Args:
            file_path: Path to the file to upload
            asset_type: Type of asset (e.g., 'video', 'image')
            name: Human-readable name for the asset
            author: Creator/author of the asset
            metadata: Additional metadata to store with the asset
            topics: Optional list of topics for categorization
            
        Returns:
            Response containing assetId, ipfsCid, and other details
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Prepare metadata with file hash
        full_metadata = {
            **metadata,
            "fileHash": file_hash,
            "originalFilePath": file_path,
            "uploadedAt": datetime.utcnow().isoformat() + "Z"
        }
        
        # Prepare form data
        form_data = aiohttp.FormData()
        
        # Add file
        with open(file_path, 'rb') as f:
            file_content = f.read()
            mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            form_data.add_field(
                'file',
                file_content,
                filename=os.path.basename(file_path),
                content_type=mime_type
            )
            
        # Add other fields
        form_data.add_field('type', asset_type)
        form_data.add_field('name', name)
        form_data.add_field('author', author)
        form_data.add_field('metadata', json.dumps(full_metadata))
        
        if topics:
            form_data.add_field('topics', json.dumps(topics))
            
        try:
            async with self.session.post(
                f"{self.provenance_endpoint}/assets",
                data=form_data
            ) as response:
                response_text = await response.text()
                
                if response.status == 201:
                    result = json.loads(response_text)
                    logger.info(f"Successfully uploaded original asset: {result.get('assetId')}")
                    return result
                else:
                    logger.error(f"Failed to upload asset: {response.status} - {response_text}")
                    raise Exception(f"Asset upload failed: {response_text}")
                    
        except Exception as e:
            logger.error(f"Error uploading original asset: {e}")
            raise
            
    async def create_derived_asset(
        self,
        parent_id: str,
        asset_type: str,
        name: str,
        author: str,
        metadata: Dict[str, Any],
        chip_bundle_path: Optional[str] = None,
        topics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a derived asset linked to a parent asset.
        
        Args:
            parent_id: ID of the parent asset
            asset_type: Type of asset (e.g., 'face_analysis')
            name: Human-readable name for the asset
            author: Creator/author of the asset
            metadata: Additional metadata (including chip information)
            chip_bundle_path: Optional path to chip bundle zip file
            topics: Optional list of topics for categorization
            
        Returns:
            Response containing assetId, ipfsCid, and other details
        """
        # Prepare form data
        form_data = aiohttp.FormData()
        
        # If chip bundle provided, add it as file
        if chip_bundle_path and os.path.exists(chip_bundle_path):
            with open(chip_bundle_path, 'rb') as f:
                form_data.add_field(
                    'file',
                    f.read(),
                    filename=os.path.basename(chip_bundle_path),
                    content_type='application/zip'
                )
        else:
            # For metadata-only derived assets, create a JSON file
            metadata_content = json.dumps(metadata, indent=2).encode('utf-8')
            form_data.add_field(
                'file',
                metadata_content,
                filename=f"{name}_metadata.json",
                content_type='application/json'
            )
            
        # Add other fields
        form_data.add_field('type', asset_type)
        form_data.add_field('name', name)
        form_data.add_field('author', author)
        form_data.add_field('parentId', parent_id)
        form_data.add_field('metadata', json.dumps(metadata))
        
        if topics:
            form_data.add_field('topics', json.dumps(topics))
            
        try:
            async with self.session.post(
                f"{self.provenance_endpoint}/assets",
                data=form_data
            ) as response:
                response_text = await response.text()
                
                if response.status == 201:
                    result = json.loads(response_text)
                    logger.info(f"Successfully created derived asset: {result.get('assetId')}")
                    return result
                else:
                    logger.error(f"Failed to create derived asset: {response.status} - {response_text}")
                    raise Exception(f"Derived asset creation failed: {response_text}")
                    
        except Exception as e:
            logger.error(f"Error creating derived asset: {e}")
            raise
            
    async def upload_face_analysis_results(
        self,
        parent_asset_id: str,
        face_chips_metadata: List[Dict],
        analysis_metadata: Dict,
        chip_bundle_path: Optional[str] = None,
        author: str = "facial-vision-system"
    ) -> Dict[str, Any]:
        """
        Upload face analysis results as a derived asset.
        
        Args:
            parent_asset_id: ID of the original video/image asset
            face_chips_metadata: List of metadata for each face chip
            analysis_metadata: Overall analysis metadata
            chip_bundle_path: Optional path to zip file containing all chips
            author: System/user creating the analysis
            
        Returns:
            Response from blockchain containing derived asset details
        """
        # Prepare complete metadata
        full_metadata = {
            **analysis_metadata,
            "analysisType": "face_detection",
            "totalFaces": len(face_chips_metadata),
            "chips": face_chips_metadata,
            "analysisTimestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Determine topics with strict limit enforcement
        topics = ["face_detected", "analysis_complete"]
        
        # Extract unique cluster IDs
        cluster_ids = set()
        for chip in face_chips_metadata:
            if 'clusterId' in chip:
                cluster_ids.add(chip['clusterId'])
        
        # Add cluster topics but strictly limit to avoid blockchain 15-topic limit
        # Conservative limit: keep total topics under 10 to be safe
        max_total_topics = 8  # Very conservative limit
        available_slots = max_total_topics - len(topics)  # 6 slots available
        cluster_topics = [f"cluster_{cluster_id}" for cluster_id in sorted(list(cluster_ids))[:available_slots]]
        topics.extend(cluster_topics)
        
        # Log topics for debugging
        logger.info(f"Topics being sent to blockchain ({len(topics)} total): {topics}")
        
        # Ensure we never exceed the limit
        if len(topics) > max_total_topics:
            topics = topics[:max_total_topics]
            logger.warning(f"Topics truncated to {max_total_topics} to avoid blockchain limit")
        
        return await self.create_derived_asset(
            parent_id=parent_asset_id,
            asset_type="face_analysis",
            name=f"Face Analysis - {analysis_metadata.get('sourceFile', 'Unknown')}",
            author=author,
            metadata=full_metadata,
            chip_bundle_path=chip_bundle_path,
            topics=topics
        )
        
    async def get_asset_provenance(self, asset_id: str) -> Dict[str, Any]:
        """
        Get the full provenance chain for an asset.
        
        Args:
            asset_id: ID of the asset
            
        Returns:
            Provenance chain data
        """
        try:
            async with self.session.get(
                f"{self.provenance_endpoint}/assets/{asset_id}/provenance"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get provenance: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting asset provenance: {e}")
            return None
            
    async def download_asset_file(self, asset_id: str, output_path: str) -> bool:
        """
        Download an asset file from IPFS.
        
        Args:
            asset_id: ID of the asset to download
            output_path: Path where to save the downloaded file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.session.get(
                f"{self.provenance_endpoint}/assets/{asset_id}/download"
            ) as response:
                if response.status == 200:
                    # Save file
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    logger.info(f"Downloaded asset {asset_id} to {output_path}")
                    return True
                else:
                    logger.error(f"Failed to download asset: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error downloading asset: {e}")
            return False


# Example usage function
async def example_blockchain_workflow():
    """Example of how to use the blockchain integration"""
    
    async with BlockchainIntegration() as blockchain:
        # 1. Check if file already processed
        file_hash = blockchain._calculate_file_hash("path/to/video.mp4")
        existing = await blockchain.check_asset_exists_by_hash(file_hash)
        
        if existing:
            print(f"File already processed: {existing['id']}")
            return
            
        # 2. Upload original asset
        original_response = await blockchain.upload_original_asset(
            file_path="path/to/video.mp4",
            asset_type="video",
            name="Security Footage 2024-01-15",
            author="camera-01",
            metadata={
                "GPS": {"lat": 39.2557, "lon": -76.7112},
                "deviceId": "AXIS-W120"
            },
            topics=["security", "front_door"]
        )
        
        parent_id = original_response['assetId']
        
        # 3. Process faces (your existing code)
        # ... face detection and clustering ...
        
        # 4. Upload analysis results
        face_chips_metadata = [
            {
                "file": "person_1/chip_001.jpg",
                "type": "image",
                "name": "chip_001",
                "timestamp": "2024-01-15T13:24:00Z",
                "clusterId": "person_1",
                "face_bounds": {"x": 84, "y": 122, "w": 64, "h": 64}
            },
            # ... more chips ...
        ]
        
        analysis_response = await blockchain.upload_face_analysis_results(
            parent_asset_id=parent_id,
            face_chips_metadata=face_chips_metadata,
            analysis_metadata={
                "sourceFile": "video.mp4",
                "processingDuration": 45.2,
                "algorithmVersion": "1.0"
            },
            chip_bundle_path="output/chips_bundle.zip"
        )
        
        print(f"Analysis uploaded: {analysis_response['assetId']}")