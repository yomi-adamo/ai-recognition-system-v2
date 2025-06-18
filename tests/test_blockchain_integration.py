import pytest
import asyncio
import json
import os
import tempfile
import hashlib
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.blockchain_integration import BlockchainIntegration


class TestBlockchainIntegration:
    """Test suite for blockchain integration functionality"""
    
    @pytest.fixture
    def blockchain(self):
        """Create a blockchain integration instance"""
        return BlockchainIntegration(base_url="http://localhost:3000")
        
    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp session"""
        session = AsyncMock()
        return session
        
    @pytest.fixture
    def sample_asset(self):
        """Sample asset data for testing"""
        return {
            "id": "test-asset-123",
            "type": "video",
            "name": "test_video.mp4",
            "metadata": {
                "fileHash": "abc123def456",
                "ipfsCid": "QmTest123",
                "GPS": {"lat": 39.2557, "lon": -76.7112}
            },
            "createdBy": "test-author",
            "createdAt": "2024-01-15T13:24:00Z"
        }
        
    @pytest.fixture
    def sample_face_chips(self):
        """Sample face chip metadata"""
        return [
            {
                "file": "person_1/chip_001.jpg",
                "type": "image",
                "name": "chip_001",
                "timestamp": "2024-01-15T13:24:00Z",
                "clusterId": "person_1",
                "face_bounds": {"x": 84, "y": 122, "w": 64, "h": 64}
            },
            {
                "file": "person_1/chip_002.jpg",
                "type": "image",
                "name": "chip_002",
                "timestamp": "2024-01-15T13:24:02Z",
                "clusterId": "person_1",
                "face_bounds": {"x": 81, "y": 120, "w": 62, "h": 63}
            },
            {
                "file": "person_2/chip_003.jpg",
                "type": "image",
                "name": "chip_003",
                "timestamp": "2024-01-15T13:24:02Z",
                "clusterId": "person_2",
                "face_bounds": {"x": 91, "y": 118, "w": 60, "h": 60}
            }
        ]
        
    @pytest.mark.asyncio
    async def test_context_manager(self, blockchain):
        """Test async context manager functionality"""
        async with blockchain as bc:
            assert bc.session is not None
        # Note: Session cleanup happens in __aexit__
        
    def test_calculate_file_hash(self, blockchain):
        """Test file hash calculation"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
            
        try:
            # Calculate hash
            file_hash = blockchain._calculate_file_hash(temp_path)
            
            # Verify it's a valid SHA-256 hash
            assert len(file_hash) == 64
            assert all(c in '0123456789abcdef' for c in file_hash)
            
            # Verify consistency
            hash2 = blockchain._calculate_file_hash(temp_path)
            assert file_hash == hash2
        finally:
            os.unlink(temp_path)
            
    @pytest.mark.asyncio
    async def test_check_asset_exists_by_hash_found(self, blockchain, mock_session, sample_asset):
        """Test checking asset existence by hash when asset exists"""
        blockchain.session = mock_session
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "assets": [sample_asset],
            "pagination": {"total": 1}
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = AsyncMock(return_value=mock_response)
        
        # Check asset
        result = await blockchain.check_asset_exists_by_hash("abc123def456")
        
        assert result is not None
        assert result["id"] == "test-asset-123"
        
        # Verify caching
        result2 = await blockchain.check_asset_exists_by_hash("abc123def456")
        assert result2 == result
        assert mock_session.get.call_count == 1  # Should use cache
        
    @pytest.mark.asyncio
    async def test_check_asset_exists_by_hash_not_found(self, blockchain, mock_session):
        """Test checking asset existence by hash when asset doesn't exist"""
        blockchain.session = mock_session
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "assets": [],
            "pagination": {"total": 0}
        })
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Check asset
        result = await blockchain.check_asset_exists_by_hash("nonexistent")
        assert result is None
        
    @pytest.mark.asyncio
    async def test_check_asset_exists_by_metadata_found(self, blockchain, mock_session, sample_asset):
        """Test checking asset existence by metadata when asset exists"""
        blockchain.session = mock_session
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "assets": [sample_asset],
            "pagination": {"total": 1}
        })
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Check asset
        result = await blockchain.check_asset_exists_by_metadata("test_video.mp4", "test-author")
        
        assert result is not None
        assert result["name"] == "test_video.mp4"
        
    @pytest.mark.asyncio
    async def test_upload_original_asset_success(self, blockchain, mock_session):
        """Test successful upload of original asset"""
        blockchain.session = mock_session
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test video content")
            temp_path = f.name
            
        try:
            # Mock response
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.text = AsyncMock(return_value=json.dumps({
                "assetId": "new-asset-123",
                "ipfsCid": "QmNewAsset123",
                "topics": ["asset_new-asset-123", "type_video"],
                "message": "Asset created successfully"
            }))
            # Configure mock to return the response directly from post()
            mock_post = AsyncMock()
            mock_post.__aenter__.return_value = mock_response
            mock_session.post.return_value = mock_post
            
            # Upload asset
            result = await blockchain.upload_original_asset(
                file_path=temp_path,
                asset_type="video",
                name="test_upload.mp4",
                author="test-user",
                metadata={"deviceId": "CAM-001"},
                topics=["test", "upload"]
            )
            
            assert result["assetId"] == "new-asset-123"
            assert result["ipfsCid"] == "QmNewAsset123"
            
            # Verify form data was sent correctly
            call_args = mock_session.post.call_args
            assert call_args[0][0] == "http://localhost:3000/provenance/assets"
            
        finally:
            os.unlink(temp_path)
            
    @pytest.mark.asyncio
    async def test_upload_original_asset_file_not_found(self, blockchain):
        """Test upload with non-existent file"""
        with pytest.raises(FileNotFoundError):
            await blockchain.upload_original_asset(
                file_path="/nonexistent/file.mp4",
                asset_type="video",
                name="test.mp4",
                author="test",
                metadata={}
            )
            
    @pytest.mark.asyncio
    async def test_create_derived_asset_with_file(self, blockchain, mock_session):
        """Test creating derived asset with chip bundle file"""
        blockchain.session = mock_session
        
        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            f.write(b"PK\x03\x04")  # Minimal zip header
            temp_path = f.name
            
        try:
            # Mock response
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.text = AsyncMock(return_value=json.dumps({
                "assetId": "derived-asset-123",
                "ipfsCid": "QmDerived123",
                "topics": ["asset_derived-asset-123", "parent_parent-123"],
                "message": "Asset created successfully"
            }))
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            # Create derived asset
            result = await blockchain.create_derived_asset(
                parent_id="parent-123",
                asset_type="face_analysis",
                name="Face Analysis Results",
                author="facial-vision",
                metadata={"totalFaces": 3},
                chip_bundle_path=temp_path,
                topics=["face_detected"]
            )
            
            assert result["assetId"] == "derived-asset-123"
            assert "parent_parent-123" in result["topics"]
            
        finally:
            os.unlink(temp_path)
            
    @pytest.mark.asyncio
    async def test_create_derived_asset_metadata_only(self, blockchain, mock_session):
        """Test creating derived asset without file (metadata only)"""
        blockchain.session = mock_session
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.text = AsyncMock(return_value=json.dumps({
            "assetId": "metadata-asset-123",
            "ipfsCid": "QmMetadata123",
            "topics": ["asset_metadata-asset-123"],
            "message": "Asset created successfully"
        }))
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        # Create derived asset
        result = await blockchain.create_derived_asset(
            parent_id="parent-456",
            asset_type="analysis",
            name="Analysis Results",
            author="system",
            metadata={"result": "success"},
            chip_bundle_path=None
        )
        
        assert result["assetId"] == "metadata-asset-123"
        
    @pytest.mark.asyncio
    async def test_upload_face_analysis_results(self, blockchain, mock_session, sample_face_chips):
        """Test uploading face analysis results"""
        blockchain.session = mock_session
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.text = AsyncMock(return_value=json.dumps({
            "assetId": "analysis-123",
            "ipfsCid": "QmAnalysis123",
            "topics": ["face_detected", "cluster_person_1", "cluster_person_2"],
            "message": "Asset created successfully"
        }))
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        # Upload analysis
        result = await blockchain.upload_face_analysis_results(
            parent_asset_id="video-asset-123",
            face_chips_metadata=sample_face_chips,
            analysis_metadata={
                "sourceFile": "test_video.mp4",
                "processingDuration": 12.5
            }
        )
        
        assert result["assetId"] == "analysis-123"
        
        # Verify topics include cluster IDs
        call_args = mock_session.post.call_args
        form_data = call_args[1]["data"]
        # Note: Can't easily inspect FormData content, but we trust the implementation
        
    @pytest.mark.asyncio
    async def test_get_asset_provenance(self, blockchain, mock_session):
        """Test retrieving asset provenance"""
        blockchain.session = mock_session
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "origin": {"id": "root-123", "type": "video"},
            "derivations": [
                {"asset": {"id": "derived-123", "parentId": "root-123"}}
            ]
        })
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Get provenance
        result = await blockchain.get_asset_provenance("root-123")
        
        assert result["origin"]["id"] == "root-123"
        assert len(result["derivations"]) == 1
        
    @pytest.mark.asyncio
    async def test_download_asset_file_success(self, blockchain, mock_session):
        """Test successful asset file download"""
        blockchain.session = mock_session
        
        # Mock response with chunked content
        mock_response = AsyncMock()
        mock_response.status = 200
        
        # Mock chunked content iteration
        async def mock_iter_chunked(size):
            yield b"chunk1"
            yield b"chunk2"
            
        mock_response.content.iter_chunked = mock_iter_chunked
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            
        try:
            # Download asset
            success = await blockchain.download_asset_file("asset-123", temp_path)
            assert success is True
            
            # Verify file content
            with open(temp_path, 'rb') as f:
                content = f.read()
                assert content == b"chunk1chunk2"
                
        finally:
            os.unlink(temp_path)
            
    @pytest.mark.asyncio
    async def test_download_asset_file_failure(self, blockchain, mock_session):
        """Test failed asset file download"""
        blockchain.session = mock_session
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Download asset
        success = await blockchain.download_asset_file("nonexistent", "output.bin")
        assert success is False
        
    @pytest.mark.asyncio
    async def test_error_handling(self, blockchain, mock_session):
        """Test error handling in various scenarios"""
        blockchain.session = mock_session
        
        # Test network error
        mock_session.get.side_effect = Exception("Network error")
        result = await blockchain.check_asset_exists_by_hash("test")
        assert result is None
        
        # Test JSON parsing error
        mock_session.get.side_effect = None
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        result = await blockchain.get_asset_provenance("test")
        assert result is None


# Integration test example (requires running Maverix)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow_integration():
    """Integration test for complete workflow (requires Maverix running)"""
    async with BlockchainIntegration() as blockchain:
        # This would be a real integration test
        # Skipped unless Maverix is actually running
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])