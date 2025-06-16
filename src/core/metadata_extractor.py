import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import exifread
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from src.utils.logger import get_facial_vision_logger, timing_decorator

# Try to import ffprobe for enhanced video metadata
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

logger = get_facial_vision_logger(__name__)


class MetadataExtractor:
    """Extract metadata from images and videos including EXIF, GPS, and timestamps"""

    def __init__(self):
        self.logger = get_facial_vision_logger(__name__)
        
        # Device ID patterns for common camera manufacturers
        self.device_patterns = [
            # Axis cameras
            (r'AXIS[_\-\s]?([A-Z0-9\-]+)', 'AXIS'),
            # Hikvision
            (r'DS[_\-\s]?([0-9A-Z\-]+)', 'Hikvision'),
            # Dahua
            (r'DH[_\-\s]?([0-9A-Z\-]+)', 'Dahua'),
            # Bosch
            (r'NBN[_\-\s]?([0-9A-Z\-]+)', 'Bosch'),
            # Generic patterns
            (r'CAM[_\-\s]?([0-9A-Z\-]+)', 'Generic'),
            (r'CAMERA[_\-\s]?([0-9A-Z\-]+)', 'Generic'),
        ]

    @timing_decorator
    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract metadata from image or video file

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type
        extension = file_path.suffix.lower()

        if extension in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            return self.extract_image_metadata(file_path)
        elif extension in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            return self.extract_video_metadata(file_path)
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return self._get_basic_metadata(file_path)

    def extract_image_metadata(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from image file"""
        image_path = Path(image_path)
        metadata = self._get_basic_metadata(image_path)

        # Try multiple methods to extract EXIF
        exif_data = self._extract_exif_with_exifread(image_path)
        pil_exif = self._extract_exif_with_pil(image_path)

        # Merge EXIF data
        if exif_data:
            metadata.update(exif_data)
        if pil_exif:
            metadata.update(pil_exif)

        # Extract GPS if available
        gps_data = self._extract_gps_from_exif(metadata.get("exif", {}))
        if gps_data:
            metadata["gps"] = gps_data

        # Extract device ID from filename or EXIF
        device_info = self._extract_device_info(image_path, metadata)
        if device_info:
            metadata["device"] = device_info

        logger.debug(f"Extracted metadata for {image_path.name}: {list(metadata.keys())}")
        return metadata

    def extract_video_metadata(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from video file with enhanced ffmpeg support"""
        video_path = Path(video_path)
        metadata = self._get_basic_metadata(video_path)

        # Try ffmpeg first for comprehensive metadata
        if FFMPEG_AVAILABLE:
            try:
                ffmpeg_metadata = self._extract_ffmpeg_metadata(video_path)
                if ffmpeg_metadata:
                    metadata.update(ffmpeg_metadata)
            except Exception as e:
                logger.debug(f"FFmpeg metadata extraction failed: {e}")

        # Fallback to OpenCV for basic video info
        try:
            cap = cv2.VideoCapture(str(video_path))

            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                video_info = {
                    "fps": fps,
                    "frame_count": frame_count,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "duration_seconds": frame_count / fps if fps > 0 else 0,
                    "codec": self._get_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
                }
                
                # Merge with existing video metadata
                if "video" in metadata:
                    metadata["video"].update(video_info)
                else:
                    metadata["video"] = video_info
                    
                cap.release()

        except Exception as e:
            logger.error(f"Error extracting video metadata with OpenCV: {e}")

        # Extract device info from filename
        device_info = self._extract_device_info(video_path, metadata)
        if device_info:
            metadata["device"] = device_info

        return metadata

    def _get_basic_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get basic file metadata"""
        stat = file_path.stat()

        return {
            "filename": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size": stat.st_size,
            "creation_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modification_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),  # Default timestamp
        }

    def _extract_exif_with_exifread(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF data using exifread library"""
        try:
            with open(image_path, "rb") as f:
                tags = exifread.process_file(f, details=False)

            exif_data = {}
            for tag, value in tags.items():
                if tag not in ["JPEGThumbnail", "TIFFThumbnail"]:
                    exif_data[tag] = str(value)

            # Extract timestamp from EXIF
            timestamp = None
            for date_tag in ["EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"]:
                if date_tag in exif_data:
                    try:
                        timestamp = datetime.strptime(
                            exif_data[date_tag], "%Y:%m:%d %H:%M:%S"
                        ).isoformat()
                        break
                    except:
                        pass

            return {"exif": exif_data, "timestamp": timestamp or datetime.now().isoformat()}

        except Exception as e:
            logger.debug(f"Could not extract EXIF with exifread: {e}")
            return {}

    def _extract_exif_with_pil(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF data using PIL"""
        try:
            image = Image.open(image_path)
            exifdata = image.getexif()

            if not exifdata:
                return {}

            exif_dict = {}
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_dict[tag] = value

            # Extract camera info
            camera_info = {}
            if "Make" in exif_dict:
                camera_info["make"] = exif_dict["Make"]
            if "Model" in exif_dict:
                camera_info["model"] = exif_dict["Model"]
            if "LensModel" in exif_dict:
                camera_info["lens"] = exif_dict["LensModel"]

            result = {}
            if camera_info:
                result["camera"] = camera_info

            return result

        except Exception as e:
            logger.debug(f"Could not extract EXIF with PIL: {e}")
            return {}

    def _extract_gps_from_exif(self, exif_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract GPS coordinates from EXIF data"""
        gps_tags = {
            "GPS GPSLatitude": "lat",
            "GPS GPSLongitude": "lon",
            "GPS GPSAltitude": "alt",
            "GPS GPSLatitudeRef": "lat_ref",
            "GPS GPSLongitudeRef": "lon_ref",
        }

        gps_data = {}
        for exif_tag, gps_key in gps_tags.items():
            if exif_tag in exif_data:
                gps_data[gps_key] = exif_data[exif_tag]

        if "lat" in gps_data and "lon" in gps_data:
            try:
                # Convert GPS coordinates to decimal degrees
                lat = self._convert_to_degrees(gps_data["lat"])
                lon = self._convert_to_degrees(gps_data["lon"])

                # Apply hemisphere correction
                if gps_data.get("lat_ref") == "S":
                    lat = -lat
                if gps_data.get("lon_ref") == "W":
                    lon = -lon

                result = {"lat": lat, "lon": lon}

                # Add altitude if available
                if "alt" in gps_data:
                    try:
                        result["altitude"] = float(str(gps_data["alt"]).split("/")[0])
                    except:
                        pass

                return result

            except Exception as e:
                logger.debug(f"Could not convert GPS coordinates: {e}")

        return None

    def _convert_to_degrees(self, value: str) -> float:
        """Convert GPS coordinate string to decimal degrees"""
        # Handle format like "[45, 30, 15.5]" or "45/1, 30/1, 155/10"
        value = str(value).strip("[]")
        parts = value.split(",")

        if len(parts) == 3:
            # Extract numeric values
            deg = self._parse_rational(parts[0].strip())
            min = self._parse_rational(parts[1].strip())
            sec = self._parse_rational(parts[2].strip())

            return deg + (min / 60.0) + (sec / 3600.0)
        else:
            raise ValueError(f"Invalid GPS coordinate format: {value}")

    def _parse_rational(self, value: str) -> float:
        """Parse rational number format (e.g., '45/1' or '45')"""
        if "/" in value:
            num, den = value.split("/")
            return float(num) / float(den)
        else:
            return float(value)

    def _get_fourcc(self, fourcc_code: float) -> str:
        """Convert fourcc code to string"""
        try:
            fourcc = int(fourcc_code)
            return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        except:
            return "unknown"

    def get_timestamp(self, metadata: Dict[str, Any]) -> str:
        """Get the best available timestamp from metadata"""
        # Priority order for timestamps
        if "timestamp" in metadata and metadata["timestamp"]:
            return metadata["timestamp"]
        elif "modification_time" in metadata:
            return metadata["modification_time"]
        elif "creation_time" in metadata:
            return metadata["creation_time"]
        else:
            return datetime.now().isoformat()

    def get_gps_coordinates(self, metadata: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get GPS coordinates from metadata if available"""
        return metadata.get("gps", None)

    def _extract_ffmpeg_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract comprehensive metadata using ffprobe"""
        try:
            probe = ffmpeg.probe(str(video_path))
            
            metadata = {}
            
            # General format info
            if 'format' in probe:
                format_info = probe['format']
                metadata['video'] = {
                    'duration_seconds': float(format_info.get('duration', 0)),
                    'size_bytes': int(format_info.get('size', 0)),
                    'format_name': format_info.get('format_name', 'unknown'),
                    'bit_rate': int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
                }
                
                # Extract creation time if available
                if 'tags' in format_info:
                    tags = format_info['tags']
                    for time_key in ['creation_time', 'date', 'DATE']:
                        if time_key in tags:
                            try:
                                # Parse various datetime formats
                                time_str = tags[time_key]
                                if 'T' in time_str:
                                    dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                                else:
                                    dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                                metadata['creation_time'] = dt.isoformat()
                                metadata['timestamp'] = dt.isoformat()
                                break
                            except Exception:
                                continue

            # Video stream info
            video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
            if video_streams:
                stream = video_streams[0]
                video_info = metadata.get('video', {})
                video_info.update({
                    'width': stream.get('width'),
                    'height': stream.get('height'),
                    'fps': eval(stream.get('r_frame_rate', '0/1')),
                    'codec_name': stream.get('codec_name'),
                    'pixel_format': stream.get('pix_fmt'),
                    'aspect_ratio': stream.get('display_aspect_ratio'),
                })
                metadata['video'] = video_info

            # Audio stream info
            audio_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
            if audio_streams:
                stream = audio_streams[0]
                metadata['audio'] = {
                    'codec_name': stream.get('codec_name'),
                    'sample_rate': stream.get('sample_rate'),
                    'channels': stream.get('channels'),
                    'duration_seconds': float(stream.get('duration', 0)),
                }

            return metadata

        except Exception as e:
            logger.debug(f"Failed to extract ffmpeg metadata: {e}")
            return {}

    def _extract_device_info(self, file_path: Path, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract device information from filename and metadata"""
        device_info = {}
        
        # Try to extract from filename
        filename = file_path.name
        for pattern, manufacturer in self.device_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                device_info.update({
                    'id': match.group(1) if match.groups() else match.group(0),
                    'manufacturer': manufacturer,
                    'source': 'filename'
                })
                break
        
        # Try to extract from EXIF data (for images)
        if not device_info and 'camera' in metadata:
            camera = metadata['camera']
            device_info = {
                'manufacturer': camera.get('make', 'unknown'),
                'model': camera.get('model', 'unknown'),
                'source': 'exif'
            }
            
            # Create device ID from make/model
            if 'make' in camera and 'model' in camera:
                device_info['id'] = f"{camera['make']}-{camera['model']}".replace(' ', '-')

        # Try to extract from parent directory name
        if not device_info:
            parent_dir = file_path.parent.name
            for pattern, manufacturer in self.device_patterns:
                match = re.search(pattern, parent_dir, re.IGNORECASE)
                if match:
                    device_info.update({
                        'id': match.group(1) if match.groups() else match.group(0),
                        'manufacturer': manufacturer,
                        'source': 'directory'
                    })
                    break

        return device_info if device_info else None

    def create_chip_metadata(
        self,
        source_file: Union[str, Path],
        chip_path: Union[str, Path],
        face_bbox: Tuple[int, int, int, int],
        cluster_id: str,
        confidence: float,
        frame_number: Optional[int] = None,
        video_timestamp: Optional[float] = None,
        parent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for a face chip
        
        Args:
            source_file: Original source file path
            chip_path: Path to the generated chip
            face_bbox: Face bounding box (top, right, bottom, left)
            cluster_id: Cluster assignment (e.g., "person_1")
            confidence: Detection confidence score
            frame_number: Frame number for video sources
            video_timestamp: Timestamp within video in seconds
            parent_id: Blockchain asset ID of parent file
            
        Returns:
            Comprehensive metadata dictionary
        """
        source_path = Path(source_file)
        chip_path = Path(chip_path)
        
        # Extract metadata from source file
        source_metadata = self.extract_metadata(source_path)
        
        # Build chip metadata
        chip_metadata = {
            'file': str(chip_path),
            'type': 'image',
            'name': chip_path.stem,
            'author': 'facial-vision-system',
            'timestamp': self.get_timestamp(source_metadata),
            'clusterId': cluster_id,
            'sourceFile': str(source_path),
            'face_bounds': {
                'x': face_bbox[3],  # left
                'y': face_bbox[0],  # top
                'w': face_bbox[1] - face_bbox[3],  # width
                'h': face_bbox[2] - face_bbox[0],  # height
            },
            'confidence': confidence,
            'topics': ['face_detected', 'clustered'],
        }
        
        # Add parent ID if provided
        if parent_id:
            chip_metadata['parentId'] = parent_id
            
        # Add device info if available
        device_info = source_metadata.get('device')
        if device_info:
            chip_metadata['deviceId'] = device_info.get('id', 'unknown')
            chip_metadata['device'] = device_info
            
        # Add GPS if available
        gps_info = source_metadata.get('gps')
        if gps_info:
            chip_metadata['gps'] = gps_info
            
        # Add video-specific metadata
        if frame_number is not None:
            chip_metadata['frameNumber'] = frame_number
            
        if video_timestamp is not None:
            chip_metadata['videoTimestamp'] = self._format_video_timestamp(video_timestamp)
            
        # Add video info if source is video
        video_info = source_metadata.get('video')
        if video_info:
            chip_metadata['video_metadata'] = {
                'fps': video_info.get('fps'),
                'duration_seconds': video_info.get('duration_seconds'),
                'resolution': f"{video_info.get('width')}x{video_info.get('height')}"
            }
            
        return chip_metadata

    def _format_video_timestamp(self, seconds: float) -> str:
        """Format video timestamp as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def extract_device_id(self, file_path: Union[str, Path]) -> Optional[str]:
        """Extract device ID from file path or metadata"""
        metadata = self.extract_metadata(file_path)
        device_info = metadata.get('device')
        return device_info.get('id') if device_info else None
