import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import cv2
import exifread
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from src.utils.logger import get_facial_vision_logger, timing_decorator
from src.core.gps_ocr_extractor import GPSOCRExtractor

# Try to import ffprobe for enhanced video metadata
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    ffmpeg = None

# Try to import gpmf for GPS track extraction
try:
    import gpmf
    GPMF_AVAILABLE = True
except ImportError:
    GPMF_AVAILABLE = False

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
        
        # GPS track cache for videos
        self._gps_track_cache = {}
        
        # Initialize GPS OCR extractor
        self.gps_ocr_extractor = None
        try:
            self.gps_ocr_extractor = GPSOCRExtractor()
        except Exception as e:
            logger.warning(f"Could not initialize GPS OCR extractor: {e}")

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

    def extract_gps_track_from_video(self, video_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Extract GPS track data from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with GPS track data or None if no GPS data found
        """
        video_path = Path(video_path)
        cache_key = str(video_path.absolute())
        
        # Check cache first
        if cache_key in self._gps_track_cache:
            return self._gps_track_cache[cache_key]
        
        gps_track = None
        
        # Try GPMF extraction (GoPro and similar)
        if GPMF_AVAILABLE:
            gps_track = self._extract_gpmf_gps_track(video_path)
        
        # Fallback: Try extracting from general MP4 metadata
        if not gps_track:
            gps_track = self._extract_mp4_gps_track(video_path)
        
        # TODO: Add other GPS extraction methods for different camera types
        # - DJI SRT files
        # - Dash cam formats
        
        # Cache the result
        self._gps_track_cache[cache_key] = gps_track
        
        if gps_track:
            logger.info(f"Extracted GPS track with {len(gps_track.get('coordinates', []))} points from {video_path.name}")
        else:
            logger.debug(f"No GPS track found in {video_path.name}")
            
        return gps_track

    def _extract_gpmf_gps_track(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """Extract GPS track using GPMF library (for GoPro videos)"""
        try:
            # Extract GPMF stream from video
            stream = gpmf.io.extract_gpmf_stream(str(video_path))
            
            if not stream:
                logger.debug(f"No GPMF stream found in {video_path.name}")
                return None
            
            # Try to parse GPS data from stream
            try:
                # Use gpmf library to extract GPS coordinates and timestamps
                gps_coords = []
                timestamps = []
                
                # The gpmf library should have methods to extract GPS data
                # This is a basic implementation that may need refinement
                
                # Check if we can extract GPS track data
                if hasattr(gpmf, 'gps') and hasattr(gpmf.gps, 'extract_gps_from_stream'):
                    gps_data = gpmf.gps.extract_gps_from_stream(stream)
                    
                    if gps_data:
                        # Convert to our format
                        for entry in gps_data:
                            if 'lat' in entry and 'lon' in entry:
                                coord = {
                                    'lat': float(entry['lat']),
                                    'lon': float(entry['lon'])
                                }
                                if 'alt' in entry:
                                    coord['alt'] = float(entry['alt'])
                                    
                                gps_coords.append(coord)
                                
                                # Extract timestamp if available
                                if 'time' in entry:
                                    timestamps.append(float(entry['time']))
                                elif 'timestamp' in entry:
                                    timestamps.append(float(entry['timestamp']))
                
                if gps_coords:
                    # Generate timestamps if not available (based on typical GoPro 18Hz GPS rate)
                    if not timestamps or len(timestamps) != len(gps_coords):
                        logger.debug("Generating synthetic timestamps for GPS coordinates")
                        gps_rate = 18  # Hz - typical GoPro GPS rate
                        timestamps = [i / gps_rate for i in range(len(gps_coords))]
                    
                    return {
                        'coordinates': gps_coords,
                        'timestamps': timestamps,
                        'source': 'gpmf',
                        'total_points': len(gps_coords)
                    }
                else:
                    logger.debug(f"No GPS coordinates found in GPMF stream from {video_path.name}")
                    return None
                    
            except Exception as parse_error:
                logger.debug(f"Failed to parse GPS from GPMF stream: {parse_error}")
                
                # Fallback: Try basic stream inspection
                logger.debug(f"GPMF stream extracted from {video_path.name}, but GPS parsing failed")
                return None
            
        except Exception as e:
            logger.debug(f"Failed to extract GPMF GPS track: {e}")
            return None

    def _extract_mp4_gps_track(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """Extract GPS track from general MP4 metadata"""
        try:
            if not FFMPEG_AVAILABLE:
                return None
                
            # Extract metadata using ffprobe
            probe = ffmpeg.probe(str(video_path))
            
            # Look for GPS data in various locations
            gps_coords = []
            timestamps = []
            
            # Check format-level metadata
            if 'format' in probe and 'tags' in probe['format']:
                tags = probe['format']['tags']
                
                # Look for single GPS coordinate (common in many cameras)
                if self._has_gps_tags(tags):
                    coord = self._extract_single_gps_coordinate(tags)
                    if coord:
                        # For single GPS coordinate, create a minimal track
                        # This represents the location where recording started
                        gps_coords = [coord]
                        timestamps = [0.0]  # At start of video
                        
                        return {
                            'coordinates': gps_coords,
                            'timestamps': timestamps,
                            'source': 'mp4_metadata',
                            'total_points': 1,
                            'type': 'single_location'  # Indicates this is not a track
                        }
            
            # Check for GPS tracks in streams (less common)
            for stream in probe.get('streams', []):
                if stream.get('codec_type') == 'data':
                    # Look for GPS data streams
                    if 'tags' in stream and self._has_gps_tags(stream['tags']):
                        coord = self._extract_single_gps_coordinate(stream['tags'])
                        if coord:
                            gps_coords.append(coord)
                            timestamps.append(0.0)
            
            if gps_coords:
                return {
                    'coordinates': gps_coords,
                    'timestamps': timestamps,
                    'source': 'mp4_streams',
                    'total_points': len(gps_coords),
                    'type': 'single_location'
                }
                
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract MP4 GPS track: {e}")
            return None

    def _has_gps_tags(self, tags: Dict[str, Any]) -> bool:
        """Check if metadata tags contain GPS information"""
        gps_keys = [
            'location', 'GPS', 'gps', 'latitude', 'longitude', 
            'lat', 'lon', 'coordinates', 'Location'
        ]
        return any(key in tags for key in gps_keys)

    def _extract_single_gps_coordinate(self, tags: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract a single GPS coordinate from metadata tags"""
        try:
            coord = {}
            
            # Try various tag formats
            if 'location' in tags:
                # Parse location string (e.g., "+37.7749-122.4194+010.000/")
                location = tags['location']
                if location.startswith('+') or location.startswith('-'):
                    # ISO 6709 format
                    import re
                    match = re.match(r'([+-]\d+\.?\d*)([+-]\d+\.?\d*)([+-]\d+\.?\d*)?', location)
                    if match:
                        coord['lat'] = float(match.group(1))
                        coord['lon'] = float(match.group(2))
                        if match.group(3):
                            coord['alt'] = float(match.group(3))
            
            # Try direct latitude/longitude tags
            for lat_key in ['latitude', 'lat']:
                if lat_key in tags:
                    coord['lat'] = float(tags[lat_key])
                    
            for lon_key in ['longitude', 'lon']:
                if lon_key in tags:
                    coord['lon'] = float(tags[lon_key])
            
            # Try altitude
            for alt_key in ['altitude', 'alt', 'elevation']:
                if alt_key in tags:
                    coord['alt'] = float(tags[alt_key])
            
            if 'lat' in coord and 'lon' in coord:
                return coord
                
            return None
            
        except Exception as e:
            logger.debug(f"Failed to parse GPS coordinate from tags: {e}")
            return None

    def get_gps_at_timestamp(self, video_path: Union[str, Path], timestamp_seconds: float) -> Optional[Dict[str, float]]:
        """
        Get GPS coordinates at a specific video timestamp
        
        Args:
            video_path: Path to video file
            timestamp_seconds: Timestamp in seconds from video start
            
        Returns:
            GPS coordinates (lat, lon, alt) or None if not available
        """
        gps_track = self.extract_gps_track_from_video(video_path)
        
        if not gps_track or 'coordinates' not in gps_track:
            return None
            
        coordinates = gps_track['coordinates']
        timestamps = gps_track.get('timestamps', [])
        track_type = gps_track.get('type', 'track')
        
        if not coordinates or not timestamps:
            return None
            
        # Handle single location (static GPS)
        if track_type == 'single_location' or len(coordinates) == 1:
            logger.debug(f"Using single GPS location for timestamp {timestamp_seconds}s")
            return coordinates[0]
            
        # Find the closest timestamp
        if len(timestamps) != len(coordinates):
            logger.warning("GPS timestamps and coordinates length mismatch")
            return None
            
        # Linear interpolation between GPS points
        return self._interpolate_gps_coordinates(coordinates, timestamps, timestamp_seconds)

    def _interpolate_gps_coordinates(self, coordinates: List[Dict[str, float]], 
                                   timestamps: List[float], 
                                   target_timestamp: float) -> Optional[Dict[str, float]]:
        """
        Interpolate GPS coordinates for a specific timestamp
        
        Args:
            coordinates: List of GPS coordinate dictionaries
            timestamps: List of timestamps corresponding to coordinates
            target_timestamp: Target timestamp to interpolate for
            
        Returns:
            Interpolated GPS coordinates or None
        """
        if not coordinates or not timestamps:
            return None
            
        # Handle edge cases
        if target_timestamp <= timestamps[0]:
            return coordinates[0]
        if target_timestamp >= timestamps[-1]:
            return coordinates[-1]
            
        # Find the two closest points
        for i in range(len(timestamps) - 1):
            if timestamps[i] <= target_timestamp <= timestamps[i + 1]:
                # Linear interpolation
                t1, t2 = timestamps[i], timestamps[i + 1]
                coord1, coord2 = coordinates[i], coordinates[i + 1]
                
                # Interpolation factor
                factor = (target_timestamp - t1) / (t2 - t1)
                
                # Interpolate each coordinate
                interpolated = {}
                for key in ['lat', 'lon', 'alt']:
                    if key in coord1 and key in coord2:
                        interpolated[key] = coord1[key] + factor * (coord2[key] - coord1[key])
                
                return interpolated
                
        return None

    def create_chip_metadata(
        self,
        source_file: Union[str, Path],
        chip_path: Union[str, Path],
        face_bbox: Tuple[int, int, int, int],
        cluster_id: str,
        confidence: float,
        frame_number: Optional[int] = None,
        video_timestamp: Optional[float] = None,
        parent_id: Optional[str] = None,
        frame_specific_gps: bool = True,
        frame: Optional[np.ndarray] = None,
        gps_roi: Optional[Tuple[int, int, int, int]] = None
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
            frame_specific_gps: Whether to extract GPS for specific video timestamp
            frame: Video frame for OCR-based GPS extraction
            gps_roi: Region of interest for GPS overlay (x, y, width, height)
            
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
        gps_info = None
        
        # For videos with frame-specific GPS enabled, try to get GPS at the specific timestamp
        if (frame_specific_gps and video_timestamp is not None and 
            source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
            try:
                # First try OCR extraction if frame is provided
                if frame is not None and self.gps_ocr_extractor:
                    gps_info = self.extract_gps_from_frame_overlay(frame, gps_roi)
                    if gps_info:
                        logger.info(f"Using OCR-extracted GPS for timestamp {video_timestamp}s: {gps_info}")
                    else:
                        # Fallback to embedded GPS track
                        gps_info = self.get_gps_at_timestamp(source_path, video_timestamp)
                        if gps_info:
                            logger.debug(f"Using embedded GPS track for timestamp {video_timestamp}s: {gps_info}")
                else:
                    # No frame provided, use embedded GPS track
                    gps_info = self.get_gps_at_timestamp(source_path, video_timestamp)
                    if gps_info:
                        logger.debug(f"Using frame-specific GPS for timestamp {video_timestamp}s: {gps_info}")
            except Exception as e:
                logger.debug(f"Failed to get frame-specific GPS: {e}")
        
        # Fallback to general source metadata GPS
        if not gps_info:
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
    
    def extract_gps_from_frame_overlay(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[Dict[str, float]]:
        """
        Extract GPS coordinates from frame overlay using OCR
        
        Args:
            frame: Video frame as numpy array
            roi: Region of interest (x, y, width, height) for GPS overlay
            
        Returns:
            Dictionary with GPS coordinates or None if not found
        """
        if not self.gps_ocr_extractor:
            return None
            
        try:
            # Extract GPS using OCR
            gps_coords = self.gps_ocr_extractor.extract_gps_from_frame(frame, roi)
            
            if gps_coords:
                logger.debug(f"Extracted GPS from frame overlay: {gps_coords}")
                
            return gps_coords
            
        except Exception as e:
            logger.error(f"Error extracting GPS from frame overlay: {e}")
            return None
