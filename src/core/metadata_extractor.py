import exifread
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from src.utils.logger import get_logger, timing_decorator

logger = get_logger(__name__)


class MetadataExtractor:
    """Extract metadata from images and videos including EXIF, GPS, and timestamps"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
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
        
        if extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return self.extract_image_metadata(file_path)
        elif extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
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
        gps_data = self._extract_gps_from_exif(metadata.get('exif', {}))
        if gps_data:
            metadata['gps'] = gps_data
            
        logger.debug(f"Extracted metadata for {image_path.name}: {list(metadata.keys())}")
        return metadata
    
    def extract_video_metadata(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from video file"""
        video_path = Path(video_path)
        metadata = self._get_basic_metadata(video_path)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if cap.isOpened():
                metadata['video'] = {
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
                    'codec': self._get_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
                }
                cap.release()
                
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")
            
        return metadata
    
    def _get_basic_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get basic file metadata"""
        stat = file_path.stat()
        
        return {
            'filename': file_path.name,
            'file_path': str(file_path.absolute()),
            'file_size': stat.st_size,
            'creation_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modification_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'timestamp': datetime.fromtimestamp(stat.st_mtime).isoformat()  # Default timestamp
        }
    
    def _extract_exif_with_exifread(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF data using exifread library"""
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
            exif_data = {}
            for tag, value in tags.items():
                if tag not in ['JPEGThumbnail', 'TIFFThumbnail']:
                    exif_data[tag] = str(value)
                    
            # Extract timestamp from EXIF
            timestamp = None
            for date_tag in ['EXIF DateTimeOriginal', 'EXIF DateTimeDigitized', 'Image DateTime']:
                if date_tag in exif_data:
                    try:
                        timestamp = datetime.strptime(
                            exif_data[date_tag], 
                            '%Y:%m:%d %H:%M:%S'
                        ).isoformat()
                        break
                    except:
                        pass
                        
            return {
                'exif': exif_data,
                'timestamp': timestamp or datetime.now().isoformat()
            }
            
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
            if 'Make' in exif_dict:
                camera_info['make'] = exif_dict['Make']
            if 'Model' in exif_dict:
                camera_info['model'] = exif_dict['Model']
            if 'LensModel' in exif_dict:
                camera_info['lens'] = exif_dict['LensModel']
                
            result = {}
            if camera_info:
                result['camera'] = camera_info
                
            return result
            
        except Exception as e:
            logger.debug(f"Could not extract EXIF with PIL: {e}")
            return {}
    
    def _extract_gps_from_exif(self, exif_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract GPS coordinates from EXIF data"""
        gps_tags = {
            'GPS GPSLatitude': 'lat',
            'GPS GPSLongitude': 'lon',
            'GPS GPSAltitude': 'alt',
            'GPS GPSLatitudeRef': 'lat_ref',
            'GPS GPSLongitudeRef': 'lon_ref'
        }
        
        gps_data = {}
        for exif_tag, gps_key in gps_tags.items():
            if exif_tag in exif_data:
                gps_data[gps_key] = exif_data[exif_tag]
                
        if 'lat' in gps_data and 'lon' in gps_data:
            try:
                # Convert GPS coordinates to decimal degrees
                lat = self._convert_to_degrees(gps_data['lat'])
                lon = self._convert_to_degrees(gps_data['lon'])
                
                # Apply hemisphere correction
                if gps_data.get('lat_ref') == 'S':
                    lat = -lat
                if gps_data.get('lon_ref') == 'W':
                    lon = -lon
                    
                result = {'lat': lat, 'lon': lon}
                
                # Add altitude if available
                if 'alt' in gps_data:
                    try:
                        result['altitude'] = float(str(gps_data['alt']).split('/')[0])
                    except:
                        pass
                        
                return result
                
            except Exception as e:
                logger.debug(f"Could not convert GPS coordinates: {e}")
                
        return None
    
    def _convert_to_degrees(self, value: str) -> float:
        """Convert GPS coordinate string to decimal degrees"""
        # Handle format like "[45, 30, 15.5]" or "45/1, 30/1, 155/10"
        value = str(value).strip('[]')
        parts = value.split(',')
        
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
        if '/' in value:
            num, den = value.split('/')
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
        if 'timestamp' in metadata and metadata['timestamp']:
            return metadata['timestamp']
        elif 'modification_time' in metadata:
            return metadata['modification_time']
        elif 'creation_time' in metadata:
            return metadata['creation_time']
        else:
            return datetime.now().isoformat()
    
    def get_gps_coordinates(self, metadata: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get GPS coordinates from metadata if available"""
        return metadata.get('gps', None)