import cv2
import numpy as np
from PIL import Image
import base64
import uuid
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Union, Dict, Any
from io import BytesIO
from src.utils.logger import get_logger, timing_decorator
from src.utils.config import get_config

logger = get_logger(__name__)


class ChipGenerator:
    """Generate face chips (cropped faces) from images with configurable padding and sizing"""
    
    def __init__(self, chip_size: Tuple[int, int] = None, padding_ratio: float = 0.2, 
                 jpeg_quality: int = None):
        """
        Initialize chip generator
        
        Args:
            chip_size: Output size (width, height). Default from config
            padding_ratio: Padding around face as ratio of face size (0.2 = 20%)
            jpeg_quality: JPEG compression quality (1-100). Default from config
        """
        config = get_config()
        output_config = config.get_output_config()
        
        self.chip_size = chip_size or tuple(output_config.get('chip_size', [224, 224]))
        self.padding_ratio = padding_ratio
        self.jpeg_quality = jpeg_quality or output_config.get('jpeg_quality', 85)
        self.use_base64 = output_config.get('use_base64', True)
        
        logger.info(f"Initialized ChipGenerator: size={self.chip_size}, "
                   f"padding={self.padding_ratio}, quality={self.jpeg_quality}")
    
    @timing_decorator
    def generate_chip(self, image: Union[np.ndarray, str, Path], 
                     face_bbox: Tuple[int, int, int, int],
                     output_path: Optional[Union[str, Path]] = None,
                     return_base64: Optional[bool] = None) -> Dict[str, Any]:
        """
        Generate face chip from image and bounding box
        
        Args:
            image: Input image (numpy array or path)
            face_bbox: Face bounding box (top, right, bottom, left)
            output_path: Optional path to save the chip
            return_base64: Override default base64 setting
            
        Returns:
            Dictionary with chip data and metadata
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        chip_name = f"face_chip_{timestamp}_{unique_id}"
        
        # Extract and process face chip
        face_chip = self._extract_face_with_padding(image, face_bbox)
        
        # Resize to standard size
        face_chip = self._resize_with_aspect_ratio(face_chip, self.chip_size)
        
        # Prepare result
        result = {
            'name': chip_name,
            'original_bbox': face_bbox,
            'chip_size': self.chip_size,
            'timestamp': datetime.now().isoformat()
        }
        
        # Handle output
        use_base64 = return_base64 if return_base64 is not None else self.use_base64
        
        if output_path:
            # Save to file
            output_path = Path(output_path)
            if output_path.suffix == '':
                output_path = output_path.with_suffix('.jpg')
            
            cv2.imwrite(str(output_path), face_chip, 
                       [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            result['file_path'] = str(output_path)
            logger.debug(f"Saved face chip to {output_path}")
        
        if use_base64 or not output_path:
            # Convert to base64
            result['base64'] = self._encode_base64(face_chip)
            result['encoding'] = 'base64'
        
        # Add chip array for further processing
        result['chip_array'] = face_chip
        
        return result
    
    def _extract_face_with_padding(self, image: np.ndarray, 
                                  bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face region with padding"""
        top, right, bottom, left = bbox
        height, width = image.shape[:2]
        
        # Calculate padding
        face_width = right - left
        face_height = bottom - top
        
        pad_x = int(face_width * self.padding_ratio)
        pad_y = int(face_height * self.padding_ratio)
        
        # Apply padding with bounds checking
        left_pad = max(0, left - pad_x)
        top_pad = max(0, top - pad_y)
        right_pad = min(width, right + pad_x)
        bottom_pad = min(height, bottom + pad_y)
        
        # Extract face region
        face_chip = image[top_pad:bottom_pad, left_pad:right_pad]
        
        if face_chip.size == 0:
            logger.warning(f"Empty face chip extracted from bbox {bbox}")
            # Return a small region as fallback
            face_chip = image[top:bottom, left:right]
        
        return face_chip
    
    def _resize_with_aspect_ratio(self, image: np.ndarray, 
                                 target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), 
                           interpolation=cv2.INTER_AREA)
        
        # Create canvas and center the image
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate centering offsets
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Place resized image on canvas
        canvas[y_offset:y_offset + new_height, 
               x_offset:x_offset + new_width] = resized
        
        return canvas
    
    def _encode_base64(self, image: np.ndarray) -> str:
        """Encode image to base64 string"""
        # Convert to RGB (PIL expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Save to bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=self.jpeg_quality)
        
        # Encode to base64
        image_bytes = buffer.getvalue()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        return base64_string
    
    def batch_generate(self, image: Union[np.ndarray, str, Path], 
                      face_bboxes: list, 
                      output_dir: Optional[Union[str, Path]] = None) -> list:
        """
        Generate chips for multiple faces in an image
        
        Args:
            image: Input image
            face_bboxes: List of face bounding boxes
            output_dir: Directory to save chips
            
        Returns:
            List of chip dictionaries
        """
        # Load image once
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image")
        
        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        chips = []
        for idx, bbox in enumerate(face_bboxes):
            try:
                output_path = None
                if output_dir:
                    output_path = output_dir / f"face_{idx:03d}.jpg"
                
                chip_data = self.generate_chip(image, bbox, output_path)
                chips.append(chip_data)
                
            except Exception as e:
                logger.error(f"Error generating chip for face {idx}: {e}")
                continue
        
        logger.info(f"Generated {len(chips)} face chips from {len(face_bboxes)} detections")
        return chips
    
    def update_config(self, chip_size: Optional[Tuple[int, int]] = None,
                     padding_ratio: Optional[float] = None,
                     jpeg_quality: Optional[int] = None):
        """Update generator configuration"""
        if chip_size:
            self.chip_size = chip_size
        if padding_ratio is not None:
            self.padding_ratio = padding_ratio
        if jpeg_quality:
            self.jpeg_quality = jpeg_quality
            
        logger.info(f"Updated ChipGenerator config: size={self.chip_size}, "
                   f"padding={self.padding_ratio}, quality={self.jpeg_quality}")