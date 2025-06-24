import re
from typing import Dict, Optional, Tuple, Any
import numpy as np
import cv2
from src.utils.logger import get_facial_vision_logger

logger = get_facial_vision_logger(__name__)

# Try to import OCR libraries
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None
    logger.warning("EasyOCR not installed. GPS OCR extraction will be disabled.")

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    logger.warning("Pytesseract not installed. Falling back to other OCR methods if available.")


class GPSOCRExtractor:
    """Extract GPS coordinates from video overlays using OCR"""
    
    def __init__(self, ocr_backend: str = "auto"):
        """
        Initialize GPS OCR extractor
        
        Args:
            ocr_backend: OCR backend to use ("easyocr", "tesseract", "auto")
        """
        self.ocr_backend = ocr_backend
        self.reader = None
        
        # GPS coordinate patterns
        self.gps_patterns = [
            # Standard decimal format: 38.9549, -77.4119
            r'([+-]?\d{1,3}\.\d{3,6})[,\s]+([+-]?\d{1,3}\.\d{3,6})',
            # Without comma: 38.9549 -77.4119
            r'([+-]?\d{1,3}\.\d{3,6})\s+([+-]?\d{1,3}\.\d{3,6})',
            # With degree symbol: 38.9549° -77.4119°
            r'([+-]?\d{1,3}\.\d{3,6})°?\s*[,\s]+([+-]?\d{1,3}\.\d{3,6})°?',
            # N/S E/W format: 38.9549N 77.4119W
            r'(\d{1,3}\.\d{3,6})\s*([NS])\s*[,\s]+(\d{1,3}\.\d{3,6})\s*([EW])',
        ]
        
        # Initialize OCR backend
        self._initialize_ocr()
        
    def _initialize_ocr(self):
        """Initialize the OCR backend"""
        if self.ocr_backend == "auto":
            if EASYOCR_AVAILABLE:
                self.ocr_backend = "easyocr"
            elif TESSERACT_AVAILABLE:
                self.ocr_backend = "tesseract"
            else:
                logger.error("No OCR backend available. Please install easyocr or pytesseract.")
                self.ocr_backend = None
                return
        
        if self.ocr_backend == "easyocr" and EASYOCR_AVAILABLE:
            try:
                # Initialize EasyOCR reader for English
                self.reader = easyocr.Reader(['en'], gpu=True)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.reader = None
        elif self.ocr_backend == "tesseract" and TESSERACT_AVAILABLE:
            logger.info("Using Tesseract OCR")
        else:
            logger.error(f"OCR backend '{self.ocr_backend}' is not available")
            
    def extract_gps_from_frame(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[Dict[str, float]]:
        """
        Extract GPS coordinates from a video frame
        
        Args:
            frame: Video frame as numpy array
            roi: Region of interest (x, y, width, height) for GPS overlay location
            
        Returns:
            Dictionary with GPS coordinates or None if not found
        """
        if self.ocr_backend is None:
            return None
            
        try:
            # Extract region of interest if specified
            if roi:
                x, y, w, h = roi
                frame_roi = frame[y:y+h, x:x+w]
            else:
                # Default: check bottom portion of frame where GPS overlay typically appears
                height, width = frame.shape[:2]
                # Bottom 20% of the frame
                y_start = int(height * 0.8)
                frame_roi = frame[y_start:, :]
            
            # Preprocess the image for better OCR
            processed_frame = self._preprocess_for_ocr(frame_roi)
            
            # Extract text using OCR
            text = self._extract_text(processed_frame)
            
            if not text:
                return None
                
            # Parse GPS coordinates from text
            gps_coords = self._parse_gps_coordinates(text)
            
            return gps_coords
            
        except Exception as e:
            logger.error(f"Error extracting GPS from frame: {e}")
            return None
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get white text on black background
        # GPS overlays are typically white text
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Denoise
        denoised = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        
        # Resize for better OCR (if image is too small)
        height, width = denoised.shape
        if height < 50:
            scale_factor = 50 / height
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            denoised = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return denoised
    
    def _extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image: Preprocessed image
            
        Returns:
            Extracted text
        """
        text = ""
        
        if self.ocr_backend == "easyocr" and self.reader:
            try:
                results = self.reader.readtext(image, detail=0)
                text = " ".join(results)
            except Exception as e:
                logger.error(f"EasyOCR failed: {e}")
                
        elif self.ocr_backend == "tesseract" and TESSERACT_AVAILABLE:
            try:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(image)
                # Use Tesseract with custom config for better number recognition
                custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.-+,NSEWnsew° '
                text = pytesseract.image_to_string(pil_image, config=custom_config)
            except Exception as e:
                logger.error(f"Tesseract failed: {e}")
        
        return text.strip()
    
    def _parse_gps_coordinates(self, text: str) -> Optional[Dict[str, float]]:
        """
        Parse GPS coordinates from extracted text
        
        Args:
            text: Extracted text from OCR
            
        Returns:
            Dictionary with lat/lon or None if not found
        """
        if not text:
            return None
            
        # Clean up text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        logger.debug(f"OCR extracted text: {text}")
        
        # Try each GPS pattern
        for pattern in self.gps_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    groups = match.groups()
                    
                    # Handle N/S E/W format
                    if len(groups) == 4:  # N/S E/W format
                        lat = float(groups[0])
                        lat_dir = groups[1].upper()
                        lon = float(groups[2])
                        lon_dir = groups[3].upper()
                        
                        # Apply hemisphere corrections
                        if lat_dir == 'S':
                            lat = -lat
                        if lon_dir == 'W':
                            lon = -lon
                            
                        return {'lat': lat, 'lon': lon}
                    
                    # Handle decimal format
                    elif len(groups) == 2:
                        lat = float(groups[0])
                        lon = float(groups[1])
                        
                        # Validate coordinates
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            return {'lat': lat, 'lon': lon}
                        else:
                            logger.debug(f"Invalid coordinates: lat={lat}, lon={lon}")
                            
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse GPS from match: {e}")
                    continue
        
        return None
    
    def detect_gps_overlay_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Automatically detect the region containing GPS overlay
        
        Args:
            frame: Video frame
            
        Returns:
            ROI tuple (x, y, width, height) or None if not detected
        """
        height, width = frame.shape[:2]
        
        # Common regions for GPS overlays
        regions_to_check = [
            # Bottom right
            (int(width * 0.6), int(height * 0.8), int(width * 0.4), int(height * 0.2)),
            # Bottom left
            (0, int(height * 0.8), int(width * 0.4), int(height * 0.2)),
            # Bottom center
            (int(width * 0.3), int(height * 0.8), int(width * 0.4), int(height * 0.2)),
            # Top right
            (int(width * 0.6), 0, int(width * 0.4), int(height * 0.2)),
        ]
        
        for roi in regions_to_check:
            # Try to extract GPS from this region
            gps = self.extract_gps_from_frame(frame, roi)
            if gps:
                logger.info(f"GPS overlay detected in region: {roi}")
                return roi
                
        return None